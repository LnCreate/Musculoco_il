"""Record joint torques (generalized forces) during a rollout.

This script is intended for LocoMuJoCo + MushroomRL agents used in this repo.
It loads a `.msh` checkpoint, runs deterministic/stochastic rollouts, and
extracts MuJoCo generalized forces for selected joints (hip/knee/ankle).

By default we record `qfrc_actuator` (actuator-generated generalized forces).
Optionally, we can also record inverse-dynamics generalized forces
(`qfrc_inverse`) at the same joints via `--record-id`.

Example:
    cd /Users/ccg/Study/Mujoco/IL/musculoco_learning/experiments/eval
    conda activate musculoco_paper
    python record_joint_torques.py \
        --checkpoint ../04_latent_with_kl_obj/logs/train_long_resume_2025-12-22_17-50-00/0/agent_epoch_169_J_952.371816.msh \
        --episodes 1 --max-steps 1000 --deterministic \
    --outdir ./runs/run_001 --record-video --record-ctrl \
    --out torques_and_ctrl.npz --csv torques_and_ctrl.csv

    python record_joint_torques.py \
        --checkpoint /Users/ccg/Study/Mujoco/IL/musculoco_learning/experiments/04_latent_with_kl_obj/logs/best_walk1222/0/best_walk.msh \
        --env-id HumanoidMuscle.walk \
        --episodes 3 --max-steps 1000 --no-absorbing --joint-set both_10 \
        --record-ctrl --record-video --record-act \
        --headless\
        --outdir ./runs/walk_001 \
        --out torques_and_ctrl.npz --csv torques_and_ctrl.csv

    python record_joint_torques.py \
        --checkpoint /Users/ccg/Study/Mujoco/IL/musculoco_learning/experiments/04_latent_with_kl_obj/logs/best_run1223/0/best_run.msh\
        --env-id HumanoidMuscle.run \
        --episodes 3 --max-steps 1000 --no-absorbing --joint-set both_10 \
        --record-ctrl --record-video --record-act --headless \
        --outdir ./runs/run_001 \
        --out torques_and_ctrl.npz --csv torques_and_ctrl.csv 
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import mujoco
from mushroom_rl.core.serialization import Serializable
from loco_mujoco import LocoEnv


DEFAULT_JOINT_NAMES = [
    # Right
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    # Left
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
]


JOINT_SETS = {
    # Main sagittal joints (one DoF each)
    "r_leg_3": ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"],
    "l_leg_3": ["hip_flexion_l", "knee_angle_l", "ankle_angle_l"],
    # Default full set used previously (kept for backward compatibility)
    "both_10": DEFAULT_JOINT_NAMES,
    # Full complex feet set (14 DoF)
    "both_14": DEFAULT_JOINT_NAMES + ["subtalar_angle_r", "mtp_angle_r", "subtalar_angle_l", "mtp_angle_l"],
}


@dataclass(frozen=True)
class JointDofSlice:
    joint_name: str
    joint_id: int
    qpos_adr: int
    qpos_num: int
    dof_adr: int
    dof_num: int


def _list_joint_names(model: mujoco.MjModel) -> List[str]:
    names: List[str] = []
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n is None:
            n = f"<joint_{j}>"
        names.append(n)
    return names


def _list_actuator_names(model: mujoco.MjModel) -> List[str]:
    names: List[str] = []
    for i in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if n is None:
            n = f"<actuator_{i}>"
        names.append(n)
    return names


def _build_act_state_names(model: mujoco.MjModel, actuator_names: Sequence[str]) -> List[str]:
    """Build human-readable names for each element in MjData.act.

    MuJoCo maps actuator activation state into data.act using:
      - model.actuator_actadr (start index per actuator)
      - model.actuator_actnum (number of activation states per actuator)

    For typical muscle models, actnum is often 1, so we name it as:
      act:<actuator_name>
    For multi-dimensional actuator activation, we name it as:
      act:<actuator_name>:k
    """

    act_dim = int(getattr(model, "na", 0))
    if act_dim <= 0:
        return []

    names: List[str] = [f"act:{i}" for i in range(act_dim)]

    if not hasattr(model, "actuator_actadr") or not hasattr(model, "actuator_actnum"):
        return names

    for i in range(int(getattr(model, "nu", 0))):
        try:
            adr = int(model.actuator_actadr[i])
            num = int(model.actuator_actnum[i])
        except Exception:
            continue

        if num <= 0:
            continue

        base = actuator_names[i] if i < len(actuator_names) else f"<actuator_{i}>"
        for k in range(num):
            idx = adr + k
            if 0 <= idx < act_dim:
                if num == 1:
                    names[idx] = f"act:{base}"
                else:
                    names[idx] = f"act:{base}:{k}"

    # Ensure uniqueness (duplicate actuator names can exist in some models)
    seen: Dict[str, int] = {}
    for i, n in enumerate(names):
        if n in seen:
            seen[n] += 1
            names[i] = f"{n}__{seen[n]}"
        else:
            seen[n] = 0

    return names


def _try_get_joint_id(model: mujoco.MjModel, name: str) -> int:
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    except Exception:
        return -1


def _resolve_joint_name(model: mujoco.MjModel, requested: str) -> Tuple[str, int]:
    """Resolve a requested joint name to an existing MuJoCo joint name + id.

    Tries exact match first. If not found, attempts simple normalization.
    """

    joint_id = _try_get_joint_id(model, requested)
    if joint_id != -1:
        return requested, joint_id

    # Common observation-key prefixes
    cleaned = requested
    for prefix in ("q_", "dq_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break

    if cleaned != requested:
        joint_id = _try_get_joint_id(model, cleaned)
        if joint_id != -1:
            return cleaned, joint_id

    # Heuristic search
    all_names = _list_joint_names(model)
    lowered = cleaned.lower()

    # 1) contains full cleaned
    candidates = [n for n in all_names if lowered in (n or "").lower()]

    # 2) token contains
    if not candidates:
        tokens = [t for t in lowered.replace("-", "_").split("_") if t]
        if tokens:
            candidates = [
                n
                for n in all_names
                if all(tok in (n or "").lower() for tok in tokens)
            ]

    if len(candidates) == 1:
        resolved = candidates[0]
        joint_id = _try_get_joint_id(model, resolved)
        return resolved, joint_id

    raise ValueError(
        "Could not resolve joint name. "
        f"requested='{requested}', cleaned='{cleaned}'. "
        f"Candidates={candidates[:10]}{'...' if len(candidates) > 10 else ''}"
    )


def _build_joint_slices(model: mujoco.MjModel, joint_names: Sequence[str]) -> List[JointDofSlice]:
    def _dof_num_from_joint_type(jnt_type: int) -> int:
        # mjJNT_FREE=0 (6 DoF), mjJNT_BALL=1 (3 DoF), mjJNT_SLIDE=2 (1 DoF), mjJNT_HINGE=3 (1 DoF)
        if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
            return 6
        if jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
            return 3
        if jnt_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            return 1
        if jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            return 1
        raise ValueError(f"Unknown MuJoCo joint type id: {jnt_type}")

    def _qpos_num_from_joint_type(jnt_type: int) -> int:
        # mjJNT_FREE=0 (7), mjJNT_BALL=1 (4), mjJNT_SLIDE=2 (1), mjJNT_HINGE=3 (1)
        if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
            return 7
        if jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
            return 4
        if jnt_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            return 1
        if jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            return 1
        raise ValueError(f"Unknown MuJoCo joint type id: {jnt_type}")

    slices: List[JointDofSlice] = []
    for requested in joint_names:
        resolved, joint_id = _resolve_joint_name(model, requested)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {requested} (resolved={resolved})")

        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos_num = _qpos_num_from_joint_type(int(model.jnt_type[joint_id]))
        dof_adr = int(model.jnt_dofadr[joint_id])
        # Some MuJoCo python builds don't expose model.jnt_dofnum; infer from joint type.
        dof_num = _dof_num_from_joint_type(int(model.jnt_type[joint_id]))
        if dof_num <= 0:
            raise ValueError(f"Joint has no DoFs: {resolved}")

        slices.append(JointDofSlice(resolved, joint_id, qpos_adr, qpos_num, dof_adr, dof_num))
    return slices


def _flatten_columns(joint_slices: Sequence[JointDofSlice], field: str = "qfrc") -> Tuple[List[str], np.ndarray]:
    """Return column names and a (n_cols, 2) array of [start, end) dof/pos indices."""
    col_names: List[str] = []
    ranges: List[Tuple[int, int]] = []
    
    prefix = ""
    if field == "qpos":
        prefix = "q:"
    elif field == "qvel":
        prefix = "dq:"
    elif field == "qfrc":
        prefix = "" # Keep original behavior for torques
    
    for sl in joint_slices:
        if field == "qpos":
            adr, num = sl.qpos_adr, sl.qpos_num
        else:
            adr, num = sl.dof_adr, sl.dof_num
            
        if num == 1:
            col_names.append(f"{prefix}{sl.joint_name}")
            ranges.append((adr, adr + 1))
        else:
            for k in range(num):
                col_names.append(f"{prefix}{sl.joint_name}:{k}")
                ranges.append((adr + k, adr + k + 1))
    return col_names, np.asarray(ranges, dtype=np.int32)


def _get_qfrc_fields(data: mujoco.MjData, field: str) -> np.ndarray:
    if not hasattr(data, field):
        raise ValueError(
            f"MjData has no field '{field}'. "
            "Try: qfrc_actuator, qfrc_passive, qfrc_applied, qfrc_constraint, qfrc_bias"
        )
    arr = getattr(data, field)
    return np.asarray(arr, dtype=np.float64)


def _get_inverse_dynamics_qfrc(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Compute and return inverse-dynamics generalized forces (qfrc_inverse).

    MuJoCo uses current (qpos, qvel, qacc) in `data`. We call `mj_inverse`
    and then read `data.qfrc_inverse`.
    """
    mujoco.mj_inverse(model, data)
    return np.asarray(data.qfrc_inverse, dtype=np.float64)


def _write_csv(path: str, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str, help="Path to .msh agent checkpoint")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1000, help="Max env steps per episode")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--no-absorbing",
        action="store_true",
        help="Disable absorbing terminations (try to run full max-steps for more gait cycles).",
    )
    p.add_argument(
        "--qfrc-field",
        type=str,
        default="qfrc_actuator",
        help="Which MjData generalized-force field to record (default: qfrc_actuator)",
    )
    p.add_argument(
        "--record-id",
        action="store_true",
        help="Also record inverse-dynamics generalized forces (qfrc_inverse) for selected joints.",
    )
    p.add_argument(
        "--record-ctrl",
        action="store_true",
        help="Record MuJoCo control signals (data.ctrl) for all actuators (muscle activations).",
    )
    p.add_argument(
        "--record-pos",
        action="store_true",
        help="Record joint positions (angles) for the selected joints.",
    )
    p.add_argument(
        "--record-vel",
        action="store_true",
        help="Record joint velocities for the selected joints.",
    )
    p.add_argument(
        "--record-actions",
        action="store_true",
        help="Record the agent action vector passed into env.step (normalized action space).",
    )
    p.add_argument(
        "--record-act",
        action="store_true",
        help=(
            "Record MuJoCo actuator activation state (data.act). Note: this is only meaningful if the model has "
            "activation dynamics (model.na > 0)."
        ),
    )
    p.add_argument(
        "--joint-set",
        type=str,
        default=None,
        choices=sorted(JOINT_SETS.keys()),
        help=(
            "Preset joint groups. If set, overrides the default joint list unless you also pass --joint. "
            "Examples: r_leg_3 (hip/knee/ankle right), l_leg_3, both_10."
        ),
    )
    p.add_argument(
        "--joint",
        action="append",
        default=None,
        help="MuJoCo joint name to record (repeatable). If not set, uses a default hip/knee/ankle list.",
    )
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Directory to store outputs for this run. If set, relative --out/--csv will be placed inside it. "
            "If --record-video is enabled and --outdir is not set, a timestamped folder under ./recordings is created."
        ),
    )
    p.add_argument(
        "--record-video",
        action="store_true",
        help="Record an MP4 video for the rollout into the same output folder.",
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Video FPS (default: ctrl_freq, i.e. one frame per env.step).",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Create the environment in headless mode (useful when recording video without opening a window).",
    )

    # Env args: keep defaults consistent with pol_runner.py
    p.add_argument("--env-id", type=str, default="HumanoidMuscle.walk")
    p.add_argument("--env-freq", type=int, default=1000)
    p.add_argument("--ctrl-freq", type=int, default=100)
    p.add_argument(
        "--use-box-feet",
        action="store_true",
        help="Use simplified box feet. If False (default), uses complex feet (44 obs dim).",
    )

    p.add_argument("--out", type=str, default=None, help="Output .npz path")
    p.add_argument("--csv", type=str, default=None, help="Optional output .csv path")

    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_substeps = args.env_freq // args.ctrl_freq
    mdp = LocoEnv.make(
        args.env_id,
        headless=bool(args.headless),
        gamma=0.99,
        horizon=args.max_steps,
        n_substeps=n_substeps,
        timestep=1 / args.env_freq,
        use_absorbing_states=not args.no_absorbing,
        use_box_feet=bool(args.use_box_feet),
        use_foot_forces=False,
        obs_mujoco_act=False,
        muscle_force_scaling=1.25,
        alpha_box_feet=0.5,
    )

    # Add passive stiffness to unactuated foot joints to match training
    if not args.use_box_feet:
        print("Restoring passive stiffness to foot joints (subtalar, mtp)...")
        foot_joints = ['subtalar_angle_l', 'subtalar_angle_r', 'mtp_angle_l', 'mtp_angle_r']
        for j_name in foot_joints:
            try:
                j_id = mdp._model.joint(j_name).id
                mdp._model.jnt_stiffness[j_id] = 30.0 
                mdp._model.dof_damping[mdp._model.jnt_dofadr[j_id]] = 2.0
            except Exception as e:
                 print(f"Warning: Could not set stiffness for {j_name}: {e}")

    agent = Serializable.load(args.checkpoint)
    agent.policy.deterministic = bool(args.deterministic)

    # Resolve joints
    if args.joint is not None:
        joint_names = args.joint
    elif args.joint_set is not None:
        joint_names = JOINT_SETS[args.joint_set]
    else:
        # Default behavior: use all 14 joints if complex feet are enabled, else 10
        if not args.use_box_feet:
            joint_names = JOINT_SETS["both_14"]
        else:
            joint_names = DEFAULT_JOINT_NAMES
    
    print(f"Recording {len(joint_names)} joints: {', '.join(joint_names)}")
    joint_slices = _build_joint_slices(mdp._model, joint_names)
    col_names, dof_ranges = _flatten_columns(joint_slices, field="qfrc")

    pos_col_names, pos_ranges = ([], np.zeros((0, 2), dtype=np.int32))
    if args.record_pos:
        pos_col_names, pos_ranges = _flatten_columns(joint_slices, field="qpos")

    vel_col_names, vel_ranges = ([], np.zeros((0, 2), dtype=np.int32))
    if args.record_vel:
        vel_col_names, vel_ranges = _flatten_columns(joint_slices, field="qvel")

    # Output paths
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir: Optional[Path]
    if args.outdir is not None:
        run_dir = Path(args.outdir)
    elif args.record_video:
        run_dir = Path("./recordings") / stamp
    else:
        run_dir = None

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    out_npz = args.out or (f"joint_torques_{stamp}.npz" if run_dir is None else "torques_and_ctrl.npz")
    if run_dir is not None and not os.path.isabs(out_npz):
        out_npz = str(run_dir / out_npz)

    out_csv = args.csv
    if out_csv is not None and run_dir is not None and not os.path.isabs(out_csv):
        out_csv = str(run_dir / out_csv)

    actuator_names = _list_actuator_names(mdp._model) if (args.record_ctrl or args.record_actions or args.record_act) else None
    act_dim = int(getattr(mdp._model, "na", 0))
    act_names = (
        _build_act_state_names(mdp._model, actuator_names or [])
        if args.record_act and act_dim > 0
        else []
    )

    # Optional video recorder
    recorder = None
    if args.record_video:
        from mushroom_rl.utils.record import VideoRecorder

        # VideoRecorder stores videos at path/tag/<video_name>.mp4
        # We want the video inside run_dir.
        if run_dir is None:
            raise RuntimeError("Internal error: run_dir must be set when --record-video is enabled.")

        fps = float(args.video_fps) if args.video_fps is not None else float(args.ctrl_freq)
        recorder = VideoRecorder(
            path=str(run_dir.parent),
            tag=run_dir.name,
            video_name="recording",
            fps=fps,
        )

    # Rollout + record
    all_rows: List[Dict[str, float]] = []

    torques_list: List[np.ndarray] = []
    id_torques_list: List[np.ndarray] = []
    pos_list: List[np.ndarray] = []
    vel_list: List[np.ndarray] = []
    times_list: List[np.ndarray] = []
    episode_index_list: List[np.ndarray] = []
    step_index_list: List[np.ndarray] = []

    ctrl_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []

    try:
        for ep in range(args.episodes):
            obs = mdp.reset()

            if recorder is not None:
                frame = mdp.render(record=True)
                recorder(frame)

            ep_t: List[float] = []
            ep_tau: List[np.ndarray] = []
            ep_tau_id: List[np.ndarray] = []
            ep_pos: List[np.ndarray] = []
            ep_vel: List[np.ndarray] = []
            ep_steps: List[int] = []

            ep_ctrl: List[np.ndarray] = []
            ep_act: List[np.ndarray] = []
            ep_act_state: List[np.ndarray] = []

            for t in range(args.max_steps):
                action = agent.draw_action(obs)
                obs, reward, absorbing, info = mdp.step(action)

                if recorder is not None:
                    frame = mdp.render(record=True)
                    recorder(frame)

                # Record AFTER stepping: mdp._data contains the new state.
                qfrc = _get_qfrc_fields(mdp._data, args.qfrc_field)

                # Pull requested dofs
                vals = np.zeros((len(col_names),), dtype=np.float64)
                for i, (start, end) in enumerate(dof_ranges):
                    vals[i] = float(qfrc[start:end].reshape(-1)[0])

                id_vals = None
                if args.record_id:
                    qfrc_inv = _get_inverse_dynamics_qfrc(mdp._model, mdp._data)
                    id_vals = np.zeros((len(col_names),), dtype=np.float64)
                    for i, (start, end) in enumerate(dof_ranges):
                        id_vals[i] = float(qfrc_inv[start:end].reshape(-1)[0])

                ep_t.append(float(mdp._data.time))
                ep_tau.append(vals)
                if id_vals is not None:
                    ep_tau_id.append(id_vals)
                ep_steps.append(t)

                if args.record_pos:
                    pvals = np.zeros((len(pos_col_names),), dtype=np.float64)
                    for i, (start, end) in enumerate(pos_ranges):
                        pvals[i] = float(mdp._data.qpos[start:end].reshape(-1)[0])
                    ep_pos.append(pvals)

                if args.record_vel:
                    vvals = np.zeros((len(vel_col_names),), dtype=np.float64)
                    for i, (start, end) in enumerate(vel_ranges):
                        vvals[i] = float(mdp._data.qvel[start:end].reshape(-1)[0])
                    ep_vel.append(vvals)

                if args.record_ctrl:
                    ep_ctrl.append(np.asarray(mdp._data.ctrl, dtype=np.float64).copy())
                if args.record_actions:
                    ep_act.append(np.asarray(action, dtype=np.float64).reshape(-1).copy())
                if args.record_act:
                    if act_dim > 0:
                        ep_act_state.append(np.asarray(mdp._data.act, dtype=np.float64).reshape(-1).copy())
                    else:
                        # Keep shapes consistent even if model.na == 0
                        ep_act_state.append(np.zeros((0,), dtype=np.float64))

                if out_csv is not None:
                    row: Dict[str, float] = {
                        "episode": float(ep),
                        "step": float(t),
                        "time": float(mdp._data.time),
                    }
                    for k, name in enumerate(col_names):
                        row[name] = float(vals[k])
                    if args.record_id and id_vals is not None:
                        for k, name in enumerate(col_names):
                            row[f"id:{name}"] = float(id_vals[k])
                    if args.record_pos:
                        for k, name in enumerate(pos_col_names):
                            row[name] = float(ep_pos[-1][k])
                    if args.record_vel:
                        for k, name in enumerate(vel_col_names):
                            row[name] = float(ep_vel[-1][k])
                    if args.record_ctrl:
                        for i, aname in enumerate(actuator_names or []):
                            row[f"ctrl:{aname}"] = float(mdp._data.ctrl[i])
                    if args.record_actions:
                        # If action dim == nu, map by actuator; else just index.
                        a_flat = np.asarray(action, dtype=np.float64).reshape(-1)
                        if actuator_names is not None and len(a_flat) == len(actuator_names):
                            for i, aname in enumerate(actuator_names):
                                row[f"action:{aname}"] = float(a_flat[i])
                        else:
                            for i in range(len(a_flat)):
                                row[f"action:{i}"] = float(a_flat[i])
                    if args.record_act and act_dim > 0:
                        astate = np.asarray(mdp._data.act, dtype=np.float64).reshape(-1)
                        for i in range(min(act_dim, astate.shape[0])):
                            key = act_names[i] if i < len(act_names) else f"act:{i}"
                            row[key] = float(astate[i])
                    all_rows.append(row)

                if absorbing:
                    break

            torques_list.append(np.stack(ep_tau, axis=0) if ep_tau else np.zeros((0, len(col_names))))
            if args.record_id:
                id_torques_list.append(np.stack(ep_tau_id, axis=0) if ep_tau_id else np.zeros((0, len(col_names))))
            if args.record_pos:
                pos_list.append(np.stack(ep_pos, axis=0) if ep_pos else np.zeros((0, len(pos_col_names))))
            if args.record_vel:
                vel_list.append(np.stack(ep_vel, axis=0) if ep_vel else np.zeros((0, len(vel_col_names))))
            times_list.append(np.asarray(ep_t, dtype=np.float64))
            episode_index_list.append(np.full((len(ep_t),), ep, dtype=np.int32))
            step_index_list.append(np.asarray(ep_steps, dtype=np.int32))

            if args.record_ctrl:
                ctrl_list.append(np.stack(ep_ctrl, axis=0) if ep_ctrl else np.zeros((0, mdp._model.nu)))
            if args.record_actions:
                if ep_act:
                    action_list.append(np.stack(ep_act, axis=0))
                else:
                    action_list.append(np.zeros((0, 0), dtype=np.float64))
            if args.record_act:
                if ep_act_state:
                    act_list.append(np.stack(ep_act_state, axis=0))
                else:
                    act_list.append(np.zeros((0, act_dim), dtype=np.float64))

    finally:
        try:
            mdp.stop()
        except Exception:
            pass
        if recorder is not None:
            recorder.stop()

    # Concatenate episodes (variable lengths)
    torques = np.concatenate(torques_list, axis=0) if torques_list else np.zeros((0, len(col_names)))
    id_torques = (
        np.concatenate(id_torques_list, axis=0)
        if args.record_id and id_torques_list
        else np.zeros((torques.shape[0], 0), dtype=np.float64)
    )
    pos = (
        np.concatenate(pos_list, axis=0)
        if args.record_pos and pos_list
        else np.zeros((torques.shape[0], 0), dtype=np.float64)
    )
    vel = (
        np.concatenate(vel_list, axis=0)
        if args.record_vel and vel_list
        else np.zeros((torques.shape[0], 0), dtype=np.float64)
    )
    times = np.concatenate(times_list, axis=0) if times_list else np.zeros((0,), dtype=np.float64)
    ep_idx = np.concatenate(episode_index_list, axis=0) if episode_index_list else np.zeros((0,), dtype=np.int32)
    step_idx = np.concatenate(step_index_list, axis=0) if step_index_list else np.zeros((0,), dtype=np.int32)

    ctrl = (
        np.concatenate(ctrl_list, axis=0)
        if args.record_ctrl and ctrl_list
        else np.zeros((torques.shape[0], 0), dtype=np.float64)
    )
    actions = (
        np.concatenate(action_list, axis=0)
        if args.record_actions and action_list
        else np.zeros((torques.shape[0], 0), dtype=np.float64)
    )
    act = (
        np.concatenate(act_list, axis=0)
        if args.record_act and act_list
        else np.zeros((torques.shape[0], 0), dtype=np.float64)
    )

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(
        out_npz,
        checkpoint=args.checkpoint,
        env_id=args.env_id,
        qfrc_field=args.qfrc_field,
        columns=np.asarray(col_names, dtype=object),
        torques=torques,
        id_columns=np.asarray([f"id:{c}" for c in col_names], dtype=object) if args.record_id else np.asarray([], dtype=object),
        id_torques=id_torques,
        qpos=pos,
        qvel=vel,
        pos_columns=np.asarray(pos_col_names, dtype=object),
        vel_columns=np.asarray(vel_col_names, dtype=object),
        actuator_names=np.asarray(actuator_names, dtype=object) if actuator_names is not None else np.asarray([], dtype=object),
        ctrl=ctrl,
        actions=actions,
        act=act,
        act_names=np.asarray(act_names, dtype=object),
        act_dim=np.asarray([act_dim], dtype=np.int32),
        time=times,
        episode=ep_idx,
        step=step_idx,
        joint_names=np.asarray([s.joint_name for s in joint_slices], dtype=object),
        joint_dofadr=np.asarray([s.dof_adr for s in joint_slices], dtype=np.int32),
        joint_dofnum=np.asarray([s.dof_num for s in joint_slices], dtype=np.int32),
    )

    if out_csv is not None:
        fieldnames = ["episode", "step", "time"] + col_names
        if args.record_id:
            fieldnames += [f"id:{n}" for n in col_names]
        if args.record_pos:
            fieldnames += pos_col_names
        if args.record_vel:
            fieldnames += vel_col_names
        if args.record_ctrl:
            fieldnames += [f"ctrl:{n}" for n in (actuator_names or [])]
        if args.record_actions:
            # Prefer naming by actuator if same length.
            if actuator_names is not None and actions.shape[1] == len(actuator_names):
                fieldnames += [f"action:{n}" for n in actuator_names]
            else:
                fieldnames += [f"action:{i}" for i in range(actions.shape[1])]
        if args.record_act and act_dim > 0:
            fieldnames += list(act_names) if act_names else [f"act:{i}" for i in range(act_dim)]
        _write_csv(out_csv, all_rows, fieldnames)

    print(f"Saved NPZ: {out_npz}")
    if out_csv is not None:
        print(f"Saved CSV: {out_csv}")
    if recorder is not None and run_dir is not None:
        print(f"Saved MP4: {str(run_dir / 'recording.mp4')}")
        if (run_dir / 'recording-1.mp4').exists():
            print("Note: multiple MP4 segments exist (recording-1.mp4, ...)")
    msg = f"Recorded {torques.shape[0]} steps; torque_columns={len(col_names)}; field={args.qfrc_field}"
    if args.record_pos:
        msg += f"; pos_dim={pos.shape[1]}"
    if args.record_id:
        msg += f"; id_dim={id_torques.shape[1]}"
    if args.record_vel:
        msg += f"; vel_dim={vel.shape[1]}"
    if args.record_ctrl:
        msg += f"; ctrl_dim={ctrl.shape[1]}"
    if args.record_actions:
        msg += f"; action_dim={actions.shape[1]}"
    if args.record_act:
        msg += f"; act_dim={act.shape[1]}"
    if args.no_absorbing:
        msg += "; absorbing_disabled=True"
    print(msg)


if __name__ == "__main__":
    main()
