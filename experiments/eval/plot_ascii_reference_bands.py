"""Plot paper-style reference bands from ASCII files.

Current default behavior:
- Single side only (L or R)
- Aggregate all participants, all speeds, all trials
- Draw gray mean±std error bands only (no per-speed mean curves)
- 6x3 panel layout similar to the paper figure
"""

"""
uv run python experiments/eval/plot_ascii_reference_bands.py \
--layout sagittal --ascii-root ASCII-files --side L \
--overlay-band-npz runs/walk_004_id/bands_from_csv_id/bands_summary.npz \
--model-mass-kg 120 --auto-align-phase \
--ref-plot trials \
--metrics-out runs/ascii_reference/sagittal_metrics_allspeeds_L.csv \
--out runs/ascii_reference/paper_style_single_side_all_trials_ALLSPEEDS_L.png
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ANGLE_LAYOUT: List[Tuple[str, str, str]] = [
    ("St1_Ankle_X", "St1_Ankle_Y", "St1_Ankle_Z"),
    ("St1_Knee_X", "St1_Knee_Y", "St1_Knee_Z"),
    ("St1_Hip_X", "St1_Hip_Y", "St1_Hip_Z"),
]

TORQUE_LAYOUT: List[Tuple[str, str, str]] = [
    ("St1_Ankle_X", "St1_Ankle_Y", "St1_Ankle_Z"),
    ("St1_Knee_X", "St1_Knee_Y", "St1_Knee_Z"),
    ("St1_Hip_X", "St1_Hip_Y", "St1_Hip_Z"),
]

ROW_LABELS_ANGLE = ["Ankle angle", "Knee angle", "Hip angle"]
ROW_LABELS_TORQUE = ["Ankle torque", "Knee torque", "Hip torque"]
COL_TITLES = ["Sagittal", "Coronal", "Transversal"]

SAGITTAL_ROW_TO_JOINT = {
    0: "ankle_angle",
    1: "knee_angle",
    2: "hip_flexion",
}

SAGITTAL_ROW_TO_NAME = {
    0: "ankle",
    1: "knee",
    2: "hip",
}

# (angle/torque row index, col index) -> NPZ joint name, or None if model has no DOF.
# Col 0 = Sagittal, Col 1 = Coronal, Col 2 = Transversal
# Ankle/Knee are hinge joints (1 DOF) in this model; Hip has 3 DOFs.
ROW_COL_TO_MODEL_JOINT: Dict[Tuple[int, int], Optional[str]] = {
    (0, 0): "ankle_angle",   # Ankle sagittal (flexion/extension)
    (0, 1): None,            # Ankle coronal  – subtalar not modelled
    (0, 2): None,            # Ankle transversal – not modelled
    (1, 0): "knee_angle",    # Knee sagittal  (flexion/extension)
    (1, 1): None,            # Knee coronal   – hinge joint, no DOF
    (1, 2): None,            # Knee transversal – not modelled
    (2, 0): "hip_flexion",   # Hip sagittal   (flexion/extension)
    (2, 1): "hip_adduction", # Hip coronal    (adduction/abduction)
    (2, 2): "hip_rotation",  # Hip transversal (internal/external rotation)
}


def resample_to_101(y: np.ndarray) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(y))
    x_new = np.linspace(0.0, 1.0, 101)
    return np.interp(x_new, x_old, y)


def collect_files(root: Path, signal: str, side: str, speed_folder: str = "") -> List[Path]:
    # signal in {"Angles", "Torques"}, side in {"L", "R"}
    if speed_folder:
        return sorted(root.glob(f"Participant*/Processed_Data/{speed_folder}/{side}/{signal}/T*.txt"))
    return sorted(root.glob(f"Participant*/Processed_Data/V*/{side}/{signal}/T*.txt"))


def _extract_participant_name(file_path: Path) -> str:
    # .../ASCII-files/ParticipantX/Processed_Data/V*/L/Torques/T1.txt
    for p in file_path.parents:
        if p.name.lower().startswith("participant"):
            return p.name
    return ""


def _read_participant_mass_kg(ascii_root: Path, participant_name: str) -> float:
    meta = ascii_root / participant_name / "Metadata.txt"
    if not meta.exists():
        raise FileNotFoundError(f"Metadata not found for {participant_name}: {meta}")

    txt = meta.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Body\s*Mass\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*kg", txt, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Could not parse body mass from {meta}")
    return float(m.group(1))


def load_column_trials(
    files: List[Path],
    col: str,
    *,
    ascii_root: Path,
    normalize_by_mass: bool = False,
) -> np.ndarray:
    trials = []
    mass_cache: Dict[str, float] = {}

    for f in files:
        try:
            df = pd.read_csv(f, sep=r"\s+|\t+", engine="python")
            if col not in df.columns:
                continue
            y = df[col].to_numpy(dtype=float)
            if len(y) < 5 or np.any(~np.isfinite(y)):
                continue

            if normalize_by_mass:
                pname = _extract_participant_name(f)
                if pname not in mass_cache:
                    mass_cache[pname] = _read_participant_mass_kg(ascii_root, pname)
                mass = mass_cache[pname]
                if mass <= 0:
                    continue
                y = y / mass

            trials.append(resample_to_101(y))
        except Exception:
            continue

    if not trials:
        return np.empty((0, 101), dtype=float)
    return np.vstack(trials)


def band_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return mean, std


def plot_gray_band(ax, arr: np.ndarray):
    if arr.shape[0] == 0:
        return
    mean, std = band_stats(arr)
    phase = np.linspace(0, 100, 101)
    ax.fill_between(phase, mean - std, mean + std, color="0.7", alpha=0.7, linewidth=0)


def plot_minmax_envelope(ax, arr: np.ndarray):
    if arr.shape[0] == 0:
        return
    phase = np.linspace(0, 100, 101)
    lo = np.min(arr, axis=0)
    hi = np.max(arr, axis=0)
    ax.fill_between(phase, lo, hi, color="0.7", alpha=0.7, linewidth=0)


def plot_all_trials(ax, arr: np.ndarray):
    if arr.shape[0] == 0:
        return
    phase = np.linspace(0, 100, 101)
    # Draw all gait cycles as thin gray lines (paper-like background cloud)
    for i in range(arr.shape[0]):
        ax.plot(phase, arr[i], color="0.75", linewidth=0.4, alpha=0.18)


def plot_reference(ax, arr: np.ndarray, mode: str):
    if mode == "std":
        plot_gray_band(ax, arr)
    elif mode == "minmax":
        plot_minmax_envelope(ax, arr)
    elif mode == "trials":
        plot_all_trials(ax, arr)
    elif mode == "trials_minmax":
        plot_all_trials(ax, arr)
        plot_minmax_envelope(ax, arr)
    else:
        raise ValueError(f"Unknown reference plot mode: {mode}")


def maybe_overlay_model_from_npz(
    ax,
    npz,
    *,
    signal_group: str,
    joint_name: str,
    side: str,
    color: str,
    line_label: Optional[str],
    torque_mass_kg: float,
    phase_shift_pct: float = 0.0,
    hip_torque_scale: float = 1.0,
    hip_torque_offset: float = 0.0,
    ref_mean: Optional[np.ndarray] = None,
    auto_fit_overlay: bool = False,
):
    side_key = "left" if side.upper() == "L" else "right"
    mean_key = f"{signal_group}/Joint {'Angle' if signal_group=='angles' else 'Torque'}:::{joint_name}/{side_key}_mean"

    band_key = f"{signal_group}/Joint {'Angle' if signal_group=='angles' else 'Torque'}:::{joint_name}/{side_key}_band"

    if mean_key not in npz:
        return

    y = np.asarray(npz[mean_key], dtype=float)
    b = np.asarray(npz[band_key], dtype=float) if band_key in npz else None
    if signal_group == "angles":
        # model angles are in radians -> degrees
        y = np.rad2deg(y)
        if b is not None:
            b = np.rad2deg(b)
        # Knee angle sign convention alignment with reference dataset
        if joint_name == "knee_angle":
            y = -y
    else:
        # Normalize model torques to N·m/kg
        y = y / float(torque_mass_kg)
        if b is not None:
            b = b / float(torque_mass_kg)
        # Knee torque sign convention alignment with reference dataset
        if joint_name == "knee_angle":
            y = -y
        # Optional adjustment for hip torque scale/offset (for visualization mapping)
        if joint_name == "hip_flexion":
            y = y * float(hip_torque_scale) + float(hip_torque_offset)

    if abs(float(phase_shift_pct)) > 1e-12:
        # 101 points => 1 point per 1% gait cycle in our pipeline
        shift_idx = int(np.round(float(phase_shift_pct) * (len(y) - 1) / 100.0))
        y = np.roll(y, shift_idx)
        if b is not None:
            b = np.roll(b, shift_idx)

    # Optional auto-fit: minimize SSD to the reference mean by applying scale + offset, or just offset.
    if auto_fit_overlay and ref_mean is not None and len(y) == len(ref_mean):
        # Find linear transformation: y_new = scale * y + offset ~ ref_mean
        A = np.vstack([y, np.ones(len(y))]).T
        m, c = np.linalg.lstsq(A, ref_mean, rcond=None)[0]
        y = m * y + c
        if b is not None:
            b = abs(m) * b
        print(f"[Auto-fit] {signal_group} {joint_name}: applied scale={m:.3f}, offset={c:.3f}")

    phase = np.asarray(npz["phase_pct"], dtype=float) if "phase_pct" in npz else np.linspace(0, 100, len(y))
    if b is not None and b.shape == y.shape:
        ax.fill_between(phase, y - b, y + b, color=color, alpha=0.18, linewidth=0)
    ax.plot(phase, y, color=color, lw=2.2, label=line_label)


def _extract_model_angle_for_alignment(npz, side: str, joint_name: str) -> np.ndarray:
    side_key = "left" if side.upper() == "L" else "right"
    key = f"angles/Joint Angle:::{joint_name}/{side_key}_mean"
    if key not in npz:
        return np.array([])
    y = np.asarray(npz[key], dtype=float)
    y = np.rad2deg(y)
    if joint_name == "knee_angle":
        y = -y
    return y


def _estimate_best_phase_shift_pct(overlay_npz, side: str, ref_sagittal_means: Dict[int, np.ndarray]) -> float:
    """Estimate phase shift (0..99%) that best aligns model sagittal angles to reference means."""
    model = {
        r: _extract_model_angle_for_alignment(overlay_npz, side, SAGITTAL_ROW_TO_JOINT[r])
        for r in [0, 1, 2]
    }

    best_shift = 0
    best_score = np.inf

    for shift in range(100):
        score = 0.0
        used = 0
        for r in [0, 1, 2]:
            if r not in ref_sagittal_means:
                continue
            y_ref = np.asarray(ref_sagittal_means[r], dtype=float)
            y_mod = np.asarray(model[r], dtype=float)
            if y_ref.size == 0 or y_mod.size == 0 or y_ref.shape != y_mod.shape:
                continue

            y_mod_s = np.roll(y_mod, shift)

            # z-score shape matching (ignore absolute scale)
            y_ref_z = (y_ref - np.mean(y_ref)) / (np.std(y_ref) + 1e-8)
            y_mod_z = (y_mod_s - np.mean(y_mod_s)) / (np.std(y_mod_s) + 1e-8)
            score += float(np.mean((y_ref_z - y_mod_z) ** 2))
            used += 1

        if used > 0 and score < best_score:
            best_score = score
            best_shift = shift

    return float(best_shift)


def _build_stance_mask(
    phase_pct: np.ndarray,
    *,
    mode: str,
    stance_start_pct: float,
    stance_end_pct: float,
    ref_ankle_torque_mean: np.ndarray,
    torque_threshold_ratio: float,
) -> np.ndarray:
    if mode == "phase":
        start = float(stance_start_pct)
        end = float(stance_end_pct)
        if end < start:
            start, end = end, start
        return (phase_pct >= start) & (phase_pct <= end)

    if mode == "torque":
        y = np.asarray(ref_ankle_torque_mean, dtype=float)
        if y.size != phase_pct.size:
            return np.zeros_like(phase_pct, dtype=bool)
        amp = float(np.max(np.abs(y)))
        if amp <= 1e-12:
            return np.zeros_like(phase_pct, dtype=bool)
        thr = float(torque_threshold_ratio) * amp
        return np.abs(y) >= thr

    raise ValueError(f"Unsupported stance mask mode: {mode}")


def _weighted_rmse(err: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    e = np.asarray(err, dtype=float)
    if w.shape != e.shape:
        return float("nan")
    denom = float(np.sum(w))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sqrt(np.sum(w * (e ** 2)) / denom))


def _compute_sagittal_torque_metrics(
    overlay_npz,
    side: str,
    ref_torque_sagittal_means: Dict[int, np.ndarray],
    *,
    torque_mass_kg: float,
    phase_shift_pct: float,
    stance_mask_mode: str,
    stance_start_pct: float,
    stance_end_pct: float,
    stance_weight: float,
    swing_weight: float,
    torque_threshold_ratio: float,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
    phase_pct = np.asarray(overlay_npz["phase_pct"], dtype=float) if "phase_pct" in overlay_npz else np.linspace(0, 100, 101)

    # Build reference-based stance mask (using ankle sagittal torque by default).
    ankle_ref = np.asarray(ref_torque_sagittal_means.get(0, np.array([])), dtype=float)
    stance_mask = _build_stance_mask(
        phase_pct,
        mode=str(stance_mask_mode),
        stance_start_pct=float(stance_start_pct),
        stance_end_pct=float(stance_end_pct),
        ref_ankle_torque_mean=ankle_ref,
        torque_threshold_ratio=float(torque_threshold_ratio),
    )
    if stance_mask.size != phase_pct.size:
        stance_mask = np.zeros_like(phase_pct, dtype=bool)
    swing_mask = ~stance_mask

    side_key = "left" if side.upper() == "L" else "right"
    metrics: Dict[str, Dict[str, float]] = {}

    for row_idx in [0, 1, 2]:
        joint = SAGITTAL_ROW_TO_JOINT[row_idx]
        display = SAGITTAL_ROW_TO_NAME[row_idx]
        key = f"torques/Joint Torque:::{joint}/{side_key}_mean"
        if key not in overlay_npz or row_idx not in ref_torque_sagittal_means:
            continue

        y_model = np.asarray(overlay_npz[key], dtype=float) / float(torque_mass_kg)
        y_ref = np.asarray(ref_torque_sagittal_means[row_idx], dtype=float)
        if joint == "knee_angle":
            y_model = -y_model

        if abs(float(phase_shift_pct)) > 1e-12:
            shift_idx = int(np.round(float(phase_shift_pct) * (len(y_model) - 1) / 100.0))
            y_model = np.roll(y_model, shift_idx)

        if y_model.shape != y_ref.shape or y_model.size != phase_pct.size:
            continue

        err = y_model - y_ref
        w = np.where(stance_mask, float(stance_weight), float(swing_weight))

        rmse_all = float(np.sqrt(np.mean(err ** 2)))
        rmse_stance = float(np.sqrt(np.mean((err[stance_mask]) ** 2))) if np.any(stance_mask) else float("nan")
        rmse_swing = float(np.sqrt(np.mean((err[swing_mask]) ** 2))) if np.any(swing_mask) else float("nan")
        wrmse = _weighted_rmse(err, w)

        metrics[display] = {
            "rmse_all": rmse_all,
            "rmse_stance": rmse_stance,
            "rmse_swing": rmse_swing,
            "rmse_weighted": wrmse,
        }

    return metrics, stance_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ascii-root", type=str, default="ASCII-files")
    p.add_argument("--out", type=str, default="runs/ascii_reference/paper_style_single_side_gray_band.png")
    p.add_argument("--side", type=str, default="L", choices=["L", "R", "l", "r"], help="Side to plot")
    p.add_argument(
        "--speed-folder",
        type=str,
        default="",
        help="Optional speed folder filter for reference data (e.g., V1, V15, V2, V25, V3, V35, V4). Empty means all speeds.",
    )
    p.add_argument(
        "--overlay-band-npz",
        type=str,
        default="",
        help="Optional model bands_summary.npz to overlay (e.g., runs/walk_004/bands/bands_summary.npz)",
    )
    p.add_argument("--model-mass-kg", type=float, default=70.0, help="Model body mass for torque normalization (N·m/kg)")
    p.add_argument("--hip-torque-scale", type=float, default=1.0, help="Visualization scalar for hip torque overlay")
    p.add_argument("--hip-torque-offset", type=float, default=0.0, help="Visualization offset for hip torque overlay")
    p.add_argument("--phase-shift-pct", type=float, default=0.0, help="Manual phase shift for overlay curve (in gait %).")
    p.add_argument("--auto-align-phase", action="store_true", help="Auto-estimate overlay phase shift from sagittal angle shape matching.")
    p.add_argument("--auto-fit-overlay", action="store_true", help="Automatically calculate and apply optimal scale/offset to match reference means.")
    p.add_argument("--no-torque-normalize", action="store_true", help="Disable torque normalization by body mass (default: normalize).")
    p.add_argument(
        "--ref-plot",
        type=str,
        default="std",
        choices=["std", "minmax", "trials", "trials_minmax"],
        help="Reference visualization: std(mean±std), minmax envelope, all trials, or all trials + minmax.",
    )
    p.add_argument(
        "--stance-mask-mode",
        type=str,
        default="phase",
        choices=["phase", "torque"],
        help="How to define stance mask for torque metrics: phase range or ref ankle-torque threshold.",
    )
    p.add_argument("--stance-start-pct", type=float, default=0.0, help="Stance phase start (%%), used when --stance-mask-mode=phase")
    p.add_argument("--stance-end-pct", type=float, default=60.0, help="Stance phase end (%%), used when --stance-mask-mode=phase")
    p.add_argument("--stance-weight", type=float, default=1.0, help="Weight for stance samples in weighted RMSE")
    p.add_argument("--swing-weight", type=float, default=0.3, help="Weight for swing samples in weighted RMSE")
    p.add_argument(
        "--torque-threshold-ratio",
        type=float,
        default=0.2,
        help="Threshold ratio for --stance-mask-mode=torque, i.e., |ankle_torque| >= ratio * max(|ankle_torque|).",
    )
    p.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional CSV path to save sagittal torque metrics (all/stance/swing/weighted RMSE).",
    )
    p.add_argument(
        "--layout",
        type=str,
        default="sagittal",
        choices=["sagittal", "full"],
        help="'sagittal': 6x1 plot of primary DOFs only; 'full': original 6x3 panel layout.",
    )
    args = p.parse_args()

    root = Path(args.ascii_root)
    side = args.side.upper()

    angle_files = collect_files(root, "Angles", side, speed_folder=str(args.speed_folder))
    normalize_torque = not bool(args.no_torque_normalize)
    torque_signal = "Torques_Norm" if normalize_torque else "Torques"
    torque_files = collect_files(root, torque_signal, side, speed_folder=str(args.speed_folder))

    if len(angle_files) == 0 or len(torque_files) == 0:
        raise RuntimeError("No ASCII angle/torque files found. Check --ascii-root path.")

    overlay_npz = None
    if args.overlay_band_npz:
        overlay_npz = np.load(args.overlay_band_npz)

    layout = str(args.layout)
    # Which columns to load: sagittal mode only needs col 0, but we always load col 0
    # for auto-alignment even in full mode. Load all cols only when needed.
    cols_to_load = [0] if layout == "sagittal" else [0, 1, 2]

    # Preload trials so we can compute reference sagittal means for auto-alignment
    angle_trials: Dict[Tuple[int, int], np.ndarray] = {}
    torque_trials: Dict[Tuple[int, int], np.ndarray] = {}
    for r in range(3):
        for c in cols_to_load:
            angle_trials[(r, c)] = load_column_trials(
                angle_files,
                ANGLE_LAYOUT[r][c],
                ascii_root=root,
                normalize_by_mass=False,
            )
            torque_trials[(r, c)] = load_column_trials(
                torque_files,
                TORQUE_LAYOUT[r][c],
                ascii_root=root,
                normalize_by_mass=False,
            )

    if layout == "sagittal":
        # 2 rows × 3 cols: row 0 = angles, row 1 = torques; cols = ankle / knee / hip
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        col_titles_to_show = [ROW_LABELS_ANGLE[r] for r in range(3)]   # column headers = joint names
    else:
        fig, axes = plt.subplots(6, 3, figsize=(12, 18), sharex=True)
        col_titles_to_show = COL_TITLES

    ref_sagittal_means: Dict[int, np.ndarray] = {}
    for r in range(3):
        arr = angle_trials[(r, 0)]
        if arr.shape[0] > 0:
            ref_sagittal_means[r] = arr.mean(axis=0)

    ref_torque_sagittal_means: Dict[int, np.ndarray] = {}
    for r in range(3):
        arr_t = torque_trials[(r, 0)]
        if arr_t.shape[0] > 0:
            ref_torque_sagittal_means[r] = arr_t.mean(axis=0)

    phase_shift_pct = float(args.phase_shift_pct)
    if overlay_npz is not None and args.auto_align_phase:
        auto_shift = _estimate_best_phase_shift_pct(overlay_npz, side, ref_sagittal_means)
        phase_shift_pct += auto_shift
        print(f"Auto phase shift estimated: {auto_shift:.1f}%")
        print(f"Total applied phase shift: {phase_shift_pct:.1f}%")

    if overlay_npz is not None:
        metrics, stance_mask = _compute_sagittal_torque_metrics(
            overlay_npz,
            side,
            ref_torque_sagittal_means,
            torque_mass_kg=float(args.model_mass_kg),
            phase_shift_pct=float(phase_shift_pct),
            stance_mask_mode=str(args.stance_mask_mode),
            stance_start_pct=float(args.stance_start_pct),
            stance_end_pct=float(args.stance_end_pct),
            stance_weight=float(args.stance_weight),
            swing_weight=float(args.swing_weight),
            torque_threshold_ratio=float(args.torque_threshold_ratio),
        )

        if metrics:
            print("Sagittal torque RMSE metrics (N·m/kg):")
            for joint_name in ["ankle", "knee", "hip"]:
                if joint_name not in metrics:
                    continue
                m = metrics[joint_name]
                print(
                    f"  {joint_name:>5s} | all={m['rmse_all']:.4f}, stance={m['rmse_stance']:.4f}, "
                    f"swing={m['rmse_swing']:.4f}, weighted={m['rmse_weighted']:.4f}"
                )

            if args.metrics_out:
                out_csv = Path(args.metrics_out)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for joint_name in ["ankle", "knee", "hip"]:
                    if joint_name in metrics:
                        rows.append({"joint": joint_name, **metrics[joint_name]})
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"Saved metrics: {out_csv}")

        stance_ratio = float(np.mean(stance_mask.astype(float))) if stance_mask.size > 0 else float("nan")
        print(f"Stance mask mode: {args.stance_mask_mode} | stance ratio: {stance_ratio:.3f}")

    if layout == "sagittal":
        # 2×3 grid: row 0 = angles, row 1 = torques; col = ankle/knee/hip
        torque_unit = "N·m/kg" if normalize_torque else "N·m"
        for col_idx in range(3):
            joint = SAGITTAL_ROW_TO_JOINT[col_idx]

            # --- angle ---
            ax_a = axes[0, col_idx]
            arr_a = angle_trials[(col_idx, 0)]
            plot_reference(ax_a, arr_a, mode=str(args.ref_plot))
            ax_a.set_title(col_titles_to_show[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax_a.set_ylabel("Angle [deg]")
            ax_a.grid(True, alpha=0.25)
            if overlay_npz is not None:
                ref_mean_a = arr_a.mean(axis=0) if arr_a.shape[0] > 0 else None
                maybe_overlay_model_from_npz(
                    ax_a, overlay_npz,
                    signal_group="angles", joint_name=joint, side=side,
                    color="tab:blue",
                    line_label="Your model (mean)" if col_idx == 0 else None,
                    torque_mass_kg=args.model_mass_kg,
                    phase_shift_pct=phase_shift_pct,
                    ref_mean=ref_mean_a,
                    auto_fit_overlay=args.auto_fit_overlay,
                )

            # --- torque ---
            ax_t = axes[1, col_idx]
            arr_t = torque_trials[(col_idx, 0)]
            plot_reference(ax_t, arr_t, mode=str(args.ref_plot))
            if col_idx == 0:
                ax_t.set_ylabel(f"Torque [{torque_unit}]")
            ax_t.set_xlabel("Gait Cycle (%)")
            ax_t.grid(True, alpha=0.25)
            if overlay_npz is not None:
                ref_mean_t = arr_t.mean(axis=0) if arr_t.shape[0] > 0 else None
                maybe_overlay_model_from_npz(
                    ax_t, overlay_npz,
                    signal_group="torques", joint_name=joint, side=side,
                    color="tab:blue", line_label=None,
                    torque_mass_kg=args.model_mass_kg,
                    phase_shift_pct=phase_shift_pct,
                    hip_torque_scale=args.hip_torque_scale,
                    hip_torque_offset=args.hip_torque_offset,
                    ref_mean=ref_mean_t,
                    auto_fit_overlay=args.auto_fit_overlay,
                )

        if overlay_npz is not None:
            axes[0, 0].legend(loc="best")

    else:
        # Original 6×3 full layout
        render_cols = [0, 1, 2]

        # Top 3 rows: angles
        for r in range(3):
            for c in render_cols:
                ax = axes[r, c]
                arr = angle_trials[(r, c)]
                plot_reference(ax, arr, mode=str(args.ref_plot))
                if r == 0 and col_titles_to_show is not None:
                    ax.set_title(col_titles_to_show[c], fontsize=11, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"{ROW_LABELS_ANGLE[r]} [deg]")
                ax.grid(True, alpha=0.25)
                if overlay_npz is not None:
                    joint = ROW_COL_TO_MODEL_JOINT.get((r, c))
                    if joint is not None:
                        maybe_overlay_model_from_npz(
                            ax, overlay_npz,
                            signal_group="angles", joint_name=joint, side=side,
                            color="tab:blue",
                            line_label="Your model (mean)" if (r == 0 and c == 0) else None,
                            torque_mass_kg=args.model_mass_kg,
                            phase_shift_pct=phase_shift_pct,
                        )

        # Bottom 3 rows: torques
        torque_unit = "N·m/kg" if normalize_torque else "N·m"
        for r in range(3):
            for c in render_cols:
                ax = axes[r + 3, c]
                arr = torque_trials[(r, c)]
                plot_reference(ax, arr, mode=str(args.ref_plot))
                if c == 0:
                    ax.set_ylabel(f"{ROW_LABELS_TORQUE[r]} [{torque_unit}]")
                ax.grid(True, alpha=0.25)
                if overlay_npz is not None:
                    joint = ROW_COL_TO_MODEL_JOINT.get((r, c))
                    if joint is not None:
                        maybe_overlay_model_from_npz(
                            ax, overlay_npz,
                            signal_group="torques", joint_name=joint, side=side,
                            color="tab:blue", line_label=None,
                            torque_mass_kg=args.model_mass_kg,
                            phase_shift_pct=phase_shift_pct,
                            hip_torque_scale=args.hip_torque_scale,
                            hip_torque_offset=args.hip_torque_offset,
                        )

        for c in render_cols:
            axes[5, c].set_xlabel("Gait Cycle (%)")

        if overlay_npz is not None:
            axes[0, 0].legend(loc="best")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)

    print(f"Side: {side}")
    print(f"Reference speed filter: {args.speed_folder if args.speed_folder else 'ALL'}")
    print(f"Angle files: {len(angle_files)}")
    print(f"Torque files: {len(torque_files)}")
    print(f"Torque source: {torque_signal}")
    print(f"Torque normalized by mass: {normalize_torque}")
    print(f"Reference plot mode: {args.ref_plot}")
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
