import time
from argparse import ArgumentParser

import numpy as np
import torch
import mujoco

from mushroom_rl.core.serialization import Serializable
from loco_mujoco import LocoEnv


np.random.seed(5011)
torch.manual_seed(5011)


def _collect_visual_groups(mdp):
    model = mdp._model

    default_rgba = np.asarray(model.geom_rgba, dtype=np.float64).copy()
    default_mat_rgba = np.asarray(model.mat_rgba, dtype=np.float64).copy() if model.nmat > 0 else None
    default_mat_specular = np.asarray(model.mat_specular, dtype=np.float64).copy() if model.nmat > 0 else None
    default_mat_shininess = np.asarray(model.mat_shininess, dtype=np.float64).copy() if model.nmat > 0 else None
    default_mat_reflectance = np.asarray(model.mat_reflectance, dtype=np.float64).copy() if model.nmat > 0 else None
    default_tex_rgb = np.asarray(model.tex_rgb, dtype=np.uint8).copy() if model.ntex > 0 else None
    default_light_diffuse = np.asarray(model.light_diffuse, dtype=np.float64).copy() if model.nlight > 0 else None
    default_light_ambient = np.asarray(model.light_ambient, dtype=np.float64).copy() if model.nlight > 0 else None
    default_light_specular = np.asarray(model.light_specular, dtype=np.float64).copy() if model.nlight > 0 else None
    body_geom_ids = []
    hidden_box_geom_ids = []
    floor_geom_ids = []
    floor_mat_ids = []
    floor_tex_ids = []

    for mid in range(model.nmat):
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mid) or ""
        mname_l = mname.lower()
        if "plane" in mname_l or "ground" in mname_l or "floor" in mname_l:
            floor_mat_ids.append(mid)
            texid = int(model.mat_texid[mid])
            if texid >= 0:
                floor_tex_ids.append(texid)

    skybox_tex_ids = []
    for tid in range(model.ntex):
        if int(model.tex_type[tid]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            skybox_tex_ids.append(tid)

    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        gname_l = gname.lower()

        if "foot_box" in gname_l or "bofoot" in gname_l:
            hidden_box_geom_ids.append(gid)

        if gname_l == "floor" or "ground" in gname_l or "plane" in gname_l:
            floor_geom_ids.append(gid)
            matid = int(model.geom_matid[gid]) if hasattr(model, "geom_matid") else -1
            if matid >= 0:
                floor_mat_ids.append(matid)
                texid = int(model.mat_texid[matid]) if matid < model.nmat else -1
                if texid >= 0:
                    floor_tex_ids.append(texid)

        bid = int(model.geom_bodyid[gid])
        if bid != 0:
            body_geom_ids.append(gid)

    return {
        "default_rgba": default_rgba,
        "default_mat_rgba": default_mat_rgba,
        "default_mat_specular": default_mat_specular,
        "default_mat_shininess": default_mat_shininess,
        "default_mat_reflectance": default_mat_reflectance,
        "default_tex_rgb": default_tex_rgb,
        "default_light_diffuse": default_light_diffuse,
        "default_light_ambient": default_light_ambient,
        "default_light_specular": default_light_specular,
        "body_geom_ids": np.asarray(sorted(set(body_geom_ids)), dtype=np.int32),
        "hidden_box_geom_ids": np.asarray(sorted(set(hidden_box_geom_ids)), dtype=np.int32),
        "floor_geom_ids": np.asarray(sorted(set(floor_geom_ids)), dtype=np.int32),
        "floor_mat_ids": np.asarray(sorted(set(floor_mat_ids)), dtype=np.int32),
        "floor_tex_ids": np.asarray(sorted(set(floor_tex_ids)), dtype=np.int32),
        "skybox_tex_ids": np.asarray(sorted(set(skybox_tex_ids)), dtype=np.int32),
    }


def _apply_soft_lighting(mdp, groups, *, light_scale=0.45, ambient_scale=0.35, specular_scale=0.06):
    model = mdp._model

    if model.nlight > 0 and groups.get("default_light_diffuse") is not None:
        model.light_diffuse[:] = np.clip(groups["default_light_diffuse"] * float(light_scale), 0.0, 1.0)
        model.light_ambient[:] = np.clip(groups["default_light_ambient"] * float(ambient_scale), 0.0, 1.0)
        model.light_specular[:] = np.clip(groups["default_light_specular"] * float(specular_scale), 0.0, 1.0)

    if model.nmat > 0 and groups.get("default_mat_specular") is not None:
        model.mat_specular[:] = np.clip(groups["default_mat_specular"] * float(specular_scale), 0.0, 1.0)
        model.mat_shininess[:] = np.clip(groups["default_mat_shininess"] * 0.2, 0.0, 1.0)
        model.mat_reflectance[:] = np.clip(groups["default_mat_reflectance"] * 0.15, 0.0, 1.0)


def _parse_rgba(s):
    if s is None or str(s).strip() == "":
        return None
    txt = str(s).replace(",", " ").strip()
    vals = [float(x) for x in txt.split() if x]
    if len(vals) != 4:
        raise ValueError("--body-rgba must contain 4 numbers: r g b a")
    arr = np.asarray(vals, dtype=np.float64)
    return np.clip(arr, 0.0, 1.0)


def _apply_visual_overrides(mdp, groups, hide_box_feet=True, body_rgba=None):
    model = mdp._model

    rgba = np.asarray(groups["default_rgba"], dtype=np.float64).copy()

    if body_rgba is not None:
        for gid in groups["body_geom_ids"]:
            rgba[gid, :] = body_rgba

    if hide_box_feet:
        for gid in groups["hidden_box_geom_ids"]:
            rgba[gid, 3] = 0.0

    model.geom_rgba[:] = rgba


def _apply_white_theme(mdp, groups, *, white_floor=True, black_floor=False, white_background=True):
    model = mdp._model

    if white_floor or black_floor:
        floor_rgb = 0.0 if black_floor else 1.0
        floor_tex_rgb = 0 if black_floor else 255

        if model.nmat > 0 and groups.get("default_mat_rgba") is not None:
            mat_rgba = np.asarray(model.mat_rgba, dtype=np.float64).copy()
            mat_spec = np.asarray(model.mat_specular, dtype=np.float64).copy()
            mat_shiny = np.asarray(model.mat_shininess, dtype=np.float64).copy()
            mat_refl = np.asarray(model.mat_reflectance, dtype=np.float64).copy()
            for mid in groups.get("floor_mat_ids", []):
                mat_rgba[int(mid), :] = np.array([floor_rgb, floor_rgb, floor_rgb, 1.0], dtype=np.float64)
                mat_spec[int(mid)] = 0.0
                mat_shiny[int(mid)] = 0.0
                mat_refl[int(mid)] = 0.0
            model.mat_rgba[:] = mat_rgba
            model.mat_specular[:] = mat_spec
            model.mat_shininess[:] = mat_shiny
            model.mat_reflectance[:] = mat_refl

        rgba = np.asarray(model.geom_rgba, dtype=np.float64).copy()
        for gid in groups.get("floor_geom_ids", []):
            rgba[int(gid), :] = np.array([floor_rgb, floor_rgb, floor_rgb, 1.0], dtype=np.float64)
        model.geom_rgba[:] = rgba

        if model.ntex > 0 and groups.get("default_tex_rgb") is not None:
            for tid in groups.get("floor_tex_ids", []):
                tid = int(tid)
                if tid < 0 or tid >= model.ntex:
                    continue
                adr = int(model.tex_adr[tid])
                tex_h = int(model.tex_height[tid])
                tex_w = int(model.tex_width[tid])
                tex_c = int(model.tex_nchannel[tid]) if hasattr(model, "tex_nchannel") else 3
                tex_size = tex_h * tex_w * tex_c
                if tex_size > 0:
                    model.tex_rgb[adr:adr + tex_size] = floor_tex_rgb

        if model.nlight > 0:
            try:
                model.light_castshadow[:] = 0
            except Exception:
                pass
            try:
                model.vis.quality.shadowsize = 0
            except Exception:
                pass

    if white_background:
        if model.ntex > 0 and groups.get("default_tex_rgb") is not None:
            for tid in groups.get("skybox_tex_ids", []):
                tid = int(tid)
                if tid < 0 or tid >= model.ntex:
                    continue
                adr = int(model.tex_adr[tid])
                tex_h = int(model.tex_height[tid])
                tex_w = int(model.tex_width[tid])
                tex_c = int(model.tex_nchannel[tid]) if hasattr(model, "tex_nchannel") else 3
                tex_size = tex_h * tex_w * tex_c
                if tex_size > 0:
                    model.tex_rgb[adr:adr + tex_size] = 255

        try:
            model.vis.rgba.fog[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            model.vis.rgba.haze[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            model.vis.map.haze = 0.0
            model.vis.map.fogstart = 1e5
            model.vis.map.fogend = 1e5 + 1.0
        except Exception:
            pass


def _apply_flat_render_flags(mdp, *, disable_reflection=True, disable_shadow=True,
                             disable_skybox=True, disable_fog=True):
    viewer = getattr(mdp, "_viewer", None)
    if viewer is None:
        return

    scene = getattr(viewer, "_scene", None)
    if scene is None or not hasattr(scene, "flags"):
        return

    try:
        if disable_reflection:
            scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        if disable_shadow:
            scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        if disable_skybox:
            scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        if disable_fog:
            scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    except Exception:
        pass


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename", required=True, help="Agent checkpoint (.msh)")
    parser.add_argument("--env-id", type=str, default="HumanoidMuscle.walk")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--ctrl-freq", type=int, default=100)
    parser.add_argument("--hide-box-feet", action="store_true", help="Hide box-feet geoms in viewer")
    parser.add_argument("--body-rgba", type=str, default="", help="Optional override color for all rigid-body geoms: 'r g b a' (0~1)")
    parser.add_argument("--white-floor", action="store_true", help="Set floor geom color to pure white")
    parser.add_argument("--black-floor", action="store_true", help="Set floor geom color to pure black")
    parser.add_argument("--white-background", action="store_true", help="Try to force white scene background (fog/haze settings)")
    parser.add_argument("--dim-light", action="store_true", help="Reduce lighting intensity and suppress specular highlights")
    parser.add_argument("--light-scale", type=float, default=0.45, help="Scale for diffuse lights when --dim-light is enabled")
    parser.add_argument("--ambient-scale", type=float, default=0.35, help="Scale for ambient lights when --dim-light is enabled")
    parser.add_argument("--specular-scale", type=float, default=0.06, help="Scale for specular lights/materials when --dim-light is enabled")
    args = parser.parse_args()

    env_freq = 1000
    n_substeps = env_freq // int(args.ctrl_freq)

    mdp = LocoEnv.make(
        args.env_id,
        gamma=0.99,
        horizon=1000,
        n_substeps=n_substeps,
        timestep=1 / env_freq,
        use_box_feet=True,
        use_foot_forces=False,
        obs_mujoco_act=False,
        muscle_force_scaling=1.25,
        alpha_box_feet=0.5,
    )

    agent = Serializable.load(args.filename)
    agent.policy.deterministic = bool(args.deterministic)

    print(f"Action Dim: {mdp.info.action_space.shape[0]}")
    print(f"State Dim : {mdp.info.observation_space.shape}")
    print(f"use_box_feet internal: True | hide_box_feet visual: {bool(args.hide_box_feet)}")
    if args.white_floor and args.black_floor:
        print("Both --white-floor and --black-floor set; using black floor.")

    body_rgba = _parse_rgba(args.body_rgba)

    groups = _collect_visual_groups(mdp)

    try:
        for ep in range(int(args.episodes)):
            obs = mdp.reset()

            if args.hide_box_feet or body_rgba is not None:
                _apply_visual_overrides(
                    mdp,
                    groups,
                    hide_box_feet=bool(args.hide_box_feet),
                    body_rgba=body_rgba,
                )

            floor_theme = bool(args.black_floor) or bool(args.white_floor and not args.black_floor)

            if floor_theme or args.white_background:
                _apply_white_theme(
                    mdp,
                    groups,
                    white_floor=bool(args.white_floor and not args.black_floor),
                    black_floor=bool(args.black_floor),
                    white_background=bool(args.white_background),
                )

            if args.dim_light:
                _apply_soft_lighting(
                    mdp,
                    groups,
                    light_scale=float(args.light_scale),
                    ambient_scale=float(args.ambient_scale),
                    specular_scale=float(args.specular_scale),
                )

            frame = mdp.render(record=True)
            if floor_theme or args.white_background:
                _apply_flat_render_flags(
                    mdp,
                    disable_reflection=True,
                    disable_shadow=True,
                    disable_skybox=bool(args.white_background),
                    disable_fog=True,
                )

            start = time.time()
            for t in range(int(args.max_steps)):
                action = agent.draw_action(obs)
                obs, reward, absorbing, info = mdp.step(action)

                frame = mdp.render(record=True)
                if floor_theme or args.white_background:
                    _apply_flat_render_flags(
                        mdp,
                        disable_reflection=True,
                        disable_shadow=True,
                        disable_skybox=bool(args.white_background),
                        disable_fog=True,
                    )

                if absorbing:
                    break

            elapsed = time.time() - start
            print(f"Episode {ep}: steps={t + 1}, elapsed={elapsed:.2f}s")

    finally:
        try:
            mdp.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
