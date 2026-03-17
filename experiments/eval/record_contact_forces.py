import os
import csv
import argparse
import numpy as np
import torch
import mujoco
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from loco_mujoco import LocoEnv
from mushroom_rl.core import Serializable

# Define body groups for legs
LEFT_FOOT_BODIES = ["talus_l", "calcn_l", "toes_l"]
RIGHT_FOOT_BODIES = ["talus_r", "calcn_r", "toes_r"]

# Joint reaction pairs (child body name)
# cfrc_int[child_id] gives the force from parent to child
JRF_BODIES = {
    "hip_r": "femur_r",
    "knee_r": "tibia_r",
    "ankle_r": "talus_r",
    "hip_l": "femur_l",
    "knee_l": "tibia_l",
    "ankle_l": "talus_l",
}

def get_grf(model, data, foot_body_names):
    """
    Calculates total Ground Reaction Force (GRF) for a set of bodies.
    Returns 3D force vector [Fx, Fy, Fz] in world coordinates.
    """
    grf = np.zeros(3)
    foot_ids = [model.body(name).id for name in foot_body_names]
    
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]
        
        # We only care about contacts between foot and everything else (usually ground)
        # Note: ground is usually body 0 (world)
        
        is_b1_foot = b1 in foot_ids
        is_b2_foot = b2 in foot_ids
        
        if not (is_b1_foot or is_b2_foot):
            continue
            
        # Get force in contact frame
        c_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, c_force)
        
        # Rotate to world frame
        # contact.frame is a 9-element array (3x3 rotation matrix)
        rot = contact.frame.reshape(3, 3)
        force_world = rot @ c_force[:3]
        
        # mj_contactForce returns force exerted BY geom1 ON geom2
        if is_b2_foot:
            grf += force_world
        else:
            grf -= force_world
            
    return grf

def get_jrf(model, data, body_name):
    """
    Returns the joint reaction force acting on body_name from its parent.
    Fallback to sensors if they exist (added to XML), otherwise use cfrc_int.
    """
    # Try sensor first (mapping body name to sensor name)
    # femur -> hip, tibia -> knee, talus -> ankle
    s_map = {"femur": "hip", "tibia": "knee", "talus": "ankle"}
    parts = body_name.split("_")
    if parts[0] in s_map and len(parts) > 1:
        sensor_name = f"{s_map[parts[0]]}_{parts[1]}_force"
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sid != -1:
                adr = model.sensor_adr[sid]
                res = data.sensordata[adr:adr+3]
                return np.linalg.norm(res), res
        except:
            pass

    # Fallback to cfrc_int
    try:
        b_id = model.body(body_name).id
        # cfrc_int is [torque_x, torque_y, torque_z, force_x, force_y, force_z]
        wrench = data.cfrc_int[b_id]
        force_com = wrench[3:]
        mag = np.linalg.norm(force_com)
        return mag, force_com
    except:
        return 0.0, np.zeros(3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to agent checkpoint")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--env-id", type=str, default="HumanoidMuscle.walk")
    parser.add_argument("--use-box-feet", action="store_true", help="Use box feet (default: False for complex)")
    parser.add_argument("--out", type=str, default="contact_forces.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use 100Hz control frequency (matches Experiment 10)
    mdp = LocoEnv.make(
        args.env_id,
        n_substeps=10,
        timestep=0.001,
        use_box_feet=args.use_box_feet,
        muscle_force_scaling=1.25
    )

    agent = Serializable.load(args.checkpoint)
    agent.policy.deterministic = True

    # Prepare results
    header = ["time", "grf_r_x", "grf_r_y", "grf_r_z", "grf_l_x", "grf_l_y", "grf_l_z"]
    for name in JRF_BODIES.keys():
        header.append(f"jrf_{name}_mag")
        header.append(f"jrf_{name}_x")
        header.append(f"jrf_{name}_y")
        header.append(f"jrf_{name}_z")

    all_rows = []

    print(f"Starting rollout for {args.episodes} episodes...")
    for ep in range(args.episodes):
        obs = mdp.reset()
        for t in range(args.max_steps):
            action = agent.draw_action(obs)
            obs, reward, absorbing, info = mdp.step(action)
            
            # Extract forces
            grf_r = get_grf(mdp._model, mdp._data, RIGHT_FOOT_BODIES)
            grf_l = get_grf(mdp._model, mdp._data, LEFT_FOOT_BODIES)
            
            row = {
                "time": mdp._data.time,
                "grf_r_x": grf_r[0], "grf_r_y": grf_r[1], "grf_r_z": grf_r[2],
                "grf_l_x": grf_l[0], "grf_l_y": grf_l[1], "grf_l_z": grf_l[2],
            }
            
            for name, body in JRF_BODIES.items():
                mag, vec = get_jrf(mdp._model, mdp._data, body)
                row[f"jrf_{name}_mag"] = mag
                row[f"jrf_{name}_x"] = vec[0]
                row[f"jrf_{name}_y"] = vec[1]
                row[f"jrf_{name}_z"] = vec[2]
                
            all_rows.append(row)
            
            if absorbing:
                break
                
    # Save to CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
        
    print(f"Done! Contact forces recorded to {args.out}")

if __name__ == "__main__":
    main()
