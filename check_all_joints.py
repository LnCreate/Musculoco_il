from loco_mujoco import LocoEnv
import numpy as np

mdp = LocoEnv.make('HumanoidMuscle.walk', use_box_feet=False)
model = mdp._model

# List some major joints to compare
joints = [
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_r', 'ankle_angle_r',
    'subtalar_angle_r', 'mtp_angle_r',
    'lumbar_extension'
]

print(f"{'Joint Name':<20} | {'Stiffness':<10} | {'Damping':<10}")
print("-" * 45)

for j_name in joints:
    try:
        j_id = model.joint(j_name).id
        stiff = model.jnt_stiffness[j_id]
        # Damping is per DOF
        dof_adr = model.jnt_dofadr[j_id]
        damp = model.dof_damping[dof_adr]
        print(f"{j_name:<20} | {stiff:<10.2f} | {damp:<10.2f}")
    except Exception as e:
        print(f"{j_name:<20} | Error: {e}")
