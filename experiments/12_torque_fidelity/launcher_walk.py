from itertools import product
from experiment_launcher import Launcher, is_local
import os


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 8

    if LOCAL:
        n_steps_per_epoch = 10000
    else:
        n_steps_per_epoch = 100000

    launcher = Launcher(exp_name='12_torque_fidelity_walk_conservative_V2',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_cores=1,
                        memory_per_core=2500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False,
                        base_dir='./experiments/12_torque_fidelity/logs',
                        project_name='PROJECT_NAME',
                        partition='PARTITION'
                        )

    default_params = dict(n_epochs=2500,
                          n_steps_per_epoch=n_steps_per_epoch,
                          n_epochs_save=10,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA,
                          )

    lrs = [(1e-5, 2e-6)]
    std_0s = [0.8]
    ctrl_freqs = [100]
    max_kls = [5e-3]
    x_stds = [0.6]
    reward_types = ['smooth_and_torque_fidelity']
    ent_coeffs = [1e-3]
    envs = ['HumanoidMuscle.walk']

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    checkpoint = os.path.join(repo_root, 'logs/09_gail_smooth_walk/1/agent_epoch_247_J_979.374208.msh')
    if not os.path.exists(checkpoint):
        print(f"Warning: checkpoint not found: {checkpoint}")
        checkpoint = None

    for lr, std_0, ctrl_hz, max_kl, r_t, ent_c, env, std_x in product(
            lrs, std_0s, ctrl_freqs, max_kls, reward_types, ent_coeffs, envs, x_stds):

        lrc, lrD = lr

        launcher.add_experiment(
            lrc=lrc,
            lrD=lrD,
            std_0=std_0,
            max_kl=max_kl,
            ctrl_freq=ctrl_hz,
            reward_type=r_t,
            env_reward_frac=0.03,
            env_reward_func_type='squared',
            env_reward_scale=0.01,
            body_mass_kg=80.0,
            torque_aux_scale=0.0002,
            torque_aux_warmup_epochs=10,
            torque_aux_ramp_epochs=50,
            torque_smooth_weight=0.01,
            torque_swing_weight=0.1,
            torque_stance_weight=0.05,
            
            torque_swing_cap_hip=0.50,
            torque_swing_cap_knee=0.40,
            torque_swing_cap_ankle=0.35,

            torque_stance_cap_hip=1.00,
            torque_stance_cap_knee=1.20,
            torque_stance_cap_ankle=1.80,
            
            standardize_obs=True,
            policy_entr_coef=ent_c,
            env_id=env,
            learn_latent_layer=False,
            std_x_0=std_x,
            use_box_feet=True,
            freeze_foot_muscles=True,
            frozen_action_value=0.0,
            checkpoint_path=checkpoint,
            **default_params,
        )

    launcher.run(LOCAL, TEST)
