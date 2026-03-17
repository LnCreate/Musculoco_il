from itertools import product
from experiment_launcher import Launcher, is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 8

    if LOCAL:
        n_steps_per_epoch = 100000
    else:
        n_steps_per_epoch = 100000

    launcher = Launcher(exp_name='12_torque_fidelity_run',
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

    default_params = dict(n_epochs=250,
                          n_steps_per_epoch=n_steps_per_epoch,
                          n_epochs_save=10,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA,
                          )

    lrs = [(2e-5, 1e-6)]
    std_0s = [0.8]
    ctrl_freqs = [50]
    max_kls = [1e-2]
    x_stds = [0.6]
    reward_types = ['torque_fidelity']
    ent_coeffs = [1e-3]
    envs = ['HumanoidMuscle.run']

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
            env_reward_frac=0.20,
            env_reward_func_type='squared',
            env_reward_scale=0.10,
            body_mass_kg=80.0,
            torque_smooth_weight=0.02,
            torque_swing_weight=1.0,
            torque_stance_weight=0.6,
            torque_balance_weight=0.2,
            torque_stance_grf_ratio=0.05,
            torque_swing_cap_hip=0.80,
            torque_swing_cap_knee=0.70,
            torque_swing_cap_ankle=0.55,
            torque_stance_floor_hip=0.15,
            torque_stance_floor_knee=0.12,
            torque_stance_floor_ankle=0.16,
            standardize_obs=True,
            policy_entr_coef=ent_c,
            env_id=env,
            learn_latent_layer=False,
            std_x_0=std_x,
            freeze_foot_muscles=True,
            frozen_action_value=0.0,
            **default_params,
        )

    launcher.run(LOCAL, TEST)
