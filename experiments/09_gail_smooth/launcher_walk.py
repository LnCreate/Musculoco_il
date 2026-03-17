
import os
from itertools import product
from experiment_launcher import Launcher, is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    JOBLIB_PARALLEL_JOBS = 1  # or os.cpu_count() to use all cores
    N_SEEDS = 1
    if LOCAL:
        n_steps_per_epoch = 100000
    else:
        n_steps_per_epoch = 100000

    launcher = Launcher(exp_name='09_gail_smooth_walk_finetune',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_cores=1,
                        memory_per_core=2500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False,
                        base_dir='./logs',
                        project_name='PROJECT_NAME',
                        partition='PARTITION'
                        )

    default_params = dict(n_epochs=250,
                          n_steps_per_epoch=n_steps_per_epoch,
                          n_epochs_save=20,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA,
                          )

    lrs = [(5e-5, 1e-5)]
    std_0s = [0.8]
    ctrl_freqs = [100] # Increased from 50
    max_kls = [4e-2]
    x_stds = [0.6]
    reward_types = ['smooth_and_magnitude']
    ent_coeffs = [1e-3]
    grfs = [False]
    envs = ['HumanoidMuscle.walk']

    # Set checkpoint path for fine-tuning
    checkpoint = 'logs/09_gail_smooth_walk/1/agent_epoch_247_J_979.374208.msh'
    # Make it absolute or ensure it exists
    if not os.path.exists(checkpoint):
        print(f"Warning: Checkpoint {checkpoint} not found. Starting from scratch.")
        checkpoint = None

    for lr, std_0, ctrl_hz, max_kl, r_t, ent_c, grf, env, std_x in product(lrs, std_0s,
                                                                ctrl_freqs,
                                                                max_kls,
                                                                reward_types,
                                                                ent_coeffs, grfs, envs, x_stds):

        if r_t == 'target_velocity':
            env_r_frac = 0.0
        else:
            env_r_frac = 0.1

        lrc, lrD = lr

        launcher.add_experiment(lrc=lrc,
                                lrD=lrD,
                                std_0=std_0,
                                max_kl=max_kl,
                                ctrl_freq=ctrl_hz,
                                reward_type=r_t,
                                env_reward_frac=env_r_frac,
                                env_reward_func_type='squared',
                                env_reward_scale=0.01,
                                standardize_obs=True,
                                policy_entr_coef=ent_c,
                                env_id=env,
                                learn_latent_layer=False,
                                std_x_0=std_x,
                                freeze_foot_muscles=True,
                                frozen_action_value=0.0,
                                checkpoint_path=checkpoint,
                                **default_params)

    launcher.run(LOCAL, TEST)
