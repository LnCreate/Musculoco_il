
from itertools import product
from datetime import datetime
from experiment_launcher import Launcher, is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    JOBLIB_PARALLEL_JOBS = 1  # or os.cpu_count() to use all cores
    N_SEEDS = 15

    if LOCAL:
        n_steps_per_epoch = 100000
    else:
        n_steps_per_epoch = 100000

    date_tag = datetime.now().strftime('%Y-%m-%d')

    launcher = Launcher(exp_name=f'15_AMP_attention_walk_{date_tag}',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_cores=1,
                        memory_per_core=2500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False,
                        base_dir='./experiments/15_AMP_attention/logs',
                        project_name='PROJECT_NAME',
                        partition='PARTITION'
                        )

    default_params = dict(n_epochs=2500,
                          n_steps_per_epoch=n_steps_per_epoch,
                          n_epochs_save=20,
                          n_eval_episodes=30,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA,
                          )

    lrs = [(5e-5, 5e-6)]
    std_0s = [0.8]
    ctrl_freqs = [50]
    max_kls = [2e-2]
    x_stds = [0.6]
    reward_types = ['target_velocity']
    ent_coeffs = [5e-4]
    grfs = [False]
    envs = ['HumanoidMuscle.walk']

    for lr, std_0, ctrl_hz, max_kl, r_t, ent_c, grf, env, std_x in product(lrs, std_0s,
                                                                ctrl_freqs,
                                                                max_kls,
                                                                reward_types,
                                                                ent_coeffs, grfs, envs, x_stds):

        if r_t == 'target_velocity':
            env_r_frac = 0.4
        else:
            env_r_frac = 0.4

        lrc, lrD = lr

        launcher.add_experiment(lrc=lrc,
                    lrD=lrD,
                    train_D_n_th_epoch=3,
                    std_0=std_0,
                    max_kl=max_kl,
                    ctrl_freq=ctrl_hz,
                    reward_type=r_t,
                    env_reward_frac=env_r_frac,
                    env_reward_func_type='squared',
                    env_reward_scale=0.05,
                                standardize_obs=True,
                    policy_entr_coef=ent_c,
                    d_entr_coef=0.0,
                    use_noisy_targets=False,
                    use_next_states=True,
                    amp_gp_weight=5.0,
                    amp_logit_reg=0.01,
                    amp_replay_size=200000,
                    amp_replay_keep_prob=0.05,
                    env_id=env,
                    learn_latent_layer=False,
                    use_attention_synergy=True,
                    n_synergies=30,
                    synergy_attn_dim=32,
                    synergy_temperature=1.3,
                    std_x_0=std_x,
                    freeze_foot_muscles=False,
                    frozen_action_value=0.0,
                                **default_params)

    launcher.run(LOCAL, TEST)
