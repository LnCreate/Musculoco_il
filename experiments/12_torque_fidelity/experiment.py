import os
import numpy as np
import mujoco

from time import perf_counter
from contextlib import contextmanager

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length, parse_dataset
from mushroom_rl.core.logger.logger import Logger

from imitation_lib.imitation import GAIL_TRPO
from imitation_lib.utils import FullyConnectedNetwork, DiscriminatorNetwork, NormcInitializer, \
                         GailDiscriminatorLoss
from imitation_lib.utils import BestAgentSaver
from experiment_launcher import run_experiment

from mushroom_rl.core.serialization import *

from loco_mujoco import LocoEnv

import time

from musculoco_il.policy.gaussian_torch_policy import OptionalGaussianTorchPolicy
from musculoco_il.policy.latent_exploration_torch_policy import LatentExplorationPolicy
from musculoco_il.util.preprocessors import StateSelectionPreprocessor
from musculoco_il.util.rewards import OutOfBoundsActionCost, ActionSmoothnessReward, ActionMagnitudePenalty, CombinedReward
from musculoco_il.util.standardizer import Standardizer
from musculoco_il.util.action_specs import HAMNER_HUMANOID_FIXED_ARMS_ACTION_SPEC


FOOT_MUSCLE_NAMES = [
    "tib_post_r", "flex_dig_r", "flex_hal_r", "tib_ant_r", "per_brev_r", "per_long_r", "per_tert_r", "ext_dig_r", "ext_hal_r",
    "tib_post_l", "flex_dig_l", "flex_hal_l", "tib_ant_l", "per_brev_l", "per_long_l", "per_tert_l", "ext_dig_l", "ext_hal_l",
]

LEFT_FOOT_BODIES = ["talus_l", "calcn_l", "toes_l"]
RIGHT_FOOT_BODIES = ["talus_r", "calcn_r", "toes_r"]


def _get_action_indices_by_name(action_dim, target_names):
    name_to_idx = {n: i for i, n in enumerate(HAMNER_HUMANOID_FIXED_ARMS_ACTION_SPEC[:action_dim])}
    idx = [name_to_idx[n] for n in target_names if n in name_to_idx]
    return np.array(sorted(set(idx)), dtype=np.int32)


def _apply_action_freeze_mask(agent, freeze_indices, fill_value=0.0):
    if freeze_indices is None or len(freeze_indices) == 0:
        return

    orig_draw_action = agent.draw_action

    def masked_draw_action(state):
        a = np.asarray(orig_draw_action(state), dtype=np.float64).reshape(-1).copy()
        a[freeze_indices] = fill_value
        return a

    agent.draw_action = masked_draw_action
    print(f"[action-mask] Frozen {len(freeze_indices)} action dims at value {fill_value}.")


def _get_grf(model, data, foot_body_names):
    grf = np.zeros(3, dtype=np.float64)
    foot_ids = [model.body(name).id for name in foot_body_names]

    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]

        is_b1_foot = b1 in foot_ids
        is_b2_foot = b2 in foot_ids
        if not (is_b1_foot or is_b2_foot):
            continue

        c_force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, c_force)
        rot = contact.frame.reshape(3, 3)
        force_world = rot @ c_force[:3]

        if is_b2_foot:
            grf += force_world
        else:
            grf -= force_world

    return grf


class TorqueFidelityReward(object):
    """Phase-aware torque fidelity prior.
    
    1) Ensures torque smoothness.
    2) Caps maximum torque during the passive 'swing' phase tightly.
    3) Caps maximum torque during the heavy-load 'stance' phase loosely (prevents excessive stiffness).
    """

    def __init__(self,
                 body_mass_kg=75.0,
                 reward_scale=0.1,
                 smooth_weight=0.02,
                 swing_weight=1.0,
                 stance_weight=0.6,
                 swing_caps=(0.60, 0.40, 0.35),
                 stance_caps=(1.50, 1.20, 1.80)):
        
        self.body_mass_kg = float(body_mass_kg)
        self.reward_scale = float(reward_scale)

        self.smooth_weight = float(smooth_weight)
        self.swing_weight = float(swing_weight)
        self.stance_weight = float(stance_weight)

        self.swing_caps = np.array(swing_caps, dtype=np.float64)
        self.stance_caps = np.array(stance_caps, dtype=np.float64)

        self._model = None
        self._data = None
        self._torque_idx = None
        self._prev_tau = None

    def set_mdp(self, mdp):
        self._model = mdp._model
        self._data = mdp._data

        def _joint_dof_idx(joint_name):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise ValueError(f"Joint not found for torque fidelity reward: {joint_name}")
            return int(self._model.jnt_dofadr[jid])

        idx = [
            _joint_dof_idx("hip_flexion_r"),
            _joint_dof_idx("knee_angle_r"),
            _joint_dof_idx("ankle_angle_r"),
            _joint_dof_idx("hip_flexion_l"),
            _joint_dof_idx("knee_angle_l"),
            _joint_dof_idx("ankle_angle_l"),
        ]
        self._torque_idx = np.asarray(idx, dtype=np.int32)

    def reset(self):
        self._prev_tau = None

    @staticmethod
    def _hinge_sq(x):
        z = np.maximum(0.0, x)
        return z * z

    def __call__(self, state, action, next_state):
        if self._model is None or self._data is None or self._torque_idx is None:
            return 0.0

        qfrc_act = np.asarray(self._data.qfrc_actuator, dtype=np.float64)
        tau = qfrc_act[self._torque_idx] / max(1e-6, self.body_mass_kg)

        tau_r = np.abs(tau[:3])
        tau_l = np.abs(tau[3:])

        grf_l = _get_grf(self._model, self._data, LEFT_FOOT_BODIES)
        grf_r = _get_grf(self._model, self._data, RIGHT_FOOT_BODIES)
        bw = self.body_mass_kg * 9.81
        fz_l = abs(float(grf_l[2]))
        fz_r = abs(float(grf_r[2]))

        # 使用软概率平滑过度
        low_thr = 0.02 * bw
        high_thr = 0.08 * bw
        
        p_stance_l = np.clip((fz_l - low_thr) / (high_thr - low_thr), 0.0, 1.0)
        p_stance_r = np.clip((fz_r - low_thr) / (high_thr - low_thr), 0.0, 1.0)

        # ---------------- 右腿判别 ----------------
        r_stance_pen = float(np.mean(self._hinge_sq(tau_r - self.stance_caps)))
        r_swing_pen  = float(np.mean(self._hinge_sq(tau_r - self.swing_caps)))
        
        pen_r = p_stance_r * (self.stance_weight * r_stance_pen) + \
                (1.0 - p_stance_r) * (self.swing_weight * r_swing_pen)

        # ---------------- 左腿判别 ----------------
        l_stance_pen = float(np.mean(self._hinge_sq(tau_l - self.stance_caps)))
        l_swing_pen  = float(np.mean(self._hinge_sq(tau_l - self.swing_caps)))
        
        pen_l = p_stance_l * (self.stance_weight * l_stance_pen) + \
                (1.0 - p_stance_l) * (self.swing_weight * l_swing_pen)

        smooth_pen = 0.0
        if self._prev_tau is not None:
            smooth_pen = float(np.mean((tau - self._prev_tau) ** 2))
        self._prev_tau = tau.copy()

        total_pen = pen_r + pen_l + self.smooth_weight * smooth_pen

        return -self.reward_scale * float(total_pen)


class RewardComposerWithMdp(CombinedReward):
    """CombinedReward that forwards mdp hooks to nested callbacks when available."""

    def set_mdp(self, mdp):
        for callback in self.reward_callbacks:
            if hasattr(callback, "set_mdp"):
                callback.set_mdp(mdp)

    def reset(self):
        for callback in self.reward_callbacks:
            if hasattr(callback, "reset"):
                callback.reset()


def _set_torque_aux_scale(callback, scale: float) -> bool:
    """Recursively set reward_scale for TorqueFidelityReward inside nested callbacks."""
    if isinstance(callback, TorqueFidelityReward):
        callback.reward_scale = float(scale)
        return True

    if hasattr(callback, "reward_callbacks"):
        found = False
        for child in callback.reward_callbacks:
            found = _set_torque_aux_scale(child, scale) or found
        return found

    return False


def _compute_torque_aux_scale(epoch: int, *, final_scale: float, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return 0.0

    if ramp_epochs <= 0:
        return float(final_scale)

    progress = float(epoch - warmup_epochs) / float(ramp_epochs)
    progress = min(1.0, max(0.0, progress))
    return float(final_scale) * progress


def initial_log(core, tb_writer, logger_stoch, logger_deter, n_eval_episodes, gamma):
    epoch = 0
    dataset = core.evaluate(n_episodes=n_eval_episodes)
    J_mean = np.mean(compute_J(dataset))
    tb_writer.add_scalar("Eval_J", J_mean, epoch)
    with catchtime() as t:
        print('Epoch %d | Time %fs ' % (epoch + 1, float(t())))

    # evaluate with deterministic policy
    core.agent.policy.deterministic = True
    dataset = core.evaluate(n_episodes=n_eval_episodes)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset, gamma=gamma))
    L = np.mean(compute_episodes_length(dataset))
    logger_deter.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
    tb_writer.add_scalar("Eval_R-deterministic", R_mean, epoch)
    tb_writer.add_scalar("Eval_J-deterministic", J_mean, epoch)
    tb_writer.add_scalar("Eval_L-deterministic", L, epoch)

    # evaluate with stochastic policy
    core.agent.policy.deterministic = False
    dataset = core.evaluate(n_episodes=n_eval_episodes)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset, gamma=gamma))
    L = np.mean(compute_episodes_length(dataset))
    E = core.agent.policy.entropy()

    a_abs_mean = log_action_mean(core.agent, dataset)

    logger_stoch.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L, E=E, a_abs_mean=a_abs_mean)
    tb_writer.add_scalar("Eval_R-stochastic", R_mean, epoch)
    tb_writer.add_scalar("Eval_J-stochastic", J_mean, epoch)
    tb_writer.add_scalar("Eval_L-stochastic", L, epoch)


def build_agent(mdp, expert_data, use_cuda, discrim_obs_mask, train_D_n_th_epoch=3,
                lrc=1e-3, lrD=0.0003, sw=None, policy_entr_coef=0.0,
                use_noisy_targets=False,
                use_next_states=True, max_kl=5e-3, d_entr_coef=1e-3,
                env_reward_frac=0.0, standardize_obs=True, learn_latent_layer=False,
                std_a_0=1.0, std_x_0=1.0, checkpoint_path=None,
                ):
    if checkpoint_path is not None:
        print(f"Loading agent from {checkpoint_path}")
        agent = GAIL_TRPO.load(checkpoint_path)
        agent._sw = sw
        if agent._sw is not None:
            setattr(agent._sw, '__deepcopy__', lambda _: None)

        if hasattr(agent, 'env_reward_frac'):
            agent.env_reward_frac = env_reward_frac

        if hasattr(agent, '_V'):
            for param_group in agent._V.model._optimizer.param_groups:
                param_group['lr'] = lrc
        if hasattr(agent, '_D'):
            for param_group in agent._D.model._optimizer.param_groups:
                param_group['lr'] = lrD
                param_group['weight_decay'] = 1e-3

        with torch.no_grad():
            if hasattr(agent.policy, '_log_sigma_a_flat'):
                agent.policy._log_sigma_a_flat.fill_(np.log(std_a_0))
            if hasattr(agent.policy, '_log_sigma_x_flat'):
                agent.policy._log_sigma_x_flat.fill_(np.log(std_x_0))
            if hasattr(agent.policy, '_log_std'):
                agent.policy._log_std.fill_(np.log(std_a_0))

        return agent

    mdp_info = deepcopy(mdp.info)

    trpo_standardizer = Standardizer(use_cuda=use_cuda) if standardize_obs else None

    print("Action DIM:")
    print(mdp_info.action_space.shape)
    print("OBS DIM:")
    print(mdp_info.observation_space.shape)
    feature_dims = [512]

    latent_dim = 256

    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=(len(discrim_obs_mask),),
                         output_shape=mdp_info.action_space.shape,
                         latent_shape=(latent_dim,),
                         learn_latent_layer=learn_latent_layer,
                         std_a_0=std_a_0,
                         std_x_0=std_x_0,
                         n_features=feature_dims,
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0)],
                         activations=['relu', 'relu'],
                         standardizer=trpo_standardizer,
                         use_cuda=use_cuda)

    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lrc,
                                               'weight_decay': 0.0}},
                         loss=F.mse_loss,
                         batch_size=256,
                         input_shape=(len(discrim_obs_mask),),
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    # remove hip rotations
    discrim_act_mask = []
    discrim_input_shape = (2 * len(discrim_obs_mask),) if use_next_states else (len(discrim_obs_mask),)
    discrim_standardizer = Standardizer() if standardize_obs else None

    discriminator_params = dict(optimizer={'class': optim.Adam,
                                           'params': {'lr': lrD,
                                                      'weight_decay': 0.0}},
                                batch_size=2000,
                                network=DiscriminatorNetwork,
                                use_next_states=use_next_states,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                squeeze_out=False,
                                n_features=[512, 256],
                                initializers=None,
                                activations=['tanh', 'tanh', 'identity'],
                                standardizer=discrim_standardizer,
                                use_actions=False,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_D_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=25,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=GailDiscriminatorLoss(entcoeff=d_entr_coef),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=max_kl,
                      use_next_states=use_next_states,
                      env_reward_frac=env_reward_frac)

    print(f'USE_NEXT_STATES: {use_next_states}')
    print(f'ENV_REWARD_FRAC: {env_reward_frac}')

    agent = GAIL_TRPO(mdp_info=mdp_info, policy_class=LatentExplorationPolicy, policy_params=policy_params, sw=sw,
                      discriminator_params=discriminator_params, critic_params=critic_params,
                      demonstrations=expert_data, **alg_params)
    return agent


def experiment(n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1024,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 500,
               horizon: int = 1000,
               gamma: float = 0.99,
               policy_entr_coef: float = 1e-3,
               train_D_n_th_epoch: int = 3,
               lrc: float = 1e-3,
               lrD: float = 0.0003,
               use_noisy_targets: bool = False,
               use_next_states: bool = False,
               use_cuda: bool = False,
               results_dir: str = './logs',
               std_0: float = 1.0,
               max_kl: float = 5e-3,
               d_entr_coef: float = 1e-3,
               env_freq: int = 1000,
               ctrl_freq: int = 100,
               reward_type: str = 'target_velocity',
               env_reward_frac: float = 0.0,
               env_reward_scale: float = 1.0,
               env_reward_func_type: str = 'abs',
               reward_action_mean: float = 0.5,
               reward_const_cost: float = 0.0,
               standardize_obs: bool = True,
               env_id: str = 'Atlas.walk',
               learn_latent_layer: bool = False,
               std_x_0: float = 1.0,
               freeze_foot_muscles: bool = False,
               frozen_action_value: float = 0.0,
               body_mass_kg: float = 75.0,
               torque_smooth_weight: float = 0.02,
               torque_swing_weight: float = 1.0,
               torque_stance_weight: float = 0.6,
               torque_swing_cap_hip: float = 0.60,
               torque_swing_cap_knee: float = 0.40,
               torque_swing_cap_ankle: float = 0.35,
               torque_stance_cap_hip: float = 1.0,
               torque_stance_cap_knee: float = 1.2,
               torque_stance_cap_ankle: float = 1.8,
               torque_aux_scale: float = 0.02,
               torque_aux_warmup_epochs: int = 0,
               torque_aux_ramp_epochs: int = 0,
               checkpoint_path: str = None,
               use_box_feet: bool = True,
               seed: int = 0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    results_dir = os.path.join(results_dir, str(seed))

    logger_stoch = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed, append=True)
    logger_deter = Logger(results_dir=results_dir, log_name="deterministic_logging", seed=seed, append=True)

    tb_writer = SummaryWriter(log_dir=results_dir)
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    n_substeps = env_freq // ctrl_freq

    reward_callback = None
    if reward_type == 'target_velocity':
        print(f'Using custom Reward: {reward_type}')
        if '4Ages' not in env_id:
            reward_type_loco = 'target_velocity'
        else:
            raise NotImplementedError
        if 'walk' in env_id:
            reward_params = dict(target_velocity=1.25)
        elif 'run' in env_id:
            reward_params = dict(target_velocity=2.5)
        else:
            reward_params = dict(target_velocity=1.25)
    elif reward_type == 'out_of_bounds_action_cost':
        print(f'Using custom Reward: {reward_type}')
        reward_type_loco = 'custom'
        reward_callback = OutOfBoundsActionCost(0.0, 1.0, reward_scale=env_reward_scale,
                                                const_cost=reward_const_cost,
                                                func_type=env_reward_func_type,)
        reward_params = dict(reward_callback=reward_callback)
    elif reward_type == 'torque_fidelity':
        print(f'Using custom Reward: {reward_type}')
        reward_type_loco = 'custom'
        reward_callback = TorqueFidelityReward(
            body_mass_kg=body_mass_kg,
            reward_scale=env_reward_scale,
            smooth_weight=torque_smooth_weight,
            swing_weight=torque_swing_weight,
            stance_weight=torque_stance_weight,
            swing_caps=(torque_swing_cap_hip, torque_swing_cap_knee, torque_swing_cap_ankle),
            stance_caps=(torque_stance_cap_hip, torque_stance_cap_knee, torque_stance_cap_ankle),
        )
        reward_params = dict(reward_callback=reward_callback)
    elif reward_type == 'smooth_and_torque_fidelity':
        print(f'Using custom Reward: {reward_type}')
        reward_type_loco = 'custom'
        reward_callback = RewardComposerWithMdp([
            ActionSmoothnessReward(reward_scale=env_reward_scale),
            ActionMagnitudePenalty(reward_scale=env_reward_scale, power=2),
            TorqueFidelityReward(
                body_mass_kg=body_mass_kg,
                reward_scale=torque_aux_scale,
                smooth_weight=torque_smooth_weight,
                swing_weight=torque_swing_weight,
                stance_weight=torque_stance_weight,
                swing_caps=(torque_swing_cap_hip, torque_swing_cap_knee, torque_swing_cap_ankle),
                stance_caps=(torque_stance_cap_hip, torque_stance_cap_knee, torque_stance_cap_ankle),
            )
        ])
        reward_params = dict(reward_callback=reward_callback)
    elif reward_type == 'action_cost':
        print(f'Using custom Reward: {reward_type}')
        reward_type_loco = 'custom'
        raise NotImplementedError
    else:
        raise Exception(f'{reward_type} is not a valid reward type!')

    env_use_box_feet = True
    if not bool(use_box_feet):
        print("[box-feet] Overriding use_box_feet=False to True for rigid-foot training consistency.")

    mdp = LocoEnv.make(env_id,
                       gamma=gamma,
                       horizon=horizon,
                       n_substeps=n_substeps,
                       timestep=1/env_freq,
                       reward_type=reward_type_loco,
                       reward_params=reward_params,
                       muscle_force_scaling=1.25,
                       use_box_feet=env_use_box_feet
                       )

    if reward_callback is not None and hasattr(reward_callback, 'set_mdp'):
        reward_callback.set_mdp(mdp)

    torque_aux_sched_active = bool(reward_type == 'smooth_and_torque_fidelity')
    if torque_aux_sched_active:
        if int(torque_aux_ramp_epochs) <= 0:
            torque_aux_ramp_epochs = max(1, int(n_epochs) - int(torque_aux_warmup_epochs))
        init_scale = _compute_torque_aux_scale(
            0,
            final_scale=float(torque_aux_scale),
            warmup_epochs=int(torque_aux_warmup_epochs),
            ramp_epochs=int(torque_aux_ramp_epochs),
        )
        _set_torque_aux_scale(reward_callback, init_scale)
        tb_writer.add_scalar("torque_aux_scale", init_scale, 0)
        print(
            f"[torque-aux-schedule] warmup={int(torque_aux_warmup_epochs)} epochs, "
            f"ramp={int(torque_aux_ramp_epochs)} epochs, final={float(torque_aux_scale):.6f}"
        )

    print('Env Action Scaling:')
    print(mdp.norm_act_mean)
    print(mdp.norm_act_delta)

    print('Reward Params:')
    print(reward_params)

    test_mdp = LocoEnv.make(env_id,
                            gamma=gamma,
                            horizon=horizon,
                            n_substeps=n_substeps,
                            timestep=1/env_freq,
                            muscle_force_scaling=1.25,
                            use_box_feet=env_use_box_feet
                            )

    print(f'DT: {mdp.dt}')
    print(f'ENV_DT: {mdp._timestep}')

    unavailable_keys = ["q_pelvis_tx", "q_pelvis_tz"]
    expert_data = mdp.create_dataset(ignore_keys=unavailable_keys)

    discrim_obs_mask = mdp.get_kinematic_obs_mask()

    print("Discrim Obs Mask:")
    print(len(discrim_obs_mask))

    agent = build_agent(mdp=mdp, expert_data=expert_data, use_cuda=use_cuda,
                        train_D_n_th_epoch=train_D_n_th_epoch, lrc=lrc,
                        lrD=lrD, sw=tb_writer, policy_entr_coef=policy_entr_coef,
                        use_noisy_targets=use_noisy_targets, use_next_states=use_next_states,
                        discrim_obs_mask=discrim_obs_mask,
                        max_kl=max_kl, d_entr_coef=d_entr_coef, env_reward_frac=env_reward_frac,
                        standardize_obs=standardize_obs, learn_latent_layer=learn_latent_layer,
                        std_a_0=std_0, std_x_0=std_x_0, checkpoint_path=checkpoint_path)

    if freeze_foot_muscles:
        action_dim = int(np.prod(mdp.info.action_space.shape))
        freeze_idx = _get_action_indices_by_name(action_dim, FOOT_MUSCLE_NAMES)
        _apply_action_freeze_mask(agent, freeze_idx, fill_value=frozen_action_value)

    core = Core(agent, mdp)
    test_core = Core(agent, test_mdp)

    if '4Ages' in env_id:
        agent.add_preprocessor(StateSelectionPreprocessor(first_n=len(discrim_obs_mask)))

    assert agent.policy._mu.model.network._stand is not None

    if agent.policy._mu.model.network._stand is not None:
        agent.policy._mu.model.network._stand.freeze()

    initial_log(test_core, tb_writer, logger_stoch, logger_deter, n_eval_episodes, gamma)

    if agent.policy._mu.model.network._stand is not None:
        agent.policy._mu.model.network._stand.unfreeze()

    for epoch in range(1, n_epochs):
        with catchtime() as t:
            start = time.time()
            core.agent.policy.deterministic = False

            if torque_aux_sched_active:
                sched_scale = _compute_torque_aux_scale(
                    int(epoch),
                    final_scale=float(torque_aux_scale),
                    warmup_epochs=int(torque_aux_warmup_epochs),
                    ramp_epochs=int(torque_aux_ramp_epochs),
                )
                _set_torque_aux_scale(reward_callback, sched_scale)
                tb_writer.add_scalar("torque_aux_scale", sched_scale, epoch)

            core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=True, render=False)

            done = time.time()
            elapsed = done - start
            tb_writer.add_scalar("time_in_learn", elapsed, epoch)

            dataset = test_core.evaluate(n_episodes=n_eval_episodes)
            J_mean = np.mean(compute_J(dataset))
            tb_writer.add_scalar("Eval_J", J_mean, epoch)
            agent_saver.save(core.agent, J_mean)
            print('Epoch %d | Time %fs ' % (epoch + 1, float(t())))

            core.agent.policy.deterministic = True

            if agent.policy._mu.model.network._stand is not None:
                agent.policy._mu.model.network._stand.freeze()

            dataset = test_core.evaluate(n_episodes=n_eval_episodes)
            R_mean = np.mean(compute_J(dataset))
            J_mean = np.mean(compute_J(dataset, gamma=gamma))
            L = np.mean(compute_episodes_length(dataset))
            logger_deter.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
            tb_writer.add_scalar("Eval_R-deterministic", R_mean, epoch)
            tb_writer.add_scalar("Eval_J-deterministic", J_mean, epoch)
            tb_writer.add_scalar("Eval_L-deterministic", L, epoch)

            core.agent.policy.deterministic = False
            dataset = test_core.evaluate(n_episodes=n_eval_episodes)
            R_mean = np.mean(compute_J(dataset))
            J_mean = np.mean(compute_J(dataset, gamma=gamma))
            L = np.mean(compute_episodes_length(dataset))
            E = agent.policy.entropy()

            tb_writer.add_scalar("Eval_R-stochastic", R_mean, epoch)
            tb_writer.add_scalar("Eval_J-stochastic", J_mean, epoch)
            tb_writer.add_scalar("Eval_L-stochastic", L, epoch)

            a_abs_mean = log_action_mean(agent, dataset)

            dataset = core.evaluate(n_episodes=n_eval_episodes)
            shaped_R_mean = np.mean(compute_J(dataset))
            tb_writer.add_scalar("Shaped_R-stochastic", shaped_R_mean, epoch)
            logger_stoch.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L, E=E,
                                   Shaped_R=shaped_R_mean, a_abs_mean=a_abs_mean)

            if agent.policy._mu.model.network._stand is not None:
                agent.policy._mu.model.network._stand.unfreeze()

    agent_saver.save_curr_best_agent()
    print("Finished.")


def log_action_mean(agent, dataset):
    s, a, r, ns, *_ = parse_dataset(dataset)
    agent.policy.deterministic = True
    a_mu = agent.draw_action(s)
    agent.policy.deterministic = False

    a_abs_mean = np.mean(np.abs(a_mu))

    return a_abs_mean


@contextmanager
def catchtime():
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":
    run_experiment(experiment)
