import torch
import torch.nn.functional as F
import numpy as np

from imitation_lib.imitation import GAIL_TRPO
from mushroom_rl.utils.minibatches import minibatch_generator

class AMPDiscriminatorLoss(torch.nn.Module):
    def __init__(self, entcoeff=0.0):
        super().__init__()
        self.entcoeff = entcoeff

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # AMP paper (Eq.6) uses LSGAN targets +1 (expert) and -1 (policy).
        # Keep compatibility with {0,1} labels produced by the parent pipeline.
        if torch.min(target) >= 0.0:
            target_ls = 2.0 * target - 1.0
        else:
            target_ls = target

        loss = torch.mean((input - target_ls) ** 2)
        return loss

    def logit_bernoulli_entropy(self, logits):
        # Keep the same entropy regularization interface as GAIL for stability.
        return (1. - torch.sigmoid(logits)) * logits - F.logsigmoid(logits)


class AMP_TRPO(GAIL_TRPO):
    def __init__(self, *args, **kwargs):
        discriminator_entcoeff = kwargs.pop('discriminator_entcoeff', 0.0)
        self._amp_gp_weight = kwargs.pop('amp_gp_weight', 10.0)
        self._amp_logit_reg = float(kwargs.pop('amp_logit_reg', 0.0))
        self._amp_replay_size = int(kwargs.pop('amp_replay_size', 200000))
        self._amp_replay_keep_prob = float(kwargs.pop('amp_replay_keep_prob', 1.0))

        # AMP discriminator is defined on state transitions D(phi(s), phi(s')).
        kwargs['use_next_states'] = True
        kwargs['use_noisy_targets'] = False
        kwargs['loss'] = AMPDiscriminatorLoss(entcoeff=discriminator_entcoeff)
        super().__init__(*args, **kwargs)

        self._amp_replay_obs = None
        self._amp_replay_next_obs = None

        self._add_save_attr(
            _amp_gp_weight='primitive',
            _amp_logit_reg='primitive',
            _amp_replay_size='primitive',
            _amp_replay_keep_prob='primitive',
            _amp_replay_obs='pickle',
            _amp_replay_next_obs='pickle'
        )

    def _to_tensor(self, array):
        device = next(self._D.model.network.parameters()).device
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    def _update_amp_replay(self, obs, next_obs):
        if self._amp_replay_keep_prob < 1.0:
            keep = np.random.rand(obs.shape[0]) < self._amp_replay_keep_prob
            if not np.any(keep):
                return
            obs = obs[keep]
            next_obs = next_obs[keep]

        if self._amp_replay_obs is None:
            self._amp_replay_obs = obs.astype(np.float32, copy=True)
            self._amp_replay_next_obs = next_obs.astype(np.float32, copy=True)
        else:
            self._amp_replay_obs = np.concatenate([self._amp_replay_obs, obs.astype(np.float32)], axis=0)
            self._amp_replay_next_obs = np.concatenate([self._amp_replay_next_obs, next_obs.astype(np.float32)], axis=0)

        if self._amp_replay_obs.shape[0] > self._amp_replay_size:
            self._amp_replay_obs = self._amp_replay_obs[-self._amp_replay_size:]
            self._amp_replay_next_obs = self._amp_replay_next_obs[-self._amp_replay_size:]

    def _sample_policy_batch(self, batch_size, plcy_obs, plcy_n_obs):
        if self._amp_replay_obs is None or self._amp_replay_obs.shape[0] < batch_size:
            return next(minibatch_generator(batch_size, plcy_obs, plcy_n_obs))

        idx = np.random.randint(0, self._amp_replay_obs.shape[0], size=batch_size)
        return self._amp_replay_obs[idx], self._amp_replay_next_obs[idx]

    def _fit_discriminator(self, plcy_obs, plcy_act, plcy_n_obs):
        del plcy_act

        plcy_obs = plcy_obs[:, self._state_mask]
        plcy_n_obs = plcy_n_obs[:, self._state_mask]

        if self._iter % self._train_D_n_th_epoch != 0:
            return

        # Store current policy transitions into replay buffer before update.
        self._update_amp_replay(plcy_obs, plcy_n_obs)

        net = self._D.model.network
        opt = self._D.model._optimizer
        loss_fn = self._D.model._loss

        for _ in range(self._n_epochs_discriminator):
            demo_obs, demo_n_obs = next(minibatch_generator(plcy_obs.shape[0],
                                                            self._demonstrations["states"],
                                                            self._demonstrations["next_states"]))
            demo_obs = demo_obs[:, self._state_mask].astype(np.float32)
            demo_n_obs = demo_n_obs[:, self._state_mask].astype(np.float32)

            pol_obs, pol_n_obs = self._sample_policy_batch(plcy_obs.shape[0], plcy_obs, plcy_n_obs)
            pol_obs = pol_obs.astype(np.float32)
            pol_n_obs = pol_n_obs.astype(np.float32)

            if self._D_standardizer is not None:
                self._D_standardizer.update_mean_std(np.concatenate([pol_obs, demo_obs], axis=0))

            pol_obs_t = self._to_tensor(pol_obs)
            pol_n_obs_t = self._to_tensor(pol_n_obs)
            demo_obs_t = self._to_tensor(demo_obs)
            demo_n_obs_t = self._to_tensor(demo_n_obs)

            demo_obs_gp = demo_obs_t.detach().clone().requires_grad_(True)
            demo_n_obs_gp = demo_n_obs_t.detach().clone().requires_grad_(True)

            d_pol = net(pol_obs_t, pol_n_obs_t)
            d_demo = net(demo_obs_t, demo_n_obs_t)

            tgt_pol = torch.zeros_like(d_pol)
            tgt_demo = torch.ones_like(d_demo)
            loss_main = loss_fn(d_pol, tgt_pol) + loss_fn(d_demo, tgt_demo)

            # AMP paper Eq.8: gradient penalty on demo transitions.
            d_demo_gp = net(demo_obs_gp, demo_n_obs_gp)
            grad_obs, grad_next_obs = torch.autograd.grad(
                outputs=d_demo_gp.sum(),
                inputs=[demo_obs_gp, demo_n_obs_gp],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            grad_pen = (grad_obs.pow(2).sum(dim=1) + grad_next_obs.pow(2).sum(dim=1)).mean()

            logit_reg = torch.tensor(0.0, device=d_demo.device)
            if self._amp_logit_reg > 0.0:
                for module in net.modules():
                    if isinstance(module, torch.nn.Linear):
                        logit_reg = logit_reg + torch.sum(module.weight.pow(2))

            loss = loss_main + self._amp_gp_weight * grad_pen + self._amp_logit_reg * logit_reg

            opt.zero_grad()
            loss.backward()
            opt.step()

            if self._sw:
                step = self._iter // max(1, self._train_D_n_th_epoch)
                d_pol_np = d_pol.detach().cpu().numpy()
                d_demo_np = d_demo.detach().cpu().numpy()
                self._sw.add_scalar('DiscrimLoss', loss.detach().item(), step)
                self._sw.add_scalar('D_Expert_Accuracy', np.mean(d_demo_np > 0.0), step)
                self._sw.add_scalar('D_Generator_Accuracy', np.mean(d_pol_np < 0.0), step)
                self._sw.add_scalar('D_Out_Expert', np.mean(d_demo_np), step)
                self._sw.add_scalar('D_Out_Generator', np.mean(d_pol_np), step)
                self._sw.add_scalar('amp/grad_penalty', grad_pen.detach().item(), step)
                self._sw.add_scalar('amp/logit_reg', logit_reg.detach().item(), step)

    @torch.no_grad()
    def make_discrim_reward(self, state, action, next_state, apply_mask=True):
        if self._use_next_state:
            d = self.discrim_output(state, next_state, apply_mask=apply_mask)
        else:
            d = self.discrim_output(state, action, apply_mask=apply_mask)

        # AMP paper Eq.7.
        reward = np.maximum(0.0, 1.0 - 0.25 * np.square(d - 1.0))

        return np.squeeze(reward).astype(np.float32)
