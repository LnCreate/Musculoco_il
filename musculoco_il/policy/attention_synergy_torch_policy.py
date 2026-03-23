from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import TorchPolicy


class SynergyAttentionMapper(nn.Module):
    def __init__(self, n_synergies, action_dim, state_dim, attn_dim=64, temperature=1.0):
        super().__init__()
        self.n_synergies = int(n_synergies)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.attn_dim = int(attn_dim)
        self.temperature = float(temperature)

        self.query_proj = nn.Linear(self.n_synergies + self.state_dim, self.attn_dim)

        # Learnable synergy basis descriptors and values.
        self.key_basis = nn.Parameter(torch.randn(self.n_synergies, self.attn_dim) * 0.1)
        self.value_basis = nn.Parameter(torch.randn(self.n_synergies, self.action_dim) * 0.1)
        self.output_scale = nn.Parameter(torch.ones(1, self.action_dim))

    def forward(self, synergy_coeffs, state):
        # synergy_coeffs: [B, S], state: [B, State_Dim]
        x = torch.cat([synergy_coeffs, state], dim=-1)
        q = self.query_proj(x)  # [B, D]
        logits = torch.matmul(q, self.key_basis.T) / np.sqrt(float(self.attn_dim))

        # Inject direct synergy intensity as a prior over attention weights.
        logits = (logits + synergy_coeffs) / max(self.temperature, 1e-6)

        attn = torch.sigmoid(logits)  # [B, S]
        action_mean = torch.matmul(attn, self.value_basis) * self.output_scale  # [B, A]
        return action_mean


class AttentionSynergyPolicy(TorchPolicy):
    def __init__(self, network, input_shape, output_shape, latent_shape,
                 std_a_0=1.0, std_x_0=1.0, learn_latent_layer=False, use_cuda=False,
                 synergy_attn_dim=64, synergy_temperature=1.0,
                 **params):
        super().__init__(use_cuda)

        del std_x_0
        del learn_latent_layer

        self._state_dim = int(input_shape[0])
        self._action_dim = int(output_shape[0])
        self._latent_dim = int(latent_shape[0])

        self._mu = Regressor(
            TorchApproximator,
            input_shape,
            latent_shape,
            network=network,
            use_cuda=use_cuda,
            **params,
        )

        self._synergy_mapper = SynergyAttentionMapper(
            n_synergies=self._latent_dim,
            action_dim=self._action_dim,
            state_dim=self._state_dim,
            attn_dim=synergy_attn_dim,
            temperature=synergy_temperature,
        )

        self._predict_params = dict()

        log_sigma_a_init = (torch.ones(self._action_dim) * np.log(std_a_0)).float()
        if self._use_cuda:
            self._synergy_mapper = self._synergy_mapper.cuda()
            log_sigma_a_init = log_sigma_a_init.cuda()

        self._log_sigma_a = nn.Parameter(log_sigma_a_init)

        self.deterministic = False

        self._mu_w_size = self._mu.weights_size
        self._mapper_w_size = int(sum(p.numel() for p in self._synergy_mapper.parameters()))

        self._add_save_attr(
            _state_dim='primitive',
            _action_dim='primitive',
            _latent_dim='primitive',
            _mu='mushroom',
            _synergy_mapper='torch',
            _predict_params='pickle',
            _log_sigma_a='torch',
            deterministic='primitive',
            _mu_w_size='primitive',
            _mapper_w_size='primitive',
        )

    def draw_action_t(self, state):
        if self.deterministic:
            return self.get_mean(state)
        return self.distribution_t(state).sample().detach()

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]

    def entropy_t(self, state):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma_a)

    def distribution_t(self, state):
        mu = self.get_mean(state)
        cov = torch.diag(torch.exp(2 * self._log_sigma_a))
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov, validate_args=False)

    def get_mean(self, state):
        synergy_coeffs = self._mu(state, **self._predict_params, output_tensor=True)
        return self._synergy_mapper(synergy_coeffs, state)

    def set_weights(self, weights):
        if not hasattr(self, '_mu_w_size'):
            self._mu_w_size = self._mu.weights_size
        if not hasattr(self, '_mapper_w_size'):
            self._mapper_w_size = int(sum(p.numel() for p in self._synergy_mapper.parameters()))

        self._mu.set_weights(weights[:self._mu_w_size])

        mapper_start = self._mu_w_size
        mapper_end = mapper_start + self._mapper_w_size
        mapper_vec = torch.from_numpy(weights[mapper_start:mapper_end]).float()
        if self.use_cuda:
            mapper_vec = mapper_vec.cuda()
        vector_to_parameters(mapper_vec, self._synergy_mapper.parameters())

        sigma_data = torch.from_numpy(weights[mapper_end:mapper_end + self._action_dim]).float()
        if self.use_cuda:
            sigma_data = sigma_data.cuda()
        self._log_sigma_a.data = sigma_data

    def get_weights(self):
        mu_weights = self._mu.get_weights()
        mapper_weights = parameters_to_vector(self._synergy_mapper.parameters()).detach().cpu().numpy()
        sigma_weights = self._log_sigma_a.data.detach().cpu().numpy()
        return np.concatenate([mu_weights, mapper_weights, sigma_weights])

    def parameters(self):
        return chain(
            self._mu.model.network.parameters(),
            self._synergy_mapper.parameters(),
            [self._log_sigma_a],
        )
