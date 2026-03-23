from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import TorchPolicy


class TCNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, history_len=5, num_channels=[128, 64], kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.history_len = history_len
        self.output_dim = output_dim

        layers = []
        in_channels = input_dim
        for out_channels in num_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size - 1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, output_dim)
        
    def forward(self, x):
        # x is expected to be [B, history_len, input_dim]
        # Conv1d expects [B, in_channels, seq_len]
        x_conv = x.transpose(1, 2) # [B, input_dim, history_len]
        out = self.network(x_conv) # [B, in_channels, seq_len + padding]
        # We only want the last timestep (causal)
        out = out[:, :, :self.history_len] # truncate padding acting as future
        last_out = out[:, :, -1] # [B, in_channels]
        return self.fc(last_out) # [B, output_dim]


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


class TcnAttentionSynergyPolicy(TorchPolicy):
    def __init__(self, network, input_shape, output_shape, latent_shape,
                 std_a_0=1.0, std_x_0=1.0, learn_latent_layer=False, use_cuda=False,
                 synergy_attn_dim=64, synergy_temperature=1.0, history_len=5,
                 **params):
        super().__init__(use_cuda)

        del std_x_0
        del learn_latent_layer
        del network # We will use TCNFeatureExtractor directly instead of mushroom_rl Network

        self._state_dim = int(input_shape[0])
        self._action_dim = int(output_shape[0])
        self._latent_dim = int(latent_shape[0])
        self._history_len = int(history_len)
        self._stand = params.get('standardizer', None)

        self._tcn = TCNFeatureExtractor(
            input_dim=self._state_dim,
            output_dim=self._latent_dim,
            history_len=self._history_len
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
            self._tcn = self._tcn.cuda()
            self._synergy_mapper = self._synergy_mapper.cuda()
            log_sigma_a_init = log_sigma_a_init.cuda()

        self._log_sigma_a = nn.Parameter(log_sigma_a_init)

        self.deterministic = False

        self._tcn_w_size = int(sum(p.numel() for p in self._tcn.parameters()))
        self._mapper_w_size = int(sum(p.numel() for p in self._synergy_mapper.parameters()))

        self._add_save_attr(
            _state_dim='primitive',
            _action_dim='primitive',
            _latent_dim='primitive',
            _history_len='primitive',
            _tcn='torch',
            _synergy_mapper='torch',
            _stand='pickle',
            _predict_params='pickle',
            _log_sigma_a='torch',
            deterministic='primitive',
            _tcn_w_size='primitive',
            _mapper_w_size='primitive',
        )
        
        # State buffer for rolling history during environment step
        self._state_buffer = None

    def _get_history_tensor(self, state):
        # state is [N, state_dim] or [state_dim]
        if state.dim() == 1:
            # Interaction step
            s = state.unsqueeze(0) # [1, state_dim]
            if self._state_buffer is None:
                self._state_buffer = s.repeat(self._history_len, 1) # [H, state_dim]
            else:
                self._state_buffer = torch.cat([self._state_buffer[1:], s], dim=0) # Shift & Append
            # Return [1, H, state_dim]
            return self._state_buffer.unsqueeze(0)
        else:
            # Batch update (N > 1)
            # Reset buffer to None just in case, though it's harmless
            self._state_buffer = None
            
            N, D = state.shape
            
            # Since TRPO evaluates on flattened sequences, we can artificially build rolling history
            # padding the start with the first element (or zeros)
            padded = torch.cat([state[0:1].repeat(self._history_len - 1, 1), state], dim=0)
            
            # unfold: creates rolling windows of size history_len
            # [N, history_len, D]
            stacked = padded.unfold(0, self._history_len, 1).transpose(1, 2)
            
            return stacked

    def reset(self):
        # Called at the end of each episode? Standard Policy doesn't have reset.
        # We can safely have a rolling buffer even across episode reset for eval, 
        # but optimally it should clear. We'll add reset capability.
        self._state_buffer = None

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
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = state.float()
            
        if self.use_cuda:
            state = state.cuda()
                
        if self._stand is not None:
            state = self._stand(state).float()

        # state could be [ObsDim] or [B, ObsDim]
        history_state = self._get_history_tensor(state) # [B, H, ObsDim]
        
        # Extracted latent synergies
        synergy_coeffs = self._tcn(history_state) # [B, S]
        
        # State used for attention mapping is the *current* state (last in history)
        current_state = history_state[:, -1, :] # [B, ObsDim]
        
        return self._synergy_mapper(synergy_coeffs, current_state)

    def set_weights(self, weights):
        if not hasattr(self, '_tcn_w_size'):
            self._tcn_w_size = int(sum(p.numel() for p in self._tcn.parameters()))
        if not hasattr(self, '_mapper_w_size'):
            self._mapper_w_size = int(sum(p.numel() for p in self._synergy_mapper.parameters()))

        tcn_end = self._tcn_w_size
        tcn_vec = torch.from_numpy(weights[:tcn_end]).float()
        
        mapper_end = tcn_end + self._mapper_w_size
        mapper_vec = torch.from_numpy(weights[tcn_end:mapper_end]).float()
        
        if self.use_cuda:
            tcn_vec = tcn_vec.cuda()
            mapper_vec = mapper_vec.cuda()
            
        vector_to_parameters(tcn_vec, self._tcn.parameters())
        vector_to_parameters(mapper_vec, self._synergy_mapper.parameters())

        sigma_data = torch.from_numpy(weights[mapper_end:mapper_end + self._action_dim]).float()
        if self.use_cuda:
            sigma_data = sigma_data.cuda()
        self._log_sigma_a.data = sigma_data

    def get_weights(self):
        tcn_weights = parameters_to_vector(self._tcn.parameters()).detach().cpu().numpy()
        mapper_weights = parameters_to_vector(self._synergy_mapper.parameters()).detach().cpu().numpy()
        sigma_weights = self._log_sigma_a.data.detach().cpu().numpy()
        return np.concatenate([tcn_weights, mapper_weights, sigma_weights])

    def parameters(self):
        return chain(
            self._tcn.parameters(),
            self._synergy_mapper.parameters(),
            [self._log_sigma_a],
        )
