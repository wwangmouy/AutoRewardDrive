"""AutoRewardDrive: Reward Learning Module"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from collections import deque, namedtuple

from reward_model import Reward, ValueFunction

Transition = namedtuple('Transition', [
    'bev_image', 'vector_state', 'action', 'sparse_reward', 'log_prob', 'overline_V',
    'next_bev', 'next_vector', 'done'
])


class RunningMeanStd:
    """Running mean and std for reward normalization"""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x) if hasattr(x, '__len__') else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RewardLearner:
    """Reward learning with EMA target encoder for stability"""
    
    def __init__(self, config, shared_encoder=None, device="cuda:0"):
        self.config = config
        self.device = device
        params = config.reward_learning_params
        
        self.gamma = params.gamma
        self.lr = params.reward_lr
        self.hidden_dim = params.hidden_dim
        self.n_samples = params.n_samples
        self.encoder_tau = params.encoder_tau
        self.steps_per_traj = params.get('steps_per_traj', 32)
        self.min_trajectories = params.get('min_trajectories', 10)
        self.min_steps_per_traj = params.get('min_steps_per_traj', 50)
        
        self.shared_encoder = shared_encoder
        self.target_encoder = None
        
        if shared_encoder is not None:
            state_feature_dim = shared_encoder.output_dim
            self._init_target_encoder(shared_encoder)
        else:
            state_feature_dim = 192
        
        self.reward_function = Reward(
            state_feature_dim=state_feature_dim,
            action_dim=2,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.value_function = ValueFunction(
            state_feature_dim=state_feature_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.reward_optimizer = optim.Adam(self.reward_function.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=self.lr)
        
        # New: V_omega for computing Advantage of learned reward
        self.learned_value_function = ValueFunction(
            state_feature_dim=state_feature_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        self.learned_value_optimizer = optim.Adam(self.learned_value_function.parameters(), lr=self.lr)
        
        self.trajectory_buffer = deque(maxlen=params.reward_buffer_size)
        
        # Reward normalization for SAC stability
        self.reward_normalizer = RunningMeanStd()
    
    def _init_target_encoder(self, encoder):
        import copy
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.to(self.device)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()  # Always eval mode for stability
    
    def update_target_encoder(self):
        if self.target_encoder is None or self.shared_encoder is None:
            return
        for target_param, param in zip(self.target_encoder.parameters(), 
                                        self.shared_encoder.parameters()):
            target_param.data.copy_(self.encoder_tau * param.data + 
                                    (1 - self.encoder_tau) * target_param.data)

    def set_shared_encoder(self, encoder):
        self.shared_encoder = encoder
        self._init_target_encoder(encoder)

    def get_state_features(self, bev, vector):
        if self.target_encoder is None:
            if self.shared_encoder is None:
                raise ValueError("Encoder not set")
            return self.shared_encoder(bev, vector)
        self.target_encoder.eval()  # Ensure eval mode
        return self.target_encoder(bev, vector)

    def get_reward(self, bev, vector, action, update_stats=True):
        if bev.dim() == 3:
            bev = bev.unsqueeze(0)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        with torch.no_grad():
            state_features = self.get_state_features(bev, vector)
            raw_reward = self.reward_function(state_features, action).cpu().numpy().squeeze()
        
        # Update stats if requested
        if update_stats:
            if raw_reward.ndim == 0:
                self.reward_normalizer.update([float(raw_reward)])
            else:
                self.reward_normalizer.update(raw_reward)
        
        if raw_reward.ndim == 0:
            normalized_reward = self.reward_normalizer.normalize(float(raw_reward))
            return float(np.clip(normalized_reward, -10, 10))
        else:
            normalized_rewards = self.reward_normalizer.normalize(raw_reward)
            return np.clip(normalized_rewards, -10, 10)

    def get_reward_from_features(self, state_features, action):
        return self.reward_function(state_features, action)

    def store_trajectory(self, trajectory):
        processed = self._compute_returns(trajectory)
        self.trajectory_buffer.append(processed)

    def _compute_returns(self, trajectory):
        processed = []
        overline_V = 0.0
        for transition in reversed(trajectory):
            overline_V = transition.sparse_reward + self.gamma * overline_V
            processed.insert(0, transition._replace(overline_V=overline_V))
        return processed

    def optimize_upper_level(self, agent):
        """Upper-level optimization with mini-batch sampling"""
        if len(self.trajectory_buffer) < self.min_trajectories:
            return None
        
        # Filter trajectories that assume minimum steps
        valid_indices = [i for i, t in enumerate(self.trajectory_buffer) 
                         if len(t) >= self.min_steps_per_traj]
        
        if len(valid_indices) < self.n_samples:
             if len(valid_indices) == 0:
                 return None
             traj_indices = np.random.choice(valid_indices, len(valid_indices), replace=False)
        else:
             traj_indices = np.random.choice(valid_indices, self.n_samples, replace=False)

        steps_per_traj = self.steps_per_traj
        
        bev_batch, vector_batch, action_batch, overline_V_batch = [], [], [], []
        next_bev_batch, next_vector_batch, done_batch, old_log_prob_batch = [], [], [], []
        
        # Optimization: cast deque to list once
        buffer_list = list(self.trajectory_buffer)
        
        for idx in traj_indices:
            trajectory = buffer_list[idx]
            # Sample M steps (or fewer if traj is short, but we filtered for min len)
            sample_size = min(len(trajectory), steps_per_traj)
            # Random uniform sampling from trajectory
            for i in np.random.choice(len(trajectory), sample_size, replace=False):
                t = trajectory[i]
                bev_batch.append(t.bev_image)
                vector_batch.append(t.vector_state)
                action_batch.append(t.action)
                overline_V_batch.append(t.overline_V)
                next_bev_batch.append(t.next_bev)
                next_vector_batch.append(t.next_vector)
                done_batch.append(t.done)
                old_log_prob_batch.append(t.log_prob)
        
        if len(bev_batch) == 0:
            return None
        
        bev_batch = torch.stack(bev_batch).to(self.device)
        vector_batch = torch.stack(vector_batch).to(self.device)
        action_batch = torch.stack(action_batch).to(self.device)
        overline_V_batch = torch.tensor(overline_V_batch, dtype=torch.float32, device=self.device)
        
        next_bev_batch = torch.stack(next_bev_batch).to(self.device)
        next_vector_batch = torch.stack(next_vector_batch).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device).view(-1)
        # log_prob is stored as float, so use tensor() instead of stack()
        old_log_prob_batch = torch.tensor(old_log_prob_batch, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            state_features = self.get_state_features(bev_batch, vector_batch)
            next_state_features = self.get_state_features(next_bev_batch, next_vector_batch)
            
            # Re-compute current log_prob for Importance Sampling
            # We use agent.evaluate_log_prob(bev, vector, action) from actor
            new_log_prob = agent.evaluate_log_prob(bev_batch, vector_batch, action_batch).view(-1)
        
        # Importance Sampling Weights (log-space clamp for numerical stability)
        is_clip_min = self.config.reward_learning_params.get("is_clip_min", 0.1)
        is_clip_max = self.config.reward_learning_params.get("is_clip_max", 10.0)
        log_is_weights = (new_log_prob - old_log_prob_batch).clamp(np.log(is_clip_min), np.log(is_clip_max))
        is_weights = torch.exp(log_is_weights)

        V_s = self.value_function(state_features).squeeze()
        advantage = overline_V_batch - V_s.detach()
        
        # Center advantage (remove mean shift)
        advantage = advantage - advantage.mean()
        
        # Robust normalization
        std = advantage.std().clamp(min=1e-3)
        normalized_advantage = advantage / (std + 1e-8)
        
        # Learned Reward and Advantage_omega
        learned_rewards = self.reward_function(state_features, action_batch).squeeze()
        V_omega_s = self.learned_value_function(state_features).squeeze()
        V_omega_next = self.learned_value_function(next_state_features).detach().squeeze()
        
        advantage_omega = learned_rewards + self.gamma * V_omega_next * (1 - done_batch) - V_omega_s.detach()
        
        # Loss = - E [ w_IS * bar_A * A_omega ]
        # We removed variance_loss as it conflicts with the pure objective
        loss = -torch.mean(is_weights * normalized_advantage * advantage_omega)
        
        self.reward_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_function.parameters(), max_norm=1.0)
        self.reward_optimizer.step()
        
        self._update_value_function(state_features.detach(), overline_V_batch)
        
        # Update V_omega
        # Target = r_omega + gamma * V_omega(s')
        v_omega_target = (learned_rewards.detach() + self.gamma * V_omega_next * (1 - done_batch)).detach()
        v_omega_loss = nn.functional.mse_loss(V_omega_s, v_omega_target)
        
        self.learned_value_optimizer.zero_grad()
        v_omega_loss.backward()
        self.learned_value_optimizer.step()

        return loss.item()
    
    def _update_value_function(self, state_features, target_V):
        pred_V = self.value_function(state_features).squeeze()
        loss = nn.functional.smooth_l1_loss(pred_V, target_V)
        
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def save(self, path):
        torch.save({
            'reward_function': self.reward_function.state_dict(),
            'value_function': self.value_function.state_dict(),
            'learned_value_function': self.learned_value_function.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.reward_function.load_state_dict(checkpoint['reward_function'])
        self.value_function.load_state_dict(checkpoint['value_function'])
        if 'learned_value_function' in checkpoint:
            self.learned_value_function.load_state_dict(checkpoint['learned_value_function'])
