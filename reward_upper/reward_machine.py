"""
AutoRewardDrive: Reward Learning Module
Upper-level optimization for learning optimal reward function using shared encoder
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from collections import deque, namedtuple

from reward_model import Reward, ValueFunction

# Transition tuple for trajectory storage
Transition = namedtuple('Transition', [
    'bev_image',
    'vector_state',
    'action',
    'sparse_reward',
    'log_prob',
    'mu',
    'overline_V'
])


class RewardLearner:
    """
    Reward learning with shared encoder
    Uses the same encoder as the policy network
    """
    
    def __init__(self, config, shared_encoder=None, device="cuda:0"):
        self.device = device
        params = config.reward_learning_params
        
        self.gamma = params.gamma
        self.lr = params.reward_lr
        self.hidden_dim = params.hidden_dim
        self.n_samples = params.n_samples
        
        # Shared encoder (passed from SAC agent)
        self.shared_encoder = shared_encoder
        
        # Get encoder output dimension
        if shared_encoder is not None:
            state_feature_dim = shared_encoder.output_dim
        else:
            state_feature_dim = 192  # default: 128 + 64
        
        # Reward and value heads (only heads, encoder is shared)
        self.reward_function = Reward(
            state_feature_dim=state_feature_dim,
            action_dim=2,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.value_function = ValueFunction(
            state_feature_dim=state_feature_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Optimizers (only for heads, not encoder)
        self.reward_optimizer = optim.Adam(self.reward_function.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=self.lr)
        
        # Trajectory buffer
        self.trajectory_buffer = deque(maxlen=params.reward_buffer_size)

    def set_shared_encoder(self, encoder):
        """Set shared encoder (called after initialization)"""
        self.shared_encoder = encoder

    def get_state_features(self, bev, vector):
        """Get state features using shared encoder"""
        if self.shared_encoder is None:
            raise ValueError("Shared encoder not set")
        return self.shared_encoder(bev, vector)

    def get_reward(self, bev, vector, action):
        """Get learned reward for state-action pair (with raw inputs)"""
        if bev.dim() == 3:
            bev = bev.unsqueeze(0)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        with torch.no_grad():
            state_features = self.get_state_features(bev, vector)
            reward = self.reward_function(state_features, action)
        return reward.cpu().numpy().squeeze()

    def get_reward_from_features(self, state_features, action):
        """Get learned reward from pre-computed state features"""
        return self.reward_function(state_features, action)

    def store_trajectory(self, trajectory):
        """Store trajectory with computed returns"""
        processed = self._compute_returns(trajectory)
        self.trajectory_buffer.append(processed)

    def _compute_returns(self, trajectory):
        """Compute discounted returns"""
        processed = []
        overline_V = 0.0
        for transition in reversed(trajectory):
            overline_V = transition.sparse_reward + self.gamma * overline_V
            processed.insert(0, transition._replace(overline_V=overline_V))
        return processed

    def optimize_upper_level(self, agent):
        """Perform upper-level optimization"""
        if len(self.trajectory_buffer) == 0:
            return None
        
        # Flatten transitions
        all_transitions = [t for traj in self.trajectory_buffer for t in traj]
        np.random.shuffle(all_transitions)
        
        # Collect batches
        bev_batch, vector_batch, action_batch = [], [], []
        sparse_reward_batch, overline_V_batch = [], []
        
        for t in all_transitions:
            bev_batch.append(t.bev_image)
            vector_batch.append(t.vector_state)
            action_batch.append(t.action)
            sparse_reward_batch.append(t.sparse_reward)
            overline_V_batch.append(t.overline_V)
        
        # Convert to tensors
        bev_batch = torch.stack(bev_batch).to(self.device)
        vector_batch = torch.stack(vector_batch).to(self.device)
        action_batch = torch.stack(action_batch).to(self.device)
        overline_V_batch = torch.tensor(overline_V_batch, dtype=torch.float32, device=self.device)
        
        # Get state features using shared encoder
        # IMPORTANT: Use no_grad to prevent reward learning from updating encoder
        # Encoder should only be updated by actor optimizer in the lower-level RL
        with torch.no_grad():
            state_features = self.get_state_features(bev_batch, vector_batch)
        
        # Compute value estimates
        V_s = self.value_function(state_features).squeeze()
        
        # Compute advantage
        advantage = overline_V_batch - V_s.detach()
        
        # Compute learned rewards
        learned_rewards = self.reward_function(state_features, action_batch).squeeze()
        
        # Upper-level loss: encourage reward to correlate with advantage
        loss = -torch.mean(advantage * learned_rewards)
        
        # Update reward function
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()
        
        # Update value function
        self._update_value_function(state_features.detach(), overline_V_batch)
        
        return loss.item()

    def _update_value_function(self, state_features, target_V):
        """Fit value function to target values"""
        pred_V = self.value_function(state_features).squeeze()
        loss = nn.functional.smooth_l1_loss(pred_V, target_V)
        
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def save(self, path):
        torch.save({
            'reward_function': self.reward_function.state_dict(),
            'value_function': self.value_function.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.reward_function.load_state_dict(checkpoint['reward_function'])
        self.value_function.load_state_dict(checkpoint['value_function'])
