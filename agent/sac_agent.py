"""
AutoRewardDrive: SAC Agent with Shared Encoder
Custom Soft Actor-Critic using shared BEV encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random

from shared_encoder import SharedStateEncoder


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, bev, vector, action, reward, next_bev, next_vector, done):
        self.buffer.append((bev, vector, action, reward, next_bev, next_vector, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        bev, vector, action, reward, next_bev, next_vector, done = zip(*batch)
        
        # Handle both tensor and numpy inputs
        def to_tensor(data, dtype=torch.float32):
            if isinstance(data[0], torch.Tensor):
                return torch.stack(data)
            else:
                return torch.tensor(np.array(data), dtype=dtype)
        
        return (
            to_tensor(bev),
            to_tensor(vector),
            to_tensor(action),
            to_tensor(reward).unsqueeze(1) if len(to_tensor(reward).shape) == 1 else to_tensor(reward),
            to_tensor(next_bev),
            to_tensor(next_vector),
            to_tensor(done).unsqueeze(1) if len(to_tensor(done).shape) == 1 else to_tensor(done)
        )
    
    def __len__(self):
        return len(self.buffer)


class ActorHead(nn.Module):
    """Actor head using shared encoder features"""
    
    def __init__(self, input_dim, action_dim=2, hidden_dim=256, 
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state_features):
        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state_features):
        mean, log_std = self.forward(state_features)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean


class CriticHead(nn.Module):
    """Critic head (Q-function) using shared encoder features"""
    
    def __init__(self, input_dim, action_dim=2, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
    
    def forward(self, state_features, action):
        x = torch.cat([state_features, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class AutoRewardDrive_SAC:
    """
    AutoRewardDrive Agent based on SAC with Shared Encoder
    """
    
    def __init__(self, config, shared_encoder=None, device="cuda:0"):
        self.device = device
        self.gamma = config.algorithm_params.gamma
        self.tau = config.algorithm_params.tau
        self.alpha = config.algorithm_params.alpha
        self.lr = config.algorithm_params.lr
        self.batch_size = config.algorithm_params.batch_size
        
        # Dimensions
        self.bev_channels = 6
        self.vector_dim = 34
        self.action_dim = 2
        
        # Shared encoder (can be passed from outside)
        if shared_encoder is None:
            self.shared_encoder = SharedStateEncoder(
                bev_channels=self.bev_channels,
                vector_dim=self.vector_dim,
                bev_features=128,
                vector_features=64
            ).to(device)
        else:
            self.shared_encoder = shared_encoder
        
        encoder_output_dim = self.shared_encoder.output_dim
        
        # Actor and Critic heads
        self.actor = ActorHead(encoder_output_dim, self.action_dim).to(device)
        self.critic1 = CriticHead(encoder_output_dim, self.action_dim).to(device)
        self.critic2 = CriticHead(encoder_output_dim, self.action_dim).to(device)
        self.critic1_target = CriticHead(encoder_output_dim, self.action_dim).to(device)
        self.critic2_target = CriticHead(encoder_output_dim, self.action_dim).to(device)
        
        # Copy weights to target
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        # Strategy: Update shared_encoder ONLY with actor (more stable for visual RL)
        # Critic uses detached features to avoid conflicting gradients
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.shared_encoder.parameters()), 
            lr=self.lr
        )
        # Critic optimizer does NOT include shared_encoder
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.lr
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=config.algorithm_params.buffer_size)
        
        # Auto-tune alpha
        self.auto_entropy_tuning = config.algorithm_params.get('auto_entropy_tuning', True)
        if self.auto_entropy_tuning:
            self.target_entropy = -self.action_dim  # Heuristic value
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
    
    def get_state_features(self, bev, vector):
        """Get encoded state features"""
        return self.shared_encoder(bev, vector)
    
    def select_action(self, bev, vector, evaluate=False):
        """Select action given state"""
        with torch.no_grad():
            bev = bev.to(self.device)
            vector = vector.to(self.device)
            state_features = self.get_state_features(bev, vector)
            if evaluate:
                mean, _ = self.actor(state_features)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state_features)
        return action.cpu()
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample from replay buffer
        bev, vector, action, reward, next_bev, next_vector, done = self.replay_buffer.sample(self.batch_size)
        bev = bev.to(self.device)
        vector = vector.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_bev = next_bev.to(self.device)
        next_vector = next_vector.to(self.device)
        done = done.to(self.device)
        
        # Update critics
        with torch.no_grad():
            # Compute next state features without gradients
            next_state_features = self.get_state_features(next_bev, next_vector)
            next_action, next_log_prob, _ = self.actor.sample(next_state_features)
            target_q1 = self.critic1_target(next_state_features, next_action)
            target_q2 = self.critic2_target(next_state_features, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Compute current state features with gradients for critic update
        # Use detached features so encoder is NOT updated by critic
        state_features = self.get_state_features(bev, vector).detach()
        current_q1 = self.critic1(state_features, action)
        current_q2 = self.critic2(state_features, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Combined critic loss - update both Q-networks together
        critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor (and shared encoder)
        # Recompute state features WITH gradients for encoder update
        state_features = self.get_state_features(bev, vector)
        new_action, log_prob, _ = self.actor.sample(state_features)
        
        # Detach state features when passing to critic to prevent critic gradients affecting actor update
        state_features_for_critic = state_features.detach()
        q1_new = self.critic1(state_features_for_critic, new_action)
        q2_new = self.critic2(state_features_for_critic, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # This updates both actor AND shared_encoder
        self.actor_optimizer.step()
        
        # Update alpha (if auto-tuning is enabled)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
                'actor_loss': actor_loss.item(),
                'alpha': self.alpha}
    
    def save(self, path):
        torch.save({
            'shared_encoder': self.shared_encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.shared_encoder.load_state_dict(checkpoint['shared_encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
