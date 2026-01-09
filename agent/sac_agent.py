"""AutoRewardDrive: SAC Agent with Shared Encoder"""

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
    """Experience replay buffer with consistent data types"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def _to_numpy(self, data, dtype=np.float32):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(dtype)
        elif isinstance(data, np.ndarray):
            return data.astype(dtype)
        else:
            return np.array(data, dtype=dtype)
    
    def push(self, bev, vector, action, reward, next_bev, next_vector, done):
        self.buffer.append((
            self._to_numpy(bev, np.float32),
            self._to_numpy(vector, np.float32),
            self._to_numpy(action, np.float32),
            np.float32(reward),
            self._to_numpy(next_bev, np.float32),
            self._to_numpy(next_vector, np.float32),
            np.float32(done)
        ))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        bev, vector, action, reward, next_bev, next_vector, done = zip(*batch)
        return (
            torch.from_numpy(np.stack(bev)),
            torch.from_numpy(np.stack(vector)),
            torch.from_numpy(np.stack(action)),
            torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(1),
            torch.from_numpy(np.stack(next_bev)),
            torch.from_numpy(np.stack(next_vector)),
            torch.from_numpy(np.array(done, dtype=np.float32)).unsqueeze(1)
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
    
    def evaluate_log_prob(self, state_features, action):
        """Evaluate log probability of a given action"""
        mean, log_std = self.forward(state_features)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # Inverse tanh: atanh(clamp(action))
        action_clamped = action.clamp(-0.9999, 0.9999) 
        x_t = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))
        
        log_prob = dist.log_prob(x_t) - torch.log(1 - action_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob

    def sample(self, state_features):
        mean, log_std = self.forward(state_features)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        action_clamped = action.clamp(-0.9999, 0.9999)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action_clamped.pow(2) + 1e-6)
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
    """SAC Agent with shared encoder for visual RL"""
    
    def __init__(self, config, shared_encoder=None, device="cuda:0"):
        self.device = device
        self.gamma = config.algorithm_params.gamma
        self.tau = config.algorithm_params.tau
        self.alpha = config.algorithm_params.alpha
        self.lr = config.algorithm_params.lr
        self.batch_size = config.algorithm_params.batch_size
        
        self.bev_channels = 6
        self.vector_dim = 34
        self.action_dim = 2
        
        if shared_encoder is None:
            self.shared_encoder = SharedStateEncoder(
                bev_channels=self.bev_channels,
                vector_dim=self.vector_dim,
                bev_features=128,
                vector_features=64
            ).to(device)
        else:
            self.shared_encoder = shared_encoder
        
        self.encoder_target = SharedStateEncoder(
            bev_channels=self.bev_channels,
            vector_dim=self.vector_dim,
            bev_features=128,
            vector_features=64
        ).to(device)
        self.encoder_target.load_state_dict(self.shared_encoder.state_dict())
        for param in self.encoder_target.parameters():
            param.requires_grad = False
        self.encoder_target.eval()  # Always eval mode
        
        encoder_output_dim = self.shared_encoder.output_dim
        
        self.actor = ActorHead(encoder_output_dim, self.action_dim).to(device)
        self.critic1 = CriticHead(encoder_output_dim, self.action_dim).to(device)
        self.critic2 = CriticHead(encoder_output_dim, self.action_dim).to(device)
        self.critic1_target = CriticHead(encoder_output_dim, self.action_dim).to(device)
        self.critic2_target = CriticHead(encoder_output_dim, self.action_dim).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_target.eval()
        self.critic2_target.eval()
        
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()) + list(self.shared_encoder.parameters()),
            lr=self.lr
        )
        
        self.replay_buffer = ReplayBuffer(capacity=config.algorithm_params.buffer_size)
        
        self.auto_entropy_tuning = config.algorithm_params.get('auto_entropy_tuning', True)
        if self.auto_entropy_tuning:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
    
    def get_state_features(self, bev, vector):
        return self.shared_encoder(bev, vector)
    
    def get_target_state_features(self, bev, vector):
        self.encoder_target.eval()  # Ensure eval mode
        return self.encoder_target(bev, vector)
    
    def select_action(self, bev, vector, evaluate=False):
        with torch.no_grad():
            bev = bev.to(self.device)
            vector = vector.to(self.device)
            state_features = self.get_state_features(bev, vector)
            if evaluate:
                mean, _ = self.actor(state_features)
                action = torch.tanh(mean)
                log_prob = torch.zeros(1, device=self.device) # Dummy log_prob for eval
            else:
                action, log_prob, _ = self.actor.sample(state_features)
        return action.cpu(), log_prob.cpu()

    def evaluate_log_prob(self, bev, vector, action):
        """Estimate log_prob of arbitrary actions given states"""
        bev = bev.to(self.device)
        vector = vector.to(self.device)
        action = action.to(self.device)
        state_features = self.get_state_features(bev, vector).detach()
        with torch.no_grad():
             log_prob = self.actor.evaluate_log_prob(state_features, action)
        return log_prob
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        bev, vector, action, reward, next_bev, next_vector, done = self.replay_buffer.sample(self.batch_size)
        bev = bev.to(self.device)
        vector = vector.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_bev = next_bev.to(self.device)
        next_vector = next_vector.to(self.device)
        done = done.to(self.device)
        
        # Compute target Q
        with torch.no_grad():
            # Use shared_encoder for policy (actor) to sample next_action
            next_feat_pi = self.get_state_features(next_bev, next_vector)
            next_action, next_log_prob, _ = self.actor.sample(next_feat_pi)
            
            # Use target encoder for target critics to evaluate Q
            next_feat_q = self.get_target_state_features(next_bev, next_vector)
            target_q1 = self.critic1_target(next_feat_q, next_action)
            target_q2 = self.critic2_target(next_feat_q, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Update critics (with encoder gradients)
        state_features = self.get_state_features(bev, vector)
        current_q1 = self.critic1(state_features, action)
        current_q2 = self.critic2(state_features, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor (detached encoder)
        # Freeze critic parameters during actor update to save computation
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False
            
        state_features = self.get_state_features(bev, vector).detach()
        new_action, log_prob, _ = self.actor.sample(state_features)
        q1_new = self.critic1(state_features, new_action)
        q2_new = self.critic2(state_features, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True
        
        # Update alpha
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update targets
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.shared_encoder.parameters(), self.encoder_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
                'actor_loss': actor_loss.item(),
                'alpha': self.alpha}
    
    def save(self, path):
        torch.save({
            'shared_encoder': self.shared_encoder.state_dict(),
            'encoder_target': self.encoder_target.state_dict(),
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.shared_encoder.load_state_dict(checkpoint['shared_encoder'])
        self.encoder_target.load_state_dict(checkpoint['encoder_target'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
