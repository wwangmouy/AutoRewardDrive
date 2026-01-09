"""AutoRewardDrive: Reward Network Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """MLP encoder for action"""
    
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, action):
        return self.mlp(action)


class RewardHead(nn.Module):
    """Reward prediction head using shared state features"""
    
    def __init__(self, state_dim, action_dim=2, hidden_dim=128):
        super().__init__()
        self.action_encoder = ActionEncoder(action_dim, 32, 32)
        self.fc1 = nn.Linear(state_dim + 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.reward = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, state_features, action):
        action_feat = self.action_encoder(action)
        x = torch.cat([state_features, action_feat], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.reward(x)


class ValueHead(nn.Module):
    """Value function head V(s) using shared state features"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, state_features):
        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        return self.value(x)


class Reward(nn.Module):
    """Reward network R_ω(s, a) using shared encoder"""
    
    def __init__(self, state_feature_dim=192, action_dim=2, hidden_dim=128):
        super().__init__()
        self.reward_head = RewardHead(state_feature_dim, action_dim, hidden_dim)
    
    def forward(self, state_features, action):
        return self.reward_head(state_features, action)


class ValueFunction(nn.Module):
    """Value function V(s) using shared encoder"""
    
    def __init__(self, state_feature_dim=192, hidden_dim=128):
        super().__init__()
        self.value_head = ValueHead(state_feature_dim, hidden_dim)
    
    def forward(self, state_features):
        return self.value_head(state_features)
