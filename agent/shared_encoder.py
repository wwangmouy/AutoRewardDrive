"""AutoRewardDrive: Shared Encoder Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedBEVEncoder(nn.Module):
    """CNN encoder for BEV semantic images with spatial awareness"""
    
    def __init__(self, input_channels=6, features_dim=256):
        super().__init__()
        # Keep 4x4 spatial resolution to preserve obstacle location info
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4 spatial grid instead of 1x1
            nn.Flatten(),
        )
        # 256 channels * 4 * 4 = 4096
        self.fc = nn.Linear(256 * 4 * 4, features_dim)
        self.features_dim = features_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        return F.relu(self.fc(self.cnn(x)))


class SharedVectorEncoder(nn.Module):
    """MLP encoder for vector observations"""
    
    def __init__(self, input_dim=34, hidden_dim=64, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.output_dim = output_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.mlp(x)


class SharedStateEncoder(nn.Module):
    """Combined state encoder (BEV + Vector)"""
    
    def __init__(self, bev_channels=6, vector_dim=34, bev_features=128, vector_features=64):
        super().__init__()
        self.bev_encoder = SharedBEVEncoder(bev_channels, bev_features)
        self.vector_encoder = SharedVectorEncoder(vector_dim, 64, vector_features)
        self.output_dim = bev_features + vector_features
    
    def forward(self, bev, vector):
        bev_feat = self.bev_encoder(bev)
        vec_feat = self.vector_encoder(vector)
        return torch.cat([bev_feat, vec_feat], dim=-1)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
