"""
AutoRewardDrive: Configuration
Configuration for automatic reward discovery in autonomous driving
"""

import torch as th
from box import Box
import numpy as np
from utils import lr_schedule

import torch.nn as nn
import torch


# ============================================================================
# AutoRewardDrive Algorithm Parameters
# ============================================================================
algorithm_params = {
    "gamma": 0.99,           # Discount factor
    "tau": 0.005,            # Soft update coefficient
    "alpha": 0.2,            # Entropy coefficient (auto-tuned)
    "lr": 3e-4,              # Learning rate
    "batch_size": 256,       # Batch size for training
    "buffer_size": 100000,   # Replay buffer size
    "learning_starts": 10000, # Steps before starting training
}

# ============================================================================
# State Representation
# ============================================================================
states = {
    "bev": ["steer", "throttle", "speed", "angle_next_waypoint", "waypoints", "seg_camera"],
}

# ============================================================================
# Reward Parameters (for sparse_reward function)
# ============================================================================
reward_params = {
    "sparse_reward": dict(
        early_stop=True,
        min_speed=20.0,
        max_speed=35.0,
        target_speed=25.0,
        max_distance=3.0,
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
}

# ============================================================================
# Bilevel Reward Learning Parameters
# ============================================================================
reward_learning_params = {
    "hidden_dim": 128,
    "encode_dim": 64,
    "reward_lr": 1e-4,
    "gamma": 0.99,
    "reward_buffer_size": 1000,
    "n_samples": 10,
    "activate_function": "relu",
    "last_activate_function": "None",
}

# ============================================================================
# Main Configuration
# ============================================================================
_CONFIG_DEFAULT = {
    "algorithm": "AutoRewardDrive",
    "algorithm_params": algorithm_params,
    "state": states["bev"],
    "action_smoothing": 0.75,
    "reward_fn": "sparse_reward",
    "reward_params": reward_params["sparse_reward"],
    "reward_learning_params": reward_learning_params,
    "obs_res": (80, 120),
    "seed": 42,
    "wrappers": [],
    "use_seg_bev": True,
    "use_rgb_bev": False,
}

CONFIGS = {
    "default": _CONFIG_DEFAULT,
}

CONFIG = None


def set_config(config_name="default"):
    global CONFIG
    CONFIG = Box(CONFIGS[config_name], default_box=True)
    return CONFIG
