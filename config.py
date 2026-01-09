"""AutoRewardDrive: Configuration"""

import torch as th
from box import Box
import numpy as np
from utils import lr_schedule

import torch.nn as nn
import torch

algorithm_params = {
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "lr": 3e-4,
    "batch_size": 256,
    "buffer_size": 100000,
    "learning_starts": 10000,
}

states = {
    "bev": ["steer", "throttle", "speed", "angle_next_waypoint", "waypoints", "seg_camera"],
}

reward_params = {
    "sparse_reward": dict(
        early_stop=True,
        max_center_deviation=3.0,
        max_speed=35.0,
        max_steps=1000,
        # max_distance removed to avoid confusion with center deviation
    ),
}

reward_learning_params = {
    "hidden_dim": 128,
    "encode_dim": 64,
    "reward_lr": 1e-4,
    "gamma": 0.99,
    "reward_buffer_size": 100,  # Small buffer to keep trajectories fresh for SAC
    "n_samples": 10,
    "min_trajectories": 10,
    "min_steps_per_traj": 10,   # Minimum trajectory length (10-20 for CARLA)
    "reward_warmup_steps": 10000,
    "reward_mixing_steps": 20000,
    "is_clip_min": 0.1,         # Importance Sampling clip min
    "is_clip_max": 10.0,        # Importance Sampling clip max
    "encoder_tau": 0.005,
    "steps_per_traj": 32,      # Number of steps to sample per trajectory (M items)
    "activate_function": "relu",
    "last_activate_function": "None",
}

_CONFIG_DEFAULT = {
    "algorithm": "AutoRewardDrive",
    "algorithm_params": algorithm_params,
    "state": states["bev"],
    "action_smoothing": 0.0,
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
