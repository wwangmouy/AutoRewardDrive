import torch as th
from box import Box
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch.nn as nn
import gymnasium as gym
import torch


class CustomCNN(nn.Module):
    def __init__(self, input_shape, features_dim=1):
        super(CustomCNN, self).__init__()
        self.channels_first, cnn_input_shape = self._canonicalize_image_shape(input_shape)
        n_input_channels = cnn_input_shape[0]

        if n_input_channels == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2),  # (16, 58, 38)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 28, 18)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (64, 13, 8)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),  # (128, 6, 4)
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 2)
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *cnn_input_shape)).view(-1).shape[0]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    @staticmethod
    def _canonicalize_image_shape(input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"CustomCNN expects a 3D image shape, got {input_shape}")

        if input_shape[0] <= 8 and input_shape[1] > 8 and input_shape[2] > 8:
            return True, input_shape

        if input_shape[2] <= 8 and input_shape[0] > 8 and input_shape[1] > 8:
            return False, (input_shape[2], input_shape[0], input_shape[1])

        raise ValueError(f"Unable to infer channel position for image shape {input_shape}")

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if not self.channels_first:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if torch.amax(x) > 1.0:
            x = x / 255.0
        x = self.cnn(x)
        x = self.linear(x)
        return x


class CustomMultiInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomMultiInputExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0
        image_keys = {"rgb_camera", "seg_camera"}

        if isinstance(observation_space, gym.spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if key in image_keys:
                    extractors[key] = CustomCNN(subspace.shape, features_dim=features_dim)
                    total_concat_size += features_dim
                else:
                    extractors[key] = nn.Flatten()
                    total_concat_size += get_flattened_obs_dim(subspace)
        else:
            extractors["default"] = CustomCNN(observation_space.shape, features_dim=features_dim)
            total_concat_size = features_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        if isinstance(observations, dict):
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(observations[key]))
        else:
            encoded_tensor_list.append(self.extractors["default"](observations))
        return torch.cat(encoded_tensor_list, dim=1)


algorithm_params = {
    "PPO": dict(
        device="cuda:0",
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])],
                           features_extractor_class=CustomMultiInputExtractor,
                           features_extractor_kwargs=dict(features_dim=256),
                           )
    ),
    "SAC": dict(
        device="cuda:0",
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
    ),
    "DDPG": dict(
        device="cuda:0",
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
    "SAC_AUTO": dict(  # New config for AutoRewardDrive
        device="cuda:0",
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=100000,
        batch_size=128,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=5000,
        use_sde=True,
        policy_kwargs=dict(
            log_std_init=-1,  # Higher initial std for more exploration (exp(-1) ≈ 0.37)
            net_arch=[400, 300],
            features_extractor_class=CustomMultiInputExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        ),
    ),
}

states = {
    "1": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver"],
    "2": ["steer", "throttle", "speed", "maneuver"],
    "3": ["steer", "throttle", "speed", "waypoints"],
    "4": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"],
    "5": ["steer", "throttle", "speed", "waypoints", "seg_camera"],
}

reward_params = {
    "reward_fn_5_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_safe_simple": dict(
        early_stop=True,
        min_speed=10.0,
        max_speed=35.0,
        target_speed=20.0,
        max_distance=2.5,
        max_std_center_lane=0.35,
        max_angle_center_lane=60.0,
        penalty_reward=-10.0,
        progress_reward_scale=20.0,
        steer_penalty_scale=0.05,
        success_reward=5.0,
    ),
    "reward_progress_simple": dict(
        early_stop=True,
        penalty_reward=-10.0,
        success_reward=5.0,
        progress_reward_scale=20.0,
    ),
    # Evaluation reward config (Ground Truth)
    "reward_eval": dict(
        early_stop=True,
        target_speed=25.0,
        reward_progress=20.0,
        penalty_collision=-20.0,
        penalty_offtrack=-12.0,
        penalty_stuck=-8.0,
        penalty_too_fast=-6.0,
        penalty_failure=-10.0,
        reward_success=100.0,
        reward_time=-0.05,
    )
}

_CONFIG_1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_space_type": "continuous",
    "action_smoothing": 0.75,
    "low_speed_threshold_kmh": 1.0,
    "low_speed_timeout_sec": 20.0,
    "low_speed_grace_sec": 5.0,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 42,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_2 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_space_type": "continuous",
    "action_smoothing": 0.75,
    "low_speed_threshold_kmh": 1.0,
    "low_speed_timeout_sec": 20.0,
    "low_speed_grace_sec": 5.0,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 42,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_3 = {
    "algorithm": "SAC_AUTO",
    "algorithm_params": algorithm_params["SAC_AUTO"],
    "gamma": 0.98,  # Discount factor for AutoReward
    "state": states["5"],
    "action_space_type": "continuous",
    "action_smoothing": 0.75,
    "low_speed_threshold_kmh": 1.0,
    "low_speed_timeout_sec": 20.0,
    "low_speed_grace_sec": 5.0,
    "reward_fn": "reward_fn_progress_simple",
    "reward_params": reward_params["reward_progress_simple"],
    "eval_reward_params": reward_params["reward_eval"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": True,
    "use_rgb_bev": False,
    # AutoReward specific parameters
    "reward_update_freq": 1024,  # Frequency of meta-learning updates
    "n_samples": 128,  # Number of action samples for reward baseline estimation
    "reward_buffer_size": 64,  # Max number of trajectories in meta-learning buffer
    "reward_state_keys": ["vehicle_measures", "waypoints"],
    "reward_output_scale": 1.0,
    "reward_lr": 1e-4,
    "value_lr": 3e-4,
    "reward_mix_beta": 0.2,
    "reward_mix_start_meta_updates": 20,
    "reward_mix_full_meta_updates": 120,
    "learned_reward_stats_momentum": 0.01,
}

CONFIGS = {
    "1": _CONFIG_1,
    "2": _CONFIG_2,
    "3": _CONFIG_3,
}

CONFIG = None


def set_config(config_name):
    global CONFIG
    CONFIG = Box(CONFIGS[config_name], default_box=True)
    return CONFIG
