import cv2
import math
import json

import gym
import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


def write_json(data, path):
    config_dict = {}
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            if isinstance(v, str) and v.isnumeric():
                config_dict[k] = int(v)
            elif isinstance(v, dict):
                config_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    config_dict[k][k_inner] = v_inner.__str__()
                config_dict[k] = str(config_dict[k])
            else:
                config_dict[k] = v.__str__()
        json.dump(config_dict, f, indent=4)


class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, int(fps), (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def add_frame_with_reward(self, frame, reward):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        reward_text = f"Reward: {reward:.2f}"

        (text_width, text_height), _ = cv2.getTextSize(
            reward_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )
        position = (frame.shape[1] - text_width - 10,  # x 坐标
                    frame.shape[0] - 10)
        cv2.putText(
            frame, reward_text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
        )
        self.video_writer.write(frame)

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


class HParamCallback(BaseCallback):
    def __init__(self, config):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()
        self.config = config

    def _on_training_start(self) -> None:
        hparam_dict = {}
        for k, v in self.config.items():
            if isinstance(v, str) and v.isnumeric():
                hparam_dict[k] = int(v)
            elif isinstance(v, dict):
                hparam_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    hparam_dict[k][k_inner] = v_inner.__str__()
                hparam_dict[k] = str(hparam_dict[k])
            else:
                hparam_dict[k] = v.__str__()
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "episode/success": 0,
            "episode/env_reward_sum": 0,
            "episode/length": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._reset_episode_buffers()

    def _reset_episode_buffers(self):
        self.episode_policy_rewards = []
        self.episode_learned_rewards = []
        self.episode_ground_truth_rewards = []

    @staticmethod
    def _to_scalar(value):
        if value is None:
            return None

        array = np.asarray(value)
        if array.size == 0:
            return None

        return float(array.reshape(-1)[0])

    def _on_step(self) -> bool:
        info = self.locals.get('infos', [{}])[0]

        if hasattr(self.model, 'auto_reward_learner'):
            policy_reward = self._to_scalar(self.locals.get('policy_rewards'))
            learned_reward = self._to_scalar(self.locals.get('r_omega_val'))
            ground_truth_reward = self._to_scalar(self.locals.get('ground_truth_rewards'))

            if policy_reward is not None:
                self.episode_policy_rewards.append(policy_reward)
            if learned_reward is not None:
                self.episode_learned_rewards.append(learned_reward)
            if ground_truth_reward is None and 'ground_truth_reward' in info:
                ground_truth_reward = self._to_scalar(info['ground_truth_reward'])
            if ground_truth_reward is not None:
                self.episode_ground_truth_rewards.append(ground_truth_reward)

        if self.locals['dones'][0]:
            terminal_reason = info.get('terminal_reason', 'Unknown')

            self.logger.record("time/num_timesteps", self.num_timesteps)
            self.logger.record("episode/env_reward_sum", info['total_reward'])
            self.logger.record("episode/env_reward_mean", info['mean_reward'])
            self.logger.record("episode/routes_completed", info['routes_completed'])
            self.logger.record("episode/route_progress", info['route_progress'])
            self.logger.record("episode/progress_delta_last", info['progress_delta'])
            self.logger.record("episode/total_distance", info['total_distance'])
            self.logger.record("episode/step_distance_last", info['step_distance'])
            self.logger.record("episode/avg_center_dev", info['avg_center_dev'])
            self.logger.record("episode/avg_speed", info['avg_speed'])
            self.logger.record("episode/length", info['episode_length'])
            self.logger.record("episode/success", float(info.get('success_state', False)))
            self.logger.record("episode/collision", float(info['collision_state']))
            self.logger.record("episode/stuck", float(terminal_reason == "Vehicle stuck"))
            self.logger.record("episode/off_track", float(terminal_reason == "Off-track"))
            self.logger.record("episode/too_fast", float(terminal_reason == "Too fast"))
            self.logger.record("episode/closed", float(info['closed']))
            self.logger.record("window/collision_rate", info['collision_rate'])

            if info['collision_state']:
                self.logger.record("collision/CPS", info['CPS'])
                self.logger.record("collision/CPM", info['CPM'])
                self.logger.record("collision/interval", info['collision_interval'])
                self.logger.record("collision/speed", info['collision_speed'])

            # AutoReward specific metrics
            if hasattr(self.model, 'auto_reward_learner'):
                learner = self.model.auto_reward_learner
                self.logger.record("autoreward/trajectory_buffer_size", len(learner.D_xi))
                if self.episode_policy_rewards:
                    self.logger.record("autoreward/ep_policy_reward_sum", np.sum(self.episode_policy_rewards))
                    self.logger.record("autoreward/ep_policy_reward_mean", np.mean(self.episode_policy_rewards))
                if self.episode_learned_rewards:
                    self.logger.record("autoreward/ep_learned_reward_sum", np.sum(self.episode_learned_rewards))
                    self.logger.record("autoreward/ep_learned_reward_mean", np.mean(self.episode_learned_rewards))
                if self.episode_ground_truth_rewards:
                    self.logger.record("autoreward/ep_ground_truth_reward_sum", np.sum(self.episode_ground_truth_rewards))
                    self.logger.record("autoreward/ep_ground_truth_reward_mean", np.mean(self.episode_ground_truth_rewards))

            self.logger.dump(self.num_timesteps)
            self._reset_episode_buffers()

        return True

class VideoRecorderCallback(BaseCallback):
    def __init__(self, video_path, frame_size, video_length=-1, fps=30, skip_frame=1, verbose=0):
        super().__init__(verbose)
        self.video_recorder = VideoRecorder(video_path, frame_size, fps)
        self.max_length = video_length
        self.skip_frame = skip_frame

    def _on_step(self) -> bool:
        # Add frame to video
        if self.max_length != -1 and self.num_timesteps > self.max_length:
            self.video_recorder.release()
            return False
        # Skip every 4 frames to reduce video size
        if self.num_timesteps % self.skip_frame != 0:
            return True
        display = self.training_env.unwrapped.envs[0].env.display
        frame = np.array(pygame.surfarray.array3d(display), dtype=np.uint8).transpose([1, 0, 2])

        self.video_recorder.add_frame(frame)
        return True

    def _on_training_end(self) -> None:
        self.video_recorder.release()


def lr_schedule(initial_value: float, end_value: float, rate: float):
    """
    Learning rate schedule:
        Exponential decay by factors of 10 from initial_value to end_value.

    :param initial_value: Initial learning rate.
    :param rate: Exponential rate of decay. High values mean fast early drop in LR
    :param end_value: The final value of the learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: A float value between 0 and 1 that represents the remaining progress.
        :return: The current learning rate.
        """
        if progress_remaining <= 0:
            return end_value

        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))

    func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    lr_schedule.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"

    return func


class HistoryWrapperObsDict(gym.Wrapper):
    # History Wrapper from rl-baselines3-zoo
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/10de3a8804b14b4ea605b487ae7d8117c52901c4/rl_zoo3/wrappers.py
    """
    History Wrapper for dict observation.
    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2, obs_key: str = 'vae_latent') -> object:
        self.obs_key = obs_key
        assert isinstance(env.observation_space.spaces[obs_key], gym.spaces.Box)
        print("Wrapping the env with HistoryWrapperObsDict.")
        wrapped_obs_space = env.observation_space.spaces[self.obs_key]
        wrapped_action_space = env.action_space

        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space.spaces[obs_key] = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs_dict = self.env.reset()
        obs = obs_dict[self.obs_key]
        self.obs_history[..., -obs.shape[-1]:] = obs

        obs_dict[self.obs_key] = self._create_obs_from_history()

        return obs_dict

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = obs_dict[self.obs_key]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action

        obs_dict[self.obs_key] = self._create_obs_from_history()

        return obs_dict, reward, done, info


class FrameSkip(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        print("Wrapping the env with FrameSkip.")
        self._skip = skip

    def step(self, action: np.ndarray):
        """
        Step the environment with the given action
        Repeat action, sum reward.
        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()


def parse_wrapper_class(wrapper_class_str: str):
    """
    Parse a string to a wrapper class.

    :param wrapper_class_str: (str) The string to parse.
    :return: (type) The wrapper class and its parameters.
    """
    wrap_class, wrap_params = wrapper_class_str.split("_", 1)
    wrap_params = wrap_params.split("_")
    wrap_params = [int(param) if param.isnumeric() else param for param in wrap_params]

    if wrap_class == "HistoryWrapperObsDict":
        return HistoryWrapperObsDict, wrap_params
    elif wrap_class == "FrameSkip":
        return FrameSkip, wrap_params
