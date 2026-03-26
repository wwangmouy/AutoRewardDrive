import cv2
import math
import json
import os
import pickle
from collections import deque

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


def save_auto_reward_state(model, path):
    learner = getattr(model, "auto_reward_learner", None)
    if learner is None:
        return False

    state = {
        "learner_class": learner.__class__.__name__,
    }
    if hasattr(model, "expert_buffer"):
        state["expert_buffer_storage"] = list(model.expert_buffer.storage)
    if hasattr(model, "shield_buffer"):
        state["shield_buffer_storage"] = list(model.shield_buffer.storage)
        state["last_shield_bc_loss"] = float(getattr(model, "last_shield_bc_loss", 0.0))
        state["last_shield_intervention_rate"] = float(getattr(model, "last_shield_intervention_rate", 0.0))
        state["last_shield_intervention_penalty_sum"] = float(getattr(model, "last_shield_intervention_penalty_sum", 0.0))

    if hasattr(learner, "reward_train_episodes") and hasattr(learner, "meta_eval_episodes"):
        state.update({
            "reward_train_episodes": list(learner.reward_train_episodes.storage),
            "meta_eval_episodes": list(learner.meta_eval_episodes.storage),
            "current_episode": list(getattr(learner, "current_episode", [])),
            "_episode_counter": int(getattr(learner, "_episode_counter", 0)),
            "total_success_trajectories_seen": int(getattr(learner, "total_success_trajectories_seen", 0)),
            "total_failure_trajectories_seen": int(getattr(learner, "total_failure_trajectories_seen", 0)),
            "reward_corr_ema": float(getattr(learner, "reward_corr_ema", 0.0)),
            "reward_ready": bool(getattr(learner, "reward_ready", False)),
            "last_gt_q_loss": float(getattr(learner, "last_gt_q_loss", 0.0)),
        })
    elif hasattr(learner, "D_xi") and hasattr(learner, "trajectory_outcomes"):
        state.update({
            "D_xi": list(learner.D_xi),
            "trajectory_outcomes": list(learner.trajectory_outcomes),
            "current_episode_data": list(getattr(learner, "current_episode_data", [])),
            "total_success_trajectories_seen": int(getattr(learner, "total_success_trajectories_seen", 0)),
            "total_failure_trajectories_seen": int(getattr(learner, "total_failure_trajectories_seen", 0)),
        })
    else:
        return False

    with open(path, "wb") as f:
        pickle.dump(state, f)
    return True


def load_auto_reward_state(model, path):
    learner = getattr(model, "auto_reward_learner", None)
    if learner is None or not path or not os.path.exists(path):
        return False

    with open(path, "rb") as f:
        state = pickle.load(f)

    if hasattr(model, "expert_buffer") and "expert_buffer_storage" in state:
        model.expert_buffer.storage.clear()
        model.expert_buffer.storage.extend(state.get("expert_buffer_storage", []))

    if hasattr(model, "shield_buffer") and "shield_buffer_storage" in state:
        model.shield_buffer.storage.clear()
        model.shield_buffer.storage.extend(state.get("shield_buffer_storage", []))
        model.last_shield_bc_loss = float(state.get("last_shield_bc_loss", 0.0))
        model.last_shield_intervention_rate = float(state.get("last_shield_intervention_rate", 0.0))
        model.last_shield_intervention_penalty_sum = float(state.get("last_shield_intervention_penalty_sum", 0.0))

    if hasattr(learner, "reward_train_episodes") and hasattr(learner, "meta_eval_episodes"):
        learner.reward_train_episodes.storage.clear()
        learner.reward_train_episodes.storage.extend(state.get("reward_train_episodes", []))
        learner.meta_eval_episodes.storage.clear()
        learner.meta_eval_episodes.storage.extend(state.get("meta_eval_episodes", []))
        learner.current_episode = list(state.get("current_episode", []))
        learner._episode_counter = int(state.get("_episode_counter", 0))
        learner.total_success_trajectories_seen = int(state.get("total_success_trajectories_seen", 0))
        learner.total_failure_trajectories_seen = int(state.get("total_failure_trajectories_seen", 0))
        learner.reward_corr_ema = float(state.get("reward_corr_ema", 0.0))
        learner.reward_ready = bool(state.get("reward_ready", False))
        learner.last_gt_q_loss = float(state.get("last_gt_q_loss", 0.0))
        return True

    if hasattr(learner, "D_xi") and hasattr(learner, "trajectory_outcomes"):
        learner.D_xi.clear()
        learner.D_xi.extend(state.get("D_xi", []))
        learner.trajectory_outcomes.clear()
        learner.trajectory_outcomes.extend(state.get("trajectory_outcomes", []))
        learner.current_episode_data = list(state.get("current_episode_data", []))
        learner.total_success_trajectories_seen = int(state.get("total_success_trajectories_seen", 0))
        learner.total_failure_trajectories_seen = int(state.get("total_failure_trajectories_seen", 0))
        return True

    return False


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
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class RobustCheckpointCallback(BaseCallback):
    """
    Save regular model checkpoints plus a rolling latest bundle that can be
    used to resume an interrupted run without depending on every checkpoint
    carrying the full replay buffer.
    """

    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix="model",
        latest_bundle_freq=None,
        save_replay_buffer=True,
        save_auto_reward_state_flag=True,
        verbose=0,
    ):
        super().__init__(verbose)
        self.save_freq = max(int(save_freq), 1)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.latest_bundle_freq = None if latest_bundle_freq is None else max(int(latest_bundle_freq), 1)
        self.save_replay_buffer = bool(save_replay_buffer)
        self.save_auto_reward_state_flag = bool(save_auto_reward_state_flag)
        self._last_step_checkpoint = -1
        self._last_bundle_step = -1

    @staticmethod
    def _cleanup_partial_artifacts(stem):
        for suffix in [".zip", "_replay_buffer.pkl", "_autoreward.pkl"]:
            path = stem + suffix
            if os.path.exists(path) and os.path.getsize(path) == 0:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _save_bundle(self, stem):
        try:
            self.model.save(stem)

            if self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
                self.model.save_replay_buffer(stem + "_replay_buffer.pkl")

            if self.save_auto_reward_state_flag:
                save_auto_reward_state(self.model, stem + "_autoreward.pkl")
        except Exception as exc:
            self._cleanup_partial_artifacts(stem)
            print(f"[RobustCheckpointCallback] Warning: failed to save bundle {stem}: {exc}")
            return False
        return True

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0:
            return True

        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps != self._last_step_checkpoint:
            checkpoint_stem = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            try:
                self.model.save(checkpoint_stem)
                self._last_step_checkpoint = self.num_timesteps
            except Exception as exc:
                self._cleanup_partial_artifacts(checkpoint_stem)
                print(f"[RobustCheckpointCallback] Warning: failed to save checkpoint {checkpoint_stem}: {exc}")

        should_save_latest = (
            self.latest_bundle_freq is not None
            and self.num_timesteps % self.latest_bundle_freq == 0
            and self.num_timesteps != self._last_bundle_step
        )
        if should_save_latest:
            latest_stem = os.path.join(self.save_path, f"{self.name_prefix}_latest")
            if self._save_bundle(latest_stem):
                self._last_bundle_step = self.num_timesteps

        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_learned_rewards = []
        self.episode_ground_truth_rewards = []
        self.rolling_success = deque(maxlen=50)
        self.rolling_route_completion = deque(maxlen=50)
        self.rolling_collision = deque(maxlen=50)
        self.rolling_speed = deque(maxlen=50)
        self.rolling_center_dev = deque(maxlen=50)

    def _on_step(self) -> bool:
        # Track rewards per step for AutoReward comparison
        if hasattr(self.model, 'auto_reward_learner'):
            # Get the last stored ground truth reward from infos if available
            if 'ground_truth_reward' in self.locals.get('infos', [{}])[0]:
                self.episode_ground_truth_rewards.append(
                    self.locals['infos'][0]['ground_truth_reward']
                )
            if 'learned_reward' in self.locals.get('infos', [{}])[0]:
                self.episode_learned_rewards.append(
                    self.locals['infos'][0]['learned_reward']
                )

        # Log scalar value (here a random variable)
        if self.locals['dones'][0]:
            self.logger.record("time/num_timesteps", self.num_timesteps)
            self.logger.record("custom/total_reward", self.locals['infos'][0]['total_reward'])
            self.logger.record("custom/routes_completed", self.locals['infos'][0]['routes_completed'])
            self.logger.record("custom/total_distance", self.locals['infos'][0]['total_distance'])
            self.logger.record("custom/avg_center_dev", self.locals['infos'][0]['avg_center_dev'])
            self.logger.record("custom/avg_speed", self.locals['infos'][0]['avg_speed'])
            self.logger.record("custom/mean_reward", self.locals['infos'][0]['mean_reward'])
            self.logger.record("custom/collision_rate", self.locals['infos'][0]['collision_rate'])
            self.logger.record("custom/collision_num", self.locals['infos'][0]['collision_num'])
            self.logger.record("custom/episode_length", self.locals['infos'][0]['episode_length'])
            if 'curriculum_phase' in self.locals['infos'][0]:
                self.logger.record("autoreward/curriculum_phase", self.locals['infos'][0]['curriculum_phase'])
            if 'expert_mix_ratio' in self.locals['infos'][0]:
                self.logger.record("autoreward/expert_mix_ratio", self.locals['infos'][0]['expert_mix_ratio'])
            if 'policy_reward_mix' in self.locals['infos'][0]:
                self.logger.record("autoreward/policy_reward_mix", self.locals['infos'][0]['policy_reward_mix'])
            if 'train_reward_source' in self.locals['infos'][0]:
                self.logger.record("autoreward/train_reward_source", self.locals['infos'][0]['train_reward_source'])
            if 'reward_ready' in self.locals['infos'][0]:
                self.logger.record("autoreward/reward_ready", self.locals['infos'][0]['reward_ready'])
            if 'bc_coef' in self.locals['infos'][0]:
                self.logger.record("autoreward/bc_coef", self.locals['infos'][0]['bc_coef'])
            if 'shield_intervention_count' in self.locals['infos'][0]:
                self.logger.record("shield/episode_intervention_count", self.locals['infos'][0]['shield_intervention_count'])
            if self.locals['infos'][0]['collision_state']:
                self.logger.record("custom/CPS", self.locals['infos'][0]['CPS'])
                self.logger.record("custom/CPM", self.locals['infos'][0]['CPM'])
                self.logger.record("custom/collision_interval", self.locals['infos'][0]['collision_interval'])
                self.logger.record("custom/collision_speed", self.locals['infos'][0]['collision_speed'])
            reward_components = self.locals['infos'][0].get('reward_components', {})
            for key, value in reward_components.items():
                self.logger.record(f"reward/{key}", value)

            # AutoReward specific metrics
            if hasattr(self.model, 'auto_reward_learner'):
                learner = self.model.auto_reward_learner
                # Trajectory buffer size
                if hasattr(learner, 'trajectory_buffer_size'):
                    self.logger.record("autoreward/trajectory_buffer_size", learner.trajectory_buffer_size)
                else:
                    self.logger.record("autoreward/trajectory_buffer_size", len(learner.D_xi))
                self.logger.record("autoreward/success_traj_count", learner.success_traj_count)
                self.logger.record("autoreward/failure_traj_count", learner.failure_traj_count)
                self.logger.record("autoreward/total_success_traj_seen", learner.total_success_trajectories_seen)
                self.logger.record("autoreward/total_failure_traj_seen", learner.total_failure_trajectories_seen)
                if 'warmup_phase' in self.locals['infos'][0]:
                    self.logger.record("autoreward/warmup_phase", float(self.locals['infos'][0]['warmup_phase']))
                if 'policy_train_enabled' in self.locals['infos'][0]:
                    self.logger.record("autoreward/policy_train_enabled", float(self.locals['infos'][0]['policy_train_enabled']))
                if 'reward_train_enabled' in self.locals['infos'][0]:
                    self.logger.record("autoreward/reward_train_enabled", float(self.locals['infos'][0]['reward_train_enabled']))
                
                # Episode-level reward comparison
                if self.episode_learned_rewards:
                    self.logger.record("autoreward/ep_mean_learned_reward", np.mean(self.episode_learned_rewards))
                    self.logger.record("autoreward/ep_sum_learned_reward", np.sum(self.episode_learned_rewards))
                if self.episode_ground_truth_rewards:
                    self.logger.record("autoreward/ep_mean_gt_reward", np.mean(self.episode_ground_truth_rewards))
                    self.logger.record("autoreward/ep_sum_gt_reward", np.sum(self.episode_ground_truth_rewards))
                
                # Reward correlation (if both available)
                if len(self.episode_learned_rewards) > 1 and len(self.episode_ground_truth_rewards) > 1:
                    if len(self.episode_learned_rewards) == len(self.episode_ground_truth_rewards):
                        correlation = np.corrcoef(self.episode_learned_rewards, self.episode_ground_truth_rewards)[0, 1]
                        if not np.isnan(correlation):
                            self.logger.record("autoreward/reward_correlation", correlation)
                if hasattr(self.model, 'last_bc_loss'):
                    self.logger.record("autoreward/bc_loss", self.model.last_bc_loss)
                if hasattr(self.model, 'last_gt_q_loss'):
                    self.logger.record("autoreward/gt_q_loss", self.model.last_gt_q_loss)
                if hasattr(self.model, 'last_meta_outer_loss'):
                    self.logger.record("autoreward/meta_outer_loss", self.model.last_meta_outer_loss)
                if hasattr(self.model, 'last_reward_corr_ema'):
                    self.logger.record("autoreward/reward_corr_ema", self.model.last_reward_corr_ema)

                # Reset episode tracking
                self.episode_learned_rewards = []
                self.episode_ground_truth_rewards = []

            if hasattr(self.model, 'replay_buffer'):
                recent_rewards = self.model.replay_buffer.rewards[max(0, self.model.replay_buffer.pos-500):self.model.replay_buffer.pos]
                if len(recent_rewards) > 0:
                    mean_recent_rewards = np.mean(recent_rewards)
                    sum_recent_rewards = np.sum(recent_rewards)
                else:
                    mean_recent_rewards = 0.0
                    sum_recent_rewards = 0.0

                # Log the results
                self.logger.record("replay_buffer/mean_recent_rewards", mean_recent_rewards)
                self.logger.record("replay_buffer/sum_recent_rewards", sum_recent_rewards)

            success_value = float(self.locals['infos'][0].get('success_state', False))
            route_completion = float(self.locals['infos'][0].get('routes_completed', 0.0))
            collision_value = float(self.locals['infos'][0].get('collision_state', False))
            speed_value = float(self.locals['infos'][0].get('avg_speed', 0.0))
            center_value = float(self.locals['infos'][0].get('avg_center_dev', 0.0))
            self.rolling_success.append(success_value)
            self.rolling_route_completion.append(route_completion)
            self.rolling_collision.append(collision_value)
            self.rolling_speed.append(speed_value)
            self.rolling_center_dev.append(center_value)
            self.logger.record("rolling/success_rate_50", np.mean(self.rolling_success))
            self.logger.record("rolling/route_completion_50", np.mean(self.rolling_route_completion))
            self.logger.record("rolling/collision_rate_50", np.mean(self.rolling_collision))
            self.logger.record("rolling/avg_speed_50", np.mean(self.rolling_speed))
            self.logger.record("rolling/avg_center_dev_50", np.mean(self.rolling_center_dev))

            self.logger.dump(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        self.logger.record("time/final_num_timesteps", self.num_timesteps)
        if hasattr(self.model, 'last_reward_ready'):
            self.logger.record("autoreward/final_reward_ready", self.model.last_reward_ready)
        if hasattr(self.model, 'last_reward_corr_ema'):
            self.logger.record("autoreward/final_reward_corr_ema", self.model.last_reward_corr_ema)
        if hasattr(self.model, 'last_policy_reward_mix'):
            self.logger.record("autoreward/final_policy_reward_mix", self.model.last_policy_reward_mix)
        if hasattr(self.model, 'last_expert_mix_ratio'):
            self.logger.record("autoreward/final_expert_mix_ratio", self.model.last_expert_mix_ratio)
        self.logger.dump(self.num_timesteps)


class CurriculumCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.config = config
        curriculum_cfg = config.get("curriculum_thresholds", {})
        self.window = int(curriculum_cfg.get("window", 50))
        self.phase_thresholds = curriculum_cfg.get("phases", [])
        self.success_hysteresis = float(curriculum_cfg.get("success_hysteresis", 0.0))
        self.route_hysteresis = float(curriculum_cfg.get("route_hysteresis", 0.0))
        self.success_history = deque(maxlen=self.window)
        self.route_history = deque(maxlen=self.window)
        self.current_phase = 0

    def _resolve_env(self):
        if not hasattr(self.training_env, "envs") or len(self.training_env.envs) == 0:
            return None
        env = self.training_env.envs[0]
        for _ in range(10):
            if env is None:
                return None
            if hasattr(env, "set_curriculum_phase"):
                return env
            if hasattr(env, "env"):
                env = env.env
                continue
            if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "set_curriculum_phase"):
                return env.unwrapped
            break
        return env if hasattr(env, "set_curriculum_phase") else None

    def _on_step(self) -> bool:
        if not self.phase_thresholds or 'infos' not in self.locals or 'dones' not in self.locals:
            return True
        if not self.locals['dones'][0]:
            return True

        info = self.locals['infos'][0]
        self.success_history.append(float(info.get('success_state', False)))
        self.route_history.append(float(info.get('routes_completed', 0.0)))
        if len(self.success_history) == 0:
            return True

        success_rate = float(np.mean(self.success_history))
        route_completion = float(np.mean(self.route_history))
        self.logger.record("curriculum/success_rate_window", success_rate)
        self.logger.record("curriculum/route_completion_window", route_completion)

        eligible_phase = 0
        for phase in self.phase_thresholds:
            if (
                success_rate >= float(phase.get("min_success_rate", 0.0))
                and route_completion >= float(phase.get("min_route_completion", 0.0))
            ):
                eligible_phase = int(phase.get("phase_index", eligible_phase))

        target_phase = eligible_phase
        if eligible_phase < self.current_phase:
            current_phase_cfg = next(
                (
                    phase for phase in self.phase_thresholds
                    if int(phase.get("phase_index", 0)) == self.current_phase
                ),
                None,
            )
            if current_phase_cfg is not None:
                retain_success = float(current_phase_cfg.get("min_success_rate", 0.0)) - self.success_hysteresis
                retain_route = float(current_phase_cfg.get("min_route_completion", 0.0)) - self.route_hysteresis
                if success_rate >= retain_success and route_completion >= retain_route:
                    target_phase = self.current_phase

        self.current_phase = int(target_phase)

        env = self._resolve_env()
        if env is not None:
            env.set_curriculum_phase(self.current_phase)
        self.logger.record("autoreward/curriculum_phase", self.current_phase)
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
