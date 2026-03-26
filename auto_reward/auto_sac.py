from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from auto_reward.buffers import ExpertBuffer, ShieldBuffer
from auto_reward.learner import AutoRewardLearner, AutoRewardLearnerV2
from auto_reward.networks import FrozenRewardFeatureEncoderWrapper


class AutoRewardedSAC(SAC):
    """
    SAC with AutoReward: learns reward function R_omega(s,a) via meta-learning.
    """

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        config: Any,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Set before super().__init__() since it calls _setup_model()
        self.config = config
        self.reward_update_freq = config.get('reward_update_freq', 2048)
        self.auto_reward_learner = None
        warmup_cfg = config.get("expert_warmup", {})
        self.expert_warmup_enabled = bool(warmup_cfg.get("enabled", False))
        self.expert_warmup_steps = int(warmup_cfg.get("warmup_steps", 0))
        self.seed_replay_buffer_from_warmup = bool(warmup_cfg.get("seed_replay_buffer", False))
        self.update_reward_learner_during_warmup = bool(warmup_cfg.get("update_reward_learner_during_warmup", False))
        self._warmup_replay_indices: List[int] = []
        self._warmup_rewards_refreshed = False
        shield_cfg = config.get("action_shield", {})
        self.action_shield_enabled = bool(shield_cfg.get("enabled", False))
        self.action_shield_apply_during_warmup = bool(shield_cfg.get("apply_during_warmup", False))
        self.action_shield_apply_during_training = bool(shield_cfg.get("apply_during_training", False))
        self.action_shield_front_distance_threshold = float(shield_cfg.get("front_distance_threshold", 10.0))
        self.action_shield_front_speed_threshold = float(shield_cfg.get("front_speed_threshold", 5.0))
        self.action_shield_front_scan_distance = float(shield_cfg.get("front_scan_distance", 25.0))
        self.action_shield_front_lateral_threshold = float(shield_cfg.get("front_lateral_threshold", 2.5))
        self.action_shield_lane_deviation_threshold = float(shield_cfg.get("lane_deviation_threshold", 0.8))
        self.action_shield_brake_strength = float(shield_cfg.get("brake_strength", 0.5))
        smooth_cfg = config.get("policy_smooth_reg", {})
        self.policy_smooth_reg_enabled = bool(smooth_cfg.get("enabled", False))
        self.policy_smooth_reg_coef = float(smooth_cfg.get("coef", 0.0))
        self.policy_smooth_reg_dims = smooth_cfg.get("dims", "all")
        self.policy_smooth_reg_start_after_timesteps = int(smooth_cfg.get("start_after_timesteps", 0))
        self._smooth_rollout_segments: List[List[Any]] = []
        self._smooth_current_episode_obs: List[Any] = []
        
        super().__init__(
            policy, env, learning_rate, buffer_size, learning_starts, batch_size,
            tau, gamma, train_freq, gradient_steps, action_noise,
            replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage,
            ent_coef, target_update_interval, target_entropy, use_sde,
            sde_sample_freq, use_sde_at_warmup, stats_window_size,
            tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        
        # Get state_dim from features extractor
        if hasattr(self.actor, "features_extractor") and hasattr(self.actor.features_extractor, "_features_dim"):
            state_dim = self.actor.features_extractor._features_dim
        elif hasattr(self.actor, "features_extractor") and hasattr(self.actor.features_extractor, "features_dim"):
            state_dim = self.actor.features_extractor.features_dim
        else:
            from stable_baselines3.common.preprocessing import get_flattened_obs_dim
            state_dim = get_flattened_obs_dim(self.observation_space)

        action_dim = self.action_space.shape[0]
        
        self.auto_reward_learner = AutoRewardLearner(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            config=self.config
        )
        print(f"[AutoRewardedSAC] Initialized: state_dim={state_dim}, action_dim={action_dim}")

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, torch_vars = super()._get_torch_save_params()
        state_dicts = list(state_dicts) + [
            "auto_reward_learner.reward_net",
            "auto_reward_learner.gt_value_net",
            "auto_reward_learner.learned_value_net",
            "auto_reward_learner.reward_optimizer",
            "auto_reward_learner.gt_value_optimizer",
            "auto_reward_learner.learned_value_optimizer",
        ]
        return state_dicts, torch_vars

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        config: Optional[Any] = None,
        device: Union[torch.device, str] = "auto",
        **kwargs
    ):
        """
        Load AutoRewardedSAC model from file.
        
        Args:
            path: Path to the saved model
            env: Environment (required for model loading)
            config: Config object (required for AutoRewardedSAC)
            device: Device to load the model on
            **kwargs: Additional arguments
        """
        if config is None:
            raise ValueError("config argument is required for AutoRewardedSAC.load()")
        
        # First, manually set the config as a class variable temporarily
        # so __init__ can access it during the parent's load process
        cls._temp_config = config
        
        # Use parent SAC class load method directly
        # This will create an instance but __init__ won't have config parameter
        # So we need a workaround
        
        # Load using parent class but we need to inject config before init
        # The solution: Create instance manually, then load parameters
        from stable_baselines3.common.save_util import load_from_zip_file
        
        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=kwargs.get("custom_objects"),
            print_system_info=kwargs.get("print_system_info", False)
        )
        
        # Create the model instance with config
        model = cls(
            policy=data["policy_class"],
            env=env,
            config=config,
            device=device,
            _init_setup_model=False,  # Don't setup yet
        )
        
        # Restore all saved attributes
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        
        # Now setup the model (this will initialize auto_reward_learner)
        model._setup_model()
        
        # Load the neural network parameters
        model.set_parameters(params, exact_match=True, device=device)
        
        # Restore pytorch-specific variables
        model.__dict__.update(pytorch_variables)
        
        # Set the environment if provided
        if env is not None:
            model.set_env(env, force_reset=kwargs.get("force_reset", True))
        
        # Clean up temp variable
        if hasattr(cls, '_temp_config'):
            delattr(cls, '_temp_config')
        
        return model

    @staticmethod
    def _clone_observation(obs: Any) -> Any:
        return deepcopy(obs)

    def _reset_smooth_rollout_cache(self) -> None:
        self._smooth_rollout_segments = []
        self._smooth_current_episode_obs = []

    def _append_smooth_observation(self, obs: Any) -> None:
        self._smooth_current_episode_obs.append(self._clone_observation(obs))

    def _finalize_smooth_episode(self) -> None:
        if len(self._smooth_current_episode_obs) > 1:
            self._smooth_rollout_segments.append(self._smooth_current_episode_obs)
        self._smooth_current_episode_obs = []

    @staticmethod
    def _stack_observation_sequence(obs_sequence: List[Any]) -> Any:
        first_obs = obs_sequence[0]
        if isinstance(first_obs, dict):
            return {
                key: np.concatenate([np.array(obs[key], copy=True) for obs in obs_sequence], axis=0)
                for key in first_obs.keys()
            }
        return np.concatenate([np.array(obs, copy=True) for obs in obs_sequence], axis=0)

    def _compute_policy_smooth_regularization(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = torch.zeros((), device=self.device)
        metrics = {
            "actor_smooth_loss": 0.0,
            "mean_action_delta": 0.0,
            "mean_steer_delta": 0.0,
            "mean_throttle_delta": 0.0,
        }

        if (
            not self.policy_smooth_reg_enabled
            or self.policy_smooth_reg_coef <= 0.0
            or len(self._smooth_rollout_segments) == 0
        ):
            return zero, metrics

        smooth_terms: List[torch.Tensor] = []
        action_delta_terms: List[torch.Tensor] = []
        steer_delta_terms: List[torch.Tensor] = []
        throttle_delta_terms: List[torch.Tensor] = []

        for obs_sequence in self._smooth_rollout_segments:
            if len(obs_sequence) < 2:
                continue

            stacked_obs = self._stack_observation_sequence(obs_sequence)
            obs_tensor, _ = self.policy.obs_to_tensor(stacked_obs)
            deterministic_actions = self.actor(obs_tensor, deterministic=True)
            action_deltas = deterministic_actions[1:] - deterministic_actions[:-1]
            if action_deltas.shape[0] == 0:
                continue

            if self.policy_smooth_reg_dims == "steer":
                smooth_deltas = action_deltas[:, :1]
            else:
                smooth_deltas = action_deltas

            smooth_terms.append(smooth_deltas.pow(2).sum(dim=1))
            action_delta_terms.append(action_deltas.norm(dim=1).detach())
            steer_delta_terms.append(action_deltas[:, 0].abs().detach())
            if action_deltas.shape[1] > 1:
                throttle_delta_terms.append(action_deltas[:, 1].abs().detach())

        if len(smooth_terms) == 0:
            return zero, metrics

        smooth_loss = torch.cat(smooth_terms).mean()
        metrics["actor_smooth_loss"] = float(smooth_loss.detach().item())
        metrics["mean_action_delta"] = float(torch.cat(action_delta_terms).mean().item())
        metrics["mean_steer_delta"] = float(torch.cat(steer_delta_terms).mean().item())
        if len(throttle_delta_terms) > 0:
            metrics["mean_throttle_delta"] = float(torch.cat(throttle_delta_terms).mean().item())

        return smooth_loss, metrics

    def _in_expert_warmup_phase(self) -> bool:
        return self.expert_warmup_enabled and self.num_timesteps < self.expert_warmup_steps

    def _compute_learned_reward_from_obs_action(self, obs: Any, action_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor, _ = self.policy.obs_to_tensor(obs)
            features = self.actor.extract_features(obs_tensor, self.actor.features_extractor)
            action_tensor = torch.as_tensor(action_np, device=self.device).float()
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)
            reward = self.auto_reward_learner.get_reward(features, action_tensor)
            return reward.cpu(memory_format=torch.contiguous_format).numpy().flatten().astype(np.float32)

    def _add_transition_to_replay(self, obs, next_obs, action_np, reward_array, done, info, track_warmup=False):
        replay_index = int(getattr(self.replay_buffer, "pos", 0))
        self.replay_buffer.add(obs, next_obs, action_np, reward_array, done, info)
        if track_warmup:
            self._warmup_replay_indices.append(replay_index)

    def _replay_obs_at(self, index: int) -> Any:
        obs_store = self.replay_buffer.observations
        if isinstance(obs_store, dict):
            return {key: np.array(value[index], copy=True) for key, value in obs_store.items()}
        return np.array(obs_store[index], copy=True)

    def _replay_action_at(self, index: int) -> np.ndarray:
        action = np.array(self.replay_buffer.actions[index], copy=True)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        return action

    def _refresh_warmup_replay_rewards(self):
        if self._warmup_rewards_refreshed:
            return
        if not self.seed_replay_buffer_from_warmup:
            self._warmup_replay_indices = []
            self._warmup_rewards_refreshed = True
            return
        if len(self._warmup_replay_indices) == 0:
            self._warmup_rewards_refreshed = True
            return

        for index in self._warmup_replay_indices:
            learned_reward = self._compute_learned_reward_from_obs_action(
                self._replay_obs_at(index),
                self._replay_action_at(index),
            )
            reward_slot = self.replay_buffer.rewards[index]
            self.replay_buffer.rewards[index] = learned_reward.reshape(reward_slot.shape)

        self._warmup_replay_indices = []
        self._warmup_rewards_refreshed = True

    def _log_phase_flags(self, warmup_phase: bool, policy_train_enabled: bool, reward_train_enabled: bool) -> None:
        self.logger.record("autoreward/warmup_phase", float(warmup_phase))
        self.logger.record("autoreward/policy_train_enabled", float(policy_train_enabled))
        self.logger.record("autoreward/reward_train_enabled", float(reward_train_enabled))
        self.logger.record("autoreward/success_traj_count", self.auto_reward_learner.success_traj_count)
        self.logger.record("autoreward/failure_traj_count", self.auto_reward_learner.failure_traj_count)
        self.logger.record("autoreward/total_success_traj_seen", self.auto_reward_learner.total_success_trajectories_seen)
        self.logger.record("autoreward/total_failure_traj_seen", self.auto_reward_learner.total_failure_trajectories_seen)

    def _maybe_update_reward_learner(self, allow_before_learning_starts: bool = False) -> None:
        if self.auto_reward_learner is None:
            return
        if not allow_before_learning_starts and self.num_timesteps <= self.learning_starts:
            return
        if self.num_timesteps % self.reward_update_freq != 0:
            return

        metrics = self.auto_reward_learner.optimize_reward()
        if metrics:
            self.logger.record("autoreward/meta_loss", metrics.get("meta_loss", 0.0))
            self.logger.record("autoreward/value_loss", metrics.get("value_loss", 0.0))
            self.logger.record("autoreward/mean_R", metrics.get("mean_R_omega", 0.0))
            self.logger.record("autoreward/mean_Adv", metrics.get("mean_Advantage", 0.0))
            self.logger.record("autoreward/gt_vs_learned_return_corr", metrics.get("gt_vs_learned_return_corr", 0.0))
            self.logger.record("autoreward/align_loss", metrics.get("align_loss", 0.0))
            self.logger.record("autoreward/rank_loss", metrics.get("rank_loss", 0.0))
            self.logger.record("autoreward/terminal_loss", metrics.get("terminal_loss", 0.0))
            self.logger.record("autoreward/reward_reg_loss", metrics.get("reward_reg_loss", 0.0))
            self.logger.record("autoreward/mean_success_return", metrics.get("mean_success_return", 0.0))
            self.logger.record("autoreward/mean_failure_return", metrics.get("mean_failure_return", 0.0))
            self.logger.record("autoreward/mean_success_terminal_reward", metrics.get("mean_success_terminal_reward", 0.0))
            self.logger.record("autoreward/mean_failure_terminal_reward", metrics.get("mean_failure_terminal_reward", 0.0))
            self.logger.record("autoreward/success_traj_count", metrics.get("success_traj_count", 0.0))
            self.logger.record("autoreward/failure_traj_count", metrics.get("failure_traj_count", 0.0))
            self.logger.record("autoreward/total_success_traj_seen", metrics.get("total_success_trajectories_seen", 0.0))
            self.logger.record("autoreward/total_failure_traj_seen", metrics.get("total_failure_trajectories_seen", 0.0))

    @staticmethod
    def _unwrap_single_env(vec_env: VecEnv) -> Optional[Any]:
        if not hasattr(vec_env, "envs") or len(vec_env.envs) == 0:
            return None
        env = vec_env.envs[0]
        max_depth = 10
        for _ in range(max_depth):
            if env is None:
                return None
            if hasattr(env, "get_expert_action"):
                return env
            if hasattr(env, "unwrapped") and env.unwrapped is not env and hasattr(env.unwrapped, "get_expert_action"):
                return env.unwrapped
            if hasattr(env, "env"):
                env = env.env
                continue
            break
        return env if hasattr(env, "get_expert_action") else None

    def _query_expert_action(self, vec_env: VecEnv) -> Optional[np.ndarray]:
        env = self._unwrap_single_env(vec_env)
        if env is None or not hasattr(env, "get_expert_action"):
            return None
        action = env.get_expert_action()
        if action is None:
            return None
        action_np = np.asarray(action, dtype=np.float32).reshape(1, -1)
        return action_np

    def _apply_action_shield(self, vec_env: VecEnv, raw_action_np: np.ndarray, warmup_phase: bool) -> Tuple[np.ndarray, Dict[str, float]]:
        metrics = {
            "shield_active": 0.0,
            "shield_front_brake": 0.0,
            "shield_steer_clamp": 0.0,
            "raw_safe_diff_steer": 0.0,
            "raw_safe_diff_throttle": 0.0,
        }
        if not self.action_shield_enabled:
            return raw_action_np, metrics
        if warmup_phase and not self.action_shield_apply_during_warmup:
            return raw_action_np, metrics
        if (not warmup_phase) and not self.action_shield_apply_during_training:
            return raw_action_np, metrics

        env = self._unwrap_single_env(vec_env)
        if env is None or not hasattr(env, "get_safety_signals"):
            return raw_action_np, metrics

        safety = env.get_safety_signals(
            front_scan_distance=self.action_shield_front_scan_distance,
            front_lateral_threshold=self.action_shield_front_lateral_threshold,
        )
        safe_action = np.array(raw_action_np, copy=True)
        raw_steer = float(safe_action[0, 0])
        raw_throttle = float(safe_action[0, 1]) if safe_action.shape[1] > 1 else 0.0

        if (
            safe_action.shape[1] > 1
            and
            safety["front_vehicle_distance"] < self.action_shield_front_distance_threshold
            and safety["speed"] > self.action_shield_front_speed_threshold
        ):
            safe_action[0, 1] = min(raw_throttle, -self.action_shield_brake_strength)
            metrics["shield_active"] = 1.0
            metrics["shield_front_brake"] = 1.0

        lateral_error = float(safety["signed_lateral_error"])
        if lateral_error > self.action_shield_lane_deviation_threshold and raw_steer > 0.0:
            safe_action[0, 0] = 0.0
            metrics["shield_active"] = 1.0
            metrics["shield_steer_clamp"] = 1.0
        elif lateral_error < -self.action_shield_lane_deviation_threshold and raw_steer < 0.0:
            safe_action[0, 0] = 0.0
            metrics["shield_active"] = 1.0
            metrics["shield_steer_clamp"] = 1.0

        metrics["raw_safe_diff_steer"] = abs(float(safe_action[0, 0]) - raw_steer)
        if safe_action.shape[1] > 1:
            metrics["raw_safe_diff_throttle"] = abs(float(safe_action[0, 1]) - raw_throttle)
        return safe_action, metrics

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: MaybeCallback,
        train_freq: Type[Any],
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """Collect rollouts with learned reward R_omega."""
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0
        
        assert isinstance(env, VecEnv) and env.num_envs == 1, "Only supports single env"

        self._reset_smooth_rollout_cache()
        callback.on_rollout_start()
        shield_step_count = 0
        shield_front_brake_count = 0
        shield_steer_clamp_count = 0
        shield_raw_safe_diff_steer = []
        shield_raw_safe_diff_throttle = []
        shield_episode_intervention_count = 0

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            self._append_smooth_observation(self._last_obs)
            if self.use_sde:
                self.actor.reset_noise(env.num_envs)

            warmup_phase = self._in_expert_warmup_phase()
            use_expert_policy = warmup_phase
            expert_actions_np = self._query_expert_action(env) if use_expert_policy else None

            with torch.no_grad():
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                features = self.actor.extract_features(obs_tensor, self.actor.features_extractor)
                mean_actions, log_std, _ = self.actor.get_action_dist_params(obs_tensor)
                mu = (mean_actions.detach(), log_std.detach())

                if expert_actions_np is not None:
                    actions_np = expert_actions_np
                    log_probs_np = np.zeros((env.num_envs, 1), dtype=np.float32)
                else:
                    actions, log_probs = self.actor.action_log_prob(obs_tensor)
                    actions_np = actions.cpu(memory_format=torch.contiguous_format).numpy()
                    log_probs_np = log_probs.cpu(memory_format=torch.contiguous_format).numpy()

                raw_actions_np = np.array(actions_np, copy=True)
                actions_np, shield_metrics = self._apply_action_shield(env, actions_np, warmup_phase)
                actions_tensor = torch.as_tensor(actions_np, device=self.device).float()
                
                # Compute learned reward R_omega inline (avoid redundant tensor conversion)
                r_omega = self.auto_reward_learner.get_reward(features, actions_tensor)
                r_omega_val = r_omega.cpu(memory_format=torch.contiguous_format).numpy().flatten()
                
                # Cache features on CPU (single transfer)
                features_cpu = features.cpu(memory_format=torch.contiguous_format).numpy().flatten()
                mu_cpu = (mu[0][0].cpu(), mu[1][0].cpu())

            new_obs, rewards, dones, infos = env.step(actions_np)
            gt_reward = float(rewards[0])
            learned_reward = float(r_omega_val[0])
            learned_reward_array = np.array([learned_reward], dtype=np.float32)
            infos[0]["ground_truth_reward"] = gt_reward
            infos[0]["learned_reward"] = learned_reward
            infos[0]["warmup_phase"] = float(warmup_phase)
            infos[0]["policy_train_enabled"] = float(not warmup_phase)
            infos[0]["reward_train_enabled"] = float((not warmup_phase) or self.update_reward_learner_during_warmup)
            infos[0]["shield_active"] = shield_metrics["shield_active"]
            infos[0]["shield_front_brake"] = shield_metrics["shield_front_brake"]
            infos[0]["shield_steer_clamp"] = shield_metrics["shield_steer_clamp"]
            infos[0]["shield_raw_safe_diff_steer"] = shield_metrics["raw_safe_diff_steer"]
            infos[0]["shield_raw_safe_diff_throttle"] = shield_metrics["raw_safe_diff_throttle"]
            infos[0]["raw_action_steer"] = float(raw_actions_np[0, 0])
            infos[0]["raw_action_throttle"] = float(raw_actions_np[0, 1])
            infos[0]["safe_action_steer"] = float(actions_np[0, 0])
            infos[0]["safe_action_throttle"] = float(actions_np[0, 1])

            shield_step_count += int(shield_metrics["shield_active"])
            shield_front_brake_count += int(shield_metrics["shield_front_brake"])
            shield_steer_clamp_count += int(shield_metrics["shield_steer_clamp"])
            shield_raw_safe_diff_steer.append(shield_metrics["raw_safe_diff_steer"])
            shield_raw_safe_diff_throttle.append(shield_metrics["raw_safe_diff_throttle"])
            shield_episode_intervention_count += int(shield_metrics["shield_active"])
            
            # Store for meta-learning (ground truth reward)
            self.auto_reward_learner.store_transition(
                state=features_cpu,
                action=actions_np.flatten(),
                reward=gt_reward,
                log_prob=log_probs_np.flatten()[0],
                mu=mu_cpu
            )

            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            
            # Handle episode end
            real_next_obs = new_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    self._finalize_smooth_episode()
                    self.auto_reward_learner.on_episode_end(success=bool(infos[idx].get("success_state", False)))
                    infos[idx]["shield_intervention_count"] = shield_episode_intervention_count
                    shield_episode_intervention_count = 0
                    if infos[idx].get("terminal_observation") is not None:
                        num_collected_episodes += 1
                        self._episode_num += 1
                        real_next_obs[idx] = infos[idx]["terminal_observation"]
                    else:
                        num_collected_episodes += 1
                        self._episode_num += 1
            
            if warmup_phase:
                self._add_transition_to_replay(
                    self._last_obs,
                    real_next_obs,
                    actions_np,
                    learned_reward_array,
                    dones,
                    infos,
                    track_warmup=self.seed_replay_buffer_from_warmup,
                )
            else:
                if self.seed_replay_buffer_from_warmup and not self._warmup_rewards_refreshed:
                    self._refresh_warmup_replay_rewards()
                self._add_transition_to_replay(
                    self._last_obs,
                    real_next_obs,
                    actions_np,
                    learned_reward_array,
                    dones,
                    infos,
                    track_warmup=False,
                )
            self._last_obs = new_obs
            if self.seed_replay_buffer_from_warmup and not self._warmup_rewards_refreshed and self.num_timesteps >= self.expert_warmup_steps:
                self._refresh_warmup_replay_rewards()
            
            # Update callback locals for TensorboardCallback
            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            
            if warmup_phase and self.update_reward_learner_during_warmup:
                self._maybe_update_reward_learner(allow_before_learning_starts=True)

            if callback.on_step() is False:
                self._finalize_smooth_episode()
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
        
        self._finalize_smooth_episode()
        self._log_phase_flags(
            warmup_phase=self._in_expert_warmup_phase(),
            policy_train_enabled=not self._in_expert_warmup_phase(),
            reward_train_enabled=(not self._in_expert_warmup_phase()) or self.update_reward_learner_during_warmup,
        )
        self.logger.record("shield/active_rate", shield_step_count / max(num_collected_steps, 1))
        self.logger.record("shield/front_brake_count", shield_front_brake_count)
        self.logger.record("shield/steer_clamp_count", shield_steer_clamp_count)
        self.logger.record("shield/mean_action_delta", (
            np.mean(np.sqrt(np.square(shield_raw_safe_diff_steer) + np.square(shield_raw_safe_diff_throttle)))
            if shield_raw_safe_diff_steer else 0.0
        ))
        self.logger.record("shield/raw_safe_diff_steer", np.mean(shield_raw_safe_diff_steer) if shield_raw_safe_diff_steer else 0.0)
        self.logger.record("shield/raw_safe_diff_throttle", np.mean(shield_raw_safe_diff_throttle) if shield_raw_safe_diff_throttle else 0.0)
        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=True)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Train SAC with action smoothness regularization and periodic meta-gradient updates."""
        self.policy.set_training_mode(True)

        if self._in_expert_warmup_phase():
            self._log_phase_flags(
                warmup_phase=True,
                policy_train_enabled=False,
                reward_train_enabled=self.update_reward_learner_during_warmup,
            )
            return

        if self.seed_replay_buffer_from_warmup and not self._warmup_rewards_refreshed:
            self._refresh_warmup_replay_rewards()

        self._log_phase_flags(
            warmup_phase=False,
            policy_train_enabled=True,
            reward_train_enabled=True,
        )

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        smooth_losses = []
        mean_action_deltas = []
        mean_steer_deltas = []
        mean_throttle_deltas = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            smooth_metrics = {
                "actor_smooth_loss": 0.0,
                "mean_action_delta": 0.0,
                "mean_steer_delta": 0.0,
                "mean_throttle_delta": 0.0,
            }
            smooth_reg_start = max(self.learning_starts, self.policy_smooth_reg_start_after_timesteps)
            if self.num_timesteps >= smooth_reg_start:
                smooth_loss, smooth_metrics = self._compute_policy_smooth_regularization()
                actor_loss = actor_loss + self.policy_smooth_reg_coef * smooth_loss

            actor_losses.append(actor_loss.item())
            smooth_losses.append(smooth_metrics["actor_smooth_loss"])
            mean_action_deltas.append(smooth_metrics["mean_action_delta"])
            mean_steer_deltas.append(smooth_metrics["mean_steer_delta"])
            mean_throttle_deltas.append(smooth_metrics["mean_throttle_delta"])

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("smooth/actor_smooth_loss", np.mean(smooth_losses) if smooth_losses else 0.0)
        self.logger.record("smooth/mean_action_delta", np.mean(mean_action_deltas) if mean_action_deltas else 0.0)
        self.logger.record("smooth/mean_steer_delta", np.mean(mean_steer_deltas) if mean_steer_deltas else 0.0)
        self.logger.record("smooth/mean_throttle_delta", np.mean(mean_throttle_deltas) if mean_throttle_deltas else 0.0)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        # Meta-update at specified frequency
        self._maybe_update_reward_learner()


class RunningRewardStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = float(value) - self.mean
        self.mean += delta / self.count
        delta2 = float(value) - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 1.0
        return max(self.m2 / (self.count - 1), 1e-6)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


class AutoRewardedSACV2(SAC):
    """
    AutoRewardDrive V2:
    - stores env reward in replay buffer
    - uses expert warm-start + BC regularization
    - freezes a reward feature encoder before reward learning starts
    - computes learned reward online during critic updates
    """

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        config: Any,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.config = config
        bootstrap_cfg = config.get("policy_bootstrap", {})
        self.pure_expert_steps = int(bootstrap_cfg.get("pure_expert_steps", 10000))
        self.mixed_expert_steps = int(bootstrap_cfg.get("mixed_expert_steps", 10000))
        self.bc_start_steps = int(bootstrap_cfg.get("bc_start_steps", 2000))
        self.bc_end_steps = int(bootstrap_cfg.get("bc_end_steps", 30000))
        self.bc_coef_start = float(bootstrap_cfg.get("bc_coef_start", 1.0))
        self.expert_mix_end_ratio = float(bootstrap_cfg.get("expert_mix_end_ratio", 0.2))
        self.expert_buffer_capacity = int(bootstrap_cfg.get("expert_buffer_capacity", 1024))
        self.expert_buffer_batch_size = int(bootstrap_cfg.get("expert_buffer_batch_size", 64))

        reward_schedule_cfg = config.get("reward_schedule", {})
        self.blend_start_steps = int(reward_schedule_cfg.get("blend_start_steps", 20000))
        self.blend_end_steps = int(reward_schedule_cfg.get("blend_end_steps", 50000))
        self.reward_ready_corr_threshold = float(reward_schedule_cfg.get("reward_ready_corr_threshold", 0.2))
        self.reward_ready_floor = float(reward_schedule_cfg.get("reward_ready_env_weight_floor", 0.3))

        meta_cfg = config.get("meta_optimizer", {})
        self.reward_update_freq = int(meta_cfg.get("update_freq", 1024))
        self.meta_start_steps = int(meta_cfg.get("start_steps", self.blend_start_steps))

        shield_cfg = config.get("action_shield", {})
        self.action_shield_train_enabled = bool(shield_cfg.get("train_enabled", False))
        self.action_shield_eval_enabled = bool(shield_cfg.get("eval_enabled", True))
        self.action_shield_front_distance_threshold = float(shield_cfg.get("front_distance_threshold", 10.0))
        self.action_shield_front_speed_threshold = float(shield_cfg.get("front_speed_threshold", 5.0))
        self.action_shield_front_scan_distance = float(shield_cfg.get("front_scan_distance", 25.0))
        self.action_shield_front_lateral_threshold = float(shield_cfg.get("front_lateral_threshold", 2.5))
        self.action_shield_lane_deviation_threshold = float(shield_cfg.get("lane_deviation_threshold", 0.8))
        self.action_shield_brake_strength = float(shield_cfg.get("brake_strength", 0.5))

        smooth_cfg = config.get("policy_smooth_reg", {})
        self.policy_smooth_reg_enabled = bool(smooth_cfg.get("enabled", False))
        self.policy_smooth_reg_coef = float(smooth_cfg.get("coef", 0.0))
        self.policy_smooth_reg_dims = smooth_cfg.get("dims", "steer")
        self.policy_smooth_reg_start_after_timesteps = int(smooth_cfg.get("start_after_timesteps", 0))

        shield_guidance_cfg = config.get("shield_guidance", {})
        self.shield_guidance_enabled = bool(shield_guidance_cfg.get("enabled", False))
        self.shield_guidance_buffer_capacity = int(shield_guidance_cfg.get("buffer_capacity", 8192))
        self.shield_guidance_batch_size = int(shield_guidance_cfg.get("batch_size", 64))
        self.shield_guidance_bc_coef = float(shield_guidance_cfg.get("bc_coef", 0.25))
        self.shield_guidance_start_after_timesteps = int(shield_guidance_cfg.get("start_after_timesteps", 0))
        self.shield_guidance_intervention_penalty = float(shield_guidance_cfg.get("intervention_penalty", 0.0))

        inference_chunk_cfg = config.get("inference_chunk", {})
        self.inference_chunk_enabled = bool(inference_chunk_cfg.get("enabled", False))
        self.inference_chunk_len = int(max(1, inference_chunk_cfg.get("chunk_len", 1)))
        self.inference_chunk_default_mode = str(inference_chunk_cfg.get("default_mode", "step"))

        self.auto_reward_learner = None
        self.reward_feature_extractor = None
        self.reward_feature_dim = 0
        self.reward_encoder_frozen = False
        self.reward_stats = RunningRewardStats()
        self.expert_buffer = ExpertBuffer(self.expert_buffer_capacity)
        self.shield_buffer = ShieldBuffer(self.shield_guidance_buffer_capacity)
        self._smooth_rollout_segments: List[List[Any]] = []
        self._smooth_current_episode_obs: List[Any] = []
        self._shield_episode_intervention_count = 0
        self._eval_chunk_action: Optional[np.ndarray] = None
        self._eval_chunk_remaining = 0
        self.last_bc_loss = 0.0
        self.last_shield_bc_loss = 0.0
        self.last_gt_q_loss = 0.0
        self.last_meta_outer_loss = 0.0
        self.last_reward_corr_ema = 0.0
        self.last_reward_ready = 0.0
        self.last_policy_reward_mix = 1.0
        self.last_train_reward_source = 0.0
        self.last_expert_mix_ratio = 1.0
        self.last_learned_reward_mean = 0.0
        self.last_shield_intervention_rate = 0.0
        self.last_shield_intervention_penalty_sum = 0.0
        self.last_shield_front_brake_count = 0
        self.last_shield_steer_clamp_count = 0
        self.last_shield_buffer_size = 0

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        if hasattr(self.actor.features_extractor, "_features_dim"):
            self.reward_feature_dim = int(self.actor.features_extractor._features_dim)
        else:
            self.reward_feature_dim = int(getattr(self.actor.features_extractor, "features_dim"))
        self.reward_feature_extractor = FrozenRewardFeatureEncoderWrapper(self.actor.features_extractor).to(self.device)
        self.reward_encoder_frozen = False
        self.auto_reward_learner = AutoRewardLearnerV2(
            feature_dim=self.reward_feature_dim,
            action_dim=self.action_space.shape[0],
            device=self.device,
            config=self.config,
        )

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, torch_vars = super()._get_torch_save_params()
        state_dicts = list(state_dicts) + [
            "reward_feature_extractor",
            "auto_reward_learner.reward_net",
            "auto_reward_learner.gt_q1",
            "auto_reward_learner.gt_q2",
            "auto_reward_learner.gt_q1_target",
            "auto_reward_learner.gt_q2_target",
            "auto_reward_learner.reward_optimizer",
            "auto_reward_learner.gt_q_optimizer",
        ]
        return state_dicts, torch_vars

    def _excluded_save_params(self) -> List[str]:
        return list(super()._excluded_save_params()) + [
            "expert_buffer",
            "shield_buffer",
            "_smooth_rollout_segments",
            "_smooth_current_episode_obs",
            "_shield_episode_intervention_count",
            "_eval_chunk_action",
            "_eval_chunk_remaining",
        ]

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        config: Optional[Any] = None,
        device: Union[torch.device, str] = "auto",
        **kwargs,
    ):
        if config is None:
            raise ValueError("config argument is required for AutoRewardedSACV2.load()")
        from stable_baselines3.common.save_util import load_from_zip_file

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=kwargs.get("custom_objects"),
            print_system_info=kwargs.get("print_system_info", False),
        )
        model = cls(
            policy=data["policy_class"],
            env=env,
            config=config,
            device=device,
            _init_setup_model=False,
        )
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()
        model.set_parameters(params, exact_match=True, device=device)
        model.__dict__.update(pytorch_variables)
        if env is not None:
            model.set_env(env, force_reset=kwargs.get("force_reset", True))
        model.reset_inference_controller()
        return model

    @staticmethod
    def _clone_observation(obs: Any) -> Any:
        return deepcopy(obs)

    def _reset_smooth_rollout_cache(self) -> None:
        self._smooth_rollout_segments = []
        self._smooth_current_episode_obs = []

    def _append_smooth_observation(self, obs: Any) -> None:
        self._smooth_current_episode_obs.append(self._clone_observation(obs))

    def _finalize_smooth_episode(self) -> None:
        if len(self._smooth_current_episode_obs) > 1:
            self._smooth_rollout_segments.append(self._smooth_current_episode_obs)
        self._smooth_current_episode_obs = []

    @staticmethod
    def _stack_observation_sequence(obs_sequence: List[Any]) -> Any:
        first_obs = obs_sequence[0]
        if isinstance(first_obs, dict):
            return {
                key: np.concatenate([np.array(obs[key], copy=True) for obs in obs_sequence], axis=0)
                for key in first_obs.keys()
            }
        return np.concatenate([np.array(obs, copy=True) for obs in obs_sequence], axis=0)

    def _compute_policy_smooth_regularization(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = torch.zeros((), device=self.device)
        metrics = {
            "actor_smooth_loss": 0.0,
            "mean_action_delta": 0.0,
            "mean_steer_delta": 0.0,
            "mean_throttle_delta": 0.0,
        }
        if (
            not self.policy_smooth_reg_enabled
            or self.policy_smooth_reg_coef <= 0.0
            or len(self._smooth_rollout_segments) == 0
        ):
            return zero, metrics

        smooth_terms: List[torch.Tensor] = []
        action_delta_terms: List[torch.Tensor] = []
        steer_delta_terms: List[torch.Tensor] = []
        throttle_delta_terms: List[torch.Tensor] = []

        for obs_sequence in self._smooth_rollout_segments:
            if len(obs_sequence) < 2:
                continue
            stacked_obs = self._stack_observation_sequence(obs_sequence)
            obs_tensor, _ = self.policy.obs_to_tensor(stacked_obs)
            deterministic_actions = self.actor(obs_tensor, deterministic=True)
            action_deltas = deterministic_actions[1:] - deterministic_actions[:-1]
            if action_deltas.shape[0] == 0:
                continue

            smooth_deltas = action_deltas[:, :1] if self.policy_smooth_reg_dims == "steer" else action_deltas
            smooth_terms.append(smooth_deltas.pow(2).sum(dim=1))
            action_delta_terms.append(action_deltas.norm(dim=1).detach())
            steer_delta_terms.append(action_deltas[:, 0].abs().detach())
            if action_deltas.shape[1] > 1:
                throttle_delta_terms.append(action_deltas[:, 1].abs().detach())

        if len(smooth_terms) == 0:
            return zero, metrics

        smooth_loss = torch.cat(smooth_terms).mean()
        metrics["actor_smooth_loss"] = float(smooth_loss.detach().item())
        metrics["mean_action_delta"] = float(torch.cat(action_delta_terms).mean().item())
        metrics["mean_steer_delta"] = float(torch.cat(steer_delta_terms).mean().item())
        if len(throttle_delta_terms) > 0:
            metrics["mean_throttle_delta"] = float(torch.cat(throttle_delta_terms).mean().item())
        return smooth_loss, metrics

    @staticmethod
    def _resolve_env_like(env_like: Any, required_attr: Optional[str] = None) -> Optional[Any]:
        env = env_like
        for _ in range(15):
            if env is None:
                return None
            if isinstance(env, VecEnv):
                if not hasattr(env, "envs") or len(env.envs) == 0:
                    return None
                env = env.envs[0]
                continue
            if required_attr is None or hasattr(env, required_attr):
                return env
            if hasattr(env, "unwrapped") and env.unwrapped is not env:
                if required_attr is None or hasattr(env.unwrapped, required_attr):
                    return env.unwrapped
            if hasattr(env, "env"):
                env = env.env
                continue
            break
        if env is None:
            return None
        if required_attr is not None and not hasattr(env, required_attr):
            return None
        return env

    @staticmethod
    def _unwrap_single_env(vec_env: VecEnv) -> Optional[Any]:
        return AutoRewardedSACV2._resolve_env_like(vec_env, required_attr="get_expert_action")

    @staticmethod
    def _zero_shield_metrics() -> Dict[str, float]:
        return {
            "shield_active": 0.0,
            "shield_front_brake": 0.0,
            "shield_steer_clamp": 0.0,
            "raw_safe_diff_steer": 0.0,
            "raw_safe_diff_throttle": 0.0,
        }

    def _should_use_shield_guidance(self) -> bool:
        start_step = max(self.learning_starts, self.shield_guidance_start_after_timesteps)
        return self.shield_guidance_enabled and self.num_timesteps >= start_step

    def reset_inference_controller(self) -> None:
        self._eval_chunk_action = None
        self._eval_chunk_remaining = 0

    def _predict_eval_base_action(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        action, _ = self.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32).reshape(1, -1)

    def _get_eval_raw_action(
        self,
        obs: Any,
        deterministic: bool = True,
        inference_mode: Optional[str] = None,
    ) -> np.ndarray:
        mode = inference_mode or self.inference_chunk_default_mode
        if mode not in {"step", "chunked"}:
            raise ValueError(f"Unsupported inference mode: {mode}")
        if mode == "step" or not self.inference_chunk_enabled or self.inference_chunk_len <= 1:
            return self._predict_eval_base_action(obs, deterministic=deterministic)
        if self._eval_chunk_action is None or self._eval_chunk_remaining <= 0:
            self._eval_chunk_action = self._predict_eval_base_action(obs, deterministic=deterministic)
            self._eval_chunk_remaining = self.inference_chunk_len - 1
        else:
            self._eval_chunk_remaining -= 1
        return np.array(self._eval_chunk_action, copy=True)

    def get_eval_action(
        self,
        obs: Any,
        env_like: Any,
        deterministic: bool = True,
        use_shield: bool = False,
        inference_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        raw_action = np.clip(self._get_eval_raw_action(obs, deterministic=deterministic, inference_mode=inference_mode), -1.0, 1.0)
        shield_metrics = self._zero_shield_metrics()
        safe_action = np.array(raw_action, copy=True)
        if use_shield:
            safe_action, shield_metrics = self._apply_action_shield(env_like, raw_action, training=False)
        metrics = dict(shield_metrics)
        metrics["shield_raw_safe_diff_steer"] = metrics.get("raw_safe_diff_steer", 0.0)
        metrics["shield_raw_safe_diff_throttle"] = metrics.get("raw_safe_diff_throttle", 0.0)
        metrics["raw_action_steer"] = float(raw_action[0, 0])
        metrics["raw_action_throttle"] = float(raw_action[0, 1]) if raw_action.shape[1] > 1 else 0.0
        metrics["safe_action_steer"] = float(safe_action[0, 0])
        metrics["safe_action_throttle"] = float(safe_action[0, 1]) if safe_action.shape[1] > 1 else 0.0
        metrics["inference_mode_chunked"] = float((inference_mode or self.inference_chunk_default_mode) == "chunked")
        metrics["shield_enabled_eval"] = float(use_shield)
        return safe_action.reshape(-1), metrics

    def _query_expert_action(self, vec_env: VecEnv) -> Optional[np.ndarray]:
        env = self._unwrap_single_env(vec_env)
        if env is None or not hasattr(env, "get_expert_action"):
            return None
        action = env.get_expert_action()
        if action is None:
            return None
        return np.asarray(action, dtype=np.float32).reshape(1, -1)

    def _is_pure_expert_phase(self) -> bool:
        return self.num_timesteps < self.pure_expert_steps

    def _is_mixed_expert_phase(self) -> bool:
        return self.pure_expert_steps <= self.num_timesteps < (self.pure_expert_steps + self.mixed_expert_steps)

    def _current_expert_mix_ratio(self) -> float:
        if self._is_pure_expert_phase():
            return 1.0
        if self._is_mixed_expert_phase():
            progress = (self.num_timesteps - self.pure_expert_steps) / max(float(self.mixed_expert_steps), 1.0)
            return float(1.0 + progress * (self.expert_mix_end_ratio - 1.0))
        return 0.0

    def _current_bc_coef(self) -> float:
        if self.num_timesteps < self.bc_start_steps:
            return 0.0
        if self.num_timesteps >= self.bc_end_steps:
            return 0.0
        progress = (self.num_timesteps - self.bc_start_steps) / max(float(self.bc_end_steps - self.bc_start_steps), 1.0)
        return float(self.bc_coef_start * (1.0 - progress))

    def _current_policy_reward_mix(self) -> float:
        if self.num_timesteps < self.blend_start_steps:
            return 1.0
        if self.num_timesteps < self.blend_end_steps:
            progress = (self.num_timesteps - self.blend_start_steps) / max(float(self.blend_end_steps - self.blend_start_steps), 1.0)
            return float(max(0.0, 1.0 - progress))
        if self.auto_reward_learner is not None and self.auto_reward_learner.reward_ready:
            return 0.0
        return float(self.reward_ready_floor)

    def _current_train_reward_source(self) -> float:
        mix = self._current_policy_reward_mix()
        if mix >= 0.999:
            return 0.0
        if mix <= 1e-6:
            return 2.0
        return 1.0

    def _refresh_reward_feature_extractor(self, force: bool = False) -> None:
        if self.reward_feature_extractor is None or force:
            self.reward_feature_extractor = FrozenRewardFeatureEncoderWrapper(self.actor.features_extractor).to(self.device)

    def _freeze_reward_encoder_if_needed(self) -> None:
        if self.reward_encoder_frozen or self.num_timesteps < self.meta_start_steps:
            return
        self._refresh_reward_feature_extractor(force=True)
        self.reward_encoder_frozen = True

    def _extract_reward_features(self, obs: Any) -> torch.Tensor:
        self._refresh_reward_feature_extractor(force=False)
        if isinstance(obs, dict):
            if all(torch.is_tensor(value) for value in obs.values()):
                obs_tensor = {key: value.to(self.device) for key, value in obs.items()}
            else:
                obs_tensor, _ = self.policy.obs_to_tensor(obs)
        elif torch.is_tensor(obs):
            obs_tensor = obs.to(self.device)
        else:
            obs_tensor, _ = self.policy.obs_to_tensor(obs)
        return self.actor.extract_features(obs_tensor, self.reward_feature_extractor.feature_extractor)

    def _compute_reward_features_np(self, obs: Any) -> np.ndarray:
        with torch.no_grad():
            features = self._extract_reward_features(obs)
        return features.detach().cpu().numpy().reshape(-1)

    @staticmethod
    def _replace_terminal_observation(batch_obs: Any, terminal_obs: Any, env_idx: int) -> Any:
        if isinstance(batch_obs, dict):
            updated = {key: np.array(value, copy=True) for key, value in batch_obs.items()}
            for key, value in terminal_obs.items():
                updated[key][env_idx] = np.array(value, copy=True)
            return updated
        updated = np.array(batch_obs, copy=True)
        updated[env_idx] = np.array(terminal_obs, copy=True)
        return updated

    def _compute_current_learned_reward_tensor(self, obs: Any, action_tensor: torch.Tensor) -> torch.Tensor:
        if self.auto_reward_learner is None or not self.reward_encoder_frozen:
            return torch.zeros((action_tensor.shape[0], 1), device=self.device)
        features = self._extract_reward_features(obs)
        return self.auto_reward_learner.get_reward(features, action_tensor)

    def compute_current_learned_reward(self, obs: Any, action_np: np.ndarray) -> np.ndarray:
        action_tensor = torch.as_tensor(action_np, device=self.device).float()
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        with torch.no_grad():
            reward = self._compute_current_learned_reward_tensor(obs, action_tensor)
        return reward.detach().cpu().numpy().reshape(-1)

    def _normalize_env_reward_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.reward_stats.mean, device=rewards.device, dtype=rewards.dtype)
        std = torch.as_tensor(self.reward_stats.std, device=rewards.device, dtype=rewards.dtype)
        return torch.clamp((rewards - mean) / (std + 1e-6), -5.0, 5.0)

    def _build_policy_reward_batch(self, observations: Any, actions: torch.Tensor, env_rewards: torch.Tensor) -> torch.Tensor:
        if self.num_timesteps < self.blend_start_steps:
            return env_rewards
        normalized_env_rewards = self._normalize_env_reward_tensor(env_rewards)
        if not self.reward_encoder_frozen or self.auto_reward_learner is None:
            return normalized_env_rewards
        features = self._extract_reward_features(observations)
        learned_rewards = self.auto_reward_learner.get_reward(features, actions)
        policy_mix = self._current_policy_reward_mix()
        return policy_mix * normalized_env_rewards + (1.0 - policy_mix) * learned_rewards

    def _should_update_reward_learner(self) -> bool:
        if self.auto_reward_learner is None:
            return False
        if self.num_timesteps < self.meta_start_steps:
            return False
        return self.num_timesteps % self.reward_update_freq == 0

    def _apply_action_shield(self, vec_env: VecEnv, raw_action_np: np.ndarray, training: bool) -> Tuple[np.ndarray, Dict[str, float]]:
        metrics = self._zero_shield_metrics()
        if training and not self.action_shield_train_enabled:
            return raw_action_np, metrics
        if (not training) and not self.action_shield_eval_enabled:
            return raw_action_np, metrics

        env = self._resolve_env_like(vec_env, required_attr="get_safety_signals")
        if env is None or not hasattr(env, "get_safety_signals"):
            return raw_action_np, metrics

        safety = env.get_safety_signals(
            front_scan_distance=self.action_shield_front_scan_distance,
            front_lateral_threshold=self.action_shield_front_lateral_threshold,
        )
        safe_action = np.array(raw_action_np, copy=True)
        raw_steer = float(safe_action[0, 0])
        raw_throttle = float(safe_action[0, 1]) if safe_action.shape[1] > 1 else 0.0

        if (
            safe_action.shape[1] > 1
            and
            safety["front_vehicle_distance"] < self.action_shield_front_distance_threshold
            and safety["speed"] > self.action_shield_front_speed_threshold
        ):
            safe_action[0, 1] = min(raw_throttle, -self.action_shield_brake_strength)
            metrics["shield_active"] = 1.0
            metrics["shield_front_brake"] = 1.0

        lateral_error = float(safety["signed_lateral_error"])
        if lateral_error > self.action_shield_lane_deviation_threshold and raw_steer > 0.0:
            safe_action[0, 0] = 0.0
            metrics["shield_active"] = 1.0
            metrics["shield_steer_clamp"] = 1.0
        elif lateral_error < -self.action_shield_lane_deviation_threshold and raw_steer < 0.0:
            safe_action[0, 0] = 0.0
            metrics["shield_active"] = 1.0
            metrics["shield_steer_clamp"] = 1.0

        metrics["raw_safe_diff_steer"] = abs(float(safe_action[0, 0]) - raw_steer)
        if safe_action.shape[1] > 1:
            metrics["raw_safe_diff_throttle"] = abs(float(safe_action[0, 1]) - raw_throttle)
        return safe_action, metrics

    def _record_phase_metrics(self, expert_mix_ratio: float) -> None:
        self.last_expert_mix_ratio = expert_mix_ratio
        self.last_policy_reward_mix = self._current_policy_reward_mix()
        self.last_train_reward_source = self._current_train_reward_source()
        self.logger.record("autoreward/expert_mix_ratio", self.last_expert_mix_ratio)
        self.logger.record("autoreward/policy_reward_mix", self.last_policy_reward_mix)
        self.logger.record("autoreward/train_reward_source", self.last_train_reward_source)
        self.logger.record("autoreward/reward_ready", self.last_reward_ready)
        self.logger.record("shield/buffer_size", len(self.shield_buffer))
        if self.auto_reward_learner is not None:
            self.logger.record("autoreward/trajectory_buffer_size", self.auto_reward_learner.trajectory_buffer_size)
            self.logger.record("autoreward/success_traj_count", self.auto_reward_learner.success_traj_count)
            self.logger.record("autoreward/failure_traj_count", self.auto_reward_learner.failure_traj_count)
            self.logger.record("autoreward/total_success_traj_seen", self.auto_reward_learner.total_success_trajectories_seen)
            self.logger.record("autoreward/total_failure_traj_seen", self.auto_reward_learner.total_failure_trajectories_seen)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: MaybeCallback,
        train_freq: Type[Any],
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0
        assert isinstance(env, VecEnv) and env.num_envs == 1, "Only supports single env"

        self._reset_smooth_rollout_cache()
        self._shield_episode_intervention_count = 0
        callback.on_rollout_start()
        shield_step_count = 0
        shield_front_brake_count = 0
        shield_steer_clamp_count = 0
        shield_raw_safe_diff_steer: List[float] = []
        shield_raw_safe_diff_throttle: List[float] = []
        shield_intervention_penalty_sum = 0.0
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            self._append_smooth_observation(self._last_obs)
            if self.use_sde:
                self.actor.reset_noise(env.num_envs)

            obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
            with torch.no_grad():
                policy_actions, log_probs = self.actor.action_log_prob(obs_tensor)
            policy_actions_np = policy_actions.detach().cpu().numpy()
            guidance_raw_actions_np = np.clip(np.array(policy_actions_np, copy=True), -1.0, 1.0)
            guidance_safe_actions_np, guidance_metrics = self._apply_action_shield(
                env, guidance_raw_actions_np, training=True
            )
            expert_actions_np = self._query_expert_action(env)
            expert_mix_ratio = self._current_expert_mix_ratio()

            if expert_actions_np is not None and expert_mix_ratio > 0.0:
                executed_actions_np = (
                    expert_mix_ratio * expert_actions_np
                    + (1.0 - expert_mix_ratio) * policy_actions_np
                )
            else:
                executed_actions_np = policy_actions_np

            raw_actions_np = np.clip(np.array(executed_actions_np, copy=True), -1.0, 1.0)
            actions_np, shield_metrics = self._apply_action_shield(env, raw_actions_np, training=True)
            new_obs, rewards, dones, infos = env.step(actions_np)
            raw_env_reward = float(rewards[0])
            shield_penalty = 0.0
            if self._should_use_shield_guidance() and shield_metrics["shield_active"] > 0.0:
                shield_penalty = self.shield_guidance_intervention_penalty
            env_reward = raw_env_reward - shield_penalty
            self.reward_stats.update(env_reward)

            if expert_actions_np is not None:
                self.expert_buffer.add(self._last_obs, expert_actions_np)

            if self._should_use_shield_guidance() and guidance_metrics["shield_active"] > 0.0:
                self.shield_buffer.add(
                    obs=self._last_obs,
                    raw_action=guidance_raw_actions_np.reshape(-1),
                    safe_action=guidance_safe_actions_np.reshape(-1),
                    shield_active=guidance_metrics["shield_active"],
                    shield_front_brake=guidance_metrics["shield_front_brake"],
                    shield_steer_clamp=guidance_metrics["shield_steer_clamp"],
                )

            self._freeze_reward_encoder_if_needed()
            learned_reward = 0.0
            if self.reward_encoder_frozen and self.auto_reward_learner is not None:
                learned_reward = float(self.compute_current_learned_reward(self._last_obs, actions_np)[0])

            infos[0]["ground_truth_reward"] = env_reward
            infos[0]["raw_env_reward"] = raw_env_reward
            infos[0]["learned_reward"] = learned_reward
            infos[0]["expert_mix_ratio"] = expert_mix_ratio
            infos[0]["policy_reward_mix"] = self._current_policy_reward_mix()
            infos[0]["train_reward_source"] = self._current_train_reward_source()
            infos[0]["reward_ready"] = float(self.auto_reward_learner.reward_ready) if self.auto_reward_learner is not None else 0.0
            infos[0]["shield_active"] = shield_metrics["shield_active"]
            infos[0]["shield_front_brake"] = shield_metrics["shield_front_brake"]
            infos[0]["shield_steer_clamp"] = shield_metrics["shield_steer_clamp"]
            infos[0]["shield_raw_safe_diff_steer"] = shield_metrics["raw_safe_diff_steer"]
            infos[0]["shield_raw_safe_diff_throttle"] = shield_metrics["raw_safe_diff_throttle"]
            infos[0]["shield_intervention_penalty"] = shield_penalty
            infos[0]["raw_action_steer"] = float(raw_actions_np[0, 0])
            infos[0]["raw_action_throttle"] = float(raw_actions_np[0, 1]) if raw_actions_np.shape[1] > 1 else 0.0
            infos[0]["safe_action_steer"] = float(actions_np[0, 0])
            infos[0]["safe_action_throttle"] = float(actions_np[0, 1]) if actions_np.shape[1] > 1 else 0.0
            infos[0]["bc_coef"] = self._current_bc_coef()

            shield_step_count += int(shield_metrics["shield_active"])
            shield_front_brake_count += int(shield_metrics["shield_front_brake"])
            shield_steer_clamp_count += int(shield_metrics["shield_steer_clamp"])
            shield_raw_safe_diff_steer.append(shield_metrics["raw_safe_diff_steer"])
            shield_raw_safe_diff_throttle.append(shield_metrics["raw_safe_diff_throttle"])
            shield_intervention_penalty_sum += shield_penalty
            self._shield_episode_intervention_count += int(shield_metrics["shield_active"])

            real_next_obs = new_obs.copy()
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None:
                    real_next_obs = self._replace_terminal_observation(real_next_obs, infos[idx]["terminal_observation"], idx)

            if self.reward_encoder_frozen and self.auto_reward_learner is not None:
                current_feature = self._compute_reward_features_np(self._last_obs)
                next_feature = self._compute_reward_features_np(real_next_obs)
                self.auto_reward_learner.store_transition(
                    feature=current_feature,
                    next_feature=next_feature,
                    action=actions_np.reshape(-1),
                    env_reward=env_reward,
                    done=bool(dones[0]),
                )

            self.replay_buffer.add(self._last_obs, real_next_obs, actions_np, np.array([env_reward], dtype=np.float32), dones, infos)
            self._last_obs = new_obs
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            for idx, done in enumerate(dones):
                if done:
                    self._finalize_smooth_episode()
                    num_collected_episodes += 1
                    self._episode_num += 1
                    infos[idx]["shield_intervention_count"] = self._shield_episode_intervention_count
                    self._shield_episode_intervention_count = 0
                    if self.reward_encoder_frozen and self.auto_reward_learner is not None:
                        self.auto_reward_learner.on_episode_end(
                            success=bool(infos[idx].get("success_state", False)),
                            terminal_reason=str(infos[idx].get("terminal_reason", "Running...")),
                        )

            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            self._record_phase_metrics(expert_mix_ratio)

            if callback.on_step() is False:
                self._finalize_smooth_episode()
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

        self._finalize_smooth_episode()
        self.last_shield_intervention_rate = shield_step_count / max(num_collected_steps, 1)
        self.last_shield_front_brake_count = shield_front_brake_count
        self.last_shield_steer_clamp_count = shield_steer_clamp_count
        self.last_shield_intervention_penalty_sum = float(shield_intervention_penalty_sum)
        self.last_shield_buffer_size = len(self.shield_buffer)
        self.logger.record("shield/train_intervention_rate", self.last_shield_intervention_rate)
        self.logger.record("shield/front_brake_count", shield_front_brake_count)
        self.logger.record("shield/steer_clamp_count", shield_steer_clamp_count)
        self.logger.record("shield/buffer_size", len(self.shield_buffer))
        self.logger.record("shield/intervention_penalty_sum", shield_intervention_penalty_sum)
        self.logger.record(
            "shield/raw_safe_diff_steer",
            np.mean(shield_raw_safe_diff_steer) if shield_raw_safe_diff_steer else 0.0,
        )
        self.logger.record(
            "shield/raw_safe_diff_throttle",
            np.mean(shield_raw_safe_diff_throttle) if shield_raw_safe_diff_throttle else 0.0,
        )
        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=True)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._freeze_reward_encoder_if_needed()

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses, shield_bc_losses = [], [], [], []
        smooth_losses = []
        mean_action_deltas = []
        mean_steer_deltas = []
        mean_throttle_deltas = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                reward_batch = self._build_policy_reward_batch(
                    replay_data.observations,
                    replay_data.actions,
                    replay_data.rewards,
                )
                self.last_learned_reward_mean = float(reward_batch.detach().mean().item())
                target_q_values = reward_batch + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            bc_loss_value = torch.zeros((), device=self.device)
            bc_coef = self._current_bc_coef()
            if bc_coef > 0.0 and len(self.expert_buffer) > 0:
                expert_batch = self.expert_buffer.sample(self.expert_buffer_batch_size)
                expert_obs, _ = self.policy.obs_to_tensor(expert_batch["observations"])
                expert_actions = torch.as_tensor(expert_batch["expert_actions"], device=self.device).float()
                bc_actions = self.actor(expert_obs, deterministic=True)
                bc_loss_value = F.mse_loss(bc_actions, expert_actions)
                actor_loss = actor_loss + bc_coef * bc_loss_value

            smooth_metrics = {
                "actor_smooth_loss": 0.0,
                "mean_action_delta": 0.0,
                "mean_steer_delta": 0.0,
                "mean_throttle_delta": 0.0,
            }
            smooth_reg_start = max(self.learning_starts, self.policy_smooth_reg_start_after_timesteps)
            if self.num_timesteps >= smooth_reg_start:
                smooth_loss, smooth_metrics = self._compute_policy_smooth_regularization()
                actor_loss = actor_loss + self.policy_smooth_reg_coef * smooth_loss

            shield_bc_loss_value = torch.zeros((), device=self.device)
            shield_guidance_start = max(self.learning_starts, self.shield_guidance_start_after_timesteps)
            if self.shield_guidance_enabled and self.num_timesteps >= shield_guidance_start and len(self.shield_buffer) > 0:
                shield_batch = self.shield_buffer.sample(self.shield_guidance_batch_size)
                shield_obs, _ = self.policy.obs_to_tensor(shield_batch["observations"])
                safe_actions = torch.as_tensor(shield_batch["safe_actions"], device=self.device).float()
                shield_actions = self.actor(shield_obs, deterministic=True)
                shield_bc_loss_value = F.mse_loss(shield_actions, safe_actions)
                actor_loss = actor_loss + self.shield_guidance_bc_coef * shield_bc_loss_value

            actor_losses.append(actor_loss.item())
            bc_losses.append(float(bc_loss_value.detach().item()))
            shield_bc_losses.append(float(shield_bc_loss_value.detach().item()))
            smooth_losses.append(smooth_metrics["actor_smooth_loss"])
            mean_action_deltas.append(smooth_metrics["mean_action_delta"])
            mean_steer_deltas.append(smooth_metrics["mean_steer_delta"])
            mean_throttle_deltas.append(smooth_metrics["mean_throttle_delta"])

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if self.reward_encoder_frozen and self.auto_reward_learner is not None:
                with torch.no_grad():
                    features = self._extract_reward_features(replay_data.observations)
                    next_features = self._extract_reward_features(replay_data.next_observations)
                    next_actions_det = self.actor(replay_data.next_observations, deterministic=True)
                gt_metrics = self.auto_reward_learner.update_gt_critics(
                    features=features,
                    actions=replay_data.actions,
                    env_rewards=replay_data.rewards,
                    next_features=next_features,
                    next_actions=next_actions_det,
                    dones=replay_data.dones,
                )
                self.last_gt_q_loss = gt_metrics["gt_q_loss"]

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.last_bc_loss = float(np.mean(bc_losses) if bc_losses else 0.0)
        self.last_shield_bc_loss = float(np.mean(shield_bc_losses) if shield_bc_losses else 0.0)
        self.last_shield_buffer_size = len(self.shield_buffer)
        self.last_reward_ready = float(self.auto_reward_learner.reward_ready) if self.auto_reward_learner is not None else 0.0
        self.last_reward_corr_ema = float(self.auto_reward_learner.reward_corr_ema) if self.auto_reward_learner is not None else 0.0
        self.last_policy_reward_mix = self._current_policy_reward_mix()
        self.last_train_reward_source = self._current_train_reward_source()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses) if ent_coef_losses else 0.0)
        self.logger.record("smooth/actor_smooth_loss", np.mean(smooth_losses) if smooth_losses else 0.0)
        self.logger.record("smooth/mean_action_delta", np.mean(mean_action_deltas) if mean_action_deltas else 0.0)
        self.logger.record("smooth/mean_steer_delta", np.mean(mean_steer_deltas) if mean_steer_deltas else 0.0)
        self.logger.record("smooth/mean_throttle_delta", np.mean(mean_throttle_deltas) if mean_throttle_deltas else 0.0)
        self.logger.record("autoreward/bc_loss", self.last_bc_loss)
        self.logger.record("autoreward/gt_q_loss", self.last_gt_q_loss)
        self.logger.record("autoreward/reward_ready", self.last_reward_ready)
        self.logger.record("autoreward/reward_corr_ema", self.last_reward_corr_ema)
        self.logger.record("autoreward/policy_reward_mix", self.last_policy_reward_mix)
        self.logger.record("autoreward/train_reward_source", self.last_train_reward_source)
        self.logger.record("autoreward/mean_R", self.last_learned_reward_mean)
        self.logger.record("shield/bc_loss", self.last_shield_bc_loss)
        self.logger.record("shield/buffer_size", len(self.shield_buffer))

        if self._should_update_reward_learner():
            metrics = self.auto_reward_learner.optimize_reward()
            if metrics:
                self.last_meta_outer_loss = metrics.get("meta_outer_loss", 0.0)
                self.last_reward_corr_ema = metrics.get("reward_corr_ema", self.last_reward_corr_ema)
                self.last_reward_ready = metrics.get("reward_ready", self.last_reward_ready)
                self.logger.record("autoreward/meta_outer_loss", metrics.get("meta_outer_loss", 0.0))
                self.logger.record("autoreward/gt_q_loss", metrics.get("gt_q_loss", self.last_gt_q_loss))
                self.logger.record("autoreward/mean_R", metrics.get("mean_R_omega", self.last_learned_reward_mean))
                self.logger.record("autoreward/gt_vs_learned_return_corr", metrics.get("gt_vs_learned_return_corr", 0.0))
                self.logger.record("autoreward/reward_corr_ema", metrics.get("reward_corr_ema", self.last_reward_corr_ema))
                self.logger.record("autoreward/reward_ready", metrics.get("reward_ready", self.last_reward_ready))
                self.logger.record("autoreward/meta_rank_gap", metrics.get("meta_rank_gap", 0.0))
                self.logger.record("autoreward/meta_terminal_gap", metrics.get("meta_terminal_gap", 0.0))
                self.logger.record("autoreward/rank_loss", metrics.get("rank_loss", 0.0))
                self.logger.record("autoreward/terminal_loss", metrics.get("terminal_loss", 0.0))
                self.logger.record("autoreward/mean_success_return", metrics.get("mean_success_return", 0.0))
                self.logger.record("autoreward/mean_failure_return", metrics.get("mean_failure_return", 0.0))
                self.logger.record("autoreward/trajectory_buffer_size", metrics.get("trajectory_buffer_size", 0.0))
                self.logger.record("autoreward/success_traj_count", metrics.get("success_traj_count", 0.0))
                self.logger.record("autoreward/failure_traj_count", metrics.get("failure_traj_count", 0.0))
                self.logger.record("autoreward/total_success_traj_seen", metrics.get("total_success_trajectories_seen", 0.0))
                self.logger.record("autoreward/total_failure_traj_seen", metrics.get("total_failure_trajectories_seen", 0.0))
