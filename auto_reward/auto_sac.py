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

from auto_reward.learner import AutoRewardLearner


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
        warmstart_cfg = config.get("reward_warmstart", {})
        self.reward_warmstart_enabled = bool(warmstart_cfg.get("enabled", False))
        self.reward_warmstart_min_success = int(warmstart_cfg.get("min_success_trajectories", 0))
        self.reward_warmstart_min_failure = int(warmstart_cfg.get("min_failure_trajectories", 0))
        self.expert_bootstrap_steps = int(config.get("expert_bootstrap_steps", 0))
        self.policy_collect_only_until = int(config.get("policy_collect_only_until", learning_starts))
        self.use_gt_reward_before_warmstart = bool(config.get("use_gt_reward_before_warmstart", False))
        smooth_cfg = config.get("policy_smooth_reg", {})
        self.policy_smooth_reg_enabled = bool(smooth_cfg.get("enabled", False))
        self.policy_smooth_reg_coef = float(smooth_cfg.get("coef", 0.0))
        self.policy_smooth_reg_dims = smooth_cfg.get("dims", "all")
        self.policy_smooth_reg_source = smooth_cfg.get("source", "recent_rollout")
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
            or self.policy_smooth_reg_source != "recent_rollout"
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

    def _is_reward_warmstart_ready(self) -> bool:
        if not self.reward_warmstart_enabled or self.auto_reward_learner is None:
            return True
        return self.auto_reward_learner.has_bootstrap_data(
            self.reward_warmstart_min_success,
            self.reward_warmstart_min_failure,
        )

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

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            self._append_smooth_observation(self._last_obs)
            if self.use_sde:
                self.actor.reset_noise(env.num_envs)

            use_expert_policy = self.num_timesteps < self.expert_bootstrap_steps
            expert_actions_np = self._query_expert_action(env) if use_expert_policy else None

            with torch.no_grad():
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                features = self.actor.extract_features(obs_tensor, self.actor.features_extractor)
                mean_actions, log_std, _ = self.actor.get_action_dist_params(obs_tensor)
                mu = (mean_actions.detach(), log_std.detach())

                if expert_actions_np is not None:
                    actions_np = expert_actions_np
                    actions_tensor = torch.as_tensor(actions_np, device=self.device)
                    log_probs_np = np.zeros((env.num_envs, 1), dtype=np.float32)
                else:
                    actions, log_probs = self.actor.action_log_prob(obs_tensor)
                    actions_np = actions.cpu(memory_format=torch.contiguous_format).numpy()
                    actions_tensor = actions
                    log_probs_np = log_probs.cpu(memory_format=torch.contiguous_format).numpy()
                
                # Compute learned reward R_omega inline (avoid redundant tensor conversion)
                r_omega = self.auto_reward_learner.get_reward(features, actions_tensor)
                r_omega_val = r_omega.cpu(memory_format=torch.contiguous_format).numpy().flatten()
                
                # Cache features on CPU (single transfer)
                features_cpu = features.cpu(memory_format=torch.contiguous_format).numpy().flatten()
                mu_cpu = (mu[0][0].cpu(), mu[1][0].cpu())

            new_obs, rewards, dones, infos = env.step(actions_np)
            gt_reward = float(rewards[0])
            learned_reward = float(r_omega_val[0])
            warmstart_ready = self._is_reward_warmstart_ready()
            use_gt_reward_for_training = self.use_gt_reward_before_warmstart and not warmstart_ready
            train_reward = gt_reward if use_gt_reward_for_training else learned_reward
            train_reward_array = np.array([train_reward], dtype=np.float32)
            infos[0]["ground_truth_reward"] = gt_reward
            infos[0]["learned_reward"] = learned_reward
            infos[0]["train_reward"] = train_reward
            infos[0]["warmstart_ready"] = warmstart_ready
            infos[0]["expert_bootstrap_active"] = float(expert_actions_np is not None)
            
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
                    if infos[idx].get("terminal_observation") is not None:
                        num_collected_episodes += 1
                        self._episode_num += 1
                        real_next_obs[idx] = infos[idx]["terminal_observation"]
                    else:
                        num_collected_episodes += 1
                        self._episode_num += 1
            
            # Store with GT/learned bootstrap reward
            self.replay_buffer.add(self._last_obs, real_next_obs, actions_np, train_reward_array, dones, infos)
            self._last_obs = new_obs
            
            # Update callback locals for TensorboardCallback
            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            
            if callback.on_step() is False:
                self._finalize_smooth_episode()
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
        
        self._finalize_smooth_episode()
        self.logger.record("autoreward/warmstart_ready", float(self._is_reward_warmstart_ready()))
        self.logger.record("autoreward/success_traj_count", self.auto_reward_learner.success_traj_count)
        self.logger.record("autoreward/failure_traj_count", self.auto_reward_learner.failure_traj_count)
        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=True)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Train SAC with action smoothness regularization and periodic meta-gradient updates."""
        self.policy.set_training_mode(True)

        if self.num_timesteps < self.policy_collect_only_until:
            self.logger.record("autoreward/collect_only_phase", 1.0)
            self.logger.record("autoreward/train_enabled", 0.0)
            return
        self.logger.record("autoreward/collect_only_phase", 0.0)
        self.logger.record("autoreward/train_enabled", 1.0)

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
        warmstart_ready = self._is_reward_warmstart_ready()

        if not warmstart_ready:
            self.logger.record("autoreward/warmstart_ready", 0.0)
            self.logger.record("autoreward/success_traj_count", self.auto_reward_learner.success_traj_count)
            self.logger.record("autoreward/failure_traj_count", self.auto_reward_learner.failure_traj_count)
            self.logger.record("smooth/actor_smooth_loss", 0.0)
            self.logger.record("smooth/mean_action_delta", 0.0)
            self.logger.record("smooth/mean_steer_delta", 0.0)
            self.logger.record("smooth/mean_throttle_delta", 0.0)
            return

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
        self.logger.record("autoreward/warmstart_ready", 1.0)
        self.logger.record("autoreward/success_traj_count", self.auto_reward_learner.success_traj_count)
        self.logger.record("autoreward/failure_traj_count", self.auto_reward_learner.failure_traj_count)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        
        def sample_action_from_mu(mu_batch, n_samples):
            """Resample actions from policy distribution."""
            mean, log_std = mu_batch
            mean, log_std = mean.to(self.device), log_std.to(self.device)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            
            # Get action bounds from action_space
            action_low = torch.tensor(self.action_space.low, device=self.device).float()
            action_high = torch.tensor(self.action_space.high, device=self.device).float()
            action_scale = (action_high - action_low) / 2.0
            action_bias = (action_high + action_low) / 2.0
            
            x_t = normal.rsample((n_samples,))
            y_t = torch.tanh(x_t)
            action = y_t * action_scale + action_bias
            
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return action, log_prob

        # Meta-update at specified frequency
        if self.num_timesteps > self.learning_starts and self.num_timesteps % self.reward_update_freq < gradient_steps:
            metrics = self.auto_reward_learner.optimize_reward(sample_action_from_mu)
            
            if metrics:
                self.logger.record("autoreward/meta_loss", metrics.get("meta_loss", 0.0))
                self.logger.record("autoreward/value_loss", metrics.get("value_loss", 0.0))
                self.logger.record("autoreward/mean_R", metrics.get("mean_R_omega", 0.0))
                self.logger.record("autoreward/mean_Adv", metrics.get("mean_Advantage", 0.0))
                self.logger.record("autoreward/gt_vs_learned_return_corr", metrics.get("gt_vs_learned_return_corr", 0.0))
                self.logger.record("autoreward/success_traj_count", metrics.get("success_traj_count", 0.0))
                self.logger.record("autoreward/failure_traj_count", metrics.get("failure_traj_count", 0.0))
