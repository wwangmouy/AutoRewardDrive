import torch
import numpy as np
from collections import deque
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
        self.autoreward_num_meta_updates = 0
        self.reward_state_keys = []
        self.last_reward_mix_alpha = 0.0
        self.reward_mix_beta = config.get('reward_mix_beta', 0.2)
        self.reward_mix_start_meta_updates = config.get('reward_mix_start_meta_updates', 20)
        self.reward_mix_full_meta_updates = config.get('reward_mix_full_meta_updates', 120)
        self.max_meta_updates = config.get('max_meta_updates', None)
        if self.max_meta_updates is not None:
            self.max_meta_updates = int(self.max_meta_updates)
        self.learned_reward_running_mean = 0.0
        self.learned_reward_running_sq_mean = 0.0
        self.learned_reward_running_count = 0
        self.learned_reward_stats_momentum = config.get('learned_reward_stats_momentum', 0.01)
        self.last_raw_learned_reward_mean = 0.0
        self.last_normalized_learned_reward_mean = 0.0
        self.action_smoothness_horizon = int(config.get('action_smoothness_horizon', 4))
        self.steer_smoothness_coef = float(config.get('steer_smoothness_coef', 0.2))
        self.longitudinal_smoothness_coef = float(config.get('longitudinal_smoothness_coef', 0.05))
        self._recent_policy_observations = deque(maxlen=self.action_smoothness_horizon + 1)
        self.last_action_smoothness_loss = 0.0
        self.last_steer_smoothness_loss = 0.0
        self.last_longitudinal_smoothness_loss = 0.0
        
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
        state_dim = self._get_reward_state_dim()

        action_dim = self.action_space.shape[0]
        
        self.auto_reward_learner = AutoRewardLearner(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            config=self.config
        )
        self.auto_reward_learner.num_meta_updates = getattr(self, "autoreward_num_meta_updates", 0)
        print(f"[AutoRewardedSAC] Initialized: reward_state_dim={state_dim}, action_dim={action_dim}, reward_keys={self.reward_state_keys}")

    def _get_reward_state_dim(self) -> int:
        if not hasattr(self.observation_space, "spaces"):
            raise ValueError("AutoRewardedSAC expects a dict observation space for reward learning.")

        configured_keys = list(self.config.get("reward_state_keys", []))
        available_keys = list(self.observation_space.spaces.keys())
        reward_keys = [key for key in configured_keys if key in available_keys]
        if not reward_keys:
            reward_keys = []
            for key, space in self.observation_space.spaces.items():
                if len(space.shape) <= 2:
                    reward_keys.append(key)

        if not reward_keys:
            raise ValueError("No stable non-image observation keys available for reward learning.")

        state_dim = 0
        for key in reward_keys:
            state_dim += int(np.prod(self.observation_space.spaces[key].shape))

        self.reward_state_keys = reward_keys
        return state_dim

    def _extract_reward_state(self, obs: Union[Dict[str, Any], np.ndarray]) -> torch.Tensor:
        if not isinstance(obs, dict):
            return torch.as_tensor(obs, device=self.device).float()

        parts = []
        for key in self.reward_state_keys:
            value = torch.as_tensor(obs[key], device=self.device).float()
            expected_ndim = len(self.observation_space.spaces[key].shape)
            if value.ndim == expected_ndim:
                value = value.unsqueeze(0)
            parts.append(value.flatten(start_dim=1))

        return torch.cat(parts, dim=1)

    def _normalize_learned_reward(self, learned_rewards: np.ndarray) -> np.ndarray:
        batch_mean = float(np.mean(learned_rewards))
        batch_sq_mean = float(np.mean(np.square(learned_rewards)))
        momentum = self.learned_reward_stats_momentum

        if self.learned_reward_running_count == 0:
            self.learned_reward_running_mean = batch_mean
            self.learned_reward_running_sq_mean = batch_sq_mean
        else:
            self.learned_reward_running_mean = (
                (1.0 - momentum) * self.learned_reward_running_mean + momentum * batch_mean
            )
            self.learned_reward_running_sq_mean = (
                (1.0 - momentum) * self.learned_reward_running_sq_mean + momentum * batch_sq_mean
            )

        self.learned_reward_running_count += 1
        variance = max(
            self.learned_reward_running_sq_mean - (self.learned_reward_running_mean ** 2),
            1e-6,
        )
        normalized = (learned_rewards - self.learned_reward_running_mean) / np.sqrt(variance)
        normalized = np.clip(normalized, -5.0, 5.0)

        self.last_raw_learned_reward_mean = batch_mean
        self.last_normalized_learned_reward_mean = float(np.mean(normalized))
        return normalized.astype(np.float32)

    def _clone_observation(self, obs: Union[Dict[str, Any], np.ndarray]) -> Union[Dict[str, Any], np.ndarray]:
        if isinstance(obs, dict):
            return {key: np.array(value, copy=True) for key, value in obs.items()}
        return np.array(obs, copy=True)

    def _append_recent_observation(self, obs: Union[Dict[str, Any], np.ndarray]) -> None:
        self._recent_policy_observations.append(self._clone_observation(obs))

    def _stack_recent_observations(self) -> Optional[Union[Dict[str, np.ndarray], np.ndarray]]:
        if len(self._recent_policy_observations) < 2:
            return None

        obs_sequence = list(self._recent_policy_observations)
        if isinstance(obs_sequence[0], dict):
            stacked_obs: Dict[str, np.ndarray] = {}
            for key in obs_sequence[0].keys():
                key_batches = []
                expected_ndim = len(self.observation_space.spaces[key].shape)
                for obs in obs_sequence:
                    value = np.array(obs[key], copy=False)
                    if value.ndim == expected_ndim:
                        value = np.expand_dims(value, axis=0)
                    key_batches.append(value)
                stacked_obs[key] = np.concatenate(key_batches, axis=0)
            return stacked_obs

        return np.stack(obs_sequence, axis=0)

    def _compute_action_smoothness_loss(self) -> torch.Tensor:
        obs_batch = self._stack_recent_observations()
        zero = torch.tensor(0.0, device=self.device)
        if obs_batch is None:
            self.last_action_smoothness_loss = 0.0
            self.last_steer_smoothness_loss = 0.0
            self.last_longitudinal_smoothness_loss = 0.0
            return zero

        obs_tensor, _ = self.policy.obs_to_tensor(obs_batch)
        with torch.set_grad_enabled(True):
            mean_actions, log_std, kwargs = self.actor.get_action_dist_params(obs_tensor)
            deterministic_actions = self.actor.action_dist.actions_from_params(
                mean_actions, log_std, deterministic=True, **kwargs
            )

        if deterministic_actions.shape[0] < 2:
            self.last_action_smoothness_loss = 0.0
            self.last_steer_smoothness_loss = 0.0
            self.last_longitudinal_smoothness_loss = 0.0
            return zero

        deltas = deterministic_actions[1:] - deterministic_actions[:-1]
        steer_loss = torch.mean(torch.square(deltas[:, 0]))
        if deterministic_actions.shape[1] > 1:
            longitudinal_loss = torch.mean(torch.square(deltas[:, 1]))
        else:
            longitudinal_loss = zero

        smoothness_loss = (
            self.steer_smoothness_coef * steer_loss
            + self.longitudinal_smoothness_coef * longitudinal_loss
        )
        self.last_action_smoothness_loss = float(smoothness_loss.detach().cpu().item())
        self.last_steer_smoothness_loss = float(steer_loss.detach().cpu().item())
        self.last_longitudinal_smoothness_loss = float(longitudinal_loss.detach().cpu().item())
        return smoothness_loss

    def _mix_policy_reward(self, env_rewards: np.ndarray, learned_rewards: np.ndarray) -> np.ndarray:
        meta_updates = 0 if self.auto_reward_learner is None else self.auto_reward_learner.num_meta_updates
        if meta_updates < self.reward_mix_start_meta_updates:
            alpha = 0.0
        elif meta_updates >= self.reward_mix_full_meta_updates:
            alpha = 1.0
        else:
            alpha = (
                (meta_updates - self.reward_mix_start_meta_updates)
                / max(1, self.reward_mix_full_meta_updates - self.reward_mix_start_meta_updates)
            )

        normalized_learned_rewards = self._normalize_learned_reward(learned_rewards)
        residual_scale = self.reward_mix_beta * alpha
        self.last_reward_mix_alpha = residual_scale
        return env_rewards + residual_scale * normalized_learned_rewards

    def _get_torch_save_params(self):
        state_dicts, tensors = super()._get_torch_save_params()
        state_dicts = list(state_dicts)
        state_dicts.extend([
            "auto_reward_learner.reward_net",
            "auto_reward_learner.reward_optimizer",
            "auto_reward_learner.value_net",
            "auto_reward_learner.value_optimizer",
        ])
        return state_dicts, tensors

    def _excluded_save_params(self):
        excluded = list(super()._excluded_save_params())
        excluded.extend(["auto_reward_learner"])
        return excluded

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
        model.set_parameters(params, exact_match=False, device=device)
        
        # Restore pytorch-specific variables
        model.__dict__.update(pytorch_variables)
        
        # Set the environment if provided
        if env is not None:
            model.set_env(env, force_reset=kwargs.get("force_reset", True))
        
        # Clean up temp variable
        if hasattr(cls, '_temp_config'):
            delattr(cls, '_temp_config')
        
        return model



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

        callback.on_rollout_start()

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde:
                self.actor.reset_noise(env.num_envs)
            self._append_recent_observation(self._last_obs)

            with torch.no_grad():
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                mean_actions, log_std, _ = self.actor.get_action_dist_params(obs_tensor)
                reward_state = self._extract_reward_state(self._last_obs)
                mu = (mean_actions.detach(), log_std.detach())
                actions, log_probs = self.actor.action_log_prob(obs_tensor)
                actions_np = actions.cpu(memory_format=torch.contiguous_format).numpy()
                log_probs_np = log_probs.cpu(memory_format=torch.contiguous_format).numpy()
                
                r_omega = self.auto_reward_learner.get_reward(reward_state, actions)
                r_omega_val = r_omega.cpu(memory_format=torch.contiguous_format).numpy().flatten()
                
                reward_state_cpu = reward_state[0].cpu(memory_format=torch.contiguous_format).numpy()
                mu_cpu = (mu[0][0].cpu(), mu[1][0].cpu())

            new_obs, rewards, dones, infos = env.step(actions_np)
            env_rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            ground_truth_rewards = env_rewards
            policy_rewards = self._mix_policy_reward(env_rewards, r_omega_val)
            
            self.auto_reward_learner.store_transition(
                state=reward_state_cpu,
                action=actions_np.flatten(),
                reward=ground_truth_rewards[0],
                log_prob=log_probs_np.flatten()[0],
                mu=mu_cpu
            )

            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            
            # Handle episode end
            real_next_obs = new_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    self.auto_reward_learner.on_episode_end()
                    num_collected_episodes += 1
                    self._episode_num += 1
                    self._recent_policy_observations.clear()
                    if infos[idx].get("terminal_observation") is not None:
                        real_next_obs[idx] = infos[idx]["terminal_observation"]
            
            # Store with learned reward
            self.replay_buffer.add(self._last_obs, real_next_obs, actions_np, policy_rewards, dones, infos)
            self._last_obs = new_obs
            
            # Update callback locals for TensorboardCallback
            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
        
        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=True)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Train with periodic meta-gradient updates."""
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        action_smoothness_losses, steer_smoothness_losses, longitudinal_smoothness_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
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
            critic_loss = 0.5 * sum(torch.nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_base_loss = (ent_coef * log_prob - min_qf_pi).mean()
            action_smoothness_loss = self._compute_action_smoothness_loss()
            actor_loss = actor_base_loss + action_smoothness_loss
            actor_losses.append(actor_loss.item())
            action_smoothness_losses.append(self.last_action_smoothness_loss)
            steer_smoothness_losses.append(self.last_steer_smoothness_loss)
            longitudinal_smoothness_losses.append(self.last_longitudinal_smoothness_loss)

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/action_smoothness_loss", np.mean(action_smoothness_losses))
        self.logger.record("train/steer_smoothness_loss", np.mean(steer_smoothness_losses), exclude=("stdout",))
        self.logger.record("train/longitudinal_smoothness_loss", np.mean(longitudinal_smoothness_losses), exclude=("stdout",))
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

        # Meta-update at specified frequency, with optional max-update freezing.
        should_meta_update = (
            self.num_timesteps > self.learning_starts
            and self.num_timesteps % self.reward_update_freq < gradient_steps
        )
        current_meta_updates = 0 if self.auto_reward_learner is None else self.auto_reward_learner.num_meta_updates
        can_meta_update = self.max_meta_updates is None or current_meta_updates < self.max_meta_updates

        if should_meta_update and can_meta_update:
            metrics = self.auto_reward_learner.optimize_reward(sample_action_from_mu)
            
            if metrics:
                self.autoreward_num_meta_updates = self.auto_reward_learner.num_meta_updates
                self.logger.record("autoreward/value_loss", metrics.get("value_loss", 0.0))
                self.logger.record("autoreward/mean_gt_reward", metrics.get("mean_gt_reward", 0.0))
                self.logger.record("autoreward/reward_std", metrics.get("reward_std", 0.0))
                self.logger.record("autoreward/meta_updates", metrics.get("meta_updates", 0))
                self.logger.record("autoreward/reward_mix_alpha", self.last_reward_mix_alpha)
                self.logger.record("autoreward/raw_learned_reward_mean", self.last_raw_learned_reward_mean)
                self.logger.record("autoreward/norm_learned_reward_mean", self.last_normalized_learned_reward_mean)
                self.logger.record("autoreward/reward_learner_frozen", 0.0)
                self.logger.record("autoreward/meta_loss", metrics.get("meta_loss", 0.0), exclude=("stdout",))
                self.logger.record("autoreward/mean_R", metrics.get("mean_R_omega", 0.0), exclude=("stdout",))
                self.logger.record("autoreward/mean_Adv", metrics.get("mean_Advantage", 0.0), exclude=("stdout",))
                self.logger.record("autoreward/std_Adv", metrics.get("std_Advantage", 0.0), exclude=("stdout",))
        elif should_meta_update and not can_meta_update:
            self.autoreward_num_meta_updates = current_meta_updates
            self.logger.record("autoreward/meta_updates", current_meta_updates)
            self.logger.record("autoreward/reward_learner_frozen", 1.0)
