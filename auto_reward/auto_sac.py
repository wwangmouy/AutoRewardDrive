from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from auto_reward.learner import AutoRewardLearner


class ChunkController:
    """Hold the same action for multiple steps to reduce control jitter."""

    def __init__(self, enabled: bool = False, chunk_len: int = 1, mode: str = "hold"):
        self.enabled = bool(enabled and chunk_len > 1)
        self.chunk_len = max(1, int(chunk_len))
        self.mode = mode
        if self.mode != "hold":
            raise ValueError(f"Unsupported chunk mode: {self.mode}")
        self.reset()

    def reset(self) -> None:
        self._chunk_step = 0
        self._chunk_action: Optional[np.ndarray] = None

    def apply(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float32)
        single = actions.ndim == 1
        if single:
            actions = actions.reshape(1, -1)

        if not self.enabled:
            reused = np.zeros(actions.shape[0], dtype=bool)
            return (actions[0].copy(), reused) if single else (actions.copy(), reused)

        reused = np.zeros(actions.shape[0], dtype=bool)
        output = actions.copy()

        if self._chunk_action is None or self._chunk_step == 0:
            self._chunk_action = actions[0].copy()
            self._chunk_step = 1
        else:
            output[0] = self._chunk_action.copy()
            reused[0] = True
            self._chunk_step += 1
            if self._chunk_step >= self.chunk_len:
                self._chunk_step = 0

        if not reused[0]:
            output[0] = self._chunk_action.copy()
            if self._chunk_step >= self.chunk_len:
                self._chunk_step = 0

        return (output[0].copy(), reused) if single else (output, reused)


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
        self.config = config
        self.reward_update_freq = config.get("reward_update_freq", 2048)
        self.auto_reward_learner = None

        chunk_config = config.get("chunk", {})
        chunk_train_config = config.get("chunk_train", {})
        self.chunk_enabled = bool(chunk_config.get("enabled", False))
        self.chunk_len = int(max(1, chunk_config.get("len", 1)))
        self.chunk_mode = chunk_config.get("mode", "hold")
        self.chunk_actor_lambda = float(chunk_train_config.get("actor_lambda", 0.0))
        self.chunk_critic_lambda = float(chunk_train_config.get("critic_lambda", 0.0))
        self.chunk_sample_windows = int(max(1, chunk_train_config.get("sample_windows", 1)))
        self.chunk_buffer_size = int(max(1, chunk_train_config.get("buffer_size", 256)))

        self.chunk_controller: Optional[ChunkController] = None
        self.recent_chunk_buffer: deque = deque(maxlen=self.chunk_buffer_size)
        self.use_chunk_training = self.chunk_enabled and self.chunk_len > 1
        self.use_executed_action_training = self.use_chunk_training
        self.chunk_action_weights: Optional[torch.Tensor] = None

        self.last_raw_action = None
        self.last_chunk_action = None
        self.last_chunk_reused = False
        self._prev_episode_raw_action = None
        self._episode_raw_action_delta_sum = 0.0
        self._episode_raw_action_delta_count = 0
        self._episode_chunk_reuse_count = 0

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

    @staticmethod
    def _copy_batch_obs(obs):
        if isinstance(obs, dict):
            return {key: value.copy() for key, value in obs.items()}
        return obs.copy()

    @staticmethod
    def _extract_single_env_obs(obs, idx: int):
        if isinstance(obs, dict):
            return {key: np.array(value[idx], copy=True) for key, value in obs.items()}
        return np.array(obs[idx], copy=True)

    @staticmethod
    def _set_single_env_obs(obs_batch, idx: int, obs_value) -> None:
        if isinstance(obs_batch, dict):
            for key, value in obs_value.items():
                obs_batch[key][idx] = value
        else:
            obs_batch[idx] = obs_value

    @staticmethod
    def _stack_obs_list(obs_list):
        first_obs = obs_list[0]
        if isinstance(first_obs, dict):
            return {
                key: np.stack([np.array(obs[key], copy=False) for obs in obs_list], axis=0)
                for key in first_obs.keys()
            }
        return np.stack([np.array(obs, copy=False) for obs in obs_list], axis=0)

    def _reset_episode_action_stats(self) -> None:
        self._prev_episode_raw_action = None
        self._episode_raw_action_delta_sum = 0.0
        self._episode_raw_action_delta_count = 0
        self._episode_chunk_reuse_count = 0

    def reset_chunk_state(self) -> None:
        if self.chunk_controller is not None:
            self.chunk_controller.reset()
        self._reset_episode_action_stats()

    def _record_raw_action_stats(self, raw_action: np.ndarray, chunk_reused: bool) -> None:
        if self._prev_episode_raw_action is not None:
            self._episode_raw_action_delta_sum += float(np.linalg.norm(raw_action - self._prev_episode_raw_action))
            self._episode_raw_action_delta_count += 1
        if chunk_reused:
            self._episode_chunk_reuse_count += 1
        self._prev_episode_raw_action = raw_action.copy()

    def _append_recent_chunk_transition(self, obs, action, reward, next_obs, done: bool) -> None:
        if not self.use_chunk_training:
            return
        self.recent_chunk_buffer.append(
            {
                "obs": obs,
                "action": np.asarray(action, dtype=np.float32).copy(),
                "reward": float(reward),
                "next_obs": next_obs,
                "done": bool(done),
            }
        )

    def _sample_window_starts(self, allow_terminal_before_last: bool = True) -> List[int]:
        if not self.use_chunk_training or len(self.recent_chunk_buffer) < self.chunk_len:
            return []

        valid_starts = []
        max_start = len(self.recent_chunk_buffer) - self.chunk_len + 1
        for start_idx in range(max_start):
            window = list(self.recent_chunk_buffer)[start_idx : start_idx + self.chunk_len]
            if not allow_terminal_before_last:
                if any(step["done"] for step in window[:-1]):
                    continue
            valid_starts.append(start_idx)
        if not valid_starts:
            return []

        sample_count = min(self.chunk_sample_windows, len(valid_starts))
        sampled = np.random.choice(valid_starts, size=sample_count, replace=len(valid_starts) < sample_count)
        return [int(index) for index in np.atleast_1d(sampled)]

    def _compute_chunk_actor_smooth_loss(self) -> Optional[torch.Tensor]:
        if not self.use_chunk_training or self.chunk_actor_lambda <= 0:
            return None

        start_indices = self._sample_window_starts(allow_terminal_before_last=False)
        if not start_indices:
            return None

        chunk_data = list(self.recent_chunk_buffer)
        obs_sequence = []
        for start_idx in start_indices:
            window = chunk_data[start_idx : start_idx + self.chunk_len]
            obs_sequence.append(window[0]["obs"])
            for step in window:
                obs_sequence.append(step["next_obs"])

        stacked_obs = self._stack_obs_list(obs_sequence)
        obs_tensor, _ = self.policy.obs_to_tensor(stacked_obs)
        actions = self.actor(obs_tensor, deterministic=True)
        actions = actions.reshape(len(start_indices), self.chunk_len + 1, -1)

        action_diff = actions[:, 1:] - actions[:, :-1]
        if self.chunk_action_weights is None:
            weights = torch.ones(action_diff.shape[-1], device=self.device)
        else:
            weights = self.chunk_action_weights
        return ((action_diff.pow(2)) * weights.view(1, 1, -1)).mean()

    def _compute_chunk_critic_loss(self, ent_coef: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_chunk_training or self.chunk_critic_lambda <= 0:
            return None

        start_indices = self._sample_window_starts(allow_terminal_before_last=True)
        if not start_indices:
            return None

        chunk_data = list(self.recent_chunk_buffer)
        obs_batch = []
        next_obs_batch = []
        action_batch = []
        returns = []
        discounts = []

        for start_idx in start_indices:
            window = chunk_data[start_idx : start_idx + self.chunk_len]
            cumulative_return = 0.0
            discount = 1.0
            bootstrap_discount = self.gamma ** self.chunk_len
            bootstrap_obs = window[-1]["next_obs"]
            terminated = False

            for step_idx, step in enumerate(window):
                cumulative_return += discount * float(step["reward"])
                if step["done"]:
                    bootstrap_discount = 0.0
                    bootstrap_obs = step["next_obs"]
                    terminated = True
                    break
                discount *= self.gamma

            if not terminated:
                bootstrap_obs = window[-1]["next_obs"]

            obs_batch.append(window[0]["obs"])
            next_obs_batch.append(bootstrap_obs)
            action_batch.append(np.asarray(window[0]["action"], dtype=np.float32))
            returns.append(cumulative_return)
            discounts.append(bootstrap_discount)

        obs_tensor, _ = self.policy.obs_to_tensor(self._stack_obs_list(obs_batch))
        next_obs_tensor, _ = self.policy.obs_to_tensor(self._stack_obs_list(next_obs_batch))
        action_tensor = torch.as_tensor(np.stack(action_batch), device=self.device).float()
        returns_tensor = torch.as_tensor(np.array(returns), device=self.device).float().reshape(-1, 1)
        discount_tensor = torch.as_tensor(np.array(discounts), device=self.device).float().reshape(-1, 1)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(next_obs_tensor)
            next_q_values = torch.cat(self.critic_target(next_obs_tensor, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            target_q_values = returns_tensor + discount_tensor * next_q_values

        current_q_values = self.critic(obs_tensor, action_tensor)
        return 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

    def _squashed_log_prob_from_action(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        action_low = torch.as_tensor(self.action_space.low, device=self.device).float()
        action_high = torch.as_tensor(self.action_space.high, device=self.device).float()
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

        y_t = torch.clamp((actions - action_bias) / action_scale, -0.999999, 0.999999)
        x_t = 0.5 * (torch.log1p(y_t) - torch.log1p(-y_t))

        normal = torch.distributions.Normal(mean_actions, log_std.exp())
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)

    def _setup_model(self) -> None:
        super()._setup_model()

        if hasattr(self.actor, "features_extractor") and hasattr(self.actor.features_extractor, "_features_dim"):
            state_dim = self.actor.features_extractor._features_dim
        elif hasattr(self.actor, "features_extractor") and hasattr(self.actor.features_extractor, "features_dim"):
            state_dim = self.actor.features_extractor.features_dim
        else:
            from stable_baselines3.common.preprocessing import get_flattened_obs_dim

            state_dim = get_flattened_obs_dim(self.observation_space)

        action_dim = self.action_space.shape[0]
        self.chunk_action_weights = torch.as_tensor(
            [
                self.config.get("chunk_train", {}).get("action_weights", {}).get("steer", 1.0),
                self.config.get("chunk_train", {}).get("action_weights", {}).get("longitudinal", 1.0),
            ][:action_dim],
            device=self.device,
        ).float()

        self.auto_reward_learner = AutoRewardLearner(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            config=self.config,
        )
        self.chunk_controller = ChunkController(
            enabled=self.chunk_enabled,
            chunk_len=self.chunk_len,
            mode=self.chunk_mode,
        )
        self.recent_chunk_buffer = deque(maxlen=self.chunk_buffer_size)
        self.reset_chunk_state()
        print(f"[AutoRewardedSAC] Initialized: state_dim={state_dim}, action_dim={action_dim}")

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
            raise ValueError("config argument is required for AutoRewardedSAC.load()")

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

        model.reset_chunk_state()
        return model

    def predict(
        self,
        observation,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        if episode_start is not None and np.any(episode_start):
            self.reset_chunk_state()

        raw_action, next_state = super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

        raw_action_np = np.asarray(raw_action, dtype=np.float32)
        self.last_raw_action = raw_action_np.copy()

        if self.chunk_controller is None:
            self.last_chunk_action = raw_action_np.copy()
            self.last_chunk_reused = False
            return raw_action, next_state

        chunk_action_np, reused_mask = self.chunk_controller.apply(raw_action_np)
        self.last_chunk_action = np.asarray(chunk_action_np, dtype=np.float32).copy()
        self.last_chunk_reused = bool(np.asarray(reused_mask).reshape(-1)[0])
        return chunk_action_np, next_state

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
        """Collect rollouts with optional chunk execution and executed-action feedback."""
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv) and env.num_envs == 1, "Only supports single env"
        callback.on_rollout_start()

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde:
                self.actor.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                features = self.actor.extract_features(obs_tensor, self.actor.features_extractor)
                mean_actions, log_std, _ = self.actor.get_action_dist_params(obs_tensor)
                raw_actions, sampled_log_probs = self.actor.action_log_prob(obs_tensor)
                raw_actions_np = raw_actions.cpu(memory_format=torch.contiguous_format).numpy()
                sampled_log_probs_np = sampled_log_probs.cpu(memory_format=torch.contiguous_format).numpy()
                features_cpu = features.cpu(memory_format=torch.contiguous_format).numpy()
                mu_cpu = (mean_actions[0].detach().cpu(), log_std[0].detach().cpu())

            if self.chunk_controller is not None:
                chunk_actions_np, reused_mask = self.chunk_controller.apply(raw_actions_np)
            else:
                chunk_actions_np = raw_actions_np.copy()
                reused_mask = np.zeros(env.num_envs, dtype=bool)

            self.last_raw_action = raw_actions_np[0].copy()
            self.last_chunk_action = chunk_actions_np[0].copy()
            self.last_chunk_reused = bool(reused_mask[0])
            self._record_raw_action_stats(self.last_raw_action, self.last_chunk_reused)

            new_obs, rewards, dones, infos = env.step(chunk_actions_np)
            real_next_obs = self._copy_batch_obs(new_obs)

            executed_actions_np = chunk_actions_np.copy()
            train_log_probs_np = sampled_log_probs_np.copy()
            finished_episode_indices = []

            for idx, done in enumerate(dones):
                info = infos[idx]
                info["raw_action"] = raw_actions_np[idx].copy()
                info["chunk_action"] = chunk_actions_np[idx].copy()
                info["chunk_reused"] = bool(reused_mask[idx])

                if self.use_executed_action_training and info.get("executed_action") is not None:
                    executed_actions_np[idx] = np.asarray(info["executed_action"], dtype=np.float32)

                if done:
                    terminal_obs = info.get("terminal_observation")
                    if terminal_obs is not None:
                        self._set_single_env_obs(real_next_obs, idx, terminal_obs)

                    num_collected_episodes += 1
                    self._episode_num += 1
                    finished_episode_indices.append(idx)

                    info["raw_action_delta"] = (
                        self._episode_raw_action_delta_sum / self._episode_raw_action_delta_count
                        if self._episode_raw_action_delta_count
                        else 0.0
                    )
                    episode_length = max(1, int(info.get("episode_length", 1)))
                    info["chunk_reuse_ratio"] = self._episode_chunk_reuse_count / episode_length

                    if self.chunk_controller is not None:
                        self.chunk_controller.reset()
                    self._reset_episode_action_stats()

            if self.use_executed_action_training:
                with torch.no_grad():
                    executed_actions_tensor = torch.as_tensor(executed_actions_np, device=self.device).float()
                    train_log_probs = self._squashed_log_prob_from_action(mean_actions, log_std, executed_actions_tensor)
                    train_log_probs_np = train_log_probs.cpu(memory_format=torch.contiguous_format).numpy()
                    train_reward_actions = executed_actions_tensor
                    r_omega = self.auto_reward_learner.get_reward(features, train_reward_actions)
                    replay_actions_np = executed_actions_np
            else:
                with torch.no_grad():
                    r_omega = self.auto_reward_learner.get_reward(features, raw_actions)
                    replay_actions_np = raw_actions_np

            r_omega_val = r_omega.cpu(memory_format=torch.contiguous_format).numpy().flatten()

            self.auto_reward_learner.store_transition(
                state=features_cpu[0].flatten(),
                action=replay_actions_np[0].flatten(),
                reward=rewards[0],
                log_prob=train_log_probs_np.flatten()[0],
                mu=mu_cpu,
            )

            if finished_episode_indices:
                self.auto_reward_learner.on_episode_end()

            self._append_recent_chunk_transition(
                obs=self._extract_single_env_obs(self._last_obs, 0),
                action=replay_actions_np[0],
                reward=float(r_omega_val[0]),
                next_obs=self._extract_single_env_obs(real_next_obs, 0),
                done=bool(dones[0]),
            )

            self.replay_buffer.add(self._last_obs, real_next_obs, replay_actions_np, r_omega_val, dones, infos)
            self._last_obs = new_obs

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())

            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=True)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Train SAC and add chunk-aligned auxiliary losses when enabled."""
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        chunk_actor_losses, chunk_critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if getattr(replay_data, "discounts", None) is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
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
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

            chunk_critic_loss = self._compute_chunk_critic_loss(ent_coef)
            if chunk_critic_loss is not None:
                critic_loss = critic_loss + self.chunk_critic_lambda * chunk_critic_loss
                chunk_critic_losses.append(chunk_critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(critic_loss.item())

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            chunk_actor_loss = self._compute_chunk_actor_smooth_loss()
            if chunk_actor_loss is not None:
                actor_loss = actor_loss + self.chunk_actor_lambda * chunk_actor_loss
                chunk_actor_losses.append(chunk_actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_losses.append(actor_loss.item())

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record(
            "train/chunk_actor_smooth_loss",
            float(np.mean(chunk_actor_losses)) if chunk_actor_losses else 0.0,
        )
        self.logger.record(
            "train/chunk_critic_loss",
            float(np.mean(chunk_critic_losses)) if chunk_critic_losses else 0.0,
        )
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        def sample_action_from_mu(mu_batch, n_samples):
            """Resample actions from policy distribution."""
            mean, log_std = mu_batch
            mean, log_std = mean.to(self.device), log_std.to(self.device)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)

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

        if self.num_timesteps > self.learning_starts and self.num_timesteps % self.reward_update_freq < gradient_steps:
            metrics = self.auto_reward_learner.optimize_reward(sample_action_from_mu)

            if metrics:
                self.logger.record("autoreward/meta_loss", metrics.get("meta_loss", 0.0))
                self.logger.record("autoreward/value_loss", metrics.get("value_loss", 0.0))
                self.logger.record("autoreward/mean_R", metrics.get("mean_R_omega", 0.0))
                self.logger.record("autoreward/mean_Adv", metrics.get("mean_Advantage", 0.0))
