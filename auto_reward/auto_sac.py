import torch
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from auto_reward.learner import AutoRewardLearner

class AutoRewardedSAC(SAC):
    """
    AutoRewardedSAC: An extension of standard SAC that integrates AutoRewardDrive.

    Modifications:
    1. collect_rollouts:
       - Captures (state, action, reward_ground_truth, mu) for Meta-Learning.
       - Replaces `reward` with R_omega(state, action) for SAC updates.
    2. train:
       - Periodically executes `auto_reward_learner.optimize_reward()`.
    """

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        config: Any, # Pass full config to access params
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
        self.config = config
        self.reward_update_freq = config.get('reward_update_freq', 2048)
        self.auto_reward_learner = None # Initialized in _setup_model

    def _setup_model(self) -> None:
        super()._setup_model()
        
        # Initialize AutoRewardLearner
        # Calculate state_dim from the features extractor
        # Assuming the policy uses a features extractor that outputs a flat vector (e.g., CustomMultiInputExtractor)
        if hasattr(self.policy, "features_extractor") and hasattr(self.policy.features_extractor, "features_dim"):
            state_dim = self.policy.features_extractor.features_dim
        else:
            # Fallback for simple MLPs
            state_dim = self.observation_space.shape[0]

        action_dim = self.action_space.shape[0]
        
        self.auto_reward_learner = AutoRewardLearner(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            config=self.config
        )
        print(f"[AutoRewardedSAC] Initialized Learner with State Dim: {state_dim}, Action Dim: {action_dim}")

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: MaybeCallback,
        train_freq: Type[Any], # TrainFreq
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Overridden to intercept rewards and collect meta-learning data.
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0
        
        assert isinstance(env, VecEnv), "You must use a VecEnv " + str(env)
        
        if env.num_envs > 1:
            assert False, "AutoRewardedSAC current only supports num_envs=1 due to trajectory buffering complexity."

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            
            # 1. Get Action and Policy Params (mu)
            if self.use_sde:
                self.actor.reset_noise(env.num_envs)

            # Sample action and get distribution parameters for Meta-Learning
            # We need to manually access the policy's distribution forwarding to get 'mu'
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device).float() # Raw observations
                
                # Extract features first (we need them for R_omega and storage)
                features = self.policy.features_extractor(obs_tensor)
                
                # Get action and distribution
                # Re-implementing parts of actor.forward to capture 'mean' and 'log_std' (which form mu)
                mean_actions, log_std, _ = self.actor.get_action_dist_params(obs_tensor)
                
                # Note: SB3 Actor might return flattened things.
                # AutoRewardDrive expects a tuple/list for 'mu' that can be passed to a sampling function later.
                # For GaussianPolicy, mu = (mean, log_std)
                mu = (mean_actions.cpu(), log_std.cpu())
                
                # Sample action standard SB3 way
                actions, log_probs = self.actor.action_log_prob(obs_tensor)
                actions = actions.cpu().numpy()
                log_probs = log_probs.cpu().numpy()

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # 2. Compute R_omega (Learned Reward)
            # We use the 'features' we extracted earlier
            with torch.no_grad():
                # Convert actions to tensor
                actions_tensor = torch.as_tensor(actions).to(self.device).float()
                r_omega = self.auto_reward_learner.get_reward(features, actions_tensor)
                r_omega_val = r_omega.cpu().numpy().flatten()
            
            # 3. Store Data for Meta-Learning (Ground Truth)
            # self._last_obs is NOT raw sometimes if Dict (it is raw in VecEnv usually)
            # We store the *features* in the buffer to save computation later
            self.auto_reward_learner.store_transition(
                state=features.cpu().numpy().flatten(), # Store encoded state
                action=actions.flatten(),
                reward=rewards[0], # Scalar Ground Truth
                log_prob=log_probs.flatten()[0],
                mu=(mu[0][0], mu[1][0]) # Unbatch: mean[0], log_std[0]
            )

            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            
            # 4. Store Data for SAC (Learned Reward)
            # We construct the transition using R_omega
            # Handle real_next_obs logic
            real_next_obs = new_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    if infos[idx].get("terminal_observation") is not None:
                         # For AutoReward, we might want to store end-of-episode logic
                         self.auto_reward_learner.on_episode_end()
                         num_collected_episodes += 1
                         self._episode_num += 1
                         real_next_obs[idx] = infos[idx]["terminal_observation"]
            
            # Add to Replay Buffer using LEARNED REWARD
            # Note: We pass original obs/next_obs so SAC can re-compute features if needed 
            # (though standard SAC re-computes features from raw obs during training)
            self.replay_buffer.add(self._last_obs, real_next_obs, actions, r_omega_val, dones, infos)

            self._last_obs = new_obs
            
            # Callback
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
        
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Modified train loop to include Meta-Gradient updates.
        """
        # 1. Standard SAC Update
        super().train(gradient_steps, batch_size)
        
        # 2. AutoReward Update (Meta-Learning)
        # Check frequency (e.g., every 2048 steps or handled by learner config)
        # Official code: if global_step % reward_frequency == 0
        
        
        # Helper function for resampling actions (closure)
        def sample_action_from_mu(mu_batch, n_samples):
            # mu_batch is tuple (mean, log_std)
            # We need to handle this carefully.
            # In learner.optimize_reward, it passes 'mu' for a SINGLE step.
            # mu = (mean_tensor, log_std_tensor)
            mean, log_std = mu_batch
            mean = mean.to(self.device)
            log_std = log_std.to(self.device)
            
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            
            # Sample N actions [N, dim]
            x_t = normal.rsample((n_samples,))
            y_t = torch.tanh(x_t)
            # Scale
            # Note: self.actor.action_scale is usually tensor or float
            action = y_t * self.actor.action_scale + self.actor.action_bias
            
            # Log prob
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound (SAC formula)
            log_prob -= torch.log(self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return action, log_prob

        # Run Meta-Update
        # We check modulo reward_update_freq, and ensures we are past learning_starts
        if self.num_timesteps > self.learning_starts and self.num_timesteps % self.reward_update_freq < gradient_steps: 
            # Rough check to execute once per 'freq' window
            
            metrics = self.auto_reward_learner.optimize_reward(sample_action_from_mu)
            
            # Log metrics
            if metrics:
                self.logger.record("autoreward/meta_loss", metrics.get("meta_loss", 0.0))
                self.logger.record("autoreward/value_loss", metrics.get("value_loss", 0.0))
                self.logger.record("autoreward/mean_R", metrics.get("mean_R_omega", 0.0))
                self.logger.record("autoreward/mean_Adv", metrics.get("mean_Advantage", 0.0))

