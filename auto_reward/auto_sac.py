import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn
from stable_baselines3.common.utils import should_collect_more_steps
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

            with torch.no_grad():
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                features = self.actor.extract_features(obs_tensor, self.actor.features_extractor)
                mean_actions, log_std, _ = self.actor.get_action_dist_params(obs_tensor)
                # Use non_blocking to reduce sync overhead
                mu = (mean_actions.detach(), log_std.detach())  # Keep on GPU, transfer later
                actions, log_probs = self.actor.action_log_prob(obs_tensor)
                actions_np = actions.cpu(memory_format=torch.contiguous_format).numpy()
                log_probs_np = log_probs.cpu(memory_format=torch.contiguous_format).numpy()
                
                # Compute learned reward R_omega inline (avoid redundant tensor conversion)
                r_omega = self.auto_reward_learner.get_reward(features, actions)
                r_omega_val = r_omega.cpu(memory_format=torch.contiguous_format).numpy().flatten()
                
                # Cache features on CPU (single transfer)
                features_cpu = features.cpu(memory_format=torch.contiguous_format).numpy().flatten()
                mu_cpu = (mu[0][0].cpu(), mu[1][0].cpu())

            new_obs, rewards, dones, infos = env.step(actions_np)
            
            # Store for meta-learning (ground truth reward)
            self.auto_reward_learner.store_transition(
                state=features_cpu,
                action=actions_np.flatten(),
                reward=rewards[0],
                log_prob=log_probs_np.flatten()[0],
                mu=mu_cpu
            )

            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            
            # Handle episode end
            real_next_obs = new_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    if infos[idx].get("terminal_observation") is not None:
                        self.auto_reward_learner.on_episode_end()
                        num_collected_episodes += 1
                        self._episode_num += 1
                        real_next_obs[idx] = infos[idx]["terminal_observation"]
            
            # Store with learned reward
            self.replay_buffer.add(self._last_obs, real_next_obs, actions_np, r_omega_val, dones, infos)
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
        super().train(gradient_steps, batch_size)
        
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