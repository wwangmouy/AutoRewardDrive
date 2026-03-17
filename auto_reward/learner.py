import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

from auto_reward.networks import RewardNetwork, ValueFunction

# Define Transition tuple to store all necessary data for meta-learning
Transition = namedtuple('Transition', 
                        ['state', 'action', 'reward', 'log_prob', 'mu', 'overline_V'])

class AutoRewardLearner:
    """
    Core Logic for Optimal Reward Discovery.
    
    Responsibilities:
    1. Manage RewardNetwork R_omega(s, a)
    2. Manage ValueFunction V(s) (Ground Truth)
    3. Store Trajectories in Buffer D_xi
    4. Compute Meta-Gradient Update
    """
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 device, 
                 config):
        
        self.device = device
        self.config = config
        
        # Hyperparameters
        self.gamma = config.gamma
        self.reward_lr = config.get('reward_lr', 1e-4)
        self.value_lr = config.get('value_lr', 3e-4)
        self.n_samples = config.get('n_samples', 128)
        self.reward_buffer_size = config.get('reward_buffer_size', 64)
        self.reward_output_scale = config.get('reward_output_scale', 1.0)
        self.num_meta_updates = 0
        
        # 1. Trainable Reward Function R_omega(s, a)
        self.reward_net = RewardNetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            hidden_dim=256, 
            encode_dim=64,
            output_scale=self.reward_output_scale
        ).to(device)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.reward_lr)
        
        # 2. Value Function V(s) - Predicts Ground Truth Return
        self.value_net = ValueFunction(input_dim=state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        
        # 3. Data Storage
        self.D_xi = deque(maxlen=self.reward_buffer_size) # Trajectory buffer
        self.current_episode_data = [] # Temp storage for current episode
        self._pending_transitions = []  # Batch processing buffer
        self._batch_size = 64  # Process every N steps
        
    def get_reward(self, state, action):
        """
        Returns R_omega(s, a) for the Agent's training.
        """
        with torch.no_grad():
            return self.reward_net(state, action)

    def store_transition(self, state, action, reward, log_prob, mu):
        """
        Store transition data during rollout.
        Args:
            reward: This should be the GROUND TRUTH reward (bar_R)
            mu: Policy distribution params tuple (mean, log_std, ...) for resampling
        """
        # Ensure data is on CPU/Numpy for storage to save VRAM
        if isinstance(state, torch.Tensor): state = state.detach().cpu().numpy()
        if isinstance(action, torch.Tensor): action = action.detach().cpu().numpy()
        if isinstance(log_prob, torch.Tensor): log_prob = log_prob.detach().cpu().numpy()
        
        # mu is a tuple of tensors usually
        mu_cpu = []
        if isinstance(mu, (tuple, list)):
            for m in mu:
                if isinstance(m, torch.Tensor):
                    mu_cpu.append(m.detach().cpu())
                else:
                    mu_cpu.append(m)
        else:
            mu_cpu = mu # Fallback
            
        t = Transition(state=state, action=action, reward=reward, log_prob=log_prob, mu=tuple(mu_cpu), overline_V=0.0)
        self.current_episode_data.append(t)

    def on_episode_end(self):
        """
        Called when episode finishes. 
        Computes Ground Truth Return (overline_V) and moves trajectory to D_xi.
        """
        if not self.current_episode_data:
            return

        R_bar_sum = 0
        new_trajectory = []
        
        # Backward pass to compute discounted return
        for t in reversed(self.current_episode_data):
            R_bar_sum = t.reward + self.gamma * R_bar_sum
            # Update the namedtuple with computed return
            new_t = t._replace(overline_V=R_bar_sum)
            new_trajectory.insert(0, new_t)
            
        # Store in main buffer
        self.D_xi.append(new_trajectory)
        self.current_episode_data = [] # Reset

    def optimize_reward(self, agent_policy_func):
        """
        Meta-Optimization Step.
        
        Args:
            agent_policy_func: Function to resample actions from mu. 
                               Signature: get_action_prob_from_mu(mu, n_samples)
        """
        if len(self.D_xi) == 0:
            return {}

        # 1. Flatten trajectories
        all_steps = [step for traj in self.D_xi for step in traj]
        # Shuffle for i.i.d updates
        random.shuffle(all_steps)
        
        total_steps = len(all_steps)
        
        states_list = []
        overline_V_list = []
        
        for step in all_steps:
            s_np, a_np, r_bar, log_prob_np, mu, overline_V = step
            states_list.append(s_np)
            overline_V_list.append(overline_V)
        batch_size = self._batch_size
        states_tensor_all = torch.tensor(np.array(states_list), device=self.device).float()
        targets_tensor_all = torch.tensor(np.array(overline_V_list), device=self.device).float().unsqueeze(1)
        
        value_loss_sum = 0.0
        
        # Update the baseline first so the advantage estimate is less noisy.
        self.value_optimizer.zero_grad()
        for i in range(0, total_steps, batch_size):
            end_idx = min(i + batch_size, total_steps)
            s_batch = states_tensor_all[i:end_idx]
            target_batch = targets_tensor_all[i:end_idx]
            
            preds = self.value_net(s_batch)
            v_loss = nn.functional.smooth_l1_loss(preds, target_batch)
            
            # Scale loss by batch size ratio for correct mean
            v_loss_scaled = v_loss * ((end_idx - i) / total_steps)
            v_loss_scaled.backward()
            
            value_loss_sum += v_loss.item() * (end_idx - i)
            
        self.value_optimizer.step()

        # Use per-sample centered rewards weighted by normalized advantages.
        with torch.no_grad():
            advantages_all = targets_tensor_all - self.value_net(states_tensor_all)
            advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-6)
            advantages_all = advantages_all.clamp(-5.0, 5.0)

        self.reward_optimizer.zero_grad()
        reward_loss_sum = 0.0
        centered_reward_sum = 0.0
        gt_reward_sum = 0.0
        reward_std_chunks = []
        
        for i in range(0, total_steps, batch_size):
            end_idx = min(i + batch_size, total_steps)
            batch_steps = all_steps[i:end_idx]
            weighted_terms = []
            centered_rewards = []
            
            for offset, step in enumerate(batch_steps):
                s_np, a_np, r_bar, log_prob_np, mu, overline_V = step
                s = torch.tensor(s_np, device=self.device).float().unsqueeze(0)
                a_tensor = torch.tensor(a_np, device=self.device).float().unsqueeze(0)
                
                r_omega_cur = self.reward_net(s, a_tensor).squeeze()
                action_samples, _ = agent_policy_func(mu, self.n_samples)
                s_expanded = s.repeat(self.n_samples, 1)
                r_omega_samples = self.reward_net(s_expanded, action_samples).squeeze(-1)
                reward_baseline = r_omega_samples.mean()
                centered_reward = r_omega_cur - reward_baseline
                advantage = advantages_all[i + offset].detach().squeeze()
                
                weighted_terms.append(-(advantage * centered_reward))
                centered_rewards.append(centered_reward.detach())
                gt_reward_sum += float(r_bar)
            
            reward_loss_batch = torch.stack(weighted_terms).mean()
            reward_loss_batch.backward()
            reward_loss_sum += reward_loss_batch.item() * (end_idx - i)
            centered_reward_sum += torch.stack(centered_rewards).mean().item() * (end_idx - i)
            reward_std_chunks.append(torch.stack(centered_rewards).std(unbiased=False).item())
        
        self.reward_optimizer.step()
        self.num_meta_updates += 1
        
        return {
            "meta_loss": reward_loss_sum / total_steps,
            "value_loss": value_loss_sum / total_steps,
            "mean_R_omega": centered_reward_sum / total_steps,
            "mean_Advantage": advantages_all.mean().item(),
            "std_Advantage": advantages_all.std(unbiased=False).item(),
            "mean_gt_reward": gt_reward_sum / total_steps,
            "reward_std": float(np.mean(reward_std_chunks)) if reward_std_chunks else 0.0,
            "meta_updates": self.num_meta_updates,
        }
