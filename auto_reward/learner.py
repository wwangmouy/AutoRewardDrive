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
                 config,
                 eval_reward_params=None):
        
        self.device = device
        self.config = config
        
        # Hyperparameters
        self.gamma = config.gamma
        self.reward_lr = 1e-4
        self.value_lr = 3e-4 # Usually higher than reward LR
        self.n_samples = config.algorithm_params.get('n_samples', 1000) # Number of samples for expectation estimation
        self.reward_buffer_size = config.algorithm_params.get('reward_buffer_size', 100) # Max number of trajectories
        
        # 1. Trainable Reward Function R_omega(s, a)
        self.reward_net = RewardNetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            hidden_dim=256, 
            encode_dim=64
        ).to(device)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.reward_lr)
        
        # 2. Value Function V(s) - Predicts Ground Truth Return
        self.value_net = ValueFunction(input_dim=state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        
        # 3. Data Storage
        self.D_xi = deque(maxlen=self.reward_buffer_size) # Trajectory buffer
        self.current_episode_data = [] # Temp storage for current episode
        
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
        Meta-Optimization Step .
        
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
        
        # Process step-by-step as per official code (or batched if possible)
        accumulator_1 = [] # (R_hat - E[R])
        accumulator_2 = [] # pi(a|s) * (V_bar - V(s))
        
        states_list = []
        overline_V_list = []
        
        for step in all_steps:
            # Unpack
            s_np, a_np, r_bar, log_prob_np, mu, overline_V = step
            
            # Convert to Tensor
            s = torch.tensor(s_np, device=self.device).float()
            
            prob_a = np.exp(log_prob_np) # pi(a|s) scalar
            # NOTE: Official code uses prob_a directly. 
            # If log_prob_np is log probability density, exp(log_prob) could be > 1. 
            # Code assumes this ratio/weight acts as part of the gradient scale.
            
            # 1. Estimate V(s) using our ValueFunction
            V_s = self.value_net(s.unsqueeze(0)).item() # Scalar
            
            # 2. Advantage (V_bar - V(s))
            advantage = overline_V - V_s
            
            # Accumulator 2: weighting term
            accumulator_2.append(prob_a * advantage)
            
            # 3. Reward Baseline Computation
            # Re-compute R_omega(s, a)
            a_tensor = torch.tensor(a_np, device=self.device).float().unsqueeze(0)
            r_omega_cur = self.reward_net(s.unsqueeze(0), a_tensor).squeeze(0) # [1]
            
            # Sample N actions from pi(.|s) using mu
            action_samples, log_prob_samples = agent_policy_func(mu, self.n_samples)
            
            # Expand state
            s_expanded = s.unsqueeze(0).repeat(self.n_samples, 1) # [N, state_dim]
            
            # Compute R_omega(s, a') for all samples
            r_omega_samples = self.reward_net(s_expanded, action_samples) # [N, 1]
            
            # Expectation: mean(R)
            reward_baseline = torch.mean(r_omega_samples)
            
            # Accumulator 1: (R(s,a) - Baseline)
            accumulator_1.append(r_omega_cur - reward_baseline)
            
            # Collect data for Value update
            states_list.append(s_np)
            overline_V_list.append(overline_V)

        # ---- Stack and Compute Loss ----
        # Loss = Mean(Acc2) * Mean(Acc1)
        loss_val_term = torch.stack(accumulator_2).mean() 
        loss_reward_term = torch.stack(accumulator_1).mean() 
        
        loss = loss_val_term * loss_reward_term
        
        # Optimize R_omega
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()
        
        # Optimize V(s)
        # Regress V(s) -> overline_V
        states_tensor = torch.tensor(np.array(states_list), device=self.device).float()
        targets_tensor = torch.tensor(np.array(overline_V_list), device=self.device).float().unsqueeze(1)
        
        preds = self.value_net(states_tensor)
        v_loss = nn.functional.smooth_l1_loss(preds, targets_tensor)
        
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()
        
        return {
            "meta_loss": loss.item(),
            "value_loss": v_loss.item(),
            "mean_R_omega": loss_reward_term.item(),
            "mean_Advantage": loss_val_term.item()
        }