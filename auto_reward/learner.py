import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from auto_reward.networks import RewardNetwork, ValueFunction


Transition = namedtuple("Transition", ["state", "action", "gt_reward", "gt_return"])


class AutoRewardLearner:
    """
    Trajectory-level approximation of the upper-level reward optimization.

    The learner stores trajectories using the ground-truth reward, then
    recomputes learned trajectory returns under the current reward network.
    Reward updates maximize the alignment between GT advantage and learned
    advantage, following the stationary simplification of equation (21).
    """

    def __init__(self, state_dim, action_dim, device, config, eval_reward_params=None):
        self.device = device
        self.config = config

        self.gamma = config.gamma
        self.reward_lr = 1e-4
        self.value_lr = 3e-4
        self.reward_buffer_size = config.get("reward_buffer_size", 100)
        self.trajectory_batch_size = min(16, self.reward_buffer_size)
        objective_cfg = config.get("reward_objective", {})
        self.align_coef = float(objective_cfg.get("align_coef", 1.0))
        self.rank_coef = float(objective_cfg.get("rank_coef", 1.0))
        self.terminal_coef = float(objective_cfg.get("terminal_coef", 0.5))
        self.reg_coef = float(objective_cfg.get("reg_coef", 1e-4))
        self.rank_margin = float(objective_cfg.get("rank_margin", 1.0))
        self.terminal_pos_margin = float(objective_cfg.get("terminal_pos_margin", 0.5))
        self.terminal_neg_margin = float(objective_cfg.get("terminal_neg_margin", 0.5))

        self.reward_net = RewardNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            encode_dim=64,
        ).to(device)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.reward_lr)

        self.gt_value_net = ValueFunction(input_dim=state_dim).to(device)
        self.gt_value_optimizer = optim.Adam(self.gt_value_net.parameters(), lr=self.value_lr)

        self.learned_value_net = ValueFunction(input_dim=state_dim).to(device)
        self.learned_value_optimizer = optim.Adam(self.learned_value_net.parameters(), lr=self.value_lr)

        self.D_xi = deque(maxlen=self.reward_buffer_size)
        self.trajectory_outcomes = deque(maxlen=self.reward_buffer_size)
        self.total_success_trajectories_seen = 0
        self.total_failure_trajectories_seen = 0
        self.current_episode_data = []

    @property
    def success_traj_count(self):
        return int(sum(1 for outcome in self.trajectory_outcomes if outcome))

    @property
    def failure_traj_count(self):
        return int(sum(1 for outcome in self.trajectory_outcomes if not outcome))

    def get_reward(self, state, action):
        with torch.no_grad():
            return self.reward_net(state, action)

    def store_transition(self, state, action, reward, log_prob=None, mu=None):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        transition = Transition(
            state=np.array(state, copy=True),
            action=np.array(action, copy=True),
            gt_reward=float(reward),
            gt_return=0.0,
        )
        self.current_episode_data.append(transition)

    def on_episode_end(self, success=False):
        if not self.current_episode_data:
            return

        gt_return_sum = 0.0
        trajectory = []
        for transition in reversed(self.current_episode_data):
            gt_return_sum = transition.gt_reward + self.gamma * gt_return_sum
            trajectory.insert(0, transition._replace(gt_return=gt_return_sum))

        self.D_xi.append(trajectory)
        self.trajectory_outcomes.append(bool(success))
        if success:
            self.total_success_trajectories_seen += 1
        else:
            self.total_failure_trajectories_seen += 1
        self.current_episode_data = []

    def _discounted_cumsum(self, rewards):
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        for idx in range(rewards.shape[0] - 1, -1, -1):
            running_return = rewards[idx] + self.gamma * running_return
            returns[idx] = running_return
        return returns

    def _sample_trajectories(self):
        if len(self.D_xi) <= self.trajectory_batch_size:
            return list(zip(self.D_xi, self.trajectory_outcomes))
        indices = random.sample(range(len(self.D_xi)), self.trajectory_batch_size)
        return [(self.D_xi[idx], self.trajectory_outcomes[idx]) for idx in indices]

    def optimize_reward(self, agent_policy_func=None):
        if len(self.D_xi) == 0:
            return {}

        sampled_trajectories = self._sample_trajectories()
        if len(sampled_trajectories) == 0:
            return {}

        flat_states = []
        flat_actions = []
        flat_gt_returns = []
        flat_learned_returns = []
        flat_learned_step_rewards = []
        trajectory_outcomes = []
        trajectory_learned_returns = []
        trajectory_terminal_rewards = []

        for trajectory, success in sampled_trajectories:
            states_np = np.array([step.state for step in trajectory], dtype=np.float32)
            actions_np = np.array([step.action for step in trajectory], dtype=np.float32)
            gt_returns_np = np.array([step.gt_return for step in trajectory], dtype=np.float32)

            states = torch.tensor(states_np, device=self.device)
            actions = torch.tensor(actions_np, device=self.device)
            gt_returns = torch.tensor(gt_returns_np, device=self.device)

            learned_step_rewards = self.reward_net(states, actions).squeeze(-1)
            learned_returns = self._discounted_cumsum(learned_step_rewards)

            flat_states.append(states)
            flat_actions.append(actions)
            flat_gt_returns.append(gt_returns)
            flat_learned_returns.append(learned_returns)
            flat_learned_step_rewards.append(learned_step_rewards)
            trajectory_outcomes.append(bool(success))
            trajectory_learned_returns.append(learned_returns[0])
            trajectory_terminal_rewards.append(learned_step_rewards[-1])

        states_all = torch.cat(flat_states, dim=0)
        actions_all = torch.cat(flat_actions, dim=0)
        gt_returns_all = torch.cat(flat_gt_returns, dim=0)
        learned_returns_all = torch.cat(flat_learned_returns, dim=0)
        learned_step_rewards_all = torch.cat(flat_learned_step_rewards, dim=0)
        trajectory_learned_returns = torch.stack(trajectory_learned_returns)
        trajectory_terminal_rewards = torch.stack(trajectory_terminal_rewards)
        success_mask = torch.tensor(trajectory_outcomes, device=self.device, dtype=torch.bool)
        failure_mask = ~success_mask

        self.gt_value_optimizer.zero_grad()
        gt_value_preds = self.gt_value_net(states_all).squeeze(-1)
        gt_value_loss = nn.functional.smooth_l1_loss(gt_value_preds, gt_returns_all)
        gt_value_loss.backward()
        self.gt_value_optimizer.step()

        self.learned_value_optimizer.zero_grad()
        learned_value_preds = self.learned_value_net(states_all).squeeze(-1)
        learned_value_loss = nn.functional.smooth_l1_loss(learned_value_preds, learned_returns_all.detach())
        learned_value_loss.backward()
        self.learned_value_optimizer.step()

        gt_value_baseline = self.gt_value_net(states_all).squeeze(-1).detach()
        learned_value_baseline = self.learned_value_net(states_all).squeeze(-1).detach()

        gt_advantage = (gt_returns_all - gt_value_baseline).detach()
        learned_advantage = learned_returns_all - learned_value_baseline
        gt_advantage_norm = (gt_advantage - gt_advantage.mean()) / (gt_advantage.std() + 1e-6)
        learned_advantage_norm = (learned_advantage - learned_advantage.mean()) / (learned_advantage.std() + 1e-6)
        loss_align = -(gt_advantage_norm * learned_advantage_norm).mean()

        zero = torch.zeros((), device=self.device)
        loss_rank = zero
        mean_success_return = None
        mean_failure_return = None
        if success_mask.any() and failure_mask.any():
            mean_success_return = trajectory_learned_returns[success_mask].mean()
            mean_failure_return = trajectory_learned_returns[failure_mask].mean()
            loss_rank = F.relu(self.rank_margin - (mean_success_return - mean_failure_return))

        loss_terminal_success = zero
        loss_terminal_failure = zero
        mean_success_terminal = None
        mean_failure_terminal = None
        if success_mask.any():
            success_terminal = trajectory_terminal_rewards[success_mask]
            mean_success_terminal = success_terminal.mean()
            loss_terminal_success = F.relu(self.terminal_pos_margin - success_terminal).mean()
        if failure_mask.any():
            failure_terminal = trajectory_terminal_rewards[failure_mask]
            mean_failure_terminal = failure_terminal.mean()
            loss_terminal_failure = F.relu(self.terminal_neg_margin + failure_terminal).mean()
        loss_terminal = loss_terminal_success + loss_terminal_failure

        loss_reg = (learned_step_rewards_all ** 2).mean()
        reward_loss = (
            self.align_coef * loss_align
            + self.rank_coef * loss_rank
            + self.terminal_coef * loss_terminal
            + self.reg_coef * loss_reg
        )

        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

        learned_returns_np = learned_returns_all.detach().cpu().numpy()
        gt_returns_np = gt_returns_all.detach().cpu().numpy()
        corr = 0.0
        if learned_returns_np.size > 1:
            if np.std(learned_returns_np) > 1e-8 and np.std(gt_returns_np) > 1e-8:
                corr = float(np.corrcoef(gt_returns_np, learned_returns_np)[0, 1])
                if np.isnan(corr):
                    corr = 0.0

        return {
            "meta_loss": float(reward_loss.detach().item()),
            "value_loss": float((gt_value_loss.detach().item() + learned_value_loss.detach().item()) / 2.0),
            "mean_R_omega": float(learned_step_rewards_all.detach().mean().item()),
            "mean_Advantage": float(gt_advantage.mean().item()),
            "gt_vs_learned_return_corr": corr,
            "align_loss": float(loss_align.detach().item()),
            "rank_loss": float(loss_rank.detach().item()),
            "terminal_loss": float(loss_terminal.detach().item()),
            "reward_reg_loss": float(loss_reg.detach().item()),
            "mean_success_return": float(mean_success_return.detach().item()) if mean_success_return is not None else 0.0,
            "mean_failure_return": float(mean_failure_return.detach().item()) if mean_failure_return is not None else 0.0,
            "mean_success_terminal_reward": float(mean_success_terminal.detach().item()) if mean_success_terminal is not None else 0.0,
            "mean_failure_terminal_reward": float(mean_failure_terminal.detach().item()) if mean_failure_terminal is not None else 0.0,
            "success_traj_count": self.success_traj_count,
            "failure_traj_count": self.failure_traj_count,
            "total_success_trajectories_seen": self.total_success_trajectories_seen,
            "total_failure_trajectories_seen": self.total_failure_trajectories_seen,
        }
