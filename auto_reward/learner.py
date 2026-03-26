import random
from collections import deque, namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from auto_reward.buffers import EpisodeDataset, EpisodeRecord, EpisodeTransition
from auto_reward.meta_ops import (
    functional_module_call,
    gradient_step,
    l2_from_params,
    named_parameter_dict,
    optimize_actions_with_reward,
)
from auto_reward.networks import GTQNetwork, RewardNetwork, RewardNetworkV2, ValueFunction


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


class AutoRewardLearnerV2:
    """
    Reward learner used by SAC_AUTO_V2.

    The learner keeps a frozen-feature episode dataset, trains GT twin critics on
    environment reward, and updates the reward network with a bilevel-style outer
    objective based on action improvement under the learned reward.
    """

    def __init__(self, feature_dim, action_dim, device, config):
        self.device = device
        self.config = config

        reward_cfg = config.get("reward_model", {})
        meta_cfg = config.get("meta_optimizer", {})
        schedule_cfg = config.get("reward_schedule", {})

        self.gamma = float(config.get("gamma", 0.98))
        self.tau = float(reward_cfg.get("tau", 0.01))
        self.reward_lr = float(reward_cfg.get("reward_lr", 1e-4))
        self.gt_q_lr = float(reward_cfg.get("gt_q_lr", 3e-4))
        hidden_dim = int(reward_cfg.get("hidden_dim", 256))

        self.inner_steps = int(meta_cfg.get("inner_steps", 2))
        self.meta_batch_size = int(meta_cfg.get("meta_batch_size", 16))
        self.update_freq = int(meta_cfg.get("update_freq", 1024))
        self.meta_start_steps = int(meta_cfg.get("start_steps", schedule_cfg.get("blend_start_steps", 20000)))
        self.rank_margin = float(meta_cfg.get("rank_margin", 0.5))
        self.terminal_margin = float(meta_cfg.get("terminal_margin", 0.5))
        self.rank_coef = float(meta_cfg.get("rank_coef", 0.5))
        self.terminal_coef = float(meta_cfg.get("terminal_coef", 0.1))
        self.reward_l2_coef = float(meta_cfg.get("reward_l2_coef", 1e-4))
        self.inner_reward_coef = float(meta_cfg.get("inner_reward_coef", 1.0))
        self.inner_rank_coef = float(meta_cfg.get("inner_rank_coef", 0.25))
        self.inner_terminal_coef = float(meta_cfg.get("inner_terminal_coef", 0.1))
        self.action_step_size = float(meta_cfg.get("action_step_size", 0.1))
        self.reward_corr_momentum = float(meta_cfg.get("reward_corr_momentum", 0.9))
        self.reward_ready_corr_threshold = float(schedule_cfg.get("reward_ready_corr_threshold", 0.2))
        self.required_success_episodes = int(meta_cfg.get("min_success_episodes", 8))
        self.required_failure_episodes = int(meta_cfg.get("min_failure_episodes", 8))

        self.reward_net = RewardNetworkV2(feature_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.reward_lr)

        self.gt_q1 = GTQNetwork(feature_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.gt_q2 = GTQNetwork(feature_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.gt_q1_target = deepcopy(self.gt_q1).to(device)
        self.gt_q2_target = deepcopy(self.gt_q2).to(device)
        self.gt_q_optimizer = optim.Adam(
            list(self.gt_q1.parameters()) + list(self.gt_q2.parameters()),
            lr=self.gt_q_lr,
        )

        self.reward_train_episodes = EpisodeDataset(capacity=int(meta_cfg.get("reward_train_capacity", 200)))
        self.meta_eval_episodes = EpisodeDataset(capacity=int(meta_cfg.get("meta_eval_capacity", 64)))

        self.current_episode: list[EpisodeTransition] = []
        self._episode_counter = 0
        self.total_success_trajectories_seen = 0
        self.total_failure_trajectories_seen = 0
        self.reward_corr_ema = 0.0
        self.reward_ready = False
        self.last_gt_q_loss = 0.0

    @property
    def success_traj_count(self):
        return self.reward_train_episodes.success_count() + self.meta_eval_episodes.success_count()

    @property
    def failure_traj_count(self):
        return self.reward_train_episodes.failure_count() + self.meta_eval_episodes.failure_count()

    @property
    def trajectory_buffer_size(self):
        return len(self.reward_train_episodes) + len(self.meta_eval_episodes)

    def get_reward(self, features, actions):
        return self.reward_net(features, actions)

    def store_transition(self, feature, next_feature, action, env_reward, done):
        self.current_episode.append(
            EpisodeTransition(
                feature=np.array(feature, copy=True, dtype=np.float32),
                next_feature=np.array(next_feature, copy=True, dtype=np.float32),
                action=np.array(action, copy=True, dtype=np.float32),
                env_reward=float(env_reward),
                done=bool(done),
            )
        )

    def on_episode_end(self, success=False, terminal_reason="Running..."):
        if not self.current_episode:
            return
        episode = EpisodeRecord(
            transitions=list(self.current_episode),
            success=bool(success),
            terminal_reason=str(terminal_reason),
        )
        if self._episode_counter % 4 == 3:
            self.meta_eval_episodes.add_episode(episode)
        else:
            self.reward_train_episodes.add_episode(episode)

        if success:
            self.total_success_trajectories_seen += 1
        else:
            self.total_failure_trajectories_seen += 1
        self._episode_counter += 1
        self.current_episode = []

    def update_gt_critics(self, features, actions, env_rewards, next_features, next_actions, dones):
        with torch.no_grad():
            next_q1 = self.gt_q1_target(next_features, next_actions)
            next_q2 = self.gt_q2_target(next_features, next_actions)
            next_q = torch.min(next_q1, next_q2)
            targets = env_rewards + (1.0 - dones) * self.gamma * next_q

        current_q1 = self.gt_q1(features, actions)
        current_q2 = self.gt_q2(features, actions)
        loss = F.mse_loss(current_q1, targets) + F.mse_loss(current_q2, targets)

        self.gt_q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.gt_q1.parameters()) + list(self.gt_q2.parameters()),
            max_norm=5.0,
        )
        self.gt_q_optimizer.step()

        self._soft_update(self.gt_q1_target, self.gt_q1)
        self._soft_update(self.gt_q2_target, self.gt_q2)
        self.last_gt_q_loss = float(loss.detach().item())
        return {"gt_q_loss": self.last_gt_q_loss}

    def optimize_reward(self):
        if len(self.reward_train_episodes) == 0 or len(self.meta_eval_episodes) == 0:
            return {}

        train_episodes = self.reward_train_episodes.sample_episodes(self.meta_batch_size)
        meta_episodes = self.meta_eval_episodes.sample_episodes(max(4, self.meta_batch_size // 2))
        if len(train_episodes) == 0 or len(meta_episodes) == 0:
            return {}

        train_features, train_next_features, train_actions, train_rewards, train_dones = self._episodes_to_tensors(train_episodes)
        meta_features, _, meta_actions, _, _ = self._episodes_to_tensors(meta_episodes)

        normalized_train_rewards = self._normalize_rewards(train_rewards)
        stateless_params = named_parameter_dict(self.reward_net)

        for _ in range(max(self.inner_steps, 1)):
            predicted_rewards = functional_module_call(self.reward_net, stateless_params, train_features, train_actions)
            inner_rank_loss, _, _, _ = self._rank_and_terminal_losses(train_episodes, stateless_params)
            inner_terminal_loss = self._terminal_margin_loss(train_episodes, stateless_params)
            inner_loss = (
                self.inner_reward_coef * F.smooth_l1_loss(predicted_rewards, normalized_train_rewards)
                + self.inner_rank_coef * inner_rank_loss
                + self.inner_terminal_coef * inner_terminal_loss
            )
            stateless_params = gradient_step(inner_loss, stateless_params, self.reward_lr, create_graph=True)

        improved_actions = optimize_actions_with_reward(
            self.reward_net,
            stateless_params,
            meta_features,
            meta_actions,
            steps=self.inner_steps,
            step_size=self.action_step_size,
        )
        gt_q = torch.min(
            self.gt_q1(meta_features, improved_actions),
            self.gt_q2(meta_features, improved_actions),
        )
        rank_loss, rank_gap, terminal_loss, terminal_gap = self._rank_and_terminal_losses(meta_episodes, stateless_params)
        reward_reg = l2_from_params(stateless_params)
        outer_loss = -gt_q.mean() + self.rank_coef * rank_loss + self.terminal_coef * terminal_loss + self.reward_l2_coef * reward_reg

        self.reward_optimizer.zero_grad()
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), max_norm=5.0)
        self.reward_optimizer.step()

        corr, mean_success_return, mean_failure_return = self._return_correlation(meta_episodes)
        self.reward_corr_ema = (
            self.reward_corr_momentum * self.reward_corr_ema
            + (1.0 - self.reward_corr_momentum) * corr
        )
        self.reward_ready = (
            self.meta_eval_episodes.success_count() >= self.required_success_episodes
            and self.meta_eval_episodes.failure_count() >= self.required_failure_episodes
            and self.reward_corr_ema >= self.reward_ready_corr_threshold
        )

        mean_reward = float(self.reward_net(meta_features, meta_actions).detach().mean().item())
        return {
            "meta_outer_loss": float(outer_loss.detach().item()),
            "gt_q_loss": self.last_gt_q_loss,
            "mean_R_omega": mean_reward,
            "gt_vs_learned_return_corr": corr,
            "reward_corr_ema": float(self.reward_corr_ema),
            "reward_ready": float(self.reward_ready),
            "meta_rank_gap": float(rank_gap),
            "meta_terminal_gap": float(terminal_gap),
            "rank_loss": float(rank_loss.detach().item()),
            "terminal_loss": float(terminal_loss.detach().item()),
            "mean_success_return": mean_success_return,
            "mean_failure_return": mean_failure_return,
            "success_traj_count": self.success_traj_count,
            "failure_traj_count": self.failure_traj_count,
            "trajectory_buffer_size": self.trajectory_buffer_size,
            "total_success_trajectories_seen": self.total_success_trajectories_seen,
            "total_failure_trajectories_seen": self.total_failure_trajectories_seen,
        }

    def _episodes_to_tensors(self, episodes):
        features = []
        next_features = []
        actions = []
        rewards = []
        dones = []
        for episode in episodes:
            for transition in episode.transitions:
                features.append(transition.feature)
                next_features.append(transition.next_feature)
                actions.append(transition.action)
                rewards.append([transition.env_reward])
                dones.append([float(transition.done)])
        return (
            torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device),
            torch.as_tensor(np.asarray(next_features, dtype=np.float32), device=self.device),
            torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self.device),
            torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.device),
            torch.as_tensor(np.asarray(dones, dtype=np.float32), device=self.device),
        )

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std()
        return torch.clamp((rewards - mean) / (std + 1e-6), -5.0, 5.0)

    def _rank_and_terminal_losses(self, episodes, params):
        success_returns = []
        failure_returns = []
        success_terminal = []
        failure_terminal = []

        for episode in episodes:
            rewards = []
            for transition in episode.transitions:
                feature = torch.as_tensor(transition.feature, device=self.device, dtype=torch.float32).unsqueeze(0)
                action = torch.as_tensor(transition.action, device=self.device, dtype=torch.float32).unsqueeze(0)
                reward = functional_module_call(self.reward_net, params, feature, action).squeeze()
                rewards.append(reward)
            reward_tensor = torch.stack(rewards)
            returns = self._discounted_cumsum(reward_tensor)
            if episode.success:
                success_returns.append(returns[0])
                success_terminal.append(reward_tensor[-1])
            else:
                failure_returns.append(returns[0])
                failure_terminal.append(reward_tensor[-1])

        zero = torch.zeros((), device=self.device)
        rank_gap = zero
        terminal_gap = zero
        rank_loss = zero
        terminal_loss = zero
        if success_returns and failure_returns:
            rank_gap = torch.stack(success_returns).mean() - torch.stack(failure_returns).mean()
            rank_loss = F.relu(self.rank_margin - rank_gap)
        if success_terminal and failure_terminal:
            terminal_gap = torch.stack(success_terminal).mean() - torch.stack(failure_terminal).mean()
            terminal_loss = F.relu(self.terminal_margin - terminal_gap)
        return rank_loss, rank_gap, terminal_loss, terminal_gap

    def _terminal_margin_loss(self, episodes, params):
        _, _, terminal_loss, _ = self._rank_and_terminal_losses(episodes, params)
        return terminal_loss

    def _return_correlation(self, episodes):
        gt_returns = []
        learned_returns = []
        success_returns = []
        failure_returns = []
        for episode in episodes:
            gt_reward_list = []
            learned_reward_list = []
            for transition in episode.transitions:
                gt_reward_list.append(float(transition.env_reward))
                feature = torch.as_tensor(transition.feature, device=self.device, dtype=torch.float32).unsqueeze(0)
                action = torch.as_tensor(transition.action, device=self.device, dtype=torch.float32).unsqueeze(0)
                learned_reward = self.reward_net(feature, action).detach().cpu().item()
                learned_reward_list.append(float(learned_reward))

            gt_returns.append(self._discounted_cumsum_np(gt_reward_list)[0])
            learned_episode_return = self._discounted_cumsum_np(learned_reward_list)[0]
            learned_returns.append(learned_episode_return)
            if episode.success:
                success_returns.append(learned_episode_return)
            else:
                failure_returns.append(learned_episode_return)

        corr = 0.0
        if len(gt_returns) > 1 and np.std(gt_returns) > 1e-8 and np.std(learned_returns) > 1e-8:
            corr = float(np.corrcoef(gt_returns, learned_returns)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        mean_success_return = float(np.mean(success_returns)) if success_returns else 0.0
        mean_failure_return = float(np.mean(failure_returns)) if failure_returns else 0.0
        return corr, mean_success_return, mean_failure_return

    def _discounted_cumsum(self, rewards):
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        for idx in range(rewards.shape[0] - 1, -1, -1):
            running_return = rewards[idx] + self.gamma * running_return
            returns[idx] = running_return
        return returns

    def _discounted_cumsum_np(self, rewards):
        returns = np.zeros(len(rewards), dtype=np.float32)
        running_return = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            running_return = float(rewards[idx]) + self.gamma * running_return
            returns[idx] = running_return
        return returns

    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * source_param.data)
