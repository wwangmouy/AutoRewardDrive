from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def clone_observation(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {key: np.array(value, copy=True) for key, value in obs.items()}
    return np.array(obs, copy=True)


def stack_observations(observations: Sequence[Any]) -> Any:
    if len(observations) == 0:
        raise ValueError("stack_observations() requires at least one observation")
    first_obs = observations[0]
    if isinstance(first_obs, dict):
        return {
            key: np.concatenate([np.array(obs[key], copy=True) for obs in observations], axis=0)
            for key in first_obs.keys()
        }
    return np.concatenate([np.array(obs, copy=True) for obs in observations], axis=0)


@dataclass
class ExpertTransition:
    obs: Any
    expert_action: np.ndarray


class ExpertBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.storage: deque[ExpertTransition] = deque(maxlen=self.capacity)

    def add(self, obs: Any, expert_action: np.ndarray) -> None:
        self.storage.append(
            ExpertTransition(
                obs=clone_observation(obs),
                expert_action=np.array(expert_action, copy=True, dtype=np.float32),
            )
        )

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        if len(self.storage) == 0:
            raise ValueError("Cannot sample from an empty ExpertBuffer")
        batch_size = min(int(batch_size), len(self.storage))
        batch = random.sample(list(self.storage), batch_size)
        obs_batch = stack_observations([item.obs for item in batch])
        action_batch = np.concatenate([item.expert_action for item in batch], axis=0)
        return {"observations": obs_batch, "expert_actions": action_batch}


@dataclass
class ShieldTransition:
    obs: Any
    raw_action: np.ndarray
    safe_action: np.ndarray
    shield_active: float
    shield_front_brake: float
    shield_steer_clamp: float


class ShieldBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.storage: deque[ShieldTransition] = deque(maxlen=self.capacity)

    def add(
        self,
        obs: Any,
        raw_action: np.ndarray,
        safe_action: np.ndarray,
        shield_active: float,
        shield_front_brake: float,
        shield_steer_clamp: float,
    ) -> None:
        self.storage.append(
            ShieldTransition(
                obs=clone_observation(obs),
                raw_action=np.array(raw_action, copy=True, dtype=np.float32),
                safe_action=np.array(safe_action, copy=True, dtype=np.float32),
                shield_active=float(shield_active),
                shield_front_brake=float(shield_front_brake),
                shield_steer_clamp=float(shield_steer_clamp),
            )
        )

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        if len(self.storage) == 0:
            raise ValueError("Cannot sample from an empty ShieldBuffer")
        batch_size = min(int(batch_size), len(self.storage))
        batch = random.sample(list(self.storage), batch_size)
        obs_batch = stack_observations([item.obs for item in batch])
        raw_action_batch = np.stack([item.raw_action.reshape(-1) for item in batch], axis=0)
        safe_action_batch = np.stack([item.safe_action.reshape(-1) for item in batch], axis=0)
        shield_active_batch = np.asarray([[item.shield_active] for item in batch], dtype=np.float32)
        shield_front_brake_batch = np.asarray([[item.shield_front_brake] for item in batch], dtype=np.float32)
        shield_steer_clamp_batch = np.asarray([[item.shield_steer_clamp] for item in batch], dtype=np.float32)
        return {
            "observations": obs_batch,
            "raw_actions": raw_action_batch,
            "safe_actions": safe_action_batch,
            "shield_active": shield_active_batch,
            "shield_front_brake": shield_front_brake_batch,
            "shield_steer_clamp": shield_steer_clamp_batch,
        }


@dataclass
class EpisodeTransition:
    feature: np.ndarray
    next_feature: np.ndarray
    action: np.ndarray
    env_reward: float
    done: bool


@dataclass
class EpisodeRecord:
    transitions: List[EpisodeTransition]
    success: bool
    terminal_reason: str

    @property
    def length(self) -> int:
        return len(self.transitions)


class EpisodeDataset:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.storage: deque[EpisodeRecord] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add_episode(self, episode: EpisodeRecord) -> None:
        self.storage.append(episode)

    def sample_episodes(self, batch_size: int) -> List[EpisodeRecord]:
        if len(self.storage) == 0:
            return []
        batch_size = min(int(batch_size), len(self.storage))
        return random.sample(list(self.storage), batch_size)

    def success_count(self) -> int:
        return int(sum(1 for episode in self.storage if episode.success))

    def failure_count(self) -> int:
        return int(sum(1 for episode in self.storage if not episode.success))

    def flatten(self, episodes: Iterable[EpisodeRecord] | None = None) -> List[EpisodeTransition]:
        if episodes is None:
            episodes = self.storage
        flat: List[EpisodeTransition] = []
        for episode in episodes:
            flat.extend(episode.transitions)
        return flat
