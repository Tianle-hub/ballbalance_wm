"""Fixed-length sequence windows for ball balance NPZ datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSplit:
    train_episodes: np.ndarray
    val_episodes: np.ndarray


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Sample windows where obs_seq[t] + action_seq[t] -> obs_seq[t + 1]."""

    def __init__(
        self,
        source: str | Path | Any,
        seq_len: int,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 0,
        episode_indices: np.ndarray | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")

        arrays = load_sequence_arrays(source)
        self.source = source
        self.seq_len = seq_len
        self.obs = arrays["obs"]
        self.action = arrays["action"]
        self.reward = arrays.get("reward")
        self.done = arrays.get("done")
        self.terminated = arrays.get("terminated")
        self.truncated = arrays.get("truncated")

        validate_sequence_arrays(self.obs, self.action, self.reward, self.done, self.terminated, self.truncated)
        num_episodes, obs_steps, _ = self.obs.shape
        action_steps = self.action.shape[1]
        if obs_steps != action_steps + 1:
            raise ValueError("Expected obs.shape[1] == action.shape[1] + 1")
        if seq_len > action_steps:
            raise ValueError(f"seq_len={seq_len} exceeds dataset action length={action_steps}")

        if episode_indices is None:
            split_indices = split_episodes(num_episodes, val_fraction=val_fraction, seed=seed)
            if split == "train":
                episode_indices = split_indices.train_episodes
            elif split == "val":
                episode_indices = split_indices.val_episodes
            else:
                episode_indices = np.arange(num_episodes)

        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        if self.episode_indices.size == 0:
            raise ValueError(f"No episodes selected for split={split!r}")

        self.windows: list[tuple[int, int]] = []
        for episode in self.episode_indices:
            for start in range(action_steps - seq_len + 1):
                self.windows.append((int(episode), int(start)))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, start = self.windows[index]
        end = start + self.seq_len

        sample = {
            "obs": torch.as_tensor(self.obs[episode, start : end + 1], dtype=torch.float32),
            "action": torch.as_tensor(self.action[episode, start:end], dtype=torch.float32),
        }
        if self.reward is not None:
            sample["reward"] = torch.as_tensor(self.reward[episode, start:end], dtype=torch.float32)
        if self.done is not None:
            sample["done"] = torch.as_tensor(self.done[episode, start:end], dtype=torch.float32)
        if self.terminated is not None:
            sample["terminated"] = torch.as_tensor(self.terminated[episode, start:end], dtype=torch.float32)
        if self.truncated is not None:
            sample["truncated"] = torch.as_tensor(self.truncated[episode, start:end], dtype=torch.float32)
        return sample

    def selected_obs_actions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return arrays from selected episodes for normalizer statistics."""

        return self.obs[self.episode_indices], self.action[self.episode_indices]

    def selected_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return selected observation, action, and reward arrays for normalization."""

        reward = None if self.reward is None else self.reward[self.episode_indices]
        return self.obs[self.episode_indices], self.action[self.episode_indices], reward


def load_sequence_arrays(source: str | Path | Any) -> dict[str, np.ndarray]:
    if isinstance(source, (str, Path)):
        return load_npz_arrays(source)
    if isinstance(source, dict):
        return normalize_sequence_arrays(source)
    if hasattr(source, "to_dataset"):
        return normalize_sequence_arrays(source.to_dataset())
    raise TypeError("source must be an NPZ path, a dataset dict, or a Buffer-like object")


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return normalize_sequence_arrays(loaded)


def normalize_sequence_arrays(loaded: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "obs": loaded["obs"].astype(np.float32),
        "action": loaded["action"].astype(np.float32),
    }
    for key in ("reward", "done", "terminated", "truncated"):
        if key in loaded:
            arrays[key] = loaded[key].astype(np.float32)
    return arrays


def validate_sequence_arrays(
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray | None = None,
    done: np.ndarray | None = None,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> None:
    if obs.ndim != 3:
        raise ValueError("obs must have shape [N, T + 1, obs_dim]")
    if action.ndim != 3:
        raise ValueError("action must have shape [N, T, action_dim]")
    if obs.shape[0] != action.shape[0]:
        raise ValueError("obs and action must have the same episode count")
    if obs.shape[1] != action.shape[1] + 1:
        raise ValueError("obs time dimension must be action time dimension + 1")
    if reward is not None and reward.shape[:2] != action.shape[:2]:
        raise ValueError("reward must share [N, T] with action")
    if done is not None and done.shape[:2] != action.shape[:2]:
        raise ValueError("done must share [N, T] with action")
    if terminated is not None and terminated.shape[:2] != action.shape[:2]:
        raise ValueError("terminated must share [N, T] with action")
    if truncated is not None and truncated.shape[:2] != action.shape[:2]:
        raise ValueError("truncated must share [N, T] with action")
    arrays = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
    }
    for name, arr in arrays.items():
        if arr is not None and not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf")


def split_episodes(num_episodes: int, val_fraction: float = 0.1, seed: int = 0) -> DatasetSplit:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_episodes)
    rng.shuffle(indices)
    if num_episodes == 1 or val_fraction == 0.0:
        return DatasetSplit(train_episodes=indices, val_episodes=indices)

    num_val = max(1, int(round(num_episodes * val_fraction)))
    num_val = min(num_val, num_episodes - 1)
    return DatasetSplit(train_episodes=indices[num_val:], val_episodes=indices[:num_val])


def batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
