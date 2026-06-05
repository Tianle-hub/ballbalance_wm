"""Observation/action normalization for RSSM training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Normalizer:
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    reward_mean: torch.Tensor
    reward_std: torch.Tensor
    eps: float = 1e-6

    @classmethod
    def from_arrays(cls, obs: np.ndarray, action: np.ndarray, reward: np.ndarray | None = None, eps: float = 1e-6) -> "Normalizer":
        # Statistics are computed over the selected training split and saved with
        # the checkpoint so planning uses the same scale as training.
        obs_flat = obs.reshape(-1, obs.shape[-1]).astype(np.float32)
        action_flat = action.reshape(-1, action.shape[-1]).astype(np.float32)
        if reward is None:
            reward_mean = np.zeros(1, dtype=np.float32)
            reward_std = np.ones(1, dtype=np.float32)
        else:
            reward_flat = reward.reshape(-1, reward.shape[-1]).astype(np.float32)
            reward_mean = reward_flat.mean(axis=0)
            reward_std = reward_flat.std(axis=0) + eps
        return cls(
            obs_mean=torch.from_numpy(obs_flat.mean(axis=0)),
            obs_std=torch.from_numpy(obs_flat.std(axis=0) + eps),
            action_mean=torch.from_numpy(action_flat.mean(axis=0)),
            action_std=torch.from_numpy(action_flat.std(axis=0) + eps),
            reward_mean=torch.from_numpy(reward_mean),
            reward_std=torch.from_numpy(reward_std),
            eps=eps,
        )

    def to(self, device: torch.device | str) -> "Normalizer":
        self.obs_mean = self.obs_mean.to(device)
        self.obs_std = self.obs_std.to(device)
        self.action_mean = self.action_mean.to(device)
        self.action_std = self.action_std.to(device)
        self.reward_mean = self.reward_mean.to(device)
        self.reward_std = self.reward_std.to(device)
        return self

    def normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.obs_mean) / self.obs_std

    def denormalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.obs_std + self.obs_mean

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean

    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        return (reward - self.reward_mean) / self.reward_std

    def denormalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        # MPC optimizes denormalized rewards so objective weights stay in
        # environment reward units.
        return reward * self.reward_std + self.reward_mean

    def state_dict(self) -> dict[str, object]:
        return {
            "obs_mean": self.obs_mean.detach().cpu(),
            "obs_std": self.obs_std.detach().cpu(),
            "action_mean": self.action_mean.detach().cpu(),
            "action_std": self.action_std.detach().cpu(),
            "reward_mean": self.reward_mean.detach().cpu(),
            "reward_std": self.reward_std.detach().cpu(),
            "eps": self.eps,
        }

    @classmethod
    def load_state_dict(cls, state: dict[str, object]) -> "Normalizer":
        return cls(
            obs_mean=torch.as_tensor(state["obs_mean"], dtype=torch.float32),
            obs_std=torch.as_tensor(state["obs_std"], dtype=torch.float32),
            action_mean=torch.as_tensor(state["action_mean"], dtype=torch.float32),
            action_std=torch.as_tensor(state["action_std"], dtype=torch.float32),
            reward_mean=torch.as_tensor(state["reward_mean"], dtype=torch.float32),
            reward_std=torch.as_tensor(state["reward_std"], dtype=torch.float32),
            eps=float(state.get("eps", 1e-6)),
        )
