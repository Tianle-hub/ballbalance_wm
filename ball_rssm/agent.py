"""Closed-loop Dreamer policy agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from ball_rssm.models import Actor, ActorConfig, Normalizer, WorldModel, WorldModelConfig
from ball_rssm.models.rssm import RSSMState
from ball_rssm.utils.checkpoint import load_checkpoint


class DreamerAgent:
    """Stateful actor that updates RSSM belief from real observations."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        action_space: spaces.Box,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
    ) -> None:
        if not isinstance(action_space, spaces.Box):
            raise TypeError("action_space must be gymnasium.spaces.Box")
        self.device = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, self.device)

        world_config = checkpoint.get("world_model_config", checkpoint.get("config"))
        if world_config is None:
            raise KeyError("checkpoint does not contain a world model config")
        actor_config = checkpoint.get("actor_config")
        if actor_config is None:
            raise KeyError("checkpoint does not contain an actor config")

        self.world_model = WorldModel(WorldModelConfig.from_dict(world_config)).to(self.device)
        self.world_model.load_state_dict(checkpoint.get("world_model_state_dict", checkpoint["model_state_dict"]))
        self.world_model.eval()
        self.world_model.requires_grad_(False)

        self.actor = Actor(ActorConfig.from_dict(actor_config)).to(self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor.eval()
        self.actor.requires_grad_(False)

        self.normalizer = Normalizer.load_state_dict(checkpoint["normalizer"]).to(self.device)
        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.deterministic = deterministic
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.state: RSSMState | None = None
        self.needs_update = False
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self, initial_obs: np.ndarray | None = None) -> None:
        self.state = self.world_model.initial_state(1, self.device)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.needs_update = False
        self.last_diagnostics = {}
        if initial_obs is not None:
            self.state = self._posterior_update(initial_obs, self.prev_action)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.reset(initial_obs=obs)
        elif self.needs_update:
            self.state = self._posterior_update(obs, self.prev_action)

        assert self.state is not None
        with torch.no_grad():
            # Runtime Dreamer control is just belief update plus actor inference.
            # PlaNet would run CEM here to search over future action sequences.
            features = self.world_model.features_from_state(self.state)
            action_norm = self.actor.sample(features, deterministic=self.deterministic)
            action_real = self.normalizer.denormalize_action(action_norm).reshape(-1)
        action = action_real.detach().cpu().numpy().astype(np.float32)
        action = np.clip(action, self.action_space.low.reshape(-1), self.action_space.high.reshape(-1))

        self.prev_action = action.astype(np.float32)
        self.needs_update = True
        self.last_diagnostics = {
            "action_norm": action_norm.detach().cpu().numpy().reshape(-1),
            "action": action.copy(),
        }
        return action.reshape(self.action_space.shape)

    def _posterior_update(self, obs: np.ndarray, prev_action: np.ndarray) -> RSSMState:
        assert self.state is not None
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device).reshape(1, -1)
        obs_norm = self.normalizer.normalize_obs(obs_t)
        action_norm = self.normalizer.normalize_action(action_t)
        return self.world_model.posterior_update(self.state, action_norm, obs_norm)
