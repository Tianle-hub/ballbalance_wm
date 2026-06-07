"""Replay buffer and offline collection utilities for ball RSSM training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ball_rssm.envs import BallBalanceEnv

PolicyMode = Literal["random_smooth", "pd", "mixed", "coverage"]


@dataclass(frozen=True)
class InitialStateBounds:
    pos: float = 0.20
    vel: float = 0.05
    angle: float = 0.0

    def validate(self, env: BallBalanceEnv) -> None:
        if self.pos < 0.0 or self.vel < 0.0 or self.angle < 0.0:
            raise ValueError("initial-state bounds must be non-negative")
        if self.pos >= env.config.board_size / 2.0:
            raise ValueError("pos bound must be strictly inside the board")
        if self.angle > env.config.max_angle:
            raise ValueError("angle bound must be <= env.config.max_angle")


def pd_action(
    obs: np.ndarray,
    kp: float = 0.8,
    kd: float = 0.25,
    target: np.ndarray | None = None,
) -> np.ndarray:
    if target is None:
        target = np.zeros(2, dtype=np.float32)
    x, y, vx, vy, _, _ = obs
    err_x = x - target[0]
    err_y = y - target[1]
    theta_y_cmd = -kp * err_x - kd * vx
    theta_x_cmd = kp * err_y + kd * vy
    return np.array([theta_x_cmd, theta_y_cmd], dtype=np.float32)


class Buffer:
    """Episode-major replay buffer backed by the repository NPZ dataset format."""

    def __init__(
        self,
        num_episodes: int,
        max_episode_steps: int,
        obs_shape: tuple[int, ...] = (6,),
        action_shape: tuple[int, ...] = (2,),
    ) -> None:
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.obs_buffer = np.zeros((num_episodes, max_episode_steps + 1, *obs_shape), dtype=np.float32)
        self.action_buffer = np.zeros((num_episodes, max_episode_steps, *action_shape), dtype=np.float32)
        self.reward_buffer = np.zeros((num_episodes, max_episode_steps, 1), dtype=np.float32)
        self.terminated_buffer = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)
        self.truncated_buffer = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)
        self.done_buffer = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)
        self.initial_bounds = np.zeros(3, dtype=np.float32)

    @classmethod
    def collect_data(
        cls,
        num_episodes: int,
        max_episode_steps: int,
        seed: int,
        mode: PolicyMode,
        initial_bounds: InitialStateBounds = InitialStateBounds(),
        target_bound: float = 0.12,
        action_noise_std: float = 0.03,
    ) -> "Buffer":
        buffer = cls(num_episodes=num_episodes, max_episode_steps=max_episode_steps)
        return buffer.collect(
            seed=seed,
            mode=mode,
            initial_bounds=initial_bounds,
            target_bound=target_bound,
            action_noise_std=action_noise_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Buffer":
        with np.load(path) as arrays:
            obs = arrays["obs"].astype(np.float32)
            action = arrays["action"].astype(np.float32)
            reward = (
                arrays["reward"].astype(np.float32)
                if "reward" in arrays
                else np.zeros((*action.shape[:2], 1), dtype=np.float32)
            )
            done = arrays["done"].astype(bool) if "done" in arrays else np.zeros((*action.shape[:2], 1), dtype=bool)
            terminated = arrays["terminated"].astype(bool) if "terminated" in arrays else done.copy()
            truncated = arrays["truncated"].astype(bool) if "truncated" in arrays else np.zeros_like(done, dtype=bool)
            initial_bounds = arrays["initial_bounds"].astype(np.float32) if "initial_bounds" in arrays else None

        buffer = cls(
            num_episodes=obs.shape[0],
            max_episode_steps=action.shape[1],
            obs_shape=tuple(obs.shape[2:]),
            action_shape=tuple(action.shape[2:]),
        )
        buffer.obs_buffer = obs
        buffer.action_buffer = action
        buffer.reward_buffer = reward
        buffer.terminated_buffer = terminated
        buffer.truncated_buffer = truncated
        buffer.done_buffer = done
        if initial_bounds is not None:
            buffer.initial_bounds = initial_bounds
        return buffer

    def collect(
        self,
        seed: int,
        mode: PolicyMode,
        initial_bounds: InitialStateBounds = InitialStateBounds(),
        target_bound: float = 0.12,
        action_noise_std: float = 0.03,
    ) -> "Buffer":
        if mode not in ("random_smooth", "pd", "mixed", "coverage"):
            raise ValueError(f"Unsupported mode: {mode}")
        if target_bound < 0.0:
            raise ValueError("target_bound must be non-negative")
        if action_noise_std < 0.0:
            raise ValueError("action_noise_std must be non-negative")

        env = BallBalanceEnv(config={"max_episode_steps": self.max_episode_steps})
        initial_bounds.validate(env)
        self.initial_bounds = np.array([initial_bounds.pos, initial_bounds.vel, initial_bounds.angle], dtype=np.float32)
        rng = np.random.default_rng(seed)

        try:
            for episode in range(self.num_episodes):
                self._collect_episode(env, episode, seed, rng, mode, initial_bounds, target_bound, action_noise_std)
        finally:
            env.close()
        return self

    def save(self, path: str | Path, **metadata: object) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **self.to_dataset(), **metadata)

    def to_dataset(self) -> dict[str, np.ndarray]:
        return {
            "obs": self.obs_buffer,
            "action": self.action_buffer,
            "reward": self.reward_buffer,
            "terminated": self.terminated_buffer,
            "truncated": self.truncated_buffer,
            "done": self.done_buffer,
            "initial_bounds": self.initial_bounds,
        }

    def sequence_dataset(
        self,
        seq_len: int,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 0,
        episode_indices: np.ndarray | None = None,
    ):
        from ball_rssm.data.sequence_dataset import SequenceDataset

        return SequenceDataset(
            self,
            seq_len=seq_len,
            split=split,
            val_fraction=val_fraction,
            seed=seed,
            episode_indices=episode_indices,
        )

    def selected_arrays(
        self,
        episode_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if episode_indices is None:
            return self.obs_buffer, self.action_buffer, self.reward_buffer
        return self.obs_buffer[episode_indices], self.action_buffer[episode_indices], self.reward_buffer[episode_indices]

    def _collect_episode(
        self,
        env: BallBalanceEnv,
        episode: int,
        seed: int,
        rng: np.random.Generator,
        mode: PolicyMode,
        initial_bounds: InitialStateBounds,
        target_bound: float,
        action_noise_std: float,
    ) -> None:
        episode_mode = choose_episode_mode(mode, rng)
        obs, _ = env.reset(seed=seed + episode, options={"state": sample_initial_state(rng, initial_bounds)})
        self.obs_buffer[episode, 0] = obs

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        target = np.zeros(2, dtype=np.float32)
        noise_std = 0.0
        hold_action = np.zeros(env.action_space.shape, dtype=np.float32)
        hold_steps = 0
        if episode_mode in ("pd_noisy", "pd_offset"):
            target = rng.uniform(-target_bound, target_bound, size=2).astype(np.float32)
            noise_std = action_noise_std if episode_mode == "pd_noisy" else 0.0

        done = False
        final_obs = obs.copy()
        for step in range(self.max_episode_steps):
            if done:
                self._pad_after_done(episode, step, final_obs)
                continue

            action, hold_action, hold_steps = self._policy_action(
                env, obs, action, hold_action, hold_steps, episode_mode, target, noise_std, rng
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            final_obs = next_obs.copy()

            self.obs_buffer[episode, step + 1] = next_obs
            self.action_buffer[episode, step] = np.clip(action, env.action_space.low, env.action_space.high)
            self.reward_buffer[episode, step, 0] = reward
            self.terminated_buffer[episode, step, 0] = terminated
            self.truncated_buffer[episode, step, 0] = truncated
            self.done_buffer[episode, step, 0] = done
            obs = next_obs

    def _policy_action(
        self,
        env: BallBalanceEnv,
        obs: np.ndarray,
        prev_action: np.ndarray,
        hold_action: np.ndarray,
        hold_steps: int,
        episode_mode: str,
        target: np.ndarray,
        noise_std: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if episode_mode == "random_smooth":
            random_action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            action = 0.95 * prev_action + 0.05 * random_action
        elif episode_mode == "random_uniform":
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        elif episode_mode == "random_burst":
            if hold_steps <= 0:
                hold_action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                hold_steps = int(rng.integers(5, 31))
            action = 0.70 * prev_action + 0.30 * hold_action
            hold_steps -= 1
        else:
            action = pd_action(obs, target=target)
            if noise_std > 0.0:
                action = action + rng.normal(0.0, noise_std, size=2).astype(np.float32)
        return action.astype(np.float32), hold_action, hold_steps

    def _pad_after_done(self, episode: int, step: int, final_obs: np.ndarray) -> None:
        self.obs_buffer[episode, step + 1] = final_obs
        self.action_buffer[episode, step] = 0.0
        self.reward_buffer[episode, step, 0] = 0.0
        self.terminated_buffer[episode, step, 0] = True
        self.truncated_buffer[episode, step, 0] = False
        self.done_buffer[episode, step, 0] = True


def choose_episode_mode(mode: PolicyMode, rng: np.random.Generator) -> str:
    if mode != "mixed":
        if mode != "coverage":
            return mode
        value = rng.random()
        if value < 0.25:
            return "pd"
        if value < 0.45:
            return "pd_offset"
        if value < 0.65:
            return "pd_noisy"
        if value < 0.80:
            return "random_smooth"
        if value < 0.90:
            return "random_burst"
        return "random_uniform"
    value = rng.random()
    if value < 0.4:
        return "random_smooth"
    if value < 0.8:
        return "pd"
    return "pd_noisy"


def sample_initial_state(rng: np.random.Generator, bounds: InitialStateBounds) -> np.ndarray:
    return np.array(
        [
            rng.uniform(-bounds.pos, bounds.pos),
            rng.uniform(-bounds.pos, bounds.pos),
            rng.uniform(-bounds.vel, bounds.vel),
            rng.uniform(-bounds.vel, bounds.vel),
            rng.uniform(-bounds.angle, bounds.angle),
            rng.uniform(-bounds.angle, bounds.angle),
        ],
        dtype=np.float32,
    )
