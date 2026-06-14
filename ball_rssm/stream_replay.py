"""Driver-style step-stream replay for online Dreamer training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class StreamReplay:
    """Replay buffer that samples contiguous windows from one step stream.

    The stream stores observation records and transition records separately:
    `obs[i] + action[i] -> obs[i + 1]`. Episode resets are represented by
    `is_first[i] == True` on the reset observation. When a new episode starts,
    a dummy reset transition is inserted and masked during training because its
    next observation is marked `is_first`.
    """

    def __init__(
        self,
        capacity_steps: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        max_episode_steps: int,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if capacity_steps <= 0:
            raise ValueError("capacity_steps must be positive")
        self.capacity_steps = capacity_steps
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.max_episode_steps = max_episode_steps
        self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float32)
        self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float32)
        self.metadata = {} if metadata is None else dict(metadata)
        self.obs: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.reward: list[np.ndarray] = []
        self.terminated: list[np.ndarray] = []
        self.truncated: list[np.ndarray] = []
        self.done: list[np.ndarray] = []
        self.is_first: list[np.ndarray] = []
        self.episode_count = 0

    @property
    def size(self) -> int:
        return self.episode_count

    @property
    def num_steps(self) -> int:
        return len(self.action)

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        capacity_steps: int | None = None,
        max_episode_steps: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "StreamReplay":
        """Load either native stream replay or legacy episode-major arrays."""

        if "episode_count" in arrays:
            return cls.from_stream_arrays(arrays, capacity_steps=capacity_steps, max_episode_steps=max_episode_steps)
        return cls.from_episode_arrays(
            arrays,
            capacity_steps=capacity_steps,
            max_episode_steps=max_episode_steps,
            metadata=metadata,
        )

    @classmethod
    def from_episode_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        capacity_steps: int | None = None,
        max_episode_steps: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "StreamReplay":
        obs = arrays["obs"].astype(np.float32)
        action = arrays["action"].astype(np.float32)
        reward = arrays["reward"].astype(np.float32)
        terminated = arrays["terminated"].astype(bool)
        truncated = arrays["truncated"].astype(bool)
        done = arrays["done"].astype(bool)
        capacity = capacity_steps or max(action.shape[0] * action.shape[1] * 2, 1)
        replay = cls(
            capacity_steps=capacity,
            obs_shape=tuple(obs.shape[2:]),
            action_shape=tuple(action.shape[2:]),
            max_episode_steps=max_episode_steps or action.shape[1],
            action_low=arrays.get("action_low"),
            action_high=arrays.get("action_high"),
            metadata=metadata,
        )
        for episode in range(action.shape[0]):
            replay.start_episode(obs[episode, 0])
            for step in range(action.shape[1]):
                replay.add_transition(
                    action=action[episode, step],
                    reward=float(reward[episode, step, 0]),
                    terminated=bool(terminated[episode, step, 0]),
                    truncated=bool(truncated[episode, step, 0]),
                    done=bool(done[episode, step, 0]),
                    next_obs=obs[episode, step + 1],
                )
                if done[episode, step, 0]:
                    break
        return replay

    @classmethod
    def from_stream_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        capacity_steps: int | None = None,
        max_episode_steps: int | None = None,
    ) -> "StreamReplay":
        """Restore the single-leading-dimension stream layout saved by this class."""

        data = dict(arrays)
        obs = data["obs"].astype(np.float32)
        action = data["action"].astype(np.float32)
        reward = data["reward"].astype(np.float32)
        terminated = data["terminated"].astype(bool)
        truncated = data["truncated"].astype(bool)
        done = data["done"].astype(bool)
        if obs.shape[0] != 1:
            raise ValueError("stream replay arrays must have one leading stream dimension")
        if obs.shape[1] != action.shape[1] + 1:
            raise ValueError("stream replay expects obs.shape[1] == action.shape[1] + 1")
        is_first = data.get("is_first")
        if is_first is None:
            is_first = np.zeros((*obs.shape[:2], 1), dtype=bool)
            is_first[:, 0, 0] = True
        else:
            is_first = is_first.astype(bool)
        stream_max_episode_steps = (
            int(np.asarray(data["max_episode_steps"]).item()) if "max_episode_steps" in data else action.shape[1]
        )
        replay = cls(
            capacity_steps=capacity_steps or max(int(action.shape[1] * 2), 1),
            obs_shape=tuple(obs.shape[2:]),
            action_shape=tuple(action.shape[2:]),
            max_episode_steps=max_episode_steps or stream_max_episode_steps,
            action_low=data.get("action_low"),
            action_high=data.get("action_high"),
        )
        replay.obs = [x.copy() for x in obs[0]]
        replay.action = [x.copy() for x in action[0]]
        replay.reward = [x.copy() for x in reward[0]]
        replay.terminated = [x.copy() for x in terminated[0]]
        replay.truncated = [x.copy() for x in truncated[0]]
        replay.done = [x.copy() for x in done[0]]
        replay.is_first = [x.copy() for x in is_first[0]]
        if "episode_count" in data:
            replay.episode_count = int(np.asarray(data["episode_count"]).item())
        else:
            replay.episode_count = int(is_first[0, :, 0].sum())
        replay._trim()
        return replay

    @classmethod
    def load(cls, path: str | Path, capacity_steps: int | None = None) -> "StreamReplay":
        with np.load(path) as arrays:
            data = {key: arrays[key] for key in arrays.files}
        return cls.from_arrays(data, capacity_steps=capacity_steps)

    def start_episode(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(self.obs_shape)
        if self.obs and len(self.obs) == len(self.action) + 1:
            zero_action = np.zeros(self.action_shape, dtype=np.float32)
            self.add_transition(
                action=zero_action,
                reward=0.0,
                terminated=False,
                truncated=True,
                done=True,
                next_obs=obs,
                next_is_first=True,
            )
        else:
            self.obs.append(obs)
            self.is_first.append(np.array([True], dtype=bool))
        self.episode_count += 1
        self._trim()

    def add_transition(
        self,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        done: bool,
        next_obs: np.ndarray,
        next_is_first: bool = False,
    ) -> None:
        if not self.obs:
            raise ValueError("start_episode() must be called before add_transition()")
        if len(self.obs) != len(self.action) + 1:
            raise ValueError("stream replay is missing the current observation")
        self.action.append(np.asarray(action, dtype=np.float32).reshape(self.action_shape))
        self.reward.append(np.array([reward], dtype=np.float32))
        self.terminated.append(np.array([terminated], dtype=bool))
        self.truncated.append(np.array([truncated], dtype=bool))
        self.done.append(np.array([done], dtype=bool))
        self.obs.append(np.asarray(next_obs, dtype=np.float32).reshape(self.obs_shape))
        self.is_first.append(np.array([next_is_first], dtype=bool))
        self._trim()

    def sequence_dataset(self, seq_len: int, split: str = "train", val_fraction: float = 0.1, seed: int = 0, **_: Any):
        from ball_rssm.data.sequence_dataset import SequenceDataset

        return SequenceDataset(self.to_dataset(), seq_len=seq_len, split=split, val_fraction=val_fraction, seed=seed)

    def selected_arrays(self, episode_indices: np.ndarray | None = None):
        del episode_indices
        dataset = self.to_dataset()
        return dataset["obs"], dataset["action"], dataset["reward"]

    def save(self, path: str | Path, **metadata: object) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dataset()
        np.savez_compressed(out_path, **data, **self._metadata_arrays(metadata))

    def to_dataset(self) -> dict[str, np.ndarray]:
        if len(self.obs) < 2:
            raise ValueError("stream replay needs at least one transition")
        obs = np.stack(self.obs, axis=0)[None]
        action = np.stack(self.action, axis=0)[None]
        reward = np.stack(self.reward, axis=0)[None]
        terminated = np.stack(self.terminated, axis=0)[None]
        truncated = np.stack(self.truncated, axis=0)[None]
        done = np.stack(self.done, axis=0)[None]
        is_first = np.stack(self.is_first, axis=0)[None]
        out = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "is_first": is_first,
            "episode_count": np.asarray(self.episode_count, dtype=np.int32),
            "max_episode_steps": np.asarray(self.max_episode_steps, dtype=np.int32),
        }
        if self.action_low is not None:
            out["action_low"] = self.action_low
        if self.action_high is not None:
            out["action_high"] = self.action_high
        return out

    def _metadata_arrays(self, extra: dict[str, object]) -> dict[str, np.ndarray]:
        combined = {**self.metadata, **extra}
        arrays: dict[str, np.ndarray] = {}
        for key, value in combined.items():
            if key in {"obs", "action", "reward", "terminated", "truncated", "done", "is_first"}:
                continue
            arrays[key] = np.asarray(value)
        return arrays

    def _trim(self) -> None:
        while len(self.action) > self.capacity_steps and len(self.obs) > 1:
            self.obs.pop(0)
            self.is_first.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.terminated.pop(0)
            self.truncated.pop(0)
            self.done.pop(0)
