"""Small DeepMind Control Suite adapter used by the training scripts."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

DMObsType = Literal["state", "pixel"]


@dataclass(frozen=True)
class DMControlConfig:
    domain: str = "cartpole"
    task: str = "swingup"
    obs_type: DMObsType = "state"
    action_repeat: int = 1
    height: int = 64
    width: int = 64
    camera_id: int | None = None
    camera_fovy: float | None = None
    mujoco_gl: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        camera_id = resolve_camera_id(self)
        data["camera_id"] = camera_id
        data["camera_fovy"] = resolve_camera_fovy(self, camera_id)
        return data


class DMControlEnv:
    """Expose DM-Control timesteps as flat state or CHW pixel observations."""

    def __init__(self, config: DMControlConfig, seed: int = 0) -> None:
        if config.mujoco_gl is not None:
            os.environ["MUJOCO_GL"] = config.mujoco_gl
        if config.action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        if config.obs_type not in ("state", "pixel"):
            raise ValueError("obs_type must be 'state' or 'pixel'")

        from dm_control import suite

        self.config = config
        self.camera_id = resolve_camera_id(config)
        self.camera_fovy = resolve_camera_fovy(config, self.camera_id)
        self._env = suite.load(
            domain_name=config.domain,
            task_name=config.task,
            task_kwargs={"random": seed},
        )
        self._apply_camera_overrides()
        self._obs_keys = tuple(self._env.observation_spec().keys())
        self._action_spec = self._env.action_spec()
        self.action_low = np.asarray(self._action_spec.minimum, dtype=np.float32)
        self.action_high = np.asarray(self._action_spec.maximum, dtype=np.float32)
        self.action_shape = tuple(int(dim) for dim in self._action_spec.shape)
        if config.obs_type == "pixel":
            self.obs_shape = (3, config.height, config.width)
        else:
            self.obs_shape = (
                int(sum(np.prod(self._env.observation_spec()[key].shape) for key in self._obs_keys)),
            )

    @property
    def obs_keys(self) -> tuple[str, ...]:
        return self._obs_keys

    @property
    def action_spec(self) -> Any:
        return self._action_spec

    def reset(self) -> np.ndarray:
        timestep = self._env.reset()
        return self._format_obs(timestep.observation)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, bool]:
        action = np.asarray(action, dtype=self._action_spec.dtype).reshape(self.action_shape)
        reward = 0.0
        timestep = None
        for _ in range(self.config.action_repeat):
            timestep = self._env.step(action)
            reward += float(timestep.reward or 0.0)
            if timestep.last():
                break
        if timestep is None:
            raise RuntimeError("DM-Control step loop did not execute")

        done = bool(timestep.last())
        discount = 1.0 if timestep.discount is None else float(timestep.discount)
        terminated = bool(done and discount == 0.0)
        truncated = bool(done and not terminated)
        return self._format_obs(timestep.observation), reward, terminated, truncated, done

    def render(self) -> np.ndarray:
        return self._env.physics.render(
            height=self.config.height,
            width=self.config.width,
            camera_id=self.camera_id,
        )

    def _apply_camera_overrides(self) -> None:
        if self.camera_fovy is None:
            return
        self._env.physics.model.cam_fovy[self.camera_id] = float(self.camera_fovy)

    def sample_random_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.action_low, self.action_high).astype(np.float32)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()

    def _format_obs(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        if self.config.obs_type == "pixel":
            frame = self.render()
            return preprocess_pixel_frame(frame)
        return flatten_observation(observation, self._obs_keys)


class NormalizeActionWrapper:
    """Wrap a continuous DM-Control env so policy actions live in [-1, 1]."""

    def __init__(self, env: DMControlEnv) -> None:
        self.env = env
        self.config = env.config
        self.obs_shape = env.obs_shape
        self.action_shape = env.action_shape
        self.real_action_low = env.action_low
        self.real_action_high = env.action_high
        self.action_low = -np.ones(env.action_shape, dtype=np.float32)
        self.action_high = np.ones(env.action_shape, dtype=np.float32)

    @property
    def obs_keys(self) -> tuple[str, ...]:
        return self.env.obs_keys

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, bool]:
        return self.env.step(self.denormalize_action(action))

    def render(self) -> np.ndarray:
        return self.env.render()

    def sample_random_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.action_low, self.action_high).astype(np.float32)

    def close(self) -> None:
        self.env.close()

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        action = np.clip(action, self.action_low, self.action_high)
        return self.real_action_low + 0.5 * (action + 1.0) * (self.real_action_high - self.real_action_low)


class DMControlDriver:
    """Run online DM-Control interaction and append transitions to stream replay."""

    def __init__(
        self,
        env: DMControlEnv | NormalizeActionWrapper,
        replay: StreamReplay,
        max_episode_steps: int,
    ) -> None:
        self.env = env
        self.replay = replay
        self.max_episode_steps = max_episode_steps

    def run(
        self,
        policy: Callable[[np.ndarray, bool], np.ndarray],
        num_episodes: int,
    ) -> dict[str, float]:
        rewards: list[float] = []
        lengths: list[int] = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            self.replay.start_episode(obs)
            total_reward = 0.0
            length = 0
            done = False
            is_first = True
            for step in range(self.max_episode_steps):
                action = np.asarray(policy(obs, is_first), dtype=np.float32).reshape(self.env.action_shape)
                next_obs, reward, terminated, truncated, env_done = self.env.step(action)
                reached_limit = step == self.max_episode_steps - 1 and not env_done
                done = bool(env_done or reached_limit)
                self.replay.add_transition(
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=bool(truncated or reached_limit),
                    done=done,
                    next_obs=next_obs,
                )
                total_reward += float(reward)
                length = step + 1
                obs = next_obs
                is_first = False
                if done:
                    break
            rewards.append(total_reward)
            lengths.append(length)
        reward_array = np.asarray(rewards, dtype=np.float32)
        length_array = np.asarray(lengths, dtype=np.float32)
        return {
            "avg_reward": float(reward_array.mean()) if rewards else float("nan"),
            "max_reward": float(reward_array.max()) if rewards else float("nan"),
            "min_reward": float(reward_array.min()) if rewards else float("nan"),
            "avg_length": float(length_array.mean()) if lengths else float("nan"),
            "episodes": float(num_episodes),
        }


def flatten_observation(observation: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    parts = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in keys]
    return np.concatenate(parts, axis=0).astype(np.float32)


def preprocess_pixel_frame(frame: np.ndarray) -> np.ndarray:
    """Convert an HWC uint8 RGB frame to centered CHW float pixels."""

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("pixel frame must have shape [height, width, 3]")
    return np.moveaxis(frame.astype(np.float32) / 255.0 - 0.5, -1, 0)


def resolve_camera_id(config: DMControlConfig) -> int:
    """Resolve a training-friendly camera when the caller did not choose one."""

    if config.camera_id is not None:
        return int(config.camera_id)
    return 0


def resolve_camera_fovy(config: DMControlConfig, camera_id: int | None = None) -> float | None:
    """Return a wider fixed camera FOV for cartpole pixel observations."""

    if config.camera_fovy is not None:
        return float(config.camera_fovy)
    resolved_camera_id = resolve_camera_id(config) if camera_id is None else int(camera_id)
    if config.obs_type == "pixel" and config.domain == "cartpole" and resolved_camera_id == 0:
        return 70.0
    return None


def save_dm_control_dataset(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
