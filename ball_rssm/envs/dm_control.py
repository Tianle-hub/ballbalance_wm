"""Small DeepMind Control Suite adapter used by the training scripts."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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
    camera_id: int = 0
    mujoco_gl: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
        self._env = suite.load(
            domain_name=config.domain,
            task_name=config.task,
            task_kwargs={"random": seed},
        )
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
            camera_id=self.config.camera_id,
        )

    def sample_random_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.action_low, self.action_high).astype(np.float32)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()

    def _format_obs(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        if self.config.obs_type == "pixel":
            frame = self.render()
            return np.moveaxis(frame.astype(np.float32) / 255.0, -1, 0)
        return flatten_observation(observation, self._obs_keys)


def flatten_observation(observation: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    parts = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in keys]
    return np.concatenate(parts, axis=0).astype(np.float32)


def collect_random_dm_control(
    config: DMControlConfig,
    num_episodes: int,
    max_episode_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Collect random-policy DM-Control replay in the project NPZ schema."""

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")

    env = DMControlEnv(config, seed=seed)
    rng = np.random.default_rng(seed)
    obs = np.zeros((num_episodes, max_episode_steps + 1, *env.obs_shape), dtype=np.float32)
    action = np.zeros((num_episodes, max_episode_steps, *env.action_shape), dtype=np.float32)
    reward = np.zeros((num_episodes, max_episode_steps, 1), dtype=np.float32)
    terminated = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)
    truncated = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)
    done = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)

    try:
        for episode in range(num_episodes):
            current_obs = env.reset()
            obs[episode, 0] = current_obs
            final_obs = current_obs
            episode_done = False
            for step in range(max_episode_steps):
                if episode_done:
                    obs[episode, step + 1] = final_obs
                    done[episode, step, 0] = True
                    truncated[episode, step, 0] = True
                    continue

                action_np = env.sample_random_action(rng)
                next_obs, step_reward, term, trunc, env_done = env.step(action_np)
                reached_limit = step == max_episode_steps - 1 and not env_done
                episode_done = bool(env_done or reached_limit)

                obs[episode, step + 1] = next_obs
                action[episode, step] = action_np
                reward[episode, step, 0] = step_reward
                terminated[episode, step, 0] = term
                truncated[episode, step, 0] = bool(trunc or reached_limit)
                done[episode, step, 0] = episode_done
                final_obs = next_obs
    finally:
        env.close()

    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
        "action_low": env.action_low,
        "action_high": env.action_high,
        "obs_type": np.asarray(config.obs_type),
        "domain": np.asarray(config.domain),
        "task": np.asarray(config.task),
        "action_repeat": np.asarray(config.action_repeat, dtype=np.int32),
        "height": np.asarray(config.height, dtype=np.int32),
        "width": np.asarray(config.width, dtype=np.int32),
        "camera_id": np.asarray(config.camera_id, dtype=np.int32),
        "obs_keys": np.asarray(env.obs_keys),
    }


def save_dm_control_dataset(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
