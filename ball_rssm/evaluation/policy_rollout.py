"""Closed-loop Dreamer policy rollout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ball_rssm.agent import DreamerAgent
from ball_rssm.envs import BallBalanceEnv
from ball_rssm.utils.plotting import OBS_LABELS, add_board_boundary


@dataclass(frozen=True)
class InitialConditionBounds:
    pos: float = 0.25
    vel: float = 0.20
    angle: float = 0.12

    def validate(self, env: BallBalanceEnv) -> None:
        if self.pos < 0.0 or self.vel < 0.0 or self.angle < 0.0:
            raise ValueError("initial-condition bounds must be non-negative")
        if self.pos >= env.config.board_size / 2.0:
            raise ValueError("pos bound must be strictly inside the board")
        if self.angle > env.config.max_angle:
            raise ValueError("angle bound must be <= env.config.max_angle")


def sample_initial_state(rng: np.random.Generator, bounds: InitialConditionBounds) -> np.ndarray:
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


def run_policy_episode(
    agent: DreamerAgent,
    env: BallBalanceEnv,
    initial_state: np.ndarray,
    max_steps: int,
    render: bool = False,
) -> dict[str, Any]:
    obs, info = env.reset(options={"state": initial_state.astype(np.float64)})
    agent.reset(initial_obs=obs)

    observations = [obs.copy()]
    states = [info["state"].astype(np.float32, copy=True)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    diagnostics: list[dict[str, Any]] = []

    for _ in range(max_steps):
        action = agent.act(observations[-1])
        next_obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action.astype(np.float32, copy=True))
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        diagnostics.append(dict(agent.last_diagnostics))
        observations.append(next_obs.copy())
        states.append(info["state"].astype(np.float32, copy=True))
        if render:
            env.render()
        if terminated or truncated:
            break

    terminated_arr = np.asarray(terminated_flags, dtype=bool)[:, None]
    truncated_arr = np.asarray(truncated_flags, dtype=bool)[:, None]
    return {
        "obs": np.asarray(observations, dtype=np.float32),
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32)[:, None],
        "terminated": terminated_arr,
        "truncated": truncated_arr,
        "done": terminated_arr | truncated_arr,
        "diagnostics": diagnostics,
        "initial_state": initial_state.astype(np.float32, copy=True),
    }


def episode_metrics(episode: dict[str, Any], target_xy: tuple[float, float] = (0.0, 0.0)) -> dict[str, float | bool | int]:
    obs = episode["obs"]
    state = episode.get("state", obs)
    reward = episode["reward"]
    target = np.asarray(target_xy, dtype=np.float32)
    distance = np.linalg.norm(state[:, :2] - target[None], axis=-1)
    return {
        "steps": int(max(obs.shape[0] - 1, 0)),
        "return": float(reward.sum()) if reward.size else 0.0,
        "terminated": bool(episode["terminated"].any()) if episode["terminated"].size else False,
        "truncated": bool(episode["truncated"].any()) if episode["truncated"].size else False,
        "final_distance": float(distance[-1]),
        "mean_distance": float(distance.mean()),
        "max_distance": float(distance.max()),
    }


def aggregate_metrics(metrics: list[dict[str, float | bool | int]]) -> dict[str, float]:
    if not metrics:
        return {}
    numeric_keys = [key for key, value in metrics[0].items() if isinstance(value, (int, float, bool))]
    out: dict[str, float] = {}
    for key in numeric_keys:
        values = np.asarray([float(item[key]) for item in metrics], dtype=np.float32)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    return out


def save_episode_npz(episode: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=episode["obs"],
        state=episode.get("state", episode["obs"]),
        action=episode["action"],
        reward=episode["reward"],
        terminated=episode["terminated"],
        truncated=episode["truncated"],
        done=episode["done"],
        initial_state=episode["initial_state"],
    )


def save_policy_plots(episode: dict[str, Any], out_dir: str | Path, target_xy: tuple[float, float] = (0.0, 0.0)) -> None:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = episode.get("state", episode["obs"])
    action = episode["action"]
    t_obs = np.arange(obs.shape[0])
    t_action = np.arange(action.shape[0])
    target = np.asarray(target_xy, dtype=np.float32)
    distance = np.linalg.norm(obs[:, :2] - target[None], axis=-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(obs[:, 0], obs[:, 1], marker="o", markersize=2)
    ax.scatter([target[0]], [target[1]], marker="x", color="tab:red")
    add_board_boundary(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Dreamer policy trajectory")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "xy_trajectory.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    for idx, ax in enumerate(axes.flat):
        ax.plot(t_obs, obs[:, idx])
        ax.set_ylabel(OBS_LABELS[idx])
        ax.grid(True)
    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "state_timeseries.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    if action.size:
        ax.plot(t_action, action[:, 0], label="theta_x_cmd")
        ax.plot(t_action, action[:, 1], label="theta_y_cmd")
    ax.set_xlabel("step")
    ax.set_ylabel("action [rad]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "action_timeseries.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_obs, distance)
    ax.set_xlabel("step")
    ax.set_ylabel("distance to center [m]")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "distance_to_center.png")
    plt.close(fig)
