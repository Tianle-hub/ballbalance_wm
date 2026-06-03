"""Closed-loop rollout helpers for RSSM MPC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.planning.rssm_mpc import RSSMMPCController

OBS_LABELS = ["x", "y", "vx", "vy", "theta_x", "theta_y"]


@dataclass(frozen=True)
class InitialConditionBounds:
    pos: float = 0.25
    vel: float = 0.10
    angle: float = 0.05


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


def run_mpc_episode(
    controller: RSSMMPCController,
    env: BallBalanceEnv,
    initial_state: np.ndarray,
    max_steps: int,
    target_xy: tuple[float, float] = (0.0, 0.0),
    render: bool = False,
) -> dict[str, np.ndarray | float | bool]:
    obs, _ = env.reset(options={"state": initial_state})
    controller.reset(obs)

    observations = [obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    costs: list[float] = []
    predicted_obs: list[np.ndarray] = []
    predicted_reward: list[np.ndarray] = []
    computation_times: list[float] = []
    terminated = False
    truncated = False

    import time

    for _ in range(max_steps):
        start = time.perf_counter()
        action = controller.act(obs)
        computation_times.append(time.perf_counter() - start)
        diagnostics = controller.diagnostics_dict()
        if "best_cost" in diagnostics:
            costs.append(float(diagnostics["best_cost"]))
        if "predicted_obs" in diagnostics:
            predicted_obs.append(np.asarray(diagnostics["predicted_obs"], dtype=np.float32))
        if "predicted_reward" in diagnostics:
            predicted_reward.append(np.asarray(diagnostics["predicted_reward"], dtype=np.float32))

        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        actions.append(action.astype(np.float32).copy())
        rewards.append(float(reward))
        if render:
            env.render()
        if terminated or truncated:
            break

    obs_arr = np.asarray(observations, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    reward_arr = np.asarray(rewards, dtype=np.float32)
    pred_arr = np.asarray(predicted_obs, dtype=np.float32) if predicted_obs else np.zeros((0, controller.horizon, 6), dtype=np.float32)
    pred_reward_arr = (
        np.asarray(predicted_reward, dtype=np.float32) if predicted_reward else np.zeros((0, controller.horizon, 1), dtype=np.float32)
    )
    distance = np.linalg.norm(obs_arr[:, :2] - np.asarray(target_xy, dtype=np.float32), axis=-1)

    return {
        "obs": obs_arr,
        "action": action_arr,
        "reward": reward_arr,
        "cost": np.asarray(costs, dtype=np.float32),
        "predicted_obs": pred_arr,
        "predicted_reward": pred_reward_arr,
        "distance": distance.astype(np.float32),
        "computation_time": np.asarray(computation_times, dtype=np.float32),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "total_reward": float(reward_arr.sum()) if reward_arr.size else 0.0,
    }


def episode_metrics(
    episode: dict[str, np.ndarray | float | bool],
    target_xy: tuple[float, float],
    final_threshold: float,
    last_window_threshold: float,
) -> dict[str, float | bool]:
    obs = np.asarray(episode["obs"])
    action = np.asarray(episode["action"])
    distance = np.linalg.norm(obs[:, :2] - np.asarray(target_xy, dtype=np.float32), axis=-1)
    tail = distance[-min(50, distance.shape[0]) :]
    fell = bool(episode["terminated"])
    success = (not fell) and float(distance[-1]) < final_threshold and float(tail.mean()) < last_window_threshold
    return {
        "success": bool(success),
        "fell": fell,
        "final_distance": float(distance[-1]),
        "mean_distance": float(distance.mean()),
        "max_distance": float(distance.max()),
        "last50_distance": float(tail.mean()),
        "total_reward": float(episode["total_reward"]),
        "control_magnitude": float(np.linalg.norm(action, axis=-1).mean()) if action.size else 0.0,
        "mean_compute_time": float(np.asarray(episode["computation_time"]).mean()) if len(episode["computation_time"]) else 0.0,
    }


def aggregate_metrics(metrics: list[dict[str, float | bool]]) -> dict[str, float]:
    if not metrics:
        return {}
    numeric_keys = [key for key, value in metrics[0].items() if isinstance(value, (int, float, bool))]
    out: dict[str, float] = {}
    for key in numeric_keys:
        values = np.asarray([float(m[key]) for m in metrics], dtype=np.float32)
        out[key] = float(values.mean())
    out["success_rate"] = float(np.mean([bool(m["success"]) for m in metrics]))
    out["fall_rate"] = float(np.mean([bool(m["fell"]) for m in metrics]))
    return out


def save_episode_npz(episode: dict[str, np.ndarray | float | bool], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **episode)


def save_mpc_plots(
    episode: dict[str, np.ndarray | float | bool],
    out_dir: str | Path,
    target_xy: tuple[float, float] = (0.0, 0.0),
    title_prefix: str = "RSSM MPC",
    planned_snapshot_indices: tuple[int, ...] = (0, 10, 25, 50),
) -> None:
    import matplotlib.pyplot as plt

    from ball_rssm.utils.plotting import add_board_boundary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = np.asarray(episode["obs"])
    action = np.asarray(episode["action"])
    pred = np.asarray(episode["predicted_obs"])
    target = np.asarray(target_xy, dtype=np.float32)
    distance = np.linalg.norm(obs[:, :2] - target, axis=-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(obs[:, 0], obs[:, 1], color="black", linewidth=2, label="actual")
    ax.plot([target[0]], [target[1]], "x", color="tab:red", markersize=10, label="target")
    for idx in planned_snapshot_indices:
        if idx < pred.shape[0]:
            ax.plot(pred[idx, :, 0], pred[idx, :, 1], linestyle="--", alpha=0.55, label=f"plan t={idx}")
    add_board_boundary(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title_prefix} x-y")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "xy_trajectory.png")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    t_obs = np.arange(obs.shape[0])
    for idx, label in enumerate(OBS_LABELS):
        axes.flat[idx].plot(t_obs, obs[:, idx])
        axes.flat[idx].set_title(label)
        axes.flat[idx].grid(True)
    t_action = np.arange(action.shape[0])
    axes.flat[6].plot(t_action, action[:, 0] if action.size else [])
    axes.flat[6].set_title("theta_x_cmd")
    axes.flat[6].grid(True)
    axes.flat[7].plot(t_action, action[:, 1] if action.size else [])
    axes.flat[7].set_title("theta_y_cmd")
    axes.flat[7].grid(True)
    axes.flat[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "state_action_timeseries.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(distance.shape[0]), distance)
    ax.set_xlabel("step")
    ax.set_ylabel("distance [m]")
    ax.set_title(f"{title_prefix} distance to target")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "distance_to_target.png")
    plt.close(fig)


def pd_action(obs: np.ndarray, target_xy: tuple[float, float] = (0.0, 0.0), kp: float = 0.8, kd: float = 0.25) -> np.ndarray:
    x, y, vx, vy, _, _ = obs
    target = np.asarray(target_xy, dtype=np.float32)
    theta_y_cmd = -kp * (x - target[0]) - kd * vx
    theta_x_cmd = kp * (y - target[1]) + kd * vy
    return np.array([theta_x_cmd, theta_y_cmd], dtype=np.float32)


def run_pd_episode(
    env: BallBalanceEnv,
    initial_state: np.ndarray,
    max_steps: int,
    target_xy: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray | float | bool]:
    obs, _ = env.reset(options={"state": initial_state})
    observations = [obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated = False
    truncated = False
    for _ in range(max_steps):
        action = np.clip(pd_action(obs, target_xy), env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break
    obs_arr = np.asarray(observations, dtype=np.float32)
    return {
        "obs": obs_arr,
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "distance": np.linalg.norm(obs_arr[:, :2] - np.asarray(target_xy, dtype=np.float32), axis=-1).astype(np.float32),
        "computation_time": np.zeros(len(actions), dtype=np.float32),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "total_reward": float(np.sum(rewards)),
    }
