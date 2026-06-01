"""Random initial-condition rollouts and imagined trajectory error metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.models import Normalizer, WorldModel
from ball_rssm.utils.plotting import OBS_LABELS, add_board_boundary

ActionMode = Literal["random_smooth", "pd", "mixed"]


@dataclass(frozen=True)
class RandomICBounds:
    """Bounds for random initial state [x, y, vx, vy, theta_x, theta_y]."""

    pos: float = 0.25
    vel: float = 0.20
    angle: float = 0.12

    def validate(self, env: BallBalanceEnv) -> None:
        if self.pos < 0.0 or self.vel < 0.0 or self.angle < 0.0:
            raise ValueError("RandomICBounds values must be non-negative")
        if self.pos >= env.config.board_size / 2.0:
            raise ValueError("pos bound must be strictly inside the board")
        if self.angle > env.config.max_angle:
            raise ValueError("angle bound must be <= env.config.max_angle")


def generate_random_ic_dataset(
    num_episodes: int,
    horizon: int,
    context_len: int,
    seed: int,
    bounds: RandomICBounds = RandomICBounds(),
    action_mode: ActionMode = "mixed",
) -> dict[str, np.ndarray]:
    """Generate true env trajectories from random bounded initial conditions.

    The resulting arrays follow obs[:, t] + action[:, t] -> obs[:, t + 1].
    """

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if context_len < 0:
        raise ValueError("context_len must be non-negative")
    if action_mode not in ("random_smooth", "pd", "mixed"):
        raise ValueError(f"Unsupported action_mode={action_mode!r}")

    total_steps = context_len + horizon
    env = BallBalanceEnv(config={"max_episode_steps": total_steps})
    bounds.validate(env)
    rng = np.random.default_rng(seed)

    obs = np.zeros((num_episodes, total_steps + 1, 6), dtype=np.float32)
    action = np.zeros((num_episodes, total_steps, 2), dtype=np.float32)
    reward = np.zeros((num_episodes, total_steps, 1), dtype=np.float32)
    terminated = np.zeros((num_episodes, total_steps, 1), dtype=bool)
    truncated = np.zeros((num_episodes, total_steps, 1), dtype=bool)
    done = np.zeros((num_episodes, total_steps, 1), dtype=bool)
    initial_state = np.zeros((num_episodes, 6), dtype=np.float32)

    try:
        for episode in range(num_episodes):
            env.reset(seed=seed + episode)
            init = sample_initial_state(rng, bounds)
            env.state[:] = init.astype(np.float64)
            initial_state[episode] = init
            obs[episode, 0] = init

            episode_mode = choose_action_mode(action_mode, rng)
            previous_action = np.zeros(2, dtype=np.float32)
            target = rng.uniform(-0.10, 0.10, size=2).astype(np.float32) if episode_mode == "pd_offset" else None
            final_obs = obs[episode, 0].copy()
            episode_done = False

            for t in range(total_steps):
                if episode_done:
                    obs[episode, t + 1] = final_obs
                    done[episode, t, 0] = True
                    terminated[episode, t, 0] = True
                    continue

                action_t = choose_action(obs[episode, t], previous_action, episode_mode, rng, env, target)
                next_obs, reward_t, terminated_t, truncated_t, _ = env.step(action_t)
                episode_done = terminated_t or truncated_t
                previous_action = action_t.astype(np.float32)
                final_obs = next_obs.copy()

                obs[episode, t + 1] = next_obs
                action[episode, t] = previous_action
                reward[episode, t, 0] = reward_t
                terminated[episode, t, 0] = terminated_t
                truncated[episode, t, 0] = truncated_t
                done[episode, t, 0] = episode_done
    finally:
        env.close()

    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
        "initial_state": initial_state,
        "bounds": np.array([bounds.pos, bounds.vel, bounds.angle], dtype=np.float32),
    }


def sample_initial_state(rng: np.random.Generator, bounds: RandomICBounds) -> np.ndarray:
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


def choose_action_mode(action_mode: ActionMode, rng: np.random.Generator) -> str:
    if action_mode != "mixed":
        return action_mode
    value = rng.random()
    if value < 0.4:
        return "random_smooth"
    if value < 0.8:
        return "pd"
    return "pd_offset"


def choose_action(
    obs: np.ndarray,
    previous_action: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    env: BallBalanceEnv,
    target: np.ndarray | None,
) -> np.ndarray:
    if mode == "random_smooth":
        random_action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        return np.clip(0.95 * previous_action + 0.05 * random_action, env.action_space.low, env.action_space.high)
    action = pd_action(obs, target=target)
    if mode == "pd_offset":
        action = action + rng.normal(0.0, 0.02, size=2).astype(np.float32)
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def pd_action(obs: np.ndarray, kp: float = 0.8, kd: float = 0.25, target: np.ndarray | None = None) -> np.ndarray:
    if target is None:
        target = np.zeros(2, dtype=np.float32)
    x, y, vx, vy, _, _ = obs
    theta_y_cmd = -kp * (x - target[0]) - kd * vx
    theta_x_cmd = kp * (y - target[1]) + kd * vy
    return np.array([theta_x_cmd, theta_y_cmd], dtype=np.float32)


def evaluate_imagined_trajectory_error(
    model: WorldModel,
    normalizer: Normalizer,
    dataset: dict[str, np.ndarray],
    context_len: int,
    horizon: int,
    device: torch.device | str,
    batch_size: int = 128,
) -> dict[str, object]:
    """Compare decoded RSSM prior rollout against true env future observations."""

    obs = dataset["obs"].astype(np.float32)
    action = dataset["action"].astype(np.float32)
    if context_len < 0 or horizon <= 0:
        raise ValueError("context_len must be non-negative and horizon must be positive")
    horizon = min(horizon, action.shape[1] - context_len)
    if horizon <= 0:
        raise ValueError("No future actions available after context_len")

    device = torch.device(device)
    model.eval()
    normalizer.to(device)

    sq_error_batches: list[np.ndarray] = []
    pos_error_batches: list[np.ndarray] = []
    vel_error_batches: list[np.ndarray] = []
    final_pos_errors: list[np.ndarray] = []
    final_vel_errors: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, obs.shape[0], batch_size):
            end = min(start + batch_size, obs.shape[0])
            obs_batch = torch.as_tensor(obs[start:end, : context_len + horizon + 1], dtype=torch.float32, device=device)
            action_batch = torch.as_tensor(action[start:end, : context_len + horizon], dtype=torch.float32, device=device)
            obs_norm = normalizer.normalize_obs(obs_batch)
            action_norm = normalizer.normalize_action(action_batch)
            pred_norm = model.open_loop_predict(obs_norm, action_norm, context_len=context_len, horizon=horizon)
            pred = normalizer.denormalize_obs(pred_norm).cpu().numpy()
            target = obs_batch[:, context_len + 1 : context_len + 1 + horizon].cpu().numpy()

            sq_error = (pred - target) ** 2
            pos_error = np.linalg.norm(pred[:, :, :2] - target[:, :, :2], axis=-1)
            vel_error = np.linalg.norm(pred[:, :, 2:4] - target[:, :, 2:4], axis=-1)
            sq_error_batches.append(sq_error)
            pos_error_batches.append(pos_error)
            vel_error_batches.append(vel_error)
            final_pos_errors.append(pos_error[:, -1])
            final_vel_errors.append(vel_error[:, -1])

    sq_error_all = np.concatenate(sq_error_batches, axis=0)
    pos_error_all = np.concatenate(pos_error_batches, axis=0)
    vel_error_all = np.concatenate(vel_error_batches, axis=0)
    final_pos_error_all = np.concatenate(final_pos_errors, axis=0)
    final_vel_error_all = np.concatenate(final_vel_errors, axis=0)

    return {
        "num_episodes": int(obs.shape[0]),
        "context_len": int(context_len),
        "horizon": int(horizon),
        "obs_mse": float(sq_error_all.mean()),
        "position_rmse": float(np.sqrt(sq_error_all[:, :, :2].mean())),
        "velocity_rmse": float(np.sqrt(sq_error_all[:, :, 2:4].mean())),
        "position_l2_mean": float(pos_error_all.mean()),
        "velocity_l2_mean": float(vel_error_all.mean()),
        "final_position_l2_mean": float(final_pos_error_all.mean()),
        "final_velocity_l2_mean": float(final_vel_error_all.mean()),
        "position_l2_curve": pos_error_all.mean(axis=0).tolist(),
        "velocity_l2_curve": vel_error_all.mean(axis=0).tolist(),
        "obs_mse_curve": sq_error_all.mean(axis=(0, 2)).tolist(),
        "mse_per_dim": {label: float(sq_error_all[:, :, idx].mean()) for idx, label in enumerate(OBS_LABELS)},
    }


def save_random_ic_dataset(dataset: dict[str, np.ndarray], out: str | Path, **metadata: object) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **dataset, **metadata)


def plot_random_ic_errors(metrics: dict[str, object], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    horizon = np.arange(1, len(metrics["position_l2_curve"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(horizon, metrics["position_l2_curve"], label="position L2")
    ax.plot(horizon, metrics["velocity_l2_curve"], label="velocity L2")
    ax.set_xlabel("open-loop horizon")
    ax.set_ylabel("mean error")
    ax.set_title("Random initial-condition imagined rollout error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_random_ic_trajectories(
    dataset: dict[str, np.ndarray],
    model: WorldModel,
    normalizer: Normalizer,
    context_len: int,
    horizon: int,
    device: torch.device | str,
    out_path: str | Path,
    max_episodes: int = 8,
) -> None:
    obs = dataset["obs"].astype(np.float32)
    action = dataset["action"].astype(np.float32)
    horizon = min(horizon, action.shape[1] - context_len)
    batch_count = min(max_episodes, obs.shape[0])
    device = torch.device(device)
    with torch.no_grad():
        obs_batch = torch.as_tensor(obs[:batch_count, : context_len + horizon + 1], dtype=torch.float32, device=device)
        action_batch = torch.as_tensor(action[:batch_count, : context_len + horizon], dtype=torch.float32, device=device)
        pred = model.open_loop_predict(
            normalizer.normalize_obs(obs_batch),
            normalizer.normalize_action(action_batch),
            context_len=context_len,
            horizon=horizon,
        )
        pred_np = normalizer.denormalize_obs(pred).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    for ep in range(batch_count):
        true_future = obs[ep, context_len + 1 : context_len + 1 + horizon]
        ax.plot(obs[ep, : context_len + 1, 0], obs[ep, : context_len + 1, 1], color="tab:green", alpha=0.45)
        ax.plot(true_future[:, 0], true_future[:, 1], color="black", alpha=0.45)
        ax.plot(pred_np[ep, :, 0], pred_np[ep, :, 1], color="tab:red", linestyle="--", alpha=0.65)
    add_board_boundary(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Random-IC true future vs imagined rollout")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def bounds_to_dict(bounds: RandomICBounds) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(bounds).items()}
