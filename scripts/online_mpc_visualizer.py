"""Online closed-loop RSSM MPC visualizer with multi-episode metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.planning.rollout import OBS_LABELS, aggregate_metrics, episode_metrics, save_episode_npz, save_mpc_plots
from ball_rssm.planning.rssm_mpc import RSSMMPCController

TaskMode = Literal["center", "viapoint", "random"]


@dataclass(frozen=True)
class OnlineInitialBounds:
    """Symmetric sampling bounds for the 6D simulator state."""

    x: float = 0.25
    y: float = 0.25
    vx: float = 0.10
    vy: float = 0.10
    theta_x: float = 0.05
    theta_y: float = 0.05


def sample_online_initial_state(rng: np.random.Generator, bounds: OnlineInitialBounds) -> np.ndarray:
    return np.array(
        [
            rng.uniform(-bounds.x, bounds.x),
            rng.uniform(-bounds.y, bounds.y),
            rng.uniform(-bounds.vx, bounds.vx),
            rng.uniform(-bounds.vy, bounds.vy),
            rng.uniform(-bounds.theta_x, bounds.theta_x),
            rng.uniform(-bounds.theta_y, bounds.theta_y),
        ],
        dtype=np.float32,
    )


def sample_target(rng: np.random.Generator, target_bound: float) -> tuple[float, float]:
    return (
        float(rng.uniform(-target_bound, target_bound)),
        float(rng.uniform(-target_bound, target_bound)),
    )


def select_episode_task(rng: np.random.Generator, task: TaskMode, target_bound: float) -> tuple[str, tuple[float, float]]:
    if task == "center":
        return "center", (0.0, 0.0)
    if task == "viapoint":
        return "viapoint", sample_target(rng, target_bound)
    if task == "random":
        sampled = "center" if rng.random() < 0.5 else "viapoint"
        return select_episode_task(rng, sampled, target_bound)
    raise ValueError(f"Unsupported task={task!r}")


def ensure_writable_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        config_dir = Path.home() / ".config" / "matplotlib"
        if not is_path_or_parent_writable(config_dir):
            fallback = Path(tempfile.gettempdir()) / "ballbalance_matplotlib"
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(fallback)

    if "XDG_CACHE_HOME" not in os.environ:
        cache_dir = Path.home() / ".cache"
        if not is_path_or_parent_writable(cache_dir):
            fallback_cache = Path(tempfile.gettempdir()) / "ballbalance_cache"
            fallback_cache.mkdir(parents=True, exist_ok=True)
            os.environ["XDG_CACHE_HOME"] = str(fallback_cache)


def is_path_or_parent_writable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    return path.parent.exists() and os.access(path.parent, os.W_OK)


class SlidingWindowVisualizer:
    def __init__(self, window: int, pause_s: float) -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        self.window = window
        self.pause_s = pause_s
        self._plt = plt

        self.action_fig, self.action_ax = plt.subplots(figsize=(8, 3), num="RSSM MPC control input")
        (self.theta_x_cmd_line,) = self.action_ax.plot([], [], label="theta_x_cmd")
        (self.theta_y_cmd_line,) = self.action_ax.plot([], [], label="theta_y_cmd")
        self.action_ax.set_xlabel("step")
        self.action_ax.set_ylabel("rad")
        self.action_ax.grid(True)
        self.action_ax.legend(loc="upper right")

        self.state_fig, self.state_axes = plt.subplots(3, 2, figsize=(10, 7), num="RSSM MPC states", sharex=True)
        self.state_lines = []
        for idx, label in enumerate(OBS_LABELS):
            ax = self.state_axes.flat[idx]
            (line,) = ax.plot([], [], label=label)
            ax.set_title(label)
            ax.grid(True)
            self.state_lines.append(line)
        self.state_axes.flat[-1].set_xlabel("step")

    def update(
        self,
        episode_idx: int,
        task_name: str,
        target_xy: tuple[float, float],
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        best_cost: float | None,
    ) -> None:
        obs = np.asarray(observations, dtype=np.float32)
        action = np.asarray(actions, dtype=np.float32)
        obs_start = max(0, obs.shape[0] - self.window)
        obs_steps = np.arange(obs_start, obs.shape[0])

        for idx, line in enumerate(self.state_lines):
            line.set_data(obs_steps, obs[obs_start:, idx])
            ax = self.state_axes.flat[idx]
            ax.set_xlim(obs_steps[0] if obs_steps.size else 0, max(obs_steps[-1] if obs_steps.size else 1, 1))
            values = obs[obs_start:, idx]
            self._set_padded_ylim(ax, values)

        action_start = max(0, action.shape[0] - self.window)
        action_steps = np.arange(action_start, action.shape[0])
        if action.size:
            self.theta_x_cmd_line.set_data(action_steps, action[action_start:, 0])
            self.theta_y_cmd_line.set_data(action_steps, action[action_start:, 1])
            self.action_ax.set_xlim(action_steps[0], max(action_steps[-1], 1))
            self._set_padded_ylim(self.action_ax, action[action_start:].reshape(-1))
        else:
            self.theta_x_cmd_line.set_data([], [])
            self.theta_y_cmd_line.set_data([], [])
            self.action_ax.set_xlim(0, 1)
            self.action_ax.set_ylim(-0.1, 0.1)

        cost_text = "" if best_cost is None else f" cost={best_cost:.3f}"
        self.action_ax.set_title(
            f"episode={episode_idx} task={task_name} target=({target_xy[0]:+.3f}, {target_xy[1]:+.3f}){cost_text}"
        )
        self.action_fig.canvas.draw_idle()
        self.state_fig.canvas.draw_idle()
        self._plt.pause(self.pause_s)

    @staticmethod
    def _set_padded_ylim(ax, values: np.ndarray) -> None:
        if values.size == 0 or not np.isfinite(values).all():
            ax.set_ylim(-1.0, 1.0)
            return
        low = float(values.min())
        high = float(values.max())
        if abs(high - low) < 1e-6:
            pad = max(0.05, abs(high) * 0.1)
        else:
            pad = 0.1 * (high - low)
        ax.set_ylim(low - pad, high + pad)

    def close(self) -> None:
        self._plt.close(self.action_fig)
        self._plt.close(self.state_fig)


def run_online_episode(
    controller: RSSMMPCController,
    env: BallBalanceEnv,
    visualizer: SlidingWindowVisualizer,
    initial_state: np.ndarray,
    max_steps: int,
    target_xy: tuple[float, float],
    episode_idx: int,
    task_name: str,
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

    env.render()
    visualizer.update(episode_idx, task_name, target_xy, observations, actions, best_cost=None)

    for _ in range(max_steps):
        start = time.perf_counter()
        action = controller.act(obs)
        computation_times.append(time.perf_counter() - start)

        diagnostics = controller.diagnostics_dict()
        best_cost = None
        if "best_cost" in diagnostics:
            best_cost = float(diagnostics["best_cost"])
            costs.append(best_cost)
        if "predicted_obs" in diagnostics:
            predicted_obs.append(np.asarray(diagnostics["predicted_obs"], dtype=np.float32))
        if "predicted_reward" in diagnostics:
            predicted_reward.append(np.asarray(diagnostics["predicted_reward"], dtype=np.float32))

        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        actions.append(action.astype(np.float32).copy())
        rewards.append(float(reward))

        env.render()
        visualizer.update(episode_idx, task_name, target_xy, observations, actions, best_cost=best_cost)
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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["center", "viapoint", "random"], default="center")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--num-candidates", type=int, default=1024)
    parser.add_argument("--num-elites", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--planning-objective", choices=["state_cost", "reward", "hybrid"], default="state_cost")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-bound", type=positive_float, default=0.18)
    parser.add_argument("--pos-bound", type=positive_float, default=0.25)
    parser.add_argument("--vel-bound", type=positive_float, default=0.10)
    parser.add_argument("--angle-bound", type=positive_float, default=0.05)
    parser.add_argument("--x-bound", type=positive_float, default=None)
    parser.add_argument("--y-bound", type=positive_float, default=None)
    parser.add_argument("--vx-bound", type=positive_float, default=None)
    parser.add_argument("--vy-bound", type=positive_float, default=None)
    parser.add_argument("--theta-x-bound", type=positive_float, default=None)
    parser.add_argument("--theta-y-bound", type=positive_float, default=None)
    parser.add_argument("--plot-window", type=int, default=150)
    parser.add_argument("--plot-pause", type=float, default=0.001)
    parser.add_argument("--final-threshold", type=float, default=0.06)
    parser.add_argument("--last-window-threshold", type=float, default=0.08)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--save-episodes", action="store_true")
    args = parser.parse_args()
    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.plot_window <= 0:
        parser.error("--plot-window must be positive")
    return args


def bounds_from_args(args: argparse.Namespace) -> OnlineInitialBounds:
    return OnlineInitialBounds(
        x=args.pos_bound if args.x_bound is None else args.x_bound,
        y=args.pos_bound if args.y_bound is None else args.y_bound,
        vx=args.vel_bound if args.vx_bound is None else args.vx_bound,
        vy=args.vel_bound if args.vy_bound is None else args.vy_bound,
        theta_x=args.angle_bound if args.theta_x_bound is None else args.theta_x_bound,
        theta_y=args.angle_bound if args.theta_y_bound is None else args.theta_y_bound,
    )


def main() -> None:
    args = parse_args()
    ensure_writable_matplotlib_cache()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "online_mpc_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    bounds = bounds_from_args(args)
    metrics: list[dict[str, float | bool]] = []
    episode_records: list[dict[str, float | bool | str]] = []

    env = BallBalanceEnv(render_mode="human", config={"max_episode_steps": args.max_steps})
    visualizer = SlidingWindowVisualizer(window=args.plot_window, pause_s=args.plot_pause)
    try:
        controller = RSSMMPCController(
            checkpoint_path=args.checkpoint,
            action_space=env.action_space,
            horizon=args.horizon,
            num_candidates=args.num_candidates,
            num_elites=args.num_elites,
            num_iterations=args.num_iterations,
            device=args.device,
            cost_mode="center",
            planning_objective=args.planning_objective,
            seed=args.seed,
        )

        for episode_idx in range(args.num_episodes):
            task_name, target_xy = select_episode_task(rng, args.task, args.target_bound)
            controller.cost_mode = "center" if task_name == "center" else "point"
            controller.target_xy = None if task_name == "center" else target_xy

            initial_state = sample_online_initial_state(rng, bounds)
            episode = run_online_episode(
                controller=controller,
                env=env,
                visualizer=visualizer,
                initial_state=initial_state,
                max_steps=args.max_steps,
                target_xy=target_xy,
                episode_idx=episode_idx,
                task_name=task_name,
            )

            if args.save_episodes:
                save_episode_npz(episode, out_dir / f"episode_{episode_idx:03d}.npz")
            if args.save_plots:
                save_mpc_plots(episode, out_dir / f"episode_{episode_idx:03d}", target_xy=target_xy, title_prefix=f"RSSM MPC {task_name}")

            episode_result = episode_metrics(
                episode,
                target_xy,
                final_threshold=args.final_threshold,
                last_window_threshold=args.last_window_threshold,
            )
            metrics.append(episode_result)
            episode_record: dict[str, float | bool | str] = {
                **episode_result,
                "task": task_name,
                "target_x": float(target_xy[0]),
                "target_y": float(target_xy[1]),
                "initial_x": float(initial_state[0]),
                "initial_y": float(initial_state[1]),
                "initial_vx": float(initial_state[2]),
                "initial_vy": float(initial_state[3]),
                "initial_theta_x": float(initial_state[4]),
                "initial_theta_y": float(initial_state[5]),
            }
            episode_records.append(episode_record)
            print(f"episode={episode_idx} task={task_name} target={target_xy} metrics={episode_result}")

        summary = {
            "episodes": episode_records,
            "aggregate": aggregate_metrics(metrics),
            "config": vars(args) | {"initial_bounds": bounds.__dict__},
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["aggregate"], indent=2))
        print(f"Saved metrics to {out_dir / 'metrics.json'}")
    finally:
        visualizer.close()
        env.close()


if __name__ == "__main__":
    main()
