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
from ball_rssm.planning.rollout import aggregate_metrics, episode_metrics, save_episode_npz, save_mpc_plots
from ball_rssm.planning.rssm_mpc import RSSMMPCController

TaskMode = Literal["center", "viapoint", "random"]


@dataclass(frozen=True)
class OnlineInitialBounds:
    """Symmetric sampling bounds for the 6D simulator state."""

    x: float = 0.45
    y: float = 0.45
    vx: float = 0.20
    vy: float = 0.20
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


class CombinedEpisodeVisualizer:
    """Single-window online dashboard with one row per episode."""

    def __init__(self, num_episodes: int, max_steps: int, board_size: float, pause_s: float) -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        self._plt = plt
        self.max_steps = max_steps
        self.fig, axes = plt.subplots(
            num_episodes,
            3,
            figsize=(15, max(3.5, 3.0 * num_episodes)),
            squeeze=False,
            num="RSSM MPC online episodes",
        )
        self.pause_s = pause_s
        self.rows = setup_combined_animation_axes(axes, num_episodes, max_steps, board_size)
        self.fig.tight_layout()

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
        cost_text = "" if best_cost is None else f" cost={best_cost:.3f}"
        status = f"episode={episode_idx} task={task_name} step={obs.shape[0] - 1}{cost_text}"
        update_combined_animation_row(
            self.rows[episode_idx],
            obs=obs,
            action=action,
            target_xy=target_xy,
            frame=obs.shape[0] - 1,
            status=status,
        )
        self.fig.canvas.draw_idle()
        self._plt.pause(self.pause_s)

    def close(self) -> None:
        self._plt.close(self.fig)


def setup_combined_animation_axes(axes, num_episodes: int, max_steps: int, board_size: float) -> list[dict[str, object]]:
    from matplotlib.patches import Rectangle

    rows: list[dict[str, object]] = []
    half = board_size / 2.0
    board_limit = half * 1.15
    distance_limit = board_size * np.sqrt(2.0)
    for episode_idx in range(num_episodes):
        board_ax, distance_ax, action_ax = axes[episode_idx]

        board_ax.add_patch(Rectangle((-half, -half), board_size, board_size, fill=False, linewidth=2, edgecolor="black"))
        board_ax.axhline(0.0, color="0.85", linewidth=1)
        board_ax.axvline(0.0, color="0.85", linewidth=1)
        (trajectory_line,) = board_ax.plot([], [], color="black", linewidth=1.6, alpha=0.85)
        (ball_line,) = board_ax.plot([], [], "o", color="tab:red", markersize=8)
        (target_line,) = board_ax.plot([], [], "x", color="tab:green", markersize=9, markeredgewidth=2)
        status_text = board_ax.text(0.02, 0.98, "", transform=board_ax.transAxes, va="top", fontsize=8)
        board_ax.set_xlim(-board_limit, board_limit)
        board_ax.set_ylim(-board_limit, board_limit)
        board_ax.set_aspect("equal", adjustable="box")
        board_ax.set_title(f"episode {episode_idx}")
        board_ax.set_xlabel("x [m]")
        board_ax.set_ylabel("y [m]")
        board_ax.grid(True, color="0.9", linewidth=0.8)

        (distance_line,) = distance_ax.plot([], [], color="tab:blue", linewidth=1.6)
        distance_ax.set_xlim(0, max_steps)
        distance_ax.set_ylim(0.0, max(0.1, distance_limit))
        distance_ax.set_title("distance")
        distance_ax.set_xlabel("step")
        distance_ax.set_ylabel("m")
        distance_ax.grid(True, color="0.9", linewidth=0.8)

        (theta_x_line,) = action_ax.plot([], [], color="tab:orange", linewidth=1.4, label="theta_x")
        (theta_y_line,) = action_ax.plot([], [], color="tab:purple", linewidth=1.4, label="theta_y")
        action_ax.set_xlim(0, max_steps)
        action_ax.set_ylim(-0.3, 0.3)
        action_ax.set_title("action")
        action_ax.set_xlabel("step")
        action_ax.set_ylabel("rad")
        action_ax.grid(True, color="0.9", linewidth=0.8)
        action_ax.legend(loc="upper right", fontsize=8)

        rows.append(
            {
                "trajectory_line": trajectory_line,
                "ball_line": ball_line,
                "target_line": target_line,
                "status_text": status_text,
                "distance_line": distance_line,
                "theta_x_line": theta_x_line,
                "theta_y_line": theta_y_line,
            }
        )
    return rows


def update_combined_animation_row(
    row: dict[str, object],
    obs: np.ndarray,
    action: np.ndarray,
    target_xy: tuple[float, float],
    frame: int,
    status: str,
) -> tuple[object, ...]:
    frame = min(max(frame, 0), max(obs.shape[0] - 1, 0))
    obs_until = obs[: frame + 1]
    action_until = action[: min(frame, action.shape[0])]
    target = np.asarray(target_xy, dtype=np.float32)
    distance = np.linalg.norm(obs_until[:, :2] - target, axis=-1)
    steps = np.arange(obs_until.shape[0])
    action_steps = np.arange(action_until.shape[0])

    row["trajectory_line"].set_data(obs_until[:, 0], obs_until[:, 1])
    row["ball_line"].set_data([obs_until[-1, 0]], [obs_until[-1, 1]])
    row["target_line"].set_data([target_xy[0]], [target_xy[1]])
    row["status_text"].set_text(status)
    row["distance_line"].set_data(steps, distance)
    if action_until.size:
        row["theta_x_line"].set_data(action_steps, action_until[:, 0])
        row["theta_y_line"].set_data(action_steps, action_until[:, 1])
    else:
        row["theta_x_line"].set_data([], [])
        row["theta_y_line"].set_data([], [])

    return (
        row["trajectory_line"],
        row["ball_line"],
        row["target_line"],
        row["status_text"],
        row["distance_line"],
        row["theta_x_line"],
        row["theta_y_line"],
    )


def save_combined_episode_gif(
    episodes: list[dict[str, np.ndarray | float | bool]],
    episode_records: list[dict[str, float | bool | str]],
    out_path: Path,
    fps: float,
    max_steps: int,
    board_size: float,
) -> None:
    if not episodes:
        return
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    ensure_writable_matplotlib_cache()
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    fig, axes = plt.subplots(
        len(episodes),
        3,
        figsize=(15, max(3.5, 3.0 * len(episodes))),
        squeeze=False,
    )
    rows = setup_combined_animation_axes(axes, len(episodes), max_steps, board_size)
    fig.suptitle("RSSM MPC online rollout replay", fontsize=14)
    fig.tight_layout()

    max_frames = max(np.asarray(episode["obs"]).shape[0] for episode in episodes)

    def update(frame: int) -> tuple[object, ...]:
        artists: list[object] = []
        for episode_idx, episode in enumerate(episodes):
            obs = np.asarray(episode["obs"], dtype=np.float32)
            action = np.asarray(episode["action"], dtype=np.float32)
            record = episode_records[episode_idx]
            target_xy = (float(record["target_x"]), float(record["target_y"]))
            task = str(record["task"])
            success = bool(record["success"])
            fell = bool(record["fell"])
            display_frame = min(frame, obs.shape[0] - 1)
            status = f"episode={episode_idx} task={task} step={display_frame} success={success} fell={fell}"
            artists.extend(
                update_combined_animation_row(
                    rows[episode_idx],
                    obs=obs,
                    action=action,
                    target_xy=target_xy,
                    frame=display_frame,
                    status=status,
                )
            )
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=max_frames, interval=1000.0 / fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=max(1, int(round(fps)))))
    plt.close(fig)


def run_online_episode(
    controller: RSSMMPCController,
    env: BallBalanceEnv,
    visualizer: CombinedEpisodeVisualizer | None,
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
    predicted_continue: list[np.ndarray] = []
    computation_times: list[float] = []
    terminated = False
    truncated = False

    if visualizer is not None:
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
        if "predicted_continue" in diagnostics:
            predicted_continue.append(np.asarray(diagnostics["predicted_continue"], dtype=np.float32))

        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        actions.append(action.astype(np.float32).copy())
        rewards.append(float(reward))

        if visualizer is not None:
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
    pred_continue_arr = (
        np.asarray(predicted_continue, dtype=np.float32) if predicted_continue else np.zeros((0, controller.horizon, 1), dtype=np.float32)
    )
    distance = np.linalg.norm(obs_arr[:, :2] - np.asarray(target_xy, dtype=np.float32), axis=-1)

    return {
        "obs": obs_arr,
        "action": action_arr,
        "reward": reward_arr,
        "cost": np.asarray(costs, dtype=np.float32),
        "predicted_obs": pred_arr,
        "predicted_reward": pred_reward_arr,
        "predicted_continue": pred_continue_arr,
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
    parser.add_argument("--planner-type", choices=["cem", "cem_gd"], default="cem")
    parser.add_argument("--gd-num-sequences", type=int, default=3)
    parser.add_argument("--gd-iterations", type=int, default=15)
    parser.add_argument("--gd-lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-bound", type=positive_float, default=0.18)
    parser.add_argument("--pos-bound", type=positive_float, default=0.45)
    parser.add_argument("--vel-bound", type=positive_float, default=0.25)
    parser.add_argument("--angle-bound", type=positive_float, default=0.05)
    parser.add_argument("--x-bound", type=positive_float, default=None)
    parser.add_argument("--y-bound", type=positive_float, default=None)
    parser.add_argument("--vx-bound", type=positive_float, default=None)
    parser.add_argument("--vy-bound", type=positive_float, default=None)
    parser.add_argument("--theta-x-bound", type=positive_float, default=None)
    parser.add_argument("--theta-y-bound", type=positive_float, default=None)
    parser.add_argument("--plot-window", type=int, default=150, help="Deprecated; accepted for compatibility.")
    parser.add_argument("--plot-pause", type=float, default=0.001)
    parser.add_argument(
        "--show-online",
        choices=["show", "not-show"],
        default="show",
        help="Show the single combined online dashboard while running, or use not-show for headless execution.",
    )
    parser.add_argument(
        "--gif-path",
        default=None,
        help="Path for the combined multi-episode GIF. Defaults to OUT_DIR/online_mpc_all_episodes.gif.",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=None,
        help="FPS for the saved GIF. Defaults to --render-fps.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=120.0,
        help="Interactive environment render refresh rate; does not change simulator dt.",
    )
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
    if args.render_fps <= 0.0:
        parser.error("--render-fps must be positive")
    if args.gif_fps is not None and args.gif_fps <= 0.0:
        parser.error("--gif-fps must be positive")
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
    gif_path = Path(args.gif_path) if args.gif_path else out_dir / "online_mpc_all_episodes.gif"
    gif_fps = float(args.gif_fps if args.gif_fps is not None else args.render_fps)
    show_online = args.show_online == "show"

    rng = np.random.default_rng(args.seed)
    bounds = bounds_from_args(args)
    metrics: list[dict[str, float | bool]] = []
    episode_records: list[dict[str, float | bool | str]] = []
    episodes: list[dict[str, np.ndarray | float | bool]] = []

    env = BallBalanceEnv(render_mode=None, config={"max_episode_steps": args.max_steps}, render_fps=args.render_fps)
    visualizer = (
        CombinedEpisodeVisualizer(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            board_size=env.config.board_size,
            pause_s=args.plot_pause,
        )
        if show_online
        else None
    )
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
            planner_type=args.planner_type,
            gd_num_sequences=args.gd_num_sequences,
            gd_iterations=args.gd_iterations,
            gd_lr=args.gd_lr,
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
            episodes.append(episode)

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
            "config": vars(args) | {"initial_bounds": bounds.__dict__, "gif_path": str(gif_path)},
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        save_combined_episode_gif(
            episodes=episodes,
            episode_records=episode_records,
            out_path=gif_path,
            fps=gif_fps,
            max_steps=args.max_steps,
            board_size=env.config.board_size,
        )
        print(json.dumps(summary["aggregate"], indent=2))
        print(f"Saved metrics to {out_dir / 'metrics.json'}")
        print(f"Saved combined episode GIF to {gif_path}")
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()


if __name__ == "__main__":
    main()
