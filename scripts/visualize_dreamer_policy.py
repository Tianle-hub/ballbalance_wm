"""Run and visualize a closed-loop Dreamer policy in the real environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.agent import DreamerAgent
from ball_rssm.envs import BallBalanceEnv
from ball_rssm.evaluation import (
    InitialConditionBounds,
    aggregate_metrics,
    episode_metrics,
    sample_initial_state,
    save_episode_npz,
    save_policy_plots,
)
from ball_rssm.utils.plotting import add_board_boundary
from ball_rssm.utils.seed import set_seed


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-bound", type=float, default=0.25)
    parser.add_argument("--vel-bound", type=float, default=0.20)
    parser.add_argument("--angle-bound", type=float, default=0.12)
    parser.add_argument("--stochastic", action="store_true", help="Sample from the actor instead of using its mode.")
    parser.add_argument("--play", action="store_true", help="Deprecated alias; use --show-online show.")
    parser.add_argument(
        "--show-online",
        choices=["show", "not-show"],
        default="show",
        help="Show the single combined online dashboard while running, or use not-show for headless execution.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--save-gif", action="store_true", help="Deprecated; one combined GIF is always saved.")
    parser.add_argument(
        "--gif-path",
        default=None,
        help="Path for the combined multi-episode GIF. Defaults to OUT_DIR/dreamer_policy_all_episodes.gif.",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=None,
        help="FPS for the saved GIF. Defaults to --fps.",
    )
    parser.add_argument(
        "--gif-episode-limit",
        type=int,
        default=None,
        help="Deprecated; accepted for compatibility.",
    )
    parser.add_argument("--save-episodes", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.fps <= 0.0:
        raise ValueError("fps must be positive")
    if args.gif_fps is not None and args.gif_fps <= 0.0:
        raise ValueError("gif_fps must be positive")

    ensure_writable_matplotlib_cache()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "policy_visualizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = Path(args.gif_path) if args.gif_path else out_dir / "dreamer_policy_all_episodes.gif"
    gif_fps = float(args.gif_fps if args.gif_fps is not None else args.fps)
    show_online = args.show_online == "show"

    env = BallBalanceEnv(config={"max_episode_steps": args.max_steps})
    bounds = InitialConditionBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    bounds.validate(env)
    agent = DreamerAgent(
        checkpoint_path=args.checkpoint,
        action_space=env.action_space,
        device=device,
        deterministic=not args.stochastic,
    )
    visualizer = (
        CombinedPolicyVisualizer(
            num_episodes=args.num_episodes,
            board_size=env.config.board_size,
            max_steps=args.max_steps,
            fps=args.fps,
        )
        if show_online
        else None
    )
    rng = np.random.default_rng(args.seed)
    metrics = []
    episodes: list[dict[str, Any]] = []

    try:
        for episode_idx in range(args.num_episodes):
            initial_state = sample_initial_state(rng, bounds)
            episode = run_visualized_policy_episode(
                agent=agent,
                env=env,
                initial_state=initial_state,
                max_steps=args.max_steps,
                visualizer=visualizer,
                episode_idx=episode_idx,
            )
            episodes.append(episode)
            episode_metric = episode_metrics(episode)
            metrics.append(episode_metric)
            if args.save_episodes:
                save_episode_npz(episode, out_dir / f"episode_{episode_idx:03d}.npz")
            if args.save_plots:
                save_policy_plots(episode, out_dir / f"episode_{episode_idx:03d}")
            print(
                f"episode={episode_idx:03d} return={episode_metric['return']:.3f} "
                f"steps={episode_metric['steps']} final_distance={episode_metric['final_distance']:.4f} "
                f"terminated={episode_metric['terminated']}"
            )
    finally:
        env.close()
        if visualizer is not None:
            visualizer.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "bounds": {"pos": args.pos_bound, "vel": args.vel_bound, "angle": args.angle_bound},
        "gif_path": str(gif_path),
        "episodes": metrics,
        "aggregate": aggregate_metrics(metrics),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_combined_policy_gif(
        episodes=episodes,
        out_path=gif_path,
        fps=gif_fps,
        board_size=env.config.board_size,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved combined Dreamer policy GIF to {gif_path}")
    print(f"wrote Dreamer policy visualization outputs to {out_dir}")


def run_visualized_policy_episode(
    agent: DreamerAgent,
    env: BallBalanceEnv,
    initial_state: np.ndarray,
    max_steps: int,
    visualizer: "CombinedPolicyVisualizer | None",
    episode_idx: int,
) -> dict[str, Any]:
    obs, _ = env.reset(options={"state": initial_state.astype(np.float64)})
    agent.reset(initial_obs=obs)

    observations = [obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    diagnostics: list[dict[str, Any]] = []
    total_return = 0.0
    if visualizer is not None:
        visualizer.update(
            episode_idx=episode_idx,
            step=0,
            observations=observations,
            actions=actions,
            rewards=rewards,
            total_return=total_return,
            done=False,
        )

    for step in range(max_steps):
        action = agent.act(observations[-1])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)

        actions.append(action.astype(np.float32, copy=True))
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        diagnostics.append(dict(agent.last_diagnostics))
        observations.append(next_obs.copy())

        if visualizer is not None:
            visualizer.update(
                episode_idx=episode_idx,
                step=step + 1,
                observations=observations,
                actions=actions,
                rewards=rewards,
                total_return=total_return,
                done=bool(terminated or truncated),
            )
        if terminated or truncated:
            break

    terminated_arr = np.asarray(terminated_flags, dtype=bool)[:, None]
    truncated_arr = np.asarray(truncated_flags, dtype=bool)[:, None]
    return {
        "obs": np.asarray(observations, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32)[:, None],
        "terminated": terminated_arr,
        "truncated": truncated_arr,
        "done": terminated_arr | truncated_arr,
        "diagnostics": diagnostics,
        "initial_state": initial_state.astype(np.float32, copy=True),
    }


class CombinedPolicyVisualizer:
    """Single-window online dashboard with one row per Dreamer episode."""

    def __init__(self, num_episodes: int, board_size: float, max_steps: int, fps: float) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.pause = 1.0 / fps
        plt.ion()
        self.fig, axes = plt.subplots(
            num_episodes,
            3,
            figsize=(15, max(3.5, 3.0 * num_episodes)),
            squeeze=False,
            num="Dreamer policy online episodes",
        )
        self.rows = setup_combined_policy_axes(axes, num_episodes, board_size, max_steps)
        self.fig.tight_layout()

    def update(
        self,
        episode_idx: int,
        step: int,
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        rewards: list[float],
        total_return: float,
        done: bool,
    ) -> None:
        obs = np.asarray(observations, dtype=np.float32)
        action = np.asarray(actions, dtype=np.float32)
        reward = np.asarray(rewards, dtype=np.float32)
        status = "done" if done else "running"
        update_combined_policy_row(
            self.rows[episode_idx],
            obs=obs,
            action=action,
            reward=reward,
            frame=step,
            status=f"episode={episode_idx:03d} step={step} return={total_return:.2f} {status}",
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(self.pause)

    def close(self) -> None:
        self.plt.close(self.fig)


def setup_combined_policy_axes(axes, num_episodes: int, board_size: float, max_steps: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    half = board_size / 2.0
    board_limit = half * 1.15
    for episode_idx in range(num_episodes):
        board_ax, return_ax, action_ax = axes[episode_idx]

        add_board_boundary(board_ax, board_size)
        board_ax.scatter([0.0], [0.0], marker="x", color="black", linewidths=1.5)
        (trace_line,) = board_ax.plot([], [], color="tab:blue", linewidth=2, label="trajectory")
        (ball_line,) = board_ax.plot([], [], "o", color="tab:red", markersize=8, label="ball")
        status_text = board_ax.text(0.02, 0.98, "", transform=board_ax.transAxes, va="top", fontsize=8)
        board_ax.set_xlim(-board_limit, board_limit)
        board_ax.set_ylim(-board_limit, board_limit)
        board_ax.set_aspect("equal", adjustable="box")
        board_ax.set_title(f"episode {episode_idx}")
        board_ax.set_xlabel("x [m]")
        board_ax.set_ylabel("y [m]")
        board_ax.grid(True, color="0.9")
        board_ax.legend(loc="lower right", fontsize=8)

        (return_line,) = return_ax.plot([], [], color="tab:green", linewidth=1.8)
        return_ax.set_xlim(0, max_steps)
        return_ax.set_ylim(-10.0, max(10.0, float(max_steps)))
        return_ax.set_title("cumulative return")
        return_ax.set_xlabel("step")
        return_ax.set_ylabel("return")
        return_ax.grid(True, color="0.9")

        (action_x_line,) = action_ax.plot([], [], label="theta_x", color="tab:orange", linewidth=1.4)
        (action_y_line,) = action_ax.plot([], [], label="theta_y", color="tab:purple", linewidth=1.4)
        action_ax.set_xlim(0, max_steps)
        action_ax.set_ylim(-0.30, 0.30)
        action_ax.set_title("action")
        action_ax.set_xlabel("step")
        action_ax.set_ylabel("rad")
        action_ax.grid(True, color="0.9")
        action_ax.legend(loc="upper right", fontsize=8)

        rows.append(
            {
                "trace_line": trace_line,
                "ball_line": ball_line,
                "status_text": status_text,
                "return_line": return_line,
                "return_ax": return_ax,
                "action_x_line": action_x_line,
                "action_y_line": action_y_line,
            }
        )
    return rows


def update_combined_policy_row(
    row: dict[str, object],
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    frame: int,
    status: str,
) -> tuple[object, ...]:
    frame = min(max(frame, 0), max(obs.shape[0] - 1, 0))
    obs_until = obs[: frame + 1]
    action_until = action[: min(frame, action.shape[0])]
    returns = np.concatenate([[0.0], np.cumsum(reward.reshape(-1), dtype=np.float32)])
    returns_until = returns[: frame + 1]
    steps = np.arange(obs_until.shape[0])
    action_steps = np.arange(action_until.shape[0])

    row["trace_line"].set_data(obs_until[:, 0], obs_until[:, 1])
    row["ball_line"].set_data([obs_until[-1, 0]], [obs_until[-1, 1]])
    row["status_text"].set_text(status)
    row["return_line"].set_data(steps, returns_until)
    if returns_until.size:
        return_ax = row["return_ax"]
        low = min(-10.0, float(returns_until.min()) - 5.0)
        high = max(10.0, float(returns_until.max()) + 5.0)
        return_ax.set_ylim(low, high)

    if action_until.size:
        row["action_x_line"].set_data(action_steps, action_until[:, 0])
        row["action_y_line"].set_data(action_steps, action_until[:, 1])
    else:
        row["action_x_line"].set_data([], [])
        row["action_y_line"].set_data([], [])

    return (
        row["trace_line"],
        row["ball_line"],
        row["status_text"],
        row["return_line"],
        row["action_x_line"],
        row["action_y_line"],
    )


def save_combined_policy_gif(
    episodes: list[dict[str, Any]],
    out_path: Path,
    fps: float,
    board_size: float,
    max_steps: int,
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
    rows = setup_combined_policy_axes(axes, len(episodes), board_size, max_steps)
    fig.suptitle("Dreamer policy rollout replay", fontsize=14)
    fig.tight_layout()
    max_frames = max(np.asarray(episode["obs"]).shape[0] for episode in episodes)

    def update(frame: int) -> tuple[object, ...]:
        artists: list[object] = []
        for episode_idx, episode in enumerate(episodes):
            obs = np.asarray(episode["obs"], dtype=np.float32)
            action = np.asarray(episode["action"], dtype=np.float32)
            reward = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
            display_frame = min(frame, obs.shape[0] - 1)
            terminated = bool(episode["terminated"].any()) if episode["terminated"].size else False
            truncated = bool(episode["truncated"].any()) if episode["truncated"].size else False
            status = f"episode={episode_idx:03d} step={display_frame} terminated={terminated} truncated={truncated}"
            artists.extend(
                update_combined_policy_row(
                    rows[episode_idx],
                    obs=obs,
                    action=action,
                    reward=reward,
                    frame=display_frame,
                    status=status,
                )
            )
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=max_frames, interval=1000.0 / fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=max(1, int(round(fps)))))
    plt.close(fig)


if __name__ == "__main__":
    main()
