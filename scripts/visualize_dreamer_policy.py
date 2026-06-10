"""Run and visualize a closed-loop Dreamer policy in the real environment."""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--play", action="store_true", help="Show a live matplotlib visualizer during rollout.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--save-gif", action="store_true", help="Save one animated GIF per visualized episode.")
    parser.add_argument(
        "--gif-episode-limit",
        type=int,
        default=None,
        help="Maximum number of episode GIFs to save. Defaults to all episodes when --save-gif is set.",
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

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "policy_visualizer"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BallBalanceEnv(config={"max_episode_steps": args.max_steps})
    bounds = InitialConditionBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    bounds.validate(env)
    agent = DreamerAgent(
        checkpoint_path=args.checkpoint,
        action_space=env.action_space,
        device=device,
        deterministic=not args.stochastic,
    )
    visualizer = LivePolicyVisualizer(env.config.board_size, args.max_steps, args.fps) if args.play else None
    rng = np.random.default_rng(args.seed)
    metrics = []
    gif_limit = args.num_episodes if args.gif_episode_limit is None else max(args.gif_episode_limit, 0)

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
            episode_metric = episode_metrics(episode)
            metrics.append(episode_metric)
            if args.save_episodes:
                save_episode_npz(episode, out_dir / f"episode_{episode_idx:03d}.npz")
            if args.save_plots:
                save_policy_plots(episode, out_dir / f"episode_{episode_idx:03d}")
            if args.save_gif and episode_idx < gif_limit:
                save_episode_animation(
                    episode=episode,
                    out_path=out_dir / f"episode_{episode_idx:03d}.gif",
                    fps=args.fps,
                    board_size=env.config.board_size,
                )
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
        "episodes": metrics,
        "aggregate": aggregate_metrics(metrics),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"wrote Dreamer policy visualization outputs to {out_dir}")


def run_visualized_policy_episode(
    agent: DreamerAgent,
    env: BallBalanceEnv,
    initial_state: np.ndarray,
    max_steps: int,
    visualizer: "LivePolicyVisualizer | None",
    episode_idx: int,
) -> dict[str, Any]:
    obs, _ = env.reset(options={"state": initial_state.astype(np.float64)})
    agent.reset(initial_obs=obs)
    if visualizer is not None:
        visualizer.reset_episode(episode_idx, obs)

    observations = [obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    diagnostics: list[dict[str, Any]] = []
    total_return = 0.0

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
                step=step + 1,
                obs=next_obs,
                action=action,
                reward=float(reward),
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


class LivePolicyVisualizer:
    """Small online matplotlib dashboard for a real closed-loop rollout."""

    def __init__(self, board_size: float, max_steps: int, fps: float) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.board_size = board_size
        self.max_steps = max_steps
        self.pause = 1.0 / fps
        self.obs_history: list[np.ndarray] = []
        self.action_history: list[np.ndarray] = []
        self.return_history: list[float] = []

        plt.ion()
        self.fig = plt.figure(figsize=(11, 5.5))
        grid = self.fig.add_gridspec(2, 2, width_ratios=(1.05, 1.0))
        self.ax_board = self.fig.add_subplot(grid[:, 0])
        self.ax_return = self.fig.add_subplot(grid[0, 1])
        self.ax_action = self.fig.add_subplot(grid[1, 1])
        self._setup_axes()

        (self.trace_line,) = self.ax_board.plot([], [], color="tab:blue", linewidth=2, label="trajectory")
        (self.ball_line,) = self.ax_board.plot([], [], "o", color="tab:red", markersize=10, label="ball")
        (self.return_line,) = self.ax_return.plot([], [], color="tab:green", linewidth=2)
        (self.action_x_line,) = self.ax_action.plot([], [], label="theta_x_cmd", color="tab:orange")
        (self.action_y_line,) = self.ax_action.plot([], [], label="theta_y_cmd", color="tab:purple")
        self.text = self.ax_board.text(0.02, 0.98, "", transform=self.ax_board.transAxes, va="top")
        self.ax_board.legend(loc="lower right")
        self.ax_action.legend(loc="upper right")
        self.fig.tight_layout()

    def _setup_axes(self) -> None:
        half = self.board_size / 2.0
        add_board_boundary(self.ax_board, self.board_size)
        self.ax_board.scatter([0.0], [0.0], marker="x", color="black", linewidths=1.5)
        self.ax_board.set_xlim(-half * 1.15, half * 1.15)
        self.ax_board.set_ylim(-half * 1.15, half * 1.15)
        self.ax_board.set_xlabel("x [m]")
        self.ax_board.set_ylabel("y [m]")
        self.ax_board.grid(True, color="0.9")

        self.ax_return.set_xlim(0, self.max_steps)
        self.ax_return.set_ylim(-10, self.max_steps)
        self.ax_return.set_xlabel("step")
        self.ax_return.set_ylabel("cumulative return")
        self.ax_return.grid(True, color="0.9")

        self.ax_action.set_xlim(0, self.max_steps)
        self.ax_action.set_ylim(-0.30, 0.30)
        self.ax_action.set_xlabel("step")
        self.ax_action.set_ylabel("action [rad]")
        self.ax_action.grid(True, color="0.9")

    def reset_episode(self, episode_idx: int, obs: np.ndarray) -> None:
        self.obs_history = [obs.astype(np.float32, copy=True)]
        self.action_history = []
        self.return_history = [0.0]
        self.trace_line.set_data([obs[0]], [obs[1]])
        self.ball_line.set_data([obs[0]], [obs[1]])
        self.return_line.set_data([0], [0.0])
        self.action_x_line.set_data([], [])
        self.action_y_line.set_data([], [])
        self.text.set_text(f"episode={episode_idx:03d}  step=0")
        self.fig.canvas.draw_idle()
        self.plt.pause(self.pause)

    def update(
        self,
        step: int,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        total_return: float,
        done: bool,
    ) -> None:
        self.obs_history.append(obs.astype(np.float32, copy=True))
        self.action_history.append(action.reshape(-1).astype(np.float32, copy=True))
        self.return_history.append(total_return)

        obs_arr = np.asarray(self.obs_history, dtype=np.float32)
        action_arr = np.asarray(self.action_history, dtype=np.float32)
        returns = np.asarray(self.return_history, dtype=np.float32)
        action_steps = np.arange(action_arr.shape[0])

        self.trace_line.set_data(obs_arr[:, 0], obs_arr[:, 1])
        self.ball_line.set_data([obs_arr[-1, 0]], [obs_arr[-1, 1]])
        self.return_line.set_data(np.arange(returns.shape[0]), returns)
        if action_arr.size:
            self.action_x_line.set_data(action_steps, action_arr[:, 0])
            self.action_y_line.set_data(action_steps, action_arr[:, 1])

        self.ax_return.set_ylim(min(-10.0, float(returns.min()) - 5.0), max(10.0, float(returns.max()) + 5.0))
        distance = float(np.linalg.norm(obs_arr[-1, :2]))
        status = "done" if done else "running"
        self.text.set_text(
            f"step={step}  return={total_return:.2f}  reward={reward:.2f}\n"
            f"distance={distance:.3f}  {status}"
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(self.pause)

    def close(self) -> None:
        self.plt.ioff()


def save_episode_animation(episode: dict[str, Any], out_path: Path, fps: float, board_size: float) -> None:
    """Save a replay animation of one closed-loop policy episode."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    obs = episode["obs"]
    action = episode["action"]
    reward = episode["reward"].reshape(-1)
    returns = np.concatenate([[0.0], np.cumsum(reward)])
    half = board_size / 2.0

    fig = plt.figure(figsize=(11, 5.5))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.0))
    ax_board = fig.add_subplot(grid[:, 0])
    ax_return = fig.add_subplot(grid[0, 1])
    ax_action = fig.add_subplot(grid[1, 1])

    add_board_boundary(ax_board, board_size)
    ax_board.scatter([0.0], [0.0], marker="x", color="black", linewidths=1.5)
    ax_board.set_xlim(-half * 1.15, half * 1.15)
    ax_board.set_ylim(-half * 1.15, half * 1.15)
    ax_board.set_xlabel("x [m]")
    ax_board.set_ylabel("y [m]")
    ax_board.grid(True, color="0.9")

    ax_return.set_xlim(0, max(obs.shape[0] - 1, 1))
    ax_return.set_ylim(min(-10.0, float(returns.min()) - 5.0), max(10.0, float(returns.max()) + 5.0))
    ax_return.set_xlabel("step")
    ax_return.set_ylabel("cumulative return")
    ax_return.grid(True, color="0.9")

    ax_action.set_xlim(0, max(action.shape[0], 1))
    ax_action.set_ylim(-0.30, 0.30)
    ax_action.set_xlabel("step")
    ax_action.set_ylabel("action [rad]")
    ax_action.grid(True, color="0.9")

    (trace_line,) = ax_board.plot([], [], color="tab:blue", linewidth=2, label="trajectory")
    (ball_line,) = ax_board.plot([], [], "o", color="tab:red", markersize=10, label="ball")
    (return_line,) = ax_return.plot([], [], color="tab:green", linewidth=2)
    (action_x_line,) = ax_action.plot([], [], label="theta_x_cmd", color="tab:orange")
    (action_y_line,) = ax_action.plot([], [], label="theta_y_cmd", color="tab:purple")
    text = ax_board.text(0.02, 0.98, "", transform=ax_board.transAxes, va="top")
    ax_board.legend(loc="lower right")
    ax_action.legend(loc="upper right")
    fig.tight_layout()

    def update(frame: int):
        trace_line.set_data(obs[: frame + 1, 0], obs[: frame + 1, 1])
        ball_line.set_data([obs[frame, 0]], [obs[frame, 1]])
        return_line.set_data(np.arange(frame + 1), returns[: frame + 1])
        if frame > 0 and action.size:
            action_steps = np.arange(frame)
            action_x_line.set_data(action_steps, action[:frame, 0])
            action_y_line.set_data(action_steps, action[:frame, 1])
        distance = np.linalg.norm(obs[frame, :2])
        text.set_text(f"step={frame}  return={returns[frame]:.2f}  distance={distance:.3f}")
        return trace_line, ball_line, return_line, action_x_line, action_y_line, text

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=obs.shape[0], interval=1000.0 / fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=int(round(fps))))
    plt.close(fig)
    print(f"Saved policy animation to {out_path}")


if __name__ == "__main__":
    main()
