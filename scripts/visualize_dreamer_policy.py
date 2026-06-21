"""Visualize a dreamer.py policy on the local ball-balance environment."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_config = Path.home() / ".config" / "matplotlib"
    if not os.access(matplotlib_config.parent, os.W_OK):
        fallback_config = Path(tempfile.gettempdir()) / "ballbalance_matplotlib"
        fallback_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(fallback_config)

from dreamer import Dreamer, make_env  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and visualize a trained dreamer.py policy on ball-balance."
    )
    parser.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint", required=True)
    parser.add_argument("--env", type=str, default="ball-balance")
    parser.add_argument("--algo", type=str, default="Dreamerv2", choices=["Dreamerv1", "Dreamerv2"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--show-online", choices=["show", "not-show"], default="show")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif-stride", type=int, default=1)
    parser.add_argument("--gif-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-gpu", action="store_true")

    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--action-noise", type=float, default=0.3)

    parser.add_argument("--cnn-activation-function", type=str, default="relu")
    parser.add_argument("--dense-activation-function", type=str, default="elu")
    parser.add_argument("--obs-embed-size", type=int, default=1024)
    parser.add_argument("--num-units", type=int, default=400)
    parser.add_argument("--deter-size", type=int, default=200)
    parser.add_argument("--stoch-size", type=int, default=30)
    parser.add_argument("--discrete-classes", type=int, default=32)
    parser.add_argument("--use-disc-model", action="store_true")

    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-seq-len", type=int, default=50)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument(
        "--actor-grad",
        type=str,
        default="dynamics",
        choices=["dynamics", "reinforce", "both"],
    )
    parser.add_argument("--actor-grad-mix", type=float, default=0.1)
    parser.add_argument("--actor-ent", type=float, default=1e-4)

    parser.add_argument("--free-nats", type=float, default=3.0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--kl-loss-coeff", type=float, default=1.0)
    parser.add_argument("--kl-alpha", type=float, default=0.8)
    parser.add_argument("--disc-loss-coeff", type=float, default=10.0)

    parser.add_argument("--model_learning-rate", type=float, default=6e-4)
    parser.add_argument("--actor_learning-rate", type=float, default=8e-5)
    parser.add_argument("--value_learning-rate", type=float, default=8e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-7)
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.checkpoint_path = args.checkpoint
    args.restore = True
    args.train = False
    args.evaluate = True
    args.max_episode_length = args.max_steps
    args.test = False
    args.test_interval = 10000
    args.test_episodes = args.num_episodes
    args.scalar_freq = 1000
    args.log_video_freq = -1
    args.max_videos_to_save = args.num_episodes
    args.checkpoint_interval = 10000
    args.experience_replay = ""
    args.render = False
    return args


def episode_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = Path(args.checkpoint).expanduser()
    default_out_dir = checkpoint.parent.parent / "policy_visualization"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = Path(args.gif_path).expanduser() if args.gif_path else out_dir / "dreamer_ball_policy.gif"
    metrics_path = out_dir / "dreamer_ball_policy_metrics.json"
    return gif_path, metrics_path


def find_ball_config(env):
    current = env
    while current is not None:
        if current.__class__.__name__ == "BallBalanceDreamer":
            return current._env.config
        current = getattr(current, "_env", None)
    return None


class BallPolicyDashboard:
    def __init__(self, num_episodes: int, max_steps: int, board_size: float, max_angle: float, show: bool):
        self.show = show
        if not show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.max_steps = max_steps
        self.board_size = board_size
        self.max_angle = max_angle
        self.fig, self.axes = plt.subplots(
            num_episodes,
            5,
            figsize=(18, max(3.2, 2.75 * num_episodes)),
            squeeze=False,
            constrained_layout=True,
        )
        self.rows = []
        for row in range(num_episodes):
            self.rows.append(self._setup_row(row))
        if show:
            plt.ion()
            self.fig.show()

    def _setup_row(self, row: int) -> dict[str, object]:
        board_ax, pos_ax, vel_ax, angle_ax, action_ax = self.axes[row]
        half = self.board_size / 2.0

        board_ax.set_title(f"episode {row + 1}")
        board_ax.set_xlim(-half * 1.15, half * 1.15)
        board_ax.set_ylim(-half * 1.15, half * 1.15)
        board_ax.set_aspect("equal", adjustable="box")
        board_ax.axhline(0.0, color="0.85", linewidth=1)
        board_ax.axvline(0.0, color="0.85", linewidth=1)
        board_ax.add_patch(
            self.plt.Rectangle(
                (-half, -half),
                self.board_size,
                self.board_size,
                fill=False,
                linewidth=1.8,
                edgecolor="black",
            )
        )
        board_ax.grid(True, color="0.92", linewidth=0.8)
        (trajectory_line,) = board_ax.plot([], [], color="black", linewidth=1.5)
        (ball_dot,) = board_ax.plot([], [], "o", color="#d62728", markersize=8)

        for axis in (pos_ax, vel_ax, angle_ax, action_ax):
            axis.set_xlim(0, max(1, self.max_steps))
            axis.grid(True, color="0.9", linewidth=0.8)
            axis.tick_params(labelsize=8)

        pos_ax.set_title("position")
        pos_ax.set_ylim(-half * 1.2, half * 1.2)
        (x_line,) = pos_ax.plot([], [], label="x", color="#1f77b4")
        (y_line,) = pos_ax.plot([], [], label="y", color="#ff7f0e")
        pos_ax.legend(loc="upper right", fontsize=7)

        vel_ax.set_title("velocity")
        vel_ax.set_ylim(-5.0, 5.0)
        (vx_line,) = vel_ax.plot([], [], label="vx", color="#2ca02c")
        (vy_line,) = vel_ax.plot([], [], label="vy", color="#9467bd")
        vel_ax.legend(loc="upper right", fontsize=7)

        angle_ax.set_title("board angle")
        angle_ax.set_ylim(-self.max_angle * 1.15, self.max_angle * 1.15)
        (theta_x_line,) = angle_ax.plot([], [], label="theta_x", color="#8c564b")
        (theta_y_line,) = angle_ax.plot([], [], label="theta_y", color="#e377c2")
        angle_ax.legend(loc="upper right", fontsize=7)

        action_ax.set_title("input")
        action_ax.set_ylim(-self.max_angle * 1.15, self.max_angle * 1.15)
        (ax_line,) = action_ax.plot([], [], label="cmd_x", color="#bcbd22")
        (ay_line,) = action_ax.plot([], [], label="cmd_y", color="#17becf")
        action_ax.legend(loc="upper right", fontsize=7)

        return {
            "board_ax": board_ax,
            "trajectory_line": trajectory_line,
            "ball_dot": ball_dot,
            "pos_lines": (x_line, y_line),
            "vel_lines": (vx_line, vy_line),
            "angle_lines": (theta_x_line, theta_y_line),
            "action_lines": (ax_line, ay_line),
        }

    def update_episode(self, row: int, episode: dict[str, object], upto: int | None = None) -> list[object]:
        states = np.asarray(episode["states"], dtype=np.float32)
        actions = np.asarray(episode["actions_real"], dtype=np.float32)
        rewards = np.asarray(episode["rewards"], dtype=np.float32)
        if upto is not None:
            states = states[:upto]
            actions = actions[: max(0, upto - 1)]
            rewards = rewards[: max(0, upto - 1)]
        if states.size == 0:
            return []

        row_artists = self.rows[row]
        state_steps = np.arange(states.shape[0])
        action_steps = np.arange(1, actions.shape[0] + 1)
        title = f"episode {row + 1}  return={float(np.sum(rewards)):.2f}  steps={max(0, states.shape[0] - 1)}"
        row_artists["board_ax"].set_title(title)

        row_artists["trajectory_line"].set_data(states[:, 0], states[:, 1])
        row_artists["ball_dot"].set_data([states[-1, 0]], [states[-1, 1]])
        for line, value_idx in zip(row_artists["pos_lines"], (0, 1)):
            line.set_data(state_steps, states[:, value_idx])
        for line, value_idx in zip(row_artists["vel_lines"], (2, 3)):
            line.set_data(state_steps, states[:, value_idx])
        for line, value_idx in zip(row_artists["angle_lines"], (4, 5)):
            line.set_data(state_steps, states[:, value_idx])
        if actions.size:
            for line, value_idx in zip(row_artists["action_lines"], (0, 1)):
                line.set_data(action_steps, actions[:, value_idx])
        return [
            row_artists["trajectory_line"],
            row_artists["ball_dot"],
            *row_artists["pos_lines"],
            *row_artists["vel_lines"],
            *row_artists["angle_lines"],
            *row_artists["action_lines"],
        ]

    def pause(self, seconds: float) -> None:
        if self.show:
            self.fig.canvas.draw_idle()
            self.plt.pause(seconds)

    def save_gif(self, episodes: list[dict[str, object]], gif_path: Path, fps: int, stride: int = 1) -> None:
        from matplotlib.animation import FuncAnimation, PillowWriter

        max_frames = max(len(episode["states"]) for episode in episodes)
        frame_indices = list(range(0, max_frames, max(1, stride)))
        if frame_indices[-1] != max_frames - 1:
            frame_indices.append(max_frames - 1)

        def update(frame_idx: int):
            artists = []
            for row, episode in enumerate(episodes):
                artists.extend(self.update_episode(row, episode, upto=frame_idx + 1))
            return artists

        animation = FuncAnimation(self.fig, update, frames=frame_indices, interval=1000 / fps, blit=False)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        if gif_path.exists():
            gif_path.unlink()
        animation.save(gif_path, writer=PillowWriter(fps=fps))


def run_episode(policy: Dreamer, env, max_steps: int, device: torch.device, dashboard, row: int, fps: int):
    obs = env.reset()
    prev_state = policy.rssm.init_state(1, device)
    prev_action = torch.zeros(1, policy.action_size, device=device)
    episode = {
        "states": [np.asarray(obs.get("state", np.zeros(6)), dtype=np.float32)],
        "actions_norm": [],
        "actions_real": [],
        "rewards": [],
        "terminated": False,
        "truncated": False,
        "fallen": False,
    }

    if dashboard is not None:
        dashboard.update_episode(row, episode)
        dashboard.pause(1.0 / max(1, fps))

    for _ in range(max_steps):
        with torch.no_grad():
            posterior, action_tensor = policy.act_with_world_model(obs, prev_state, prev_action)
        action = action_tensor[0].detach().cpu().numpy().astype(np.float32)
        next_obs, reward, done, info = env.step(action)

        real_action = np.asarray(info.get("action_clipped", action), dtype=np.float32)
        episode["actions_norm"].append(action.copy())
        episode["actions_real"].append(real_action.copy())
        episode["rewards"].append(float(reward))
        episode["states"].append(np.asarray(next_obs.get("state", np.zeros(6)), dtype=np.float32))
        episode["terminated"] = bool(episode["terminated"] or info.get("terminated", False))
        episode["truncated"] = bool(episode["truncated"] or info.get("truncated", False))
        episode["fallen"] = bool(episode["fallen"] or info.get("fallen", False))

        obs = next_obs
        prev_state = posterior
        prev_action = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

        if dashboard is not None:
            dashboard.update_episode(row, episode)
            dashboard.pause(1.0 / max(1, fps))
        if done:
            break

    for key in ("states", "actions_norm", "actions_real", "rewards"):
        episode[key] = np.asarray(episode[key], dtype=np.float32)
    return episode


def summarize_episodes(episodes: list[dict[str, object]]) -> dict[str, object]:
    summaries = []
    for idx, episode in enumerate(episodes, start=1):
        states = np.asarray(episode["states"], dtype=np.float32)
        rewards = np.asarray(episode["rewards"], dtype=np.float32)
        distance = np.linalg.norm(states[:, :2], axis=-1)
        tail = distance[-min(50, len(distance)) :]
        summaries.append(
            {
                "episode": idx,
                "steps": int(len(rewards)),
                "return": float(np.sum(rewards)),
                "final_distance": float(distance[-1]),
                "last50_mean_distance": float(np.mean(tail)),
                "fallen": bool(episode["fallen"]),
                "terminated": bool(episode["terminated"]),
                "truncated": bool(episode["truncated"]),
            }
        )
    returns = [item["return"] for item in summaries]
    return {
        "episodes": summaries,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "max_return": float(np.max(returns)) if returns else 0.0,
        "min_return": float(np.min(returns)) if returns else 0.0,
    }


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    gif_path, metrics_path = episode_output_paths(args)

    checkpoint_obj = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint_obj, dict) or "rssm" not in checkpoint_obj or "actor" not in checkpoint_obj:
        keys = sorted(checkpoint_obj.keys()) if isinstance(checkpoint_obj, dict) else type(checkpoint_obj).__name__
        raise KeyError(
            "This visualizer loads old dreamer.py checkpoints with keys such as "
            "'rssm', 'actor', and 'obs_encoder'. The checkpoint you passed uses a "
            f"different schema: {keys}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    env = make_env(args)
    obs_shape = env.observation_space["image"].shape
    action_size = env.action_space.shape[0]
    policy = Dreamer(args, obs_shape, action_size, device, restore=True)
    policy.rssm.eval()
    policy.actor.eval()
    policy.obs_encoder.eval()

    config = find_ball_config(env)
    board_size = float(config.board_size if config is not None else 1.0)
    max_angle = float(config.max_angle if config is not None else 0.25)
    dashboard = BallPolicyDashboard(
        args.num_episodes,
        args.max_steps,
        board_size,
        max_angle,
        show=args.show_online == "show",
    )

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Saving combined GIF to: {gif_path}")
    print(f"Saving metrics to: {metrics_path}")

    episodes = []
    for episode_idx in range(args.num_episodes):
        episode = run_episode(policy, env, args.max_steps, device, dashboard, episode_idx, args.fps)
        episodes.append(episode)
        summary = summarize_episodes([episode])["episodes"][0]
        print(
            "Episode {episode}: return={return:.3f}, steps={steps}, "
            "final_distance={final_distance:.4f}, fallen={fallen}".format(**summary)
        )

    metrics = summarize_episodes(episodes)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    dashboard.save_gif(episodes, gif_path, args.fps, stride=args.gif_stride)
    print(f"Done. Mean return: {metrics['mean_return']:.3f}")


if __name__ == "__main__":
    main()
