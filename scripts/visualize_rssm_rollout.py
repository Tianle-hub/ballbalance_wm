"""Visualize true observations, posterior reconstructions, and prior rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from ball_rssm.utils.checkpoint import load_checkpoint
from ball_rssm.utils.plotting import OBS_LABELS, add_board_boundary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--play", action="store_true", help="Interactively play decoded imagined future observations.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--save-gif", default=None, help="Optional path to save the imagined rollout animation as a GIF.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = WorldModel(WorldModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = Normalizer.load_state_dict(checkpoint["normalizer"]).to(device)

    data = np.load(args.dataset)
    episode = min(max(args.episode_index, 0), data["obs"].shape[0] - 1)
    horizon = min(args.horizon, data["action"].shape[1] - args.context_len)
    obs_np = data["obs"][episode : episode + 1, : args.context_len + horizon + 1]
    action_np = data["action"][episode : episode + 1, : args.context_len + horizon]

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    action = torch.as_tensor(action_np, dtype=torch.float32, device=device)
    obs_norm = normalizer.normalize_obs(obs)
    action_norm = normalizer.normalize_action(action)

    with torch.no_grad():
        recon_norm = model.reconstruct(obs_norm, action_norm)
        prior_norm = model.open_loop_predict(obs_norm, action_norm, context_len=args.context_len, horizon=horizon)
        recon = normalizer.denormalize_obs(recon_norm).cpu().numpy()[0]
        prior = normalizer.denormalize_obs(prior_norm).cpu().numpy()[0]

    true = obs_np[0]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_xy(true, recon, prior, args.context_len, out_dir / "rollout_xy.png")
    plot_timeseries(true, recon, prior, args.context_len, out_dir / "rollout_timeseries.png")
    plot_errors(true, prior, args.context_len, out_dir / "rollout_errors.png")
    if args.play or args.save_gif:
        play_imagined_rollout(
            true=true,
            prior=prior,
            context_len=args.context_len,
            fps=args.fps,
            save_gif=Path(args.save_gif) if args.save_gif else None,
            show=args.play,
        )
    print(f"Saved figures to {out_dir}")


def plot_xy(true: np.ndarray, recon: np.ndarray, prior: np.ndarray, context_len: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(true[:, 0], true[:, 1], label="true", color="black", linewidth=2)
    ax.plot(recon[:, 0], recon[:, 1], label="posterior recon", color="tab:blue", alpha=0.85)
    prior_x = np.arange(context_len + 1, context_len + 1 + prior.shape[0])
    del prior_x
    ax.plot(prior[:, 0], prior[:, 1], label="prior open-loop", color="tab:red", linestyle="--")
    ax.plot(true[: context_len + 1, 0], true[: context_len + 1, 1], label="context", color="tab:green", linewidth=3)
    add_board_boundary(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("RSSM trajectory rollout")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_timeseries(true: np.ndarray, recon: np.ndarray, prior: np.ndarray, context_len: int, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    true_t = np.arange(true.shape[0])
    prior_t = np.arange(context_len + 1, context_len + 1 + prior.shape[0])
    for idx, ax in enumerate(axes.flat):
        ax.plot(true_t, true[:, idx], label="true", color="black")
        ax.plot(true_t, recon[:, idx], label="posterior recon", color="tab:blue", alpha=0.85)
        ax.plot(prior_t, prior[:, idx], label="prior open-loop", color="tab:red", linestyle="--")
        ax.axvline(context_len, color="tab:green", linestyle=":", linewidth=1.5)
        ax.set_title(OBS_LABELS[idx])
        ax.grid(True)
    axes.flat[0].legend()
    axes.flat[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_errors(true: np.ndarray, prior: np.ndarray, context_len: int, out_path: Path) -> None:
    target = true[context_len + 1 : context_len + 1 + prior.shape[0]]
    pos_err = np.linalg.norm(prior[:, :2] - target[:, :2], axis=-1)
    vel_err = np.linalg.norm(prior[:, 2:4] - target[:, 2:4], axis=-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, prior.shape[0] + 1), pos_err, label="position error")
    ax.plot(np.arange(1, prior.shape[0] + 1), vel_err, label="velocity error")
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("L2 error")
    ax.set_title("Open-loop rollout error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def play_imagined_rollout(
    true: np.ndarray,
    prior: np.ndarray,
    context_len: int,
    fps: float,
    save_gif: Path | None,
    show: bool,
) -> None:
    """Play decoded prior observations against true future observations on the board."""

    if fps <= 0.0:
        raise ValueError("fps must be positive")

    from matplotlib.animation import FuncAnimation, PillowWriter

    future_true = true[context_len + 1 : context_len + 1 + prior.shape[0]]
    if future_true.shape[0] != prior.shape[0]:
        raise ValueError("true future and prior rollout lengths do not match")

    fig, ax = plt.subplots(figsize=(6, 6))
    add_board_boundary(ax)
    ax.set_xlim(-0.58, 0.58)
    ax.set_ylim(-0.58, 0.58)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)

    ax.plot(true[: context_len + 1, 0], true[: context_len + 1, 1], color="tab:green", linewidth=3, label="posterior context")
    true_trace, = ax.plot([], [], color="black", linewidth=2, alpha=0.75, label="true future")
    prior_trace, = ax.plot([], [], color="tab:red", linestyle="--", linewidth=2, label="decoded prior rollout")
    true_ball, = ax.plot([], [], "o", color="black", markersize=9)
    prior_ball, = ax.plot([], [], "o", color="tab:red", markersize=9)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="lower right")

    def update(frame: int):
        true_trace.set_data(future_true[: frame + 1, 0], future_true[: frame + 1, 1])
        prior_trace.set_data(prior[: frame + 1, 0], prior[: frame + 1, 1])
        true_ball.set_data([future_true[frame, 0]], [future_true[frame, 1]])
        prior_ball.set_data([prior[frame, 0]], [prior[frame, 1]])
        pos_err = np.linalg.norm(prior[frame, :2] - future_true[frame, :2])
        vel_err = np.linalg.norm(prior[frame, 2:4] - future_true[frame, 2:4])
        text.set_text(f"step={context_len + 1 + frame}  pos_err={pos_err:.4f}  vel_err={vel_err:.4f}")
        return true_trace, prior_trace, true_ball, prior_ball, text

    anim = FuncAnimation(fig, update, frames=prior.shape[0], interval=1000.0 / fps, blit=True)
    if save_gif is not None:
        save_gif.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_gif, writer=PillowWriter(fps=int(round(fps))))
        print(f"Saved imagined rollout animation to {save_gif}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
