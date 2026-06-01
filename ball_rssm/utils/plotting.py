"""Plotting utilities shared by RSSM scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBS_LABELS = ["x", "y", "vx", "vy", "theta_x", "theta_y"]


def save_xy_diagnostics(obs: np.ndarray, out_path: str | Path, board_size: float = 1.0, max_episodes: int = 12) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    count = min(max_episodes, obs.shape[0])
    for episode in range(count):
        ax.plot(obs[episode, :, 0], obs[episode, :, 1], alpha=0.8, linewidth=1.2)
    add_board_boundary(ax, board_size)
    ax.set_title("Dataset x-y trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def add_board_boundary(ax, board_size: float = 1.0) -> None:
    half = board_size / 2.0
    ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="black", linewidth=2)
    ax.set_aspect("equal", adjustable="box")
