"""Plot trajectories from a collected ball balance dataset."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-trajectories", type=int, default=8)
    args = parser.parse_args()

    data = np.load(args.dataset)
    obs = data["obs"]
    if obs.ndim != 3 or obs.shape[-1] != 6:
        raise ValueError("Expected obs with shape [num_episodes, max_episode_steps + 1, 6]")

    episode = min(max(args.episode, 0), obs.shape[0] - 1)
    num_traj = min(args.num_trajectories, obs.shape[0])
    dt = 0.02
    t = np.arange(obs.shape[1]) * dt

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    half_board = 0.5
    for i in range(num_traj):
        axes[0].plot(obs[i, :, 0], obs[i, :, 1], linewidth=1.5, alpha=0.8)
    axes[0].plot(
        [-half_board, half_board, half_board, -half_board, -half_board],
        [-half_board, -half_board, half_board, half_board, -half_board],
        color="black",
        linewidth=2,
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("x-y trajectories")
    axes[0].grid(True)

    labels = ["x", "y", "vx", "vy", "theta_x", "theta_y"]
    for dim, label in enumerate(labels):
        axes[1].plot(t, obs[episode, :, dim], label=label)
    axes[1].set_xlabel("time [s]")
    axes[1].set_title(f"episode {episode} state")
    axes[1].legend(ncol=3)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
