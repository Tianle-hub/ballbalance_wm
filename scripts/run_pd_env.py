"""Run a PD controller for center stabilization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv


def pd_action(obs: np.ndarray, kp: float, kd: float) -> np.ndarray:
    x, y, vx, vy, _, _ = obs
    theta_y_cmd = -kp * x - kd * vx
    theta_x_cmd = kp * y + kd * vy
    return np.array([theta_x_cmd, theta_y_cmd], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kp", type=float, default=0.8)
    parser.add_argument("--kd", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = BallBalanceEnv(render_mode="human")
    history: list[np.ndarray] = []
    total_reward = 0.0

    try:
        obs, _ = env.reset(seed=args.seed)
        history.append(obs.copy())
        done = False

        while not done:
            action = pd_action(obs, kp=args.kp, kd=args.kd)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            history.append(obs.copy())
            done = terminated or truncated
            env.render()

        print(
            f"finished step={info['step_count']} total_reward={total_reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )
    finally:
        env.close()

    data = np.asarray(history)
    t = np.arange(data.shape[0]) * env.config.dt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(data[:, 0], data[:, 1])
    half = env.config.board_size / 2.0
    axes[0].set_xlim(-half, half)
    axes[0].set_ylim(-half, half)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("PD rollout x-y")
    axes[0].grid(True)

    axes[1].plot(t, data[:, 0], label="x")
    axes[1].plot(t, data[:, 1], label="y")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position [m]")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
