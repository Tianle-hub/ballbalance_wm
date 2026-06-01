"""Run random smooth rollouts in the ball balance environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()

    env = BallBalanceEnv(render_mode="human")
    rng = np.random.default_rng(args.seed)

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            done = False
            total_reward = 0.0

            while not done:
                random_action = env.action_space.sample()
                action = 0.95 * action + 0.05 * random_action
                action += rng.normal(0.0, 0.003, size=action.shape).astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                env.render()

                if info["step_count"] % args.print_every == 0 or done:
                    print(
                        f"episode={episode} step={info['step_count']} "
                        f"obs={np.round(obs, 3)} reward={reward:.3f} "
                        f"terminated={terminated} truncated={truncated}"
                    )

            print(f"episode={episode} total_reward={total_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
