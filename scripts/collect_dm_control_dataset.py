"""Collect random-policy replay from a DeepMind Control Suite task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs.dm_control import DMControlConfig, collect_random_dm_control, save_dm_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="cartpole")
    parser.add_argument("--task", default="swingup")
    parser.add_argument("--obs-type", choices=["state", "pixel"], default="state")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = DMControlConfig(
        domain=args.domain,
        task=args.task,
        obs_type=args.obs_type,
        action_repeat=args.action_repeat,
        height=args.height,
        width=args.width,
        camera_id=args.camera_id,
        mujoco_gl=args.mujoco_gl,
    )
    arrays = collect_random_dm_control(
        config=config,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    save_dm_control_dataset(args.out, arrays)
    print(
        f"saved {args.num_episodes} episodes to {args.out}; "
        f"obs={arrays['obs'].shape}, action={arrays['action'].shape}, reward={arrays['reward'].shape}"
    )


if __name__ == "__main__":
    main()
