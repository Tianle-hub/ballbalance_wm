"""Collect offline ball balance trajectories into an NPZ file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.buffer import (
    Buffer,
    InitialStateBounds,
    PolicyMode,
    choose_episode_mode,
    pd_action,
    sample_initial_state,
)


def collect_dataset(
    num_episodes: int,
    max_episode_steps: int,
    seed: int,
    mode: PolicyMode,
    initial_bounds: InitialStateBounds = InitialStateBounds(),
    target_bound: float = 0.12,
    action_noise_std: float = 0.03,
) -> dict[str, np.ndarray]:
    buffer = Buffer.collect_data(
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        mode=mode,
        initial_bounds=initial_bounds,
        target_bound=target_bound,
        action_noise_std=action_noise_std,
    )
    return buffer.to_dataset()


def save_dataset(dataset: dict[str, np.ndarray] | Buffer, out: str | Path, **metadata: object) -> None:
    if isinstance(dataset, Buffer):
        dataset.save(out, **metadata)
        return

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset, **metadata)


def _choose_episode_mode(mode: PolicyMode, rng: np.random.Generator) -> str:
    return choose_episode_mode(mode, rng)


def _sample_initial_state(rng: np.random.Generator, bounds: InitialStateBounds) -> np.ndarray:
    return sample_initial_state(rng, bounds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--out", type=str, default="data/ball_balance_dataset.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["random_smooth", "pd", "mixed", "mpc_cover"], default="mixed")
    parser.add_argument("--pos-bound", type=float, default=0.20)
    parser.add_argument("--vel-bound", type=float, default=0.05)
    parser.add_argument("--angle-bound", type=float, default=0.0)
    parser.add_argument("--target-bound", type=float, default=0.12)
    parser.add_argument("--action-noise-std", type=float, default=0.03)
    args = parser.parse_args()

    initial_bounds = InitialStateBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    buffer = Buffer.collect_data(
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        mode=args.mode,
        initial_bounds=initial_bounds,
        target_bound=args.target_bound,
        action_noise_std=args.action_noise_std,
    )
    buffer.save(
        args.out,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        mode=args.mode,
        pos_bound=args.pos_bound,
        vel_bound=args.vel_bound,
        angle_bound=args.angle_bound,
        target_bound=args.target_bound,
        action_noise_std=args.action_noise_std,
    )
    print(f"saved {args.num_episodes} episodes to {args.out}")


if __name__ == "__main__":
    main()
