"""Visualize a saved MPC rollout or run one short rollout and plot it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.planning.rollout import InitialConditionBounds, run_mpc_episode, sample_initial_state, save_mpc_plots
from ball_rssm.planning.rssm_mpc import RSSMMPCController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-npz", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default="runs/mpc_visualization")
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--num-candidates", type=int, default=512)
    parser.add_argument("--num-elites", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    target = (args.target_x, args.target_y)
    if args.episode_npz:
        loaded = np.load(args.episode_npz, allow_pickle=True)
        episode = {key: loaded[key] for key in loaded.files}
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --episode-npz is not provided")
        env = BallBalanceEnv(config={"max_episode_steps": args.max_steps})
        try:
            controller = RSSMMPCController(
                checkpoint_path=args.checkpoint,
                action_space=env.action_space,
                horizon=args.horizon,
                num_candidates=args.num_candidates,
                num_elites=args.num_elites,
                num_iterations=args.num_iterations,
                device=args.device,
                cost_mode="center" if target == (0.0, 0.0) else "point",
                target_xy=None if target == (0.0, 0.0) else target,
                seed=args.seed,
            )
            initial_state = sample_initial_state(np.random.default_rng(args.seed), InitialConditionBounds())
            episode = run_mpc_episode(controller, env, initial_state, args.max_steps, target_xy=target)
        finally:
            env.close()
    save_mpc_plots(episode, args.out_dir, target_xy=target, title_prefix="RSSM MPC")
    print(f"Saved MPC visualization to {args.out_dir}")


if __name__ == "__main__":
    main()
