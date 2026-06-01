"""Run RSSM+CEM MPC fixed target stabilization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.planning.rollout import (
    InitialConditionBounds,
    aggregate_metrics,
    episode_metrics,
    run_mpc_episode,
    sample_initial_state,
    save_episode_npz,
    save_mpc_plots,
)
from ball_rssm.planning.rssm_mpc import RSSMMPCController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-x", type=float, default=0.15)
    parser.add_argument("--target-y", type=float, default=-0.10)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--num-candidates", type=int, default=1024)
    parser.add_argument("--num-elites", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    target = (args.target_x, args.target_y)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "mpc_viapoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metrics = []
    env = BallBalanceEnv(render_mode="human" if args.render else None, config={"max_episode_steps": args.max_steps})
    try:
        controller = RSSMMPCController(
            checkpoint_path=args.checkpoint,
            action_space=env.action_space,
            horizon=args.horizon,
            num_candidates=args.num_candidates,
            num_elites=args.num_elites,
            num_iterations=args.num_iterations,
            device=args.device,
            cost_mode="point",
            target_xy=target,
            seed=args.seed,
        )
        for episode_idx in range(args.num_episodes):
            initial_state = sample_initial_state(rng, InitialConditionBounds())
            episode = run_mpc_episode(controller, env, initial_state, args.max_steps, target_xy=target, render=args.render)
            save_episode_npz(episode, out_dir / f"episode_{episode_idx:03d}.npz")
            save_mpc_plots(episode, out_dir / f"episode_{episode_idx:03d}", target_xy=target, title_prefix="RSSM MPC target")
            metrics.append(episode_metrics(episode, target, final_threshold=0.06, last_window_threshold=0.08))
            print(f"episode={episode_idx} metrics={metrics[-1]}")
    finally:
        env.close()

    summary = {"episodes": metrics, "aggregate": aggregate_metrics(metrics), "config": vars(args)}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
