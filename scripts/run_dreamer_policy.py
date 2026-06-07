"""Evaluate a trained Dreamer actor in the true ball balance environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.agent import DreamerAgent
from ball_rssm.envs import BallBalanceEnv
from ball_rssm.evaluation import (
    InitialConditionBounds,
    aggregate_metrics,
    episode_metrics,
    run_policy_episode,
    sample_initial_state,
    save_episode_npz,
    save_policy_plots,
)
from ball_rssm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-bound", type=float, default=0.25)
    parser.add_argument("--vel-bound", type=float, default=0.20)
    parser.add_argument("--angle-bound", type=float, default=0.12)
    parser.add_argument("--stochastic", action="store_true", help="Sample from the actor instead of using its mode.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-episodes", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "policy_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BallBalanceEnv(render_mode="human" if args.render else None, config={"max_episode_steps": args.max_steps})
    bounds = InitialConditionBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    bounds.validate(env)
    agent = DreamerAgent(
        checkpoint_path=args.checkpoint,
        action_space=env.action_space,
        device=device,
        deterministic=not args.stochastic,
    )
    rng = np.random.default_rng(args.seed)
    metrics = []

    try:
        for episode_idx in range(args.num_episodes):
            initial_state = sample_initial_state(rng, bounds)
            episode = run_policy_episode(agent, env, initial_state, max_steps=args.max_steps, render=args.render)
            episode_metric = episode_metrics(episode)
            metrics.append(episode_metric)
            if args.save_episodes:
                save_episode_npz(episode, out_dir / f"episode_{episode_idx:03d}.npz")
            if args.save_plots:
                save_policy_plots(episode, out_dir / f"episode_{episode_idx:03d}")
            print(
                f"episode={episode_idx:03d} return={episode_metric['return']:.3f} "
                f"steps={episode_metric['steps']} final_distance={episode_metric['final_distance']:.4f} "
                f"terminated={episode_metric['terminated']}"
            )
    finally:
        env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "bounds": {"pos": args.pos_bound, "vel": args.vel_bound, "angle": args.angle_bound},
        "episodes": metrics,
        "aggregate": aggregate_metrics(metrics),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"wrote Dreamer policy evaluation to {out_dir}")


if __name__ == "__main__":
    main()
