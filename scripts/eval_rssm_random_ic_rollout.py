"""Evaluate RSSM imagined rollout on fresh bounded random initial conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.evaluation import RandomICBounds, evaluate_imagined_trajectory_error, generate_random_ic_dataset
from ball_rssm.evaluation.random_ic_rollout import (
    bounds_to_dict,
    plot_random_ic_errors,
    plot_random_ic_trajectories,
    save_random_ic_dataset,
)
from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from ball_rssm.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--out-dataset", default=None)
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--action-mode", choices=["random_smooth", "pd", "mixed"], default="mixed")
    parser.add_argument("--pos-bound", type=float, default=0.25)
    parser.add_argument("--vel-bound", type=float, default=0.20)
    parser.add_argument("--angle-bound", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = WorldModel(WorldModelConfig.from_dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer = Normalizer.load_state_dict(checkpoint["normalizer"]).to(device)

    bounds = RandomICBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    dataset = generate_random_ic_dataset(
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        context_len=args.context_len,
        seed=args.seed,
        bounds=bounds,
        action_mode=args.action_mode,
    )
    metrics = evaluate_imagined_trajectory_error(
        model=model,
        normalizer=normalizer,
        dataset=dataset,
        context_len=args.context_len,
        horizon=args.horizon,
        device=device,
        batch_size=args.batch_size,
    )
    metrics["initial_condition_bounds"] = bounds_to_dict(bounds)
    metrics["action_mode"] = args.action_mode
    metrics["seed"] = args.seed

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "random_ic_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "random_ic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_random_ic_errors(metrics, out_dir / "random_ic_error_curves.png")
    plot_random_ic_trajectories(dataset, model, normalizer, args.context_len, args.horizon, device, out_dir / "random_ic_trajectories.png")

    if args.out_dataset:
        save_random_ic_dataset(
            dataset,
            args.out_dataset,
            seed=args.seed,
            action_mode=args.action_mode,
            context_len=args.context_len,
            horizon=args.horizon,
            initial_condition_bounds=bounds_to_dict(bounds),
        )

    print(json.dumps(metrics, indent=2))
    print(f"Saved random-IC evaluation to {out_dir}")


if __name__ == "__main__":
    main()
