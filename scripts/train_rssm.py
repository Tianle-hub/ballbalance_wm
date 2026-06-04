"""Train a low-dimensional Gaussian RSSM on ball balance trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.buffer import Buffer
from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from ball_rssm.trainer import Trainer
from ball_rssm.utils.checkpoint import load_checkpoint
from ball_rssm.utils.seed import set_seed


def find_resume_checkpoint(run_dir: Path, explicit_path: str | None) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
        return path

    for name in ("last.pt", "latest.pt"):
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", default="runs/rssm_ball_v0")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--deter-dim", type=int, default=128)
    parser.add_argument("--stoch-dim", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from run-dir/checkpoints/last.pt or latest.pt when present (default).",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start a new training run.")
    parser.add_argument("--resume-from", default=None, help="Explicit checkpoint path to resume from.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)

    buffer = Buffer.load(args.dataset)
    train_ds = buffer.sequence_dataset(args.seq_len, split="train", val_fraction=args.val_fraction, seed=args.seed)
    train_obs, train_action, train_reward = train_ds.selected_arrays()
    normalizer = Normalizer.from_arrays(train_obs, train_action, train_reward).to(device)

    config = WorldModelConfig(
        deter_dim=args.deter_dim,
        stoch_dim=args.stoch_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        beta_kl=args.beta_kl,
        free_nats=args.free_nats,
        reward_loss_weight=args.reward_loss_weight,
    )
    model = WorldModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_val_loss = float("inf")

    resume_path = find_resume_checkpoint(run_dir, args.resume_from) if args.resume else None
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        normalizer = Normalizer.load_state_dict(checkpoint["normalizer"]).to(device)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resuming from {resume_path} at epoch {start_epoch}; target epochs={args.epochs}")

    config_dict = vars(args) | {"model": config.to_dict()}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    trainer = Trainer(
        model=model,
        buffer=buffer,
        optimizer=optimizer,
        normalizer=normalizer,
        device=device,
        run_dir=run_dir,
        grad_clip=args.grad_clip,
    )
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
        seed=args.seed,
        train_args=vars(args),
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    main()
