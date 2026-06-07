"""Train a low-dimensional DreamerV1 agent on ball balance trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.buffer import Buffer
from ball_rssm.envs import BallBalanceEnv
from ball_rssm.models import Actor, ActorConfig, Critic, CriticConfig, Normalizer, WorldModel, WorldModelConfig
from ball_rssm.trainer import DreamerTrainConfig, Trainer
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


def normalized_action_bounds(normalizer: Normalizer) -> tuple[tuple[float, ...], tuple[float, ...]]:
    env = BallBalanceEnv()
    try:
        low = torch.as_tensor(env.action_space.low.reshape(-1), dtype=torch.float32)
        high = torch.as_tensor(env.action_space.high.reshape(-1), dtype=torch.float32)
        low_norm = normalizer.normalize_action(low)
        high_norm = normalizer.normalize_action(high)
    finally:
        env.close()
    return tuple(float(x) for x in low_norm), tuple(float(x) for x in high_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", default="runs/dreamer_ball_v0")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--world-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=8e-5)
    parser.add_argument("--critic-lr", type=float, default=8e-5)
    parser.add_argument("--deter-dim", type=int, default=128)
    parser.add_argument("--stoch-dim", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--actor-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=128)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--continuation-loss-weight", type=float, default=1.0)
    parser.add_argument("--imagination-horizon", type=int, default=15)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--actor-entropy-scale", type=float, default=1e-3)
    parser.add_argument("--target-tau", type=float, default=0.01)
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
    parser.add_argument("--resume-from", default=None, help="Explicit Dreamer checkpoint path to resume from.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)

    buffer = Buffer.load(args.dataset)
    train_ds = buffer.sequence_dataset(args.seq_len, split="train", val_fraction=args.val_fraction, seed=args.seed)
    train_obs, train_action, train_reward = train_ds.selected_arrays()
    normalizer = Normalizer.from_arrays(train_obs, train_action, train_reward)

    resume_path = find_resume_checkpoint(run_dir, args.resume_from) if args.resume else None
    checkpoint: dict[str, Any] | None = load_checkpoint(resume_path, device) if resume_path is not None else None

    if checkpoint is None:
        world_config = WorldModelConfig(
            deter_dim=args.deter_dim,
            stoch_dim=args.stoch_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            beta_kl=args.beta_kl,
            free_nats=args.free_nats,
            reward_loss_weight=args.reward_loss_weight,
            continuation_loss_weight=args.continuation_loss_weight,
        )
        action_low, action_high = normalized_action_bounds(normalizer)
        actor_config = ActorConfig(
            feature_dim=world_config.deter_dim + world_config.stoch_dim,
            action_dim=world_config.action_dim,
            hidden_dim=args.actor_hidden_dim,
            action_low=action_low,
            action_high=action_high,
        )
        critic_config = CriticConfig(
            feature_dim=world_config.deter_dim + world_config.stoch_dim,
            hidden_dim=args.critic_hidden_dim,
        )
        start_epoch = 0
        best_val_loss = float("inf")
    else:
        if "actor_config" not in checkpoint:
            raise KeyError(f"{resume_path} is a world-model-only checkpoint; train Dreamer with --no-resume")
        world_config = WorldModelConfig.from_dict(checkpoint.get("world_model_config", checkpoint["config"]))
        actor_config = ActorConfig.from_dict(checkpoint["actor_config"])
        critic_config = CriticConfig.from_dict(checkpoint["critic_config"])
        normalizer = Normalizer.load_state_dict(checkpoint["normalizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resuming from {resume_path} at epoch {start_epoch}; target epochs={args.epochs}")

    normalizer = normalizer.to(device)
    world_model = WorldModel(world_config).to(device)
    actor = Actor(actor_config).to(device)
    critic = Critic(critic_config).to(device)
    target_critic = Critic(critic_config).to(device)
    target_critic.load_state_dict(critic.state_dict())

    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=args.world_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    if checkpoint is not None:
        world_model.load_state_dict(checkpoint.get("world_model_state_dict", checkpoint["model_state_dict"]))
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        target_critic.load_state_dict(checkpoint.get("target_critic_state_dict", checkpoint["critic_state_dict"]))
        if "world_optimizer_state_dict" in checkpoint:
            world_optimizer.load_state_dict(checkpoint["world_optimizer_state_dict"])
        elif "optimizer_state_dict" in checkpoint:
            world_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    dreamer_config = DreamerTrainConfig(
        imagination_horizon=args.imagination_horizon,
        discount=args.discount,
        lambda_=args.lambda_,
        actor_entropy_scale=args.actor_entropy_scale,
        grad_clip=args.grad_clip,
        target_tau=args.target_tau,
    )

    config_dict = {
        **vars(args),
        "world_model": world_config.to_dict(),
        "actor": actor_config.to_dict(),
        "critic": critic_config.to_dict(),
        "dreamer": dreamer_config.to_dict(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    trainer = Trainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        buffer=buffer,
        world_optimizer=world_optimizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        normalizer=normalizer,
        device=device,
        run_dir=run_dir,
        config=dreamer_config,
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
