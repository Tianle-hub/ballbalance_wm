"""Train a low-dimensional DreamerV1 agent on ball balance trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.buffer import Buffer, InitialStateBounds
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


def normalized_action_bounds(
    normalizer: Normalizer,
    action_arrays: np.ndarray | None = None,
    action_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    use_ball_env_bounds: bool = True,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if action_bounds is not None:
        low = torch.as_tensor(action_bounds[0].reshape(-1), dtype=torch.float32)
        high = torch.as_tensor(action_bounds[1].reshape(-1), dtype=torch.float32)
        low_norm = normalizer.normalize_action(low)
        high_norm = normalizer.normalize_action(high)
        return tuple(float(x) for x in low_norm), tuple(float(x) for x in high_norm)

    if use_ball_env_bounds:
        try:
            return normalized_ball_action_bounds(normalizer)
        except ValueError:
            if action_arrays is None:
                raise

    if action_arrays is None:
        action_dim = int(normalizer.action_mean.numel())
        low_norm = torch.full((action_dim,), -1.0)
        high_norm = torch.full((action_dim,), 1.0)
    else:
        action_flat = action_arrays.reshape(-1, action_arrays.shape[-1]).astype(np.float32)
        low = torch.from_numpy(action_flat.min(axis=0))
        high = torch.from_numpy(action_flat.max(axis=0))
        low_norm = normalizer.normalize_action(low)
        high_norm = normalizer.normalize_action(high)
    return tuple(float(x) for x in low_norm), tuple(float(x) for x in high_norm)


def normalized_ball_action_bounds(normalizer: Normalizer) -> tuple[tuple[float, ...], tuple[float, ...]]:
    env = BallBalanceEnv()
    try:
        low = torch.as_tensor(env.action_space.low.reshape(-1), dtype=torch.float32)
        high = torch.as_tensor(env.action_space.high.reshape(-1), dtype=torch.float32)
        if low.numel() != normalizer.action_mean.numel():
            raise ValueError("BallBalanceEnv action bounds do not match replay action_dim")
        low_norm = normalizer.normalize_action(low)
        high_norm = normalizer.normalize_action(high)
    finally:
        env.close()
    return tuple(float(x) for x in low_norm), tuple(float(x) for x in high_norm)


def infer_obs_type(requested: str, obs_shape: tuple[int, ...]) -> str:
    if requested != "auto":
        return requested
    return "pixel" if len(obs_shape) == 3 else "vector"


def load_dataset_action_bounds(path: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    with np.load(path) as arrays:
        if "action_low" not in arrays or "action_high" not in arrays:
            return None
        return arrays["action_low"].astype(np.float32), arrays["action_high"].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-dir", default="runs/dreamer_ball_v0")
    parser.add_argument("--train-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--dreamer-version", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--obs-type",
        choices=["auto", "vector", "pixel"],
        default="auto",
        help="Observation model type. auto treats 3D trailing obs shapes as pixels.",
    )
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--online-iterations", type=int, default=None)
    parser.add_argument("--update-steps", type=int, default=100)
    parser.add_argument("--collect-episodes", type=int, default=1)
    parser.add_argument("--seed-episodes", type=int, default=20)
    parser.add_argument("--buffer-episodes", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed-policy-mode", choices=["random_smooth", "pd", "mixed", "coverage"], default="coverage")
    parser.add_argument("--pos-bound", type=float, default=0.20)
    parser.add_argument("--vel-bound", type=float, default=0.05)
    parser.add_argument("--angle-bound", type=float, default=0.0)
    parser.add_argument("--target-bound", type=float, default=0.12)
    parser.add_argument("--action-noise-std", type=float, default=0.03)
    parser.add_argument("--world-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=8e-5)
    parser.add_argument("--critic-lr", type=float, default=8e-5)
    parser.add_argument("--deter-dim", type=int, default=128)
    parser.add_argument("--stoch-dim", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--discrete-classes", type=int, default=32)
    parser.add_argument("--actor-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=128)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--kl-alpha", type=float, default=0.8)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--continuation-loss-weight", type=float, default=1.0)
    parser.add_argument("--imagination-horizon", type=int, default=15)
    parser.add_argument(
        "--behavior-batch-size",
        type=int,
        default=4096,
        help="Maximum posterior states used for each actor/value imagination update.",
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument(
        "--actor-gradient",
        choices=["auto", "dynamics", "reinforce", "both"],
        default="auto",
        help=(
            "Actor gradient estimator. auto uses reinforce for discrete actors and dynamics for continuous actors; "
            "both mixes score-function and dynamics terms."
        ),
    )
    parser.add_argument("--actor-entropy-scale", type=float, default=1e-3)
    parser.add_argument(
        "--exploration-mode",
        choices=["auto", "noise", "policy_entropy"],
        default="auto",
        help="Data-collection exploration. auto uses noise for V1 and policy entropy for V2.",
    )
    parser.add_argument("--exploration-noise", type=float, default=0.3)
    parser.add_argument("--exploration-decay", type=float, default=1.0)
    parser.add_argument("--min-exploration-noise", type=float, default=0.0)
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

    if args.train_mode == "offline" and args.dataset is None:
        parser.error("--dataset is required for --train-mode offline")

    initial_bounds = InitialStateBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    if args.train_mode == "online":
        replay_resume_path = run_dir / "replay" / "latest.npz"
        if args.dataset is None and args.resume and replay_resume_path.exists():
            seed_buffer = Buffer.load(replay_resume_path)
            print(f"resuming online replay from {replay_resume_path} with {seed_buffer.size} episodes")
        elif args.dataset is not None:
            seed_buffer = Buffer.load(args.dataset)
        else:
            seed_buffer = Buffer.collect_data(
                num_episodes=args.seed_episodes,
                max_episode_steps=args.max_episode_steps,
                seed=args.seed,
                mode=args.seed_policy_mode,
                initial_bounds=initial_bounds,
                target_bound=args.target_bound,
                action_noise_std=args.action_noise_std,
            )
        if seed_buffer.max_episode_steps != args.max_episode_steps:
            args.max_episode_steps = seed_buffer.max_episode_steps
        buffer_capacity = max(args.buffer_episodes, seed_buffer.size)
        buffer = Buffer.empty(
            capacity_episodes=buffer_capacity,
            max_episode_steps=seed_buffer.max_episode_steps,
            obs_shape=tuple(seed_buffer.obs_buffer.shape[2:]),
            action_shape=tuple(seed_buffer.action_buffer.shape[2:]),
        )
        buffer.append_buffer(seed_buffer)
    else:
        assert args.dataset is not None
        buffer = Buffer.load(args.dataset)

    train_ds = buffer.sequence_dataset(args.seq_len, split="train", val_fraction=args.val_fraction, seed=args.seed)
    train_obs, train_action, train_reward = train_ds.selected_arrays()
    normalizer = Normalizer.from_arrays(train_obs, train_action, train_reward)
    obs_shape = tuple(int(dim) for dim in train_obs.shape[2:])
    action_dim = int(train_action.shape[-1])
    obs_type = infer_obs_type(args.obs_type, obs_shape)
    action_bounds = load_dataset_action_bounds(args.dataset)
    if obs_type == "vector" and len(obs_shape) != 1:
        parser.error(f"--obs-type vector requires replay obs shape [N, T, obs_dim], got trailing shape {obs_shape}")
    if obs_type == "pixel" and len(obs_shape) != 3:
        parser.error(f"--obs-type pixel requires replay obs shape [N, T, C, H, W], got trailing shape {obs_shape}")

    resume_path = find_resume_checkpoint(run_dir, args.resume_from) if args.resume else None
    checkpoint: dict[str, Any] | None = load_checkpoint(resume_path, device) if resume_path is not None else None

    if checkpoint is None:
        world_config = WorldModelConfig(
            obs_dim=int(prod(obs_shape)),
            obs_shape=obs_shape,
            obs_type=obs_type,
            action_dim=action_dim,
            deter_dim=args.deter_dim,
            stoch_dim=args.stoch_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            dreamer_version=args.dreamer_version,
            discrete_classes=args.discrete_classes,
            beta_kl=args.beta_kl,
            free_nats=args.free_nats,
            kl_alpha=args.kl_alpha,
            reward_loss_weight=args.reward_loss_weight,
            continuation_loss_weight=args.continuation_loss_weight,
        )
        use_ball_env_bounds = obs_shape == (6,) and action_dim == 2
        action_low, action_high = normalized_action_bounds(
            normalizer,
            action_arrays=train_action,
            action_bounds=action_bounds,
            use_ball_env_bounds=use_ball_env_bounds,
        )
        actor_config = ActorConfig(
            feature_dim=world_config.feature_dim,
            action_dim=world_config.action_dim,
            hidden_dim=args.actor_hidden_dim,
            action_low=action_low,
            action_high=action_high,
        )
        critic_config = CriticConfig(
            feature_dim=world_config.feature_dim,
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
        behavior_batch_size=args.behavior_batch_size,
        discount=args.discount,
        lambda_=args.lambda_,
        actor_gradient=args.actor_gradient,
        actor_entropy_scale=args.actor_entropy_scale,
        exploration_mode=args.exploration_mode,
        exploration_noise=args.exploration_noise,
        exploration_decay=args.exploration_decay,
        min_exploration_noise=args.min_exploration_noise,
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
    if args.train_mode == "online":
        iterations = args.online_iterations if args.online_iterations is not None else args.epochs
        trainer.train_online(
            iterations=iterations,
            update_steps=args.update_steps,
            collect_episodes=args.collect_episodes,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            val_fraction=args.val_fraction,
            seed=args.seed,
            train_args=vars(args),
            initial_bounds=initial_bounds,
            start_iteration=start_epoch,
            best_val_loss=best_val_loss,
        )
    else:
        trainer.train_offline(
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
