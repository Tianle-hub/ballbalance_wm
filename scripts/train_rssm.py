"""Train a low-dimensional Gaussian RSSM on ball balance trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.data.sequence_dataset import SequenceDataset, batch_to_device
from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from ball_rssm.utils.checkpoint import save_checkpoint
from ball_rssm.utils.seed import set_seed


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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SequenceDataset(args.dataset, seq_len=args.seq_len, split="train", val_fraction=args.val_fraction, seed=args.seed)
    val_ds = SequenceDataset(args.dataset, seq_len=args.seq_len, split="val", val_fraction=args.val_fraction, seed=args.seed)
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    writer = make_writer(run_dir)
    config_dict = vars(args) | {"model": config.to_dict()}
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_metrics = run_epoch(model, train_loader, normalizer, device, optimizer, args.grad_clip, desc=f"epoch {epoch} train")
        model.eval()
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, normalizer, device, None, args.grad_clip, desc=f"epoch {epoch} val")
            open_loop = validation_open_loop_losses(model, val_loader, normalizer, device, horizons=(1, 5, 10, 25, 50))

        val_loss = val_metrics["total_loss"]
        is_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "config": config.to_dict(),
            "train_args": vars(args),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        save_checkpoint(state, ckpt_dir / "latest.pt")
        if is_best:
            save_checkpoint(state, ckpt_dir / "best.pt")

        log_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
        log_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
        log_metrics.update({f"val_open_loop/{k}": v for k, v in open_loop.items()})
        if writer is not None:
            for key, value in log_metrics.items():
                writer.add_scalar(key, value, epoch)

        open_loop_str = " ".join(f"{k}={v:.5f}" for k, v in open_loop.items())
        print(
            f"epoch={epoch:03d} train={train_metrics['total_loss']:.5f} "
            f"val={val_loss:.5f} recon={val_metrics['recon_loss']:.5f} "
            f"reward={val_metrics['reward_loss']:.5f} kl={val_metrics['kl_loss']:.5f} "
            f"raw_kl={val_metrics['raw_kl']:.5f} {open_loop_str}"
        )

    if writer is not None:
        writer.close()


def run_epoch(
    model: WorldModel,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    desc: str,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch_to_device(batch, device)
        obs = normalizer.normalize_obs(batch["obs"])
        action = normalizer.normalize_action(batch["action"])
        reward = normalizer.normalize_reward(batch["reward"]) if "reward" in batch else None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(obs, action, reward, batch.get("done"))
        grad_norm = torch.tensor(0.0)
        if optimizer is not None:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        batch_size = obs.shape[0]
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm.detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def validation_open_loop_losses(
    model: WorldModel,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    horizons: tuple[int, ...],
) -> dict[str, float]:
    batch = next(iter(loader), None)
    if batch is None:
        return {f"obs_h{h}": float("nan") for h in horizons}
    batch = batch_to_device(batch, device)
    obs = normalizer.normalize_obs(batch["obs"])
    action = normalizer.normalize_action(batch["action"])
    reward = normalizer.normalize_reward(batch["reward"]) if "reward" in batch else None
    out: dict[str, float] = {}
    context_len = min(10, action.shape[1] - 1)
    for horizon in horizons:
        if context_len + horizon > action.shape[1]:
            continue
        pred = model.open_loop_predict(obs, action, context_len=context_len, horizon=horizon)
        target = obs[:, context_len + 1 : context_len + 1 + horizon]
        obs_loss = torch.mean((pred - target) ** 2)
        if reward is not None:
            reward_pred = model.open_loop_predict_rewards(obs, action, context_len=context_len, horizon=horizon)
            reward_target = reward[:, context_len : context_len + horizon]
            reward_loss = torch.mean((reward_pred - reward_target) ** 2)
            out[f"obs_h{horizon}"] = float(obs_loss.detach().cpu())
            out[f"reward_h{horizon}"] = float(reward_loss.detach().cpu())
        else:
            out[f"obs_h{horizon}"] = float(obs_loss.detach().cpu())
    return out


def make_writer(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(run_dir / "tb")


if __name__ == "__main__":
    main()
