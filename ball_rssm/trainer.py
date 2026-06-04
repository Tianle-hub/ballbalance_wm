"""Trainer orchestration for low-dimensional ball RSSM models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ball_rssm.buffer import Buffer
from ball_rssm.data.sequence_dataset import batch_to_device
from ball_rssm.models import Normalizer, WorldModel
from ball_rssm.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: WorldModel,
        buffer: Buffer,
        optimizer: torch.optim.Optimizer,
        normalizer: Normalizer,
        device: torch.device,
        run_dir: str | Path,
        grad_clip: float = 100.0,
    ) -> None:
        self.model = model
        self.buffer = buffer
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.device = device
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.grad_clip = grad_clip
        self.writer = make_writer(self.run_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        epochs: int,
        batch_size: int,
        seq_len: int,
        val_fraction: float,
        seed: int,
        train_args: dict[str, object],
        start_epoch: int = 0,
        best_val_loss: float = float("inf"),
    ) -> None:
        train_loader, val_loader = self.make_dataloaders(batch_size, seq_len, val_fraction, seed)

        if start_epoch >= epochs:
            print(f"checkpoint is already at epoch {start_epoch}; target epochs={epochs}, nothing to train")
            if self.writer is not None:
                self.writer.close()
            return

        for epoch in range(start_epoch + 1, epochs + 1):
            self.model.train()
            train_metrics = self.run_epoch(train_loader, optimizer=self.optimizer, desc=f"epoch {epoch} train")
            self.model.eval()
            with torch.no_grad():
                val_metrics = self.run_epoch(val_loader, optimizer=None, desc=f"epoch {epoch} val")
                open_loop = self.validation_open_loop_losses(val_loader, horizons=(1, 5, 10, 25, 50))

            val_loss = val_metrics["total_loss"]
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            self.save_checkpoint(epoch, best_val_loss, train_args, is_best)
            self.log_metrics(epoch, train_metrics, val_metrics, open_loop)
            self.print_epoch(epoch, train_metrics, val_metrics, open_loop)

        if self.writer is not None:
            self.writer.close()

    def make_dataloaders(
        self,
        batch_size: int,
        seq_len: int,
        val_fraction: float,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        train_dataset = self.buffer.sequence_dataset(seq_len=seq_len, split="train", val_fraction=val_fraction, seed=seed)
        val_dataset = self.buffer.sequence_dataset(seq_len=seq_len, split="val", val_fraction=val_fraction, seed=seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_loader, val_loader

    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        desc: str,
    ) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch_to_device(batch, self.device)
            obs = self.normalizer.normalize_obs(batch["obs"])
            action = self.normalizer.normalize_action(batch["action"])
            reward = self.normalizer.normalize_reward(batch["reward"]) if "reward" in batch else None
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = self.model.loss(obs, action, reward, batch.get("done"), batch.get("terminated"))
            grad_norm = torch.tensor(0.0, device=self.device)
            if optimizer is not None:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
            batch_size = obs.shape[0]
            count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm.detach().cpu()) * batch_size
        return {key: value / max(count, 1) for key, value in totals.items()}

    def validation_open_loop_losses(self, loader: DataLoader, horizons: tuple[int, ...]) -> dict[str, float]:
        batch = next(iter(loader), None)
        if batch is None:
            return {f"obs_h{h}": float("nan") for h in horizons}
        batch = batch_to_device(batch, self.device)
        obs = self.normalizer.normalize_obs(batch["obs"])
        action = self.normalizer.normalize_action(batch["action"])
        reward = self.normalizer.normalize_reward(batch["reward"]) if "reward" in batch else None
        out: dict[str, float] = {}
        context_len = min(10, action.shape[1] - 1)
        for horizon in horizons:
            if context_len + horizon > action.shape[1]:
                continue
            pred = self.model.open_loop_predict(obs, action, context_len=context_len, horizon=horizon)
            target = obs[:, context_len + 1 : context_len + 1 + horizon]
            obs_loss = torch.mean((pred - target) ** 2)
            if reward is not None:
                reward_pred = self.model.open_loop_predict_rewards(obs, action, context_len=context_len, horizon=horizon)
                reward_target = reward[:, context_len : context_len + horizon]
                reward_loss = torch.mean((reward_pred - reward_target) ** 2)
                out[f"obs_h{horizon}"] = float(obs_loss.detach().cpu())
                out[f"reward_h{horizon}"] = float(reward_loss.detach().cpu())
            else:
                out[f"obs_h{horizon}"] = float(obs_loss.detach().cpu())
        return out

    def save_checkpoint(
        self,
        epoch: int,
        best_val_loss: float,
        train_args: dict[str, object],
        is_best: bool,
    ) -> None:
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "config": self.model.config.to_dict(),
            "train_args": train_args,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        save_checkpoint(state, self.ckpt_dir / "last.pt")
        save_checkpoint(state, self.ckpt_dir / "latest.pt")
        if is_best:
            save_checkpoint(state, self.ckpt_dir / "best.pt")

    def log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        open_loop: dict[str, float],
    ) -> None:
        if self.writer is None:
            return
        log_metrics = {f"train/{key}": value for key, value in train_metrics.items()}
        log_metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
        log_metrics.update({f"val_open_loop/{key}": value for key, value in open_loop.items()})
        for key, value in log_metrics.items():
            self.writer.add_scalar(key, value, epoch)

    @staticmethod
    def print_epoch(
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        open_loop: dict[str, float],
    ) -> None:
        open_loop_str = " ".join(f"{key}={value:.5f}" for key, value in open_loop.items())
        print(
            f"epoch={epoch:03d} train={train_metrics['total_loss']:.5f} "
            f"val={val_metrics['total_loss']:.5f} recon={val_metrics['recon_loss']:.5f} "
            f"reward={val_metrics['reward_loss']:.5f} kl={val_metrics['kl_loss']:.5f} "
            f"raw_kl={val_metrics['raw_kl']:.5f} {open_loop_str}"
        )


def make_writer(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(run_dir / "tb")
