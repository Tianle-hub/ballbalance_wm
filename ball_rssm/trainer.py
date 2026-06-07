"""DreamerV1-style training loop for the ball balance RSSM."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ball_rssm.buffer import Buffer
from ball_rssm.data.sequence_dataset import batch_to_device
from ball_rssm.models import Actor, Critic, Normalizer, WorldModel
from ball_rssm.models.behavior import discount_weights, lambda_return
from ball_rssm.models.rssm import RSSMState, stack_states
from ball_rssm.utils.checkpoint import save_checkpoint


@dataclass
class DreamerTrainConfig:
    imagination_horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95
    actor_entropy_scale: float = 1e-3
    grad_clip: float = 100.0
    target_tau: float = 0.01

    def __post_init__(self) -> None:
        if self.imagination_horizon < 2:
            raise ValueError("imagination_horizon must be at least 2")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("lambda_ must be in [0, 1]")
        if self.actor_entropy_scale < 0.0:
            raise ValueError("actor_entropy_scale must be non-negative")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        if not 0.0 <= self.target_tau <= 1.0:
            raise ValueError("target_tau must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Trainer:
    """Train world dynamics, actor, and value from the same replay batches."""

    def __init__(
        self,
        world_model: WorldModel,
        actor: Actor,
        critic: Critic,
        target_critic: Critic,
        buffer: Buffer,
        world_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        normalizer: Normalizer,
        device: torch.device,
        run_dir: str | Path,
        config: DreamerTrainConfig,
    ) -> None:
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.buffer = buffer
        self.world_optimizer = world_optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.normalizer = normalizer
        self.device = device
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.config = config
        self.writer = make_writer(self.run_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.target_critic.requires_grad_(False)

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
            self.close()
            return

        for epoch in range(start_epoch + 1, epochs + 1):
            self.world_model.train()
            self.actor.train()
            self.critic.train()
            train_metrics = self.run_epoch(train_loader, train=True, desc=f"epoch {epoch} train")

            self.world_model.eval()
            self.actor.eval()
            self.critic.eval()
            with torch.no_grad():
                val_metrics = self.run_epoch(val_loader, train=False, desc=f"epoch {epoch} val")
                open_loop = self.validation_open_loop_losses(val_loader, horizons=(1, 5, 10, 25, 50))

            val_loss = val_metrics["total_loss"]
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            self.save_checkpoint(epoch, best_val_loss, train_args, is_best)
            self.log_metrics(epoch, train_metrics, val_metrics, open_loop)
            self.print_epoch(epoch, train_metrics, val_metrics, open_loop)

        self.close()

    def close(self) -> None:
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

    def run_epoch(self, loader: DataLoader, train: bool, desc: str) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch_to_device(batch, self.device)
            obs = self.normalizer.normalize_obs(batch["obs"])
            action = self.normalizer.normalize_action(batch["action"])
            reward = self.normalizer.normalize_reward(batch["reward"]) if "reward" in batch else None

            if train:
                metrics = self.train_batch(obs, action, reward, batch.get("done"), batch.get("terminated"))
            else:
                metrics = self.evaluate_batch(obs, action, reward, batch.get("done"), batch.get("terminated"))

            batch_size = obs.shape[0]
            count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        return {key: value / max(count, 1) for key, value in totals.items()}

    def train_batch(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor | None,
        done: torch.Tensor | None,
        terminated: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        self.world_optimizer.zero_grad(set_to_none=True)
        world_loss, metrics = self.world_model.loss(obs, action, reward, done, terminated)
        world_loss.backward()
        world_grad = torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)
        self.world_optimizer.step()

        with torch.no_grad():
            posterior = self.world_model.forward(obs, action)["posterior"]
            assert isinstance(posterior, dict)
            start = flatten_state_sequence(posterior, drop_last=True)

        actor_loss, actor_metrics = self.actor_loss(start)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        critic_loss, critic_metrics = self.critic_loss(start)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()
        soft_update(self.critic, self.target_critic, self.config.target_tau)

        metrics = dict(metrics)
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        metrics["world_grad_norm"] = world_grad.detach()
        metrics["actor_grad_norm"] = actor_grad.detach()
        metrics["critic_grad_norm"] = critic_grad.detach()
        return metrics

    def evaluate_batch(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor | None,
        done: torch.Tensor | None,
        terminated: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        world_loss, metrics = self.world_model.loss(obs, action, reward, done, terminated)
        posterior = self.world_model.forward(obs, action)["posterior"]
        assert isinstance(posterior, dict)
        start = flatten_state_sequence(posterior, drop_last=True)
        actor_loss, actor_metrics = self.actor_loss(start)
        critic_loss, critic_metrics = self.critic_loss(start)
        metrics = dict(metrics)
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        metrics["world_grad_norm"] = torch.zeros((), device=obs.device)
        metrics["actor_grad_norm"] = torch.zeros((), device=obs.device)
        metrics["critic_grad_norm"] = torch.zeros((), device=obs.device)
        metrics["total_loss"] = world_loss.detach()
        return metrics

    def actor_loss(self, start: RSSMState) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with freeze_parameters(self.world_model, self.critic):
            states, _, entropy = self.imagine(start, deterministic=False)
            features = self.world_model.features_from_sequence(states)
            reward = self.world_model.predict_reward_sequence(states)
            continuation = self.world_model.predict_continuation_sequence(states)
            pcont = self.config.discount * continuation
            value = self.critic(features)
            returns = lambda_return(
                reward[:, :-1],
                value[:, :-1],
                value[:, -1],
                pcont[:, :-1],
                self.config.lambda_,
                stop_gradient=False,
            )
            weights = discount_weights(pcont[:, :-1]).detach()
            objective = (weights * returns).mean()
            entropy_bonus = entropy[:, :-1].mean()
            loss = -objective - self.config.actor_entropy_scale * entropy_bonus
        metrics = {
            "actor_loss": loss.detach(),
            "actor_objective": objective.detach(),
            "actor_entropy": entropy_bonus.detach(),
            "imagined_reward_mean": reward.detach().mean(),
            "imagined_continue_mean": continuation.detach().mean(),
        }
        return loss, metrics

    def critic_loss(self, start: RSSMState) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            states, _, _ = self.imagine(start, deterministic=False)
            features = self.world_model.features_from_sequence(states)
            reward = self.world_model.predict_reward_sequence(states)
            continuation = self.world_model.predict_continuation_sequence(states)
            pcont = self.config.discount * continuation
            target_value = self.target_critic(features)
            returns = lambda_return(
                reward[:, :-1],
                target_value[:, :-1],
                target_value[:, -1],
                pcont[:, :-1],
                self.config.lambda_,
                stop_gradient=True,
            )
            weights = discount_weights(pcont[:, :-1]).detach()
            features = features[:, :-1].detach()
            returns = returns.detach()

        pred = self.critic(features)
        loss = (weights * (pred - returns) ** 2).mean()
        metrics = {
            "critic_loss": loss.detach(),
            "critic_value_mean": pred.detach().mean(),
            "critic_target_mean": returns.detach().mean(),
        }
        return loss, metrics

    def imagine(
        self,
        start: RSSMState,
        deterministic: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        prev = start
        states: list[RSSMState] = []
        actions: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for _ in range(self.config.imagination_horizon):
            features = self.world_model.features_from_state(prev)
            dist = self.actor(features)
            action = dist.mode() if deterministic else dist.rsample()
            prev, _ = self.world_model.rssm.img_step(prev, action, deterministic=False)
            states.append(prev)
            actions.append(action)
            entropies.append(dist.entropy())
        return stack_states(states), torch.stack(actions, dim=1), torch.stack(entropies, dim=1).unsqueeze(-1)

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
            pred = self.world_model.open_loop_predict(obs, action, context_len=context_len, horizon=horizon)
            target = obs[:, context_len + 1 : context_len + 1 + horizon]
            obs_loss = torch.mean((pred - target) ** 2)
            out[f"obs_h{horizon}"] = float(obs_loss.detach().cpu())
            if reward is not None:
                reward_pred = self.world_model.open_loop_predict_rewards(obs, action, context_len=context_len, horizon=horizon)
                reward_target = reward[:, context_len : context_len + horizon]
                reward_loss = torch.mean((reward_pred - reward_target) ** 2)
                out[f"reward_h{horizon}"] = float(reward_loss.detach().cpu())
        return out

    def save_checkpoint(
        self,
        epoch: int,
        best_val_loss: float,
        train_args: dict[str, object],
        is_best: bool,
    ) -> None:
        state = {
            "world_model_state_dict": self.world_model.state_dict(),
            "model_state_dict": self.world_model.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "world_optimizer_state_dict": self.world_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "optimizer_state_dict": self.world_optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "config": self.world_model.config.to_dict(),
            "world_model_config": self.world_model.config.to_dict(),
            "actor_config": self.actor.config.to_dict(),
            "critic_config": self.critic.config.to_dict(),
            "dreamer_config": self.config.to_dict(),
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
            f"actor={val_metrics['actor_loss']:.5f} critic={val_metrics['critic_loss']:.5f} "
            f"{open_loop_str}"
        )


def flatten_state_sequence(states: dict[str, torch.Tensor], drop_last: bool) -> RSSMState:
    """Flatten `[batch, time, dim]` RSSM state tensors into one start-state batch."""

    time_slice = slice(None, -1) if drop_last else slice(None)

    def flatten(name: str) -> torch.Tensor:
        value = states[name][:, time_slice]
        return value.reshape(-1, value.shape[-1]).detach()

    return RSSMState(
        h=flatten("h"),
        z=flatten("z"),
        mean=flatten("mean"),
        std=flatten("std"),
    )


@contextmanager
def freeze_parameters(*modules: nn.Module) -> Iterator[None]:
    params = [param for module in modules for param in module.parameters()]
    previous = [param.requires_grad for param in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, requires_grad in zip(params, previous):
            param.requires_grad_(requires_grad)


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def make_writer(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(run_dir / "tb")
