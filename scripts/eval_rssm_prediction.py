"""Evaluate posterior, one-step prior, and open-loop RSSM predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.data.sequence_dataset import SequenceDataset, batch_to_device
from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from ball_rssm.utils.checkpoint import load_checkpoint

OBS_LABELS = ["x", "y", "vx", "vy", "theta_x", "theta_y"]


def ensure_writable_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        config_dir = Path.home() / ".config" / "matplotlib"
        if not is_path_or_parent_writable(config_dir):
            fallback = Path(tempfile.gettempdir()) / "ballbalance_matplotlib"
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(fallback)

    if "XDG_CACHE_HOME" not in os.environ:
        cache_dir = Path.home() / ".cache"
        if not is_path_or_parent_writable(cache_dir):
            fallback_cache = Path(tempfile.gettempdir()) / "ballbalance_cache"
            fallback_cache.mkdir(parents=True, exist_ok=True)
            os.environ["XDG_CACHE_HOME"] = str(fallback_cache)


def is_path_or_parent_writable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    return path.parent.exists() and os.access(path.parent, os.W_OK)


def main() -> None:
    ensure_writable_matplotlib_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = WorldModel(WorldModelConfig.from_dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = Normalizer.load_state_dict(checkpoint["normalizer"]).to(device)

    seq_len = args.context_len + args.horizon
    dataset = SequenceDataset(args.dataset, seq_len=seq_len, split="all")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate(model, loader, normalizer, device, args.context_len, args.horizon, args.num_batches)
    out_dir = Path(args.checkpoint).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_curve(metrics["open_loop_mse_curve"], out_dir / "open_loop_mse_curve.png")
    if "open_loop_reward_mse_curve" in metrics:
        plot_curve(metrics["open_loop_reward_mse_curve"], out_dir / "open_loop_reward_mse_curve.png", ylabel="reward MSE")
    print(json.dumps(metrics, indent=2))


def evaluate(
    model: WorldModel,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    context_len: int,
    horizon: int,
    num_batches: int,
) -> dict[str, object]:
    recon_losses: list[torch.Tensor] = []
    one_step_losses: list[torch.Tensor] = []
    open_loop_losses: list[torch.Tensor] = []
    reward_losses: list[torch.Tensor] = []
    open_loop_reward_losses: list[torch.Tensor] = []
    curve_sum = torch.zeros(horizon, device=device)
    reward_curve_sum = torch.zeros(horizon, device=device)
    curve_count = 0
    reward_curve_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break
            batch = batch_to_device(batch, device)
            obs = normalizer.normalize_obs(batch["obs"])
            action = normalizer.normalize_action(batch["action"])
            reward = normalizer.normalize_reward(batch["reward"]) if "reward" in batch else None
            out = model.forward(obs, action)
            recon = out["recon"]
            prior_recon = out["prior_recon"]
            reward_pred = out["reward_pred"]
            assert isinstance(recon, torch.Tensor)
            assert isinstance(prior_recon, torch.Tensor)
            assert isinstance(reward_pred, torch.Tensor)

            pred = model.open_loop_predict(obs, action, context_len=context_len, horizon=horizon)
            target_open = obs[:, context_len + 1 : context_len + 1 + horizon]
            pred_denorm = normalizer.denormalize_obs(pred)
            target_denorm = normalizer.denormalize_obs(target_open)
            recon_denorm = normalizer.denormalize_obs(recon)
            prior_denorm = normalizer.denormalize_obs(prior_recon)
            obs_denorm = normalizer.denormalize_obs(obs)

            recon_losses.append((recon_denorm[:, 1:] - obs_denorm[:, 1:]) ** 2)
            one_step_losses.append((prior_denorm[:, 1:] - obs_denorm[:, 1:]) ** 2)
            open_err = (pred_denorm - target_denorm) ** 2
            open_loop_losses.append(open_err)
            curve_sum += open_err.mean(dim=(0, 2))
            curve_count += 1

            if reward is not None:
                reward_pred_denorm = normalizer.denormalize_reward(reward_pred[:, 1:])
                reward_target_denorm = normalizer.denormalize_reward(reward)
                reward_losses.append((reward_pred_denorm - reward_target_denorm) ** 2)

                open_reward = model.open_loop_predict_rewards(obs, action, context_len=context_len, horizon=horizon)
                target_reward = reward[:, context_len : context_len + horizon]
                open_reward_denorm = normalizer.denormalize_reward(open_reward)
                target_reward_denorm = normalizer.denormalize_reward(target_reward)
                open_reward_err = (open_reward_denorm - target_reward_denorm) ** 2
                open_loop_reward_losses.append(open_reward_err)
                reward_curve_sum += open_reward_err.mean(dim=(0, 2))
                reward_curve_count += 1

    recon_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in recon_losses], dim=0)
    one_step_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in one_step_losses], dim=0)
    open_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in open_loop_losses], dim=0)
    metrics: dict[str, object] = {
        "posterior_reconstruction_mse": float(recon_all.mean().cpu()),
        "one_step_prior_mse": float(one_step_all.mean().cpu()),
        "open_loop_mse": float(open_all.mean().cpu()),
        "open_loop_mse_per_dim": {label: float(open_all[:, i].mean().cpu()) for i, label in enumerate(OBS_LABELS)},
        "open_loop_mse_curve": [float(x) for x in (curve_sum / max(curve_count, 1)).detach().cpu()],
    }
    if reward_losses:
        reward_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in reward_losses], dim=0)
        open_reward_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in open_loop_reward_losses], dim=0)
        metrics.update(
            {
                "posterior_reward_mse": float(reward_all.mean().cpu()),
                "open_loop_reward_mse": float(open_reward_all.mean().cpu()),
                "open_loop_reward_mse_curve": [
                    float(x) for x in (reward_curve_sum / max(reward_curve_count, 1)).detach().cpu()
                ],
            }
        )
    return metrics


def plot_curve(values: list[float], out_path: Path, ylabel: str = "MSE") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(values) + 1), values)
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Open-loop RSSM {ylabel} vs horizon")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
