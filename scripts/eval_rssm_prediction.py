"""Evaluate posterior, one-step prior, and open-loop RSSM predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.data.sequence_dataset import SequenceDataset, batch_to_device
from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from ball_rssm.utils.checkpoint import load_checkpoint
from ball_rssm.utils.plotting import OBS_LABELS


def main() -> None:
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
    model = WorldModel(WorldModelConfig(**checkpoint["config"])).to(device)
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
    curve_sum = torch.zeros(horizon, device=device)
    curve_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break
            batch = batch_to_device(batch, device)
            obs = normalizer.normalize_obs(batch["obs"])
            action = normalizer.normalize_action(batch["action"])
            out = model.forward(obs, action)
            recon = out["recon"]
            prior_recon = out["prior_recon"]
            assert isinstance(recon, torch.Tensor)
            assert isinstance(prior_recon, torch.Tensor)

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

    recon_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in recon_losses], dim=0)
    one_step_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in one_step_losses], dim=0)
    open_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in open_loop_losses], dim=0)
    return {
        "posterior_reconstruction_mse": float(recon_all.mean().cpu()),
        "one_step_prior_mse": float(one_step_all.mean().cpu()),
        "open_loop_mse": float(open_all.mean().cpu()),
        "open_loop_mse_per_dim": {label: float(open_all[:, i].mean().cpu()) for i, label in enumerate(OBS_LABELS)},
        "open_loop_mse_curve": [float(x) for x in (curve_sum / max(curve_count, 1)).detach().cpu()],
    }


def plot_curve(values: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(values) + 1), values)
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("MSE")
    ax.set_title("Open-loop RSSM MSE vs horizon")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
