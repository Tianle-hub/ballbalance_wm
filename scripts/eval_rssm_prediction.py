"""Evaluate posterior, one-step prior, and open-loop RSSM predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

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

    metrics, curves = evaluate(model, loader, normalizer, device, args.context_len, args.horizon, args.num_batches)
    out_dir = Path(args.checkpoint).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_curve(curves["open_loop_mse_curve"], out_dir / "open_loop_mse_curve.png", ylabel="observation MSE")
    reward_continuation_curves = {
        label: curves[label]
        for label in ("open_loop_reward_mse_curve", "open_loop_continuation_mse_curve")
        if label in curves
    }
    if reward_continuation_curves:
        plot_curves(
            reward_continuation_curves,
            out_dir / "open_loop_reward_continuation_mse_curve.png",
            ylabel="mean MSE",
        )
    if "open_loop_reward_mse_curve" in curves:
        plot_curve(
            curves["open_loop_reward_mse_curve"],
            out_dir / "open_loop_reward_mse_curve.png",
            ylabel="reward MSE",
        )
    if "open_loop_continuation_mse_curve" in curves:
        plot_curve(
            curves["open_loop_continuation_mse_curve"],
            out_dir / "open_loop_continuation_mse_curve.png",
            ylabel="continuation MSE",
        )
    print(json.dumps(metrics, indent=2))


def evaluate(
    model: WorldModel,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    context_len: int,
    horizon: int,
    num_batches: int,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    recon_losses: list[torch.Tensor] = []
    one_step_losses: list[torch.Tensor] = []
    open_loop_losses: list[torch.Tensor] = []
    reward_losses: list[torch.Tensor] = []
    open_loop_reward_losses: list[torch.Tensor] = []
    continuation_losses: list[torch.Tensor] = []
    open_loop_continuation_losses: list[torch.Tensor] = []

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
            continuation_logit = out["continuation_logit"]
            assert isinstance(recon, torch.Tensor)
            assert isinstance(prior_recon, torch.Tensor)
            assert isinstance(reward_pred, torch.Tensor)
            assert isinstance(continuation_logit, torch.Tensor)

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

            continuation_source = batch.get("terminated", batch.get("done"))
            if continuation_source is not None:
                continuation_target = 1.0 - continuation_source
                continuation_prob = torch.sigmoid(continuation_logit[:, 1:])
                continuation_losses.append((continuation_prob - continuation_target) ** 2)

                open_continuation = model.open_loop_predict_continuation(
                    obs,
                    action,
                    context_len=context_len,
                    horizon=horizon,
                )
                target_continuation = continuation_target[:, context_len : context_len + horizon]
                open_loop_continuation_losses.append((open_continuation - target_continuation) ** 2)

    recon_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in recon_losses], dim=0)
    one_step_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in one_step_losses], dim=0)
    open_stacked = torch.cat(open_loop_losses, dim=0)
    open_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in open_loop_losses], dim=0)
    metrics: dict[str, object] = {
        "posterior_reconstruction_mse": float(recon_all.mean().cpu()),
        "one_step_prior_mse": float(one_step_all.mean().cpu()),
        "open_loop_mse": float(open_all.mean().cpu()),
        "open_loop_mse_per_dim": {label: float(open_all[:, i].mean().cpu()) for i, label in enumerate(OBS_LABELS)},
    }
    curves = {
        "open_loop_mse_curve": tensor_to_float_list(open_stacked.mean(dim=(0, 2))),
    }
    if reward_losses:
        reward_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in reward_losses], dim=0)
        open_reward_stacked = torch.cat(open_loop_reward_losses, dim=0)
        open_reward_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in open_loop_reward_losses], dim=0)
        metrics.update(
            {
                "posterior_reward_mse": float(reward_all.mean().cpu()),
                "open_loop_reward_mse": float(open_reward_all.mean().cpu()),
            }
        )
        curves["open_loop_reward_mse_curve"] = tensor_to_float_list(open_reward_stacked.mean(dim=(0, 2)))
    if continuation_losses:
        continuation_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in continuation_losses], dim=0)
        open_continuation_stacked = torch.cat(open_loop_continuation_losses, dim=0)
        open_continuation_all = torch.cat(
            [x.reshape(-1, x.shape[-1]) for x in open_loop_continuation_losses],
            dim=0,
        )
        metrics.update(
            {
                "posterior_continuation_mse": float(continuation_all.mean().cpu()),
                "open_loop_continuation_mse": float(open_continuation_all.mean().cpu()),
            }
        )
        curves["open_loop_continuation_mse_curve"] = tensor_to_float_list(open_continuation_stacked.mean(dim=(0, 2)))
    return metrics, curves


def tensor_to_float_list(values: torch.Tensor) -> list[float]:
    return [float(x) for x in values.detach().cpu()]


def pretty_curve_label(label: str) -> str:
    return label.removeprefix("open_loop_").removesuffix("_curve").replace("_", " ")


def plot_curve(values: list[float], out_path: Path, ylabel: str = "MSE") -> None:
    plot_curves({ylabel: values}, out_path, ylabel=ylabel)


def plot_curves(curves: dict[str, list[float]], out_path: Path, ylabel: str = "MSE") -> None:
    ensure_writable_matplotlib_cache()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in curves.items():
        steps = range(1, len(values) + 1)
        ax.plot(steps, values, label=pretty_curve_label(label))
    ax.set_xlabel("prediction horizon step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Open-loop {ylabel} vs horizon")
    if len(curves) > 1:
        ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
