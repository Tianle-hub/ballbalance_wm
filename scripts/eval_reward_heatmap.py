"""Evaluate learned reward as a position-binned heatmap over buffer transitions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--position-limit", type=float, default=0.5)
    parser.add_argument("--max-transitions", type=int, default=200_000)
    parser.add_argument("--include-padding", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bins <= 1:
        parser.error("--bins must be greater than 1")
    if args.position_limit <= 0.0:
        parser.error("--position-limit must be positive")
    if args.max_transitions <= 0:
        parser.error("--max-transitions must be positive")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, device)
    model = WorldModel(WorldModelConfig.from_dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = Normalizer.load_state_dict(checkpoint["normalizer"]).to(device)

    arrays = load_buffer_arrays(args.dataset)
    episode_indices = sample_episode_indices(arrays["obs"].shape[0], args.num_episodes, rng)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "reward_heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap = RewardHeatmapAccumulator(
        bins=args.bins,
        position_limit=args.position_limit,
        has_true_reward="reward" in arrays,
    )
    transition_chunks: list[dict[str, np.ndarray]] = []

    for start in range(0, len(episode_indices), args.batch_size):
        batch_indices = episode_indices[start : start + args.batch_size]
        batch = predict_batch(
            model=model,
            normalizer=normalizer,
            arrays=arrays,
            episode_indices=batch_indices,
            device=device,
            include_padding=args.include_padding,
        )
        heatmap.update(batch)
        transition_chunks.append(batch)

    transitions = concat_transition_chunks(transition_chunks)
    transitions = sample_transition_table(transitions, args.max_transitions, rng)
    surfaces = heatmap.surfaces()
    metrics = compute_metrics(transitions, heatmap.counts)

    np.savez_compressed(out_dir / "reward_transition_sample.npz", **transitions)
    np.savez_compressed(
        out_dir / "reward_heatmap_data.npz",
        x_edges=heatmap.x_edges,
        y_edges=heatmap.y_edges,
        counts=heatmap.counts,
        true_reward=surfaces["true_reward"],
        posterior_reward=surfaces["posterior_reward"],
        prior_reward=surfaces["prior_reward"],
    )
    (out_dir / "reward_heatmap_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_heatmaps(
        out_path=out_dir / "reward_position_heatmaps.png",
        x_edges=heatmap.x_edges,
        y_edges=heatmap.y_edges,
        counts=heatmap.counts,
        surfaces=surfaces,
    )
    print(json.dumps(metrics, indent=2))
    print(f"wrote reward heatmap outputs to {out_dir}")


def load_buffer_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        arrays = {
            "obs": loaded["obs"].astype(np.float32),
            "action": loaded["action"].astype(np.float32),
        }
        for key in ("reward", "done", "terminated", "truncated"):
            if key in loaded:
                arrays[key] = loaded[key].astype(np.float32)

    obs = arrays["obs"]
    action = arrays["action"]
    if obs.ndim != 3 or action.ndim != 3:
        raise ValueError("dataset must contain obs [N, T+1, obs_dim] and action [N, T, action_dim]")
    if obs.shape[0] != action.shape[0] or obs.shape[1] != action.shape[1] + 1:
        raise ValueError("expected obs.shape[:2] to be [N, T+1] and action.shape[:2] to be [N, T]")
    return arrays


def sample_episode_indices(num_episodes: int, requested: int, rng: np.random.Generator) -> np.ndarray:
    count = min(requested, num_episodes)
    return np.sort(rng.choice(num_episodes, size=count, replace=False))


def predict_batch(
    model: WorldModel,
    normalizer: Normalizer,
    arrays: dict[str, np.ndarray],
    episode_indices: np.ndarray,
    device: torch.device,
    include_padding: bool,
) -> dict[str, np.ndarray]:
    obs_np = arrays["obs"][episode_indices]
    action_np = arrays["action"][episode_indices]
    reward_np = arrays.get("reward")
    done_np = arrays.get("done")
    terminated_np = arrays.get("terminated")

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    action = torch.as_tensor(action_np, dtype=torch.float32, device=device)
    obs_norm = normalizer.normalize_obs(obs)
    action_norm = normalizer.normalize_action(action)

    with torch.no_grad():
        out = model.forward(obs_norm, action_norm)
        posterior_reward = out["reward_pred"]
        prior_reward = out["prior_reward_pred"]
        assert isinstance(posterior_reward, torch.Tensor)
        assert isinstance(prior_reward, torch.Tensor)
        posterior_reward = normalizer.denormalize_reward(posterior_reward[:, 1:]).squeeze(-1)
        prior_reward = normalizer.denormalize_reward(prior_reward[:, 1:]).squeeze(-1)

    mask = transition_mask(done_np[episode_indices] if done_np is not None else None, action_np.shape[:2])
    if include_padding:
        mask = np.ones_like(mask)

    next_obs = obs_np[:, 1:]
    flat_mask = mask.reshape(-1)
    out_batch = {
        "x": next_obs[..., 0].reshape(-1)[flat_mask],
        "y": next_obs[..., 1].reshape(-1)[flat_mask],
        "vx": next_obs[..., 2].reshape(-1)[flat_mask],
        "vy": next_obs[..., 3].reshape(-1)[flat_mask],
        "theta_x": next_obs[..., 4].reshape(-1)[flat_mask],
        "theta_y": next_obs[..., 5].reshape(-1)[flat_mask],
        "action_x": action_np[..., 0].reshape(-1)[flat_mask],
        "action_y": action_np[..., 1].reshape(-1)[flat_mask],
        "posterior_reward": posterior_reward.detach().cpu().numpy().reshape(-1)[flat_mask],
        "prior_reward": prior_reward.detach().cpu().numpy().reshape(-1)[flat_mask],
    }
    if reward_np is not None:
        out_batch["true_reward"] = reward_np[episode_indices].squeeze(-1).reshape(-1)[flat_mask]
    else:
        out_batch["true_reward"] = np.full(flat_mask.sum(), np.nan, dtype=np.float32)
    if done_np is not None:
        out_batch["done"] = done_np[episode_indices].squeeze(-1).reshape(-1)[flat_mask].astype(np.float32)
    else:
        out_batch["done"] = np.zeros(flat_mask.sum(), dtype=np.float32)
    if terminated_np is not None:
        out_batch["terminated"] = terminated_np[episode_indices].squeeze(-1).reshape(-1)[flat_mask].astype(np.float32)
    else:
        out_batch["terminated"] = out_batch["done"].copy()
    return {key: value.astype(np.float32, copy=False) for key, value in out_batch.items()}


def transition_mask(done: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if done is None:
        return np.ones(shape, dtype=bool)
    done_bool = done.squeeze(-1).astype(bool)
    effective = np.ones_like(done_bool, dtype=bool)
    effective[:, 1:] = ~done_bool[:, :-1]
    return effective


class RewardHeatmapAccumulator:
    def __init__(self, bins: int, position_limit: float, has_true_reward: bool) -> None:
        self.x_edges = np.linspace(-position_limit, position_limit, bins + 1, dtype=np.float32)
        self.y_edges = np.linspace(-position_limit, position_limit, bins + 1, dtype=np.float32)
        self.counts = np.zeros((bins, bins), dtype=np.int64)
        self.posterior_sum = np.zeros((bins, bins), dtype=np.float64)
        self.prior_sum = np.zeros((bins, bins), dtype=np.float64)
        self.true_sum = np.zeros((bins, bins), dtype=np.float64)
        self.has_true_reward = has_true_reward

    def update(self, transitions: dict[str, np.ndarray]) -> None:
        x = transitions["x"]
        y = transitions["y"]
        ix = np.searchsorted(self.x_edges, x, side="right") - 1
        iy = np.searchsorted(self.y_edges, y, side="right") - 1
        valid = (ix >= 0) & (ix < self.counts.shape[1]) & (iy >= 0) & (iy < self.counts.shape[0])
        if not np.any(valid):
            return

        rows = iy[valid]
        cols = ix[valid]
        np.add.at(self.counts, (rows, cols), 1)
        np.add.at(self.posterior_sum, (rows, cols), transitions["posterior_reward"][valid])
        np.add.at(self.prior_sum, (rows, cols), transitions["prior_reward"][valid])
        if self.has_true_reward:
            np.add.at(self.true_sum, (rows, cols), transitions["true_reward"][valid])

    def surfaces(self) -> dict[str, np.ndarray]:
        return {
            "true_reward": safe_divide(self.true_sum, self.counts),
            "posterior_reward": safe_divide(self.posterior_sum, self.counts),
            "prior_reward": safe_divide(self.prior_sum, self.counts),
        }


def safe_divide(total: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, counts, out=out, where=counts > 0)
    return out


def concat_transition_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("no transitions were collected")
    keys = chunks[0].keys()
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def sample_transition_table(
    transitions: dict[str, np.ndarray],
    max_transitions: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    total = len(next(iter(transitions.values())))
    if total <= max_transitions:
        return transitions
    keep = np.sort(rng.choice(total, size=max_transitions, replace=False))
    return {key: value[keep] for key, value in transitions.items()}


def compute_metrics(transitions: dict[str, np.ndarray], counts: np.ndarray) -> dict[str, object]:
    true_reward = transitions["true_reward"]
    has_true = np.isfinite(true_reward).any()
    metrics: dict[str, object] = {
        "num_sampled_transitions": int(len(transitions["x"])),
        "num_filled_position_bins": int((counts > 0).sum()),
        "max_bin_count": int(counts.max()) if counts.size else 0,
        "mean_posterior_reward": float(np.mean(transitions["posterior_reward"])),
        "mean_prior_reward": float(np.mean(transitions["prior_reward"])),
    }
    if has_true:
        metrics.update(
            {
                "mean_true_reward": float(np.mean(true_reward)),
                "posterior_reward_mse": float(np.mean((transitions["posterior_reward"] - true_reward) ** 2)),
                "prior_reward_mse": float(np.mean((transitions["prior_reward"] - true_reward) ** 2)),
            }
        )
    return metrics


def plot_heatmaps(
    out_path: Path,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    counts: np.ndarray,
    surfaces: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    panels = [
        ("true_reward", "True reward"),
        ("posterior_reward", "Posterior predicted reward"),
        ("prior_reward", "One-step prior predicted reward"),
    ]
    values = [surfaces[key] for key, _ in panels if np.isfinite(surfaces[key]).any()]
    vmin = min(float(np.nanmin(value)) for value in values)
    vmax = max(float(np.nanmax(value)) for value in values)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for ax, (key, title) in zip(axes[:3], panels, strict=True):
        image = ax.imshow(
            surfaces[key],
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("ball x")
        ax.set_ylabel("ball y")
        fig.colorbar(image, ax=ax, shrink=0.8)

    positive_counts = counts[counts > 0]
    count_norm = LogNorm(vmin=1, vmax=float(positive_counts.max())) if positive_counts.size else None
    count_image = axes[3].imshow(
        np.ma.masked_where(counts <= 0, counts),
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="magma",
        norm=count_norm,
        aspect="equal",
    )
    axes[3].set_title("Transition count (log)")
    axes[3].set_xlabel("ball x")
    axes[3].set_ylabel("ball y")
    fig.colorbar(count_image, ax=axes[3], shrink=0.8)
    fig.suptitle("Reward averaged by next ball position, marginalizing board angles and actions")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
