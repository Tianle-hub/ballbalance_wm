"""Plot DM-Control Dreamer training curves from metrics.jsonl logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = [
    "runs/dmc_cartpole_swingup_pixel_v1_0614",
    "runs/dmc_cartpole_swingup_pixel_v2_0614",
    "runs/dmc_cartpole_swingup_v1_0614",
    "runs/dmc_cartpole_swingup_v2_0614",
]

CURVES = {
    "recon": ("recon_loss", "Reconstruction loss"),
    "reward": ("reward_loss", "Reward loss"),
    "kl": ("kl_loss", "KL loss"),
    "actor": ("actor_loss", "Actor loss"),
    "critic": ("critic_loss", "Critic loss"),
    "return": ("collect_avg_reward", "Return"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Run directories containing metrics.jsonl.")
    parser.add_argument("--out-dir", default="runs/dm_control_loss_plots", help="Directory for PNG plots.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def load_metrics(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows_by_step: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows_by_step[int(row["step"])] = row
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def series(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray] | None:
    points = [(row["step"], row[key]) for row in rows if key in row and np.isfinite(row[key])]
    if not points:
        return None
    x, y = zip(*points)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def plot_one(metric_name: str, metric_key: str, title: str, run_rows: dict[str, list[dict[str, float]]], out_dir: Path, dpi: int) -> bool:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    plotted = False
    panels = [
        ("train", f"train/{metric_key}", axes[0]),
        ("val", f"val/{metric_key}", axes[1]),
    ]
    if metric_name == "return":
        panels = [
            ("collect", "collect/collect_avg_reward", axes[0]),
            ("val", "val/return", axes[1]),
        ]

    for panel_name, key, ax in panels:
        panel_plotted = False
        for label, rows in run_rows.items():
            values = series(rows, key)
            if values is None:
                continue
            x, y = values
            ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.4, label=label)
            panel_plotted = True
            plotted = True
        ax.set_title(f"{title} - {panel_name}")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.25)
        if panel_plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "not logged", ha="center", va="center", transform=ax.transAxes)
    axes[1].set_xlabel("iteration")
    fig.suptitle(title)
    if plotted:
        out_path = out_dir / f"{metric_name}.png"
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return plotted


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = {Path(run).name: load_metrics(Path(run)) for run in args.runs}
    missing = [name for name, rows in run_rows.items() if not rows]
    if missing:
        print("No metrics.jsonl for:", ", ".join(missing))

    wrote = []
    for metric_name, (metric_key, title) in CURVES.items():
        if plot_one(metric_name, metric_key, title, run_rows, out_dir, args.dpi):
            wrote.append(out_dir / f"{metric_name}.png")
    if wrote:
        print("Wrote plots:")
        for path in wrote:
            print(f"  {path}")
    else:
        print("No plots written because no matching metrics were found.")


if __name__ == "__main__":
    main()
