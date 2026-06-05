"""Small MLP building blocks for low-dimensional RSSM models."""

from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int = 2,
    activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
    """Build a small fully connected network with repeated hidden blocks."""

    if depth < 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(activation())
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


def softplus_std(raw_std: torch.Tensor, min_std: float) -> torch.Tensor:
    """Map raw standard-deviation logits to positive Gaussian std values."""

    return torch.nn.functional.softplus(raw_std) + min_std
