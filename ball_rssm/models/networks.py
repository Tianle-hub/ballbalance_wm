"""Neural network building blocks for RSSM world models."""

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


class ConvEncoder(nn.Module):
    """Encode image observations into Dreamer observation embeddings."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        embed_dim: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError("ConvEncoder obs_shape must be [channels, height, width]")
        if not channels:
            raise ValueError("ConvEncoder requires at least one convolution channel")

        in_channels, height, width = obs_shape
        layers: list[nn.Module] = []
        current_channels = in_channels
        spatial_shapes = [(height, width)]
        for out_channels in channels:
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2))
            layers.append(activation())
            height, width = conv2d_hw(height, width, kernel_size=4, stride=2)
            if height <= 0 or width <= 0:
                raise ValueError("obs_shape is too small for the configured ConvEncoder")
            spatial_shapes.append((height, width))
            current_channels = out_channels

        self.obs_shape = obs_shape
        self.channels = channels
        self.spatial_shapes = tuple(spatial_shapes)
        self.conv = nn.Sequential(*layers)
        self.conv_out_shape = (channels[-1], height, width)
        self.conv_out_dim = channels[-1] * height * width
        self.fc = nn.Linear(self.conv_out_dim, embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError("ConvEncoder input must have shape [batch, channels, height, width]")
        conv = self.conv(obs)
        return self.fc(conv.reshape(conv.shape[0], -1))


class ConvDecoder(nn.Module):
    """Decode Dreamer features back into image observations."""

    def __init__(
        self,
        feature_dim: int,
        obs_shape: tuple[int, int, int],
        channels: tuple[int, ...] = (32, 64, 128, 256),
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError("ConvDecoder obs_shape must be [channels, height, width]")
        if not channels:
            raise ValueError("ConvDecoder requires at least one convolution channel")

        image_channels, height, width = obs_shape
        spatial_shapes = [(height, width)]
        for _ in channels:
            height, width = conv2d_hw(height, width, kernel_size=4, stride=2)
            if height <= 0 or width <= 0:
                raise ValueError("obs_shape is too small for the configured ConvDecoder")
            spatial_shapes.append((height, width))

        self.obs_shape = obs_shape
        self.channels = channels
        self.spatial_shapes = tuple(spatial_shapes)
        self.fc = nn.Linear(feature_dim, channels[-1] * height * width)
        self.activation = activation()

        decoder_layers: list[nn.ConvTranspose2d] = []
        reversed_channels = tuple(reversed(channels))
        for index, in_channels in enumerate(reversed_channels):
            is_last = index == len(reversed_channels) - 1
            out_channels = image_channels if is_last else reversed_channels[index + 1]
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2))
        self.deconvs = nn.ModuleList(decoder_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("ConvDecoder input must have shape [batch, feature_dim]")

        batch_size = features.shape[0]
        last_height, last_width = self.spatial_shapes[-1]
        x = self.fc(features).reshape(batch_size, self.channels[-1], last_height, last_width)
        target_shapes = tuple(reversed(self.spatial_shapes[:-1]))
        for index, (layer, target_hw) in enumerate(zip(self.deconvs, target_shapes)):
            target_channels = layer.out_channels
            x = layer(x, output_size=(batch_size, target_channels, *target_hw))
            if index < len(self.deconvs) - 1:
                x = self.activation(x)
        return x


def conv2d_hw(height: int, width: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """Return the spatial size after a no-padding Conv2d."""

    height_out = (height - kernel_size) // stride + 1
    width_out = (width - kernel_size) // stride + 1
    return height_out, width_out
