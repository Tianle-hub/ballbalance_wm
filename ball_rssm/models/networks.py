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
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build a small fully connected network with repeated hidden blocks."""

    if depth < 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(current, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
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
        channels: tuple[int, ...] | None = None,
        kernels: tuple[int, ...] = (4, 4, 4, 4),
        cnn_depth: int = 48,
        activation: type[nn.Module] = nn.ELU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError("ConvEncoder obs_shape must be [channels, height, width]")
        if channels is None:
            channels = tuple(cnn_depth * (2**index) for index in range(len(kernels)))
        if not channels:
            raise ValueError("ConvEncoder requires at least one convolution channel")
        if len(channels) != len(kernels):
            raise ValueError("ConvEncoder channels and kernels must have the same length")

        in_channels, height, width = obs_shape
        layers: list[nn.Module] = []
        current_channels = in_channels
        spatial_shapes = [(height, width)]
        for out_channels, kernel in zip(channels, kernels):
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=kernel, stride=2))
            if layer_norm:
                layers.append(nn.GroupNorm(1, out_channels))
            layers.append(activation())
            height, width = conv2d_hw(height, width, kernel_size=kernel, stride=2)
            if height <= 0 or width <= 0:
                raise ValueError("obs_shape is too small for the configured ConvEncoder")
            spatial_shapes.append((height, width))
            current_channels = out_channels

        self.obs_shape = obs_shape
        self.channels = channels
        self.kernels = kernels
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
        channels: tuple[int, ...] | None = None,
        kernels: tuple[int, ...] = (5, 5, 6, 6),
        cnn_depth: int = 48,
        activation: type[nn.Module] = nn.ELU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError("ConvDecoder obs_shape must be [channels, height, width]")
        if channels is None:
            channels = (
                32 * cnn_depth,
                *(cnn_depth * (2 ** (len(kernels) - index - 2)) for index in range(len(kernels) - 1)),
            )
        if not channels:
            raise ValueError("ConvDecoder requires at least one hidden convolution channel")
        if len(kernels) < 1:
            raise ValueError("ConvDecoder requires at least one kernel")
        if len(channels) != len(kernels):
            raise ValueError("ConvDecoder channels and kernels must have the same length")

        image_channels, target_height, target_width = obs_shape
        height, width = 1, 1
        spatial_shapes = [(height, width)]
        for kernel in kernels:
            height, width = conv_transpose2d_hw(height, width, kernel_size=kernel, stride=2)
            spatial_shapes.append((height, width))
        if (height, width) != (target_height, target_width):
            raise ValueError(
                "ConvDecoder kernels must expand from 1x1 to the image size; "
                f"kernels={kernels} produce {(height, width)} for target {(target_height, target_width)}"
            )

        self.obs_shape = obs_shape
        self.channels = channels
        self.kernels = kernels
        self.spatial_shapes = tuple(spatial_shapes)
        self.fc = nn.Linear(feature_dim, channels[0])
        self.activation = activation()

        deconvs: list[nn.ConvTranspose2d] = []
        norms: list[nn.Module] = []
        in_channels = channels[0]
        out_channels_sequence = (*channels[1:], image_channels)
        for index, (out_channels, kernel) in enumerate(zip(out_channels_sequence, kernels)):
            deconvs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=2))
            if index < len(kernels) - 1:
                norms.append(nn.GroupNorm(1, out_channels) if layer_norm else nn.Identity())
            in_channels = out_channels
        self.deconvs = nn.ModuleList(deconvs)
        self.norms = nn.ModuleList(norms)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("ConvDecoder input must have shape [batch, feature_dim]")

        batch_size = features.shape[0]
        x = self.fc(features).reshape(batch_size, self.channels[0], 1, 1)
        for index, layer in enumerate(self.deconvs):
            x = layer(x)
            if index < len(self.deconvs) - 1:
                x = self.norms[index](x)
                x = self.activation(x)
        return x


def conv2d_hw(height: int, width: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """Return the spatial size after a no-padding Conv2d."""

    height_out = (height - kernel_size) // stride + 1
    width_out = (width - kernel_size) // stride + 1
    return height_out, width_out


def conv_transpose2d_hw(height: int, width: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """Return the spatial size after a no-padding ConvTranspose2d."""

    height_out = (height - 1) * stride + kernel_size
    width_out = (width - 1) * stride + kernel_size
    return height_out, width_out
