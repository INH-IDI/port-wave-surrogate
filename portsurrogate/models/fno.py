"""Fourier Neural Operator (FNO) 2D for high-fidelity coastal wave simulation emulation.

Based on Li et al. (2020) with water-masked relative L2 loss.
"""
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier layer: performs discrete Fourier transform, linear transform in frequency, and inverse transform."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication using Einstein summation convention: (batch, in, x, y) * (in, out, x, y) -> (batch, out, x, y)."""
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        # Real FFT2
        x_ft = torch.fft.rfft2(x)

        # Allocate frequency output tensor
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Multiply relevant Fourier modes
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Inverse Real FFT2
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class ConvMLP(nn.Module):
    """Pointwise MLP implemented via 1x1 convolutions."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super(ConvMLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    """2D Fourier Neural Operator emulating significant wave height (Hm0).

    Args:
        modes1: Retained Fourier modes in row dimension.
        modes2: Retained Fourier modes in col dimension.
        width: Latent channel dimension (e.g. 32 for compact v2, 64 for baseline v1).
        input_channels: Number of physical input features (default = 4: [Hmo, Fp, Theta, Depth]).
        output_channels: Number of target output variables (default = 1: [Hm0]).
        padding: Zero-padding along boundary edges to handle non-periodic conditions (e.g. 4 or 9).
    """

    def __init__(
        self,
        modes1: int = 45,
        modes2: int = 45,
        width: int = 32,
        input_channels: int = 4,
        output_channels: int = 1,
        padding: int = 4,
    ):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Linear lifting from physical space (+2 coordinate grid channels) to latent width
        self.p = nn.Linear(2 + self.input_channels, self.width)

        # 4 Fourier spectral layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Intermediate MLPs
        self.mlp0 = ConvMLP(self.width, self.width, self.width)
        self.mlp1 = ConvMLP(self.width, self.width, self.width)
        self.mlp2 = ConvMLP(self.width, self.width, self.width)
        self.mlp3 = ConvMLP(self.width, self.width, self.width)

        # Bypass 1x1 convolutions
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Final projection MLP
        self.q = ConvMLP(self.width, self.output_channels, self.width * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects x of shape (batch, height, width, input_channels)."""
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # Layer 1
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        # Layer 2
        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        # Layer 3
        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        # Layer 4
        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.q(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, output_channels)
        return x

    @staticmethod
    def get_grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Normalized spatial coordinates grid [0, 1] x [0, 1]."""
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


class CustomNormalizer:
    """Z-score channel-wise normalizer for tensors."""

    def __init__(self, mean: Union[torch.Tensor, np.ndarray], std: Union[torch.Tensor, np.ndarray], eps: float = 1e-5):
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean.astype(np.float32))
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std.astype(np.float32))

        self.mean = mean
        self.std = std
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode physical values: (x - mean) / (std + eps)."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / (std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode normalized values back to physical units."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x * (std + self.eps)) + mean


class MaskedRelL2(nn.Module):
    """Mean relative L2 loss computed strictly over water cells."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:
            pred: (B, H, W, C)
            target: (B, H, W, C)
            mask: (H, W) or (B, H, W, 1) binary mask
        """
        if mask.ndim == 2:
            m = mask[None, :, :, None]
        else:
            m = mask

        diff = (pred - target) * m
        # Reshape to (B, C, H*W)
        diff_flat = diff.permute(0, 3, 1, 2).reshape(diff.shape[0], diff.shape[3], -1)
        targ_flat = (target * m).permute(0, 3, 1, 2).reshape(target.shape[0], target.shape[3], -1)

        num = torch.norm(diff_flat, dim=2)
        den = torch.norm(targ_flat, dim=2).clamp_min(1e-8)
        return (num / den).mean()
