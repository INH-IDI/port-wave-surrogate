"""Dataset building, tensor assembly, normalization statistics, and spatial pooling."""
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from .config import PortConfig, CropConfig


def coarsen_mean(X: np.ndarray, Y: np.ndarray, water_mask: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact cell-average spatial pooling (kernel = stride = factor) masked strictly over water cells.

    Prevents contamination from dry beach artifacts.
    Args:
        X: (N, H, W, C) physical input tensor (channel 3 is Depth).
        Y: (N, H, W, Cy) or (N, H, W) physical target tensor (hm0).
        water_mask: (H, W) binary water mask.
        factor: Downsampling integer factor (e.g. 2 for 360->180, 4 for 360->90).
    Returns:
        (Xc, Yc, mc): coarsened input, target, and coarsened water mask.
    """
    sq = False
    if Y.ndim == 3:
        Y = Y[..., None]
        sq = True

    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).permute(0, 3, 1, 2)
    Yt = torch.from_numpy(np.asarray(Y, dtype=np.float32)).permute(0, 3, 1, 2)
    mt = torch.from_numpy(np.asarray(water_mask, dtype=np.float32))[None, None]

    k = factor
    frac = F.avg_pool2d(mt, k, stride=k)  # Water fraction in coarse cell
    mc = (frac[0, 0] >= 0.5).float()
    cnt = frac.clamp_min(1e-8)

    # Channels 0-2 of X are spatially uniform (direct avg pool)
    Xc = F.avg_pool2d(Xt, k, stride=k)
    # Depth channel is averaged only over water cells
    Xc[:, 3:4] = F.avg_pool2d(Xt[:, 3:4] * mt, k, stride=k) / cnt
    # Target Hm0 is averaged only over water cells
    Yc = F.avg_pool2d(Yt * mt, k, stride=k) / cnt

    Yc_np = Yc.permute(0, 2, 3, 1).numpy()
    if sq:
        Yc_np = Yc_np[..., 0]

    return (Xc.permute(0, 2, 3, 1).numpy(), Yc_np, mc.numpy())


class WaveDatasetBuilder:
    """Builds and normalizes multi-channel tensors for neural operators and convolutional surrogates."""

    def __init__(self, depth_crop: np.ndarray, crop_slices: Tuple[slice, slice]):
        self.depth_crop = depth_crop.astype(np.float32)
        self.crop_slices = crop_slices
        self.H, self.W = self.depth_crop.shape

    @classmethod
    def from_config(cls, cfg: PortConfig, bathy_path: Optional[str] = None) -> "WaveDatasetBuilder":
        if bathy_path is None:
            bathy_path = cfg.resolve_path("bathy_mod3")
        dep_full = np.loadtxt(bathy_path, dtype=np.float32)
        dep_crop = dep_full[cfg.grid.crop.slices]
        return cls(dep_crop, cfg.grid.crop.slices)

    def build_xy(
        self,
        hmo_vals: np.ndarray,
        fp_vals: np.ndarray,
        theta_vals: np.ndarray,
        hm0_fields: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble 4-channel inputs [Hmo, Fp, Theta, Depth] and cropped Hm0 targets.

        Returns:
            X: (N, H, W, 4) float32
            Y: (N, H, W, 1) float32
        """
        N = len(hmo_vals) if n_samples is None else min(len(hmo_vals), n_samples)
        X = np.empty((N, self.H, self.W, 4), dtype=np.float32)

        X[..., 0] = np.asarray(hmo_vals[:N], dtype=np.float32)[:, None, None]
        X[..., 1] = np.asarray(fp_vals[:N], dtype=np.float32)[:, None, None]
        X[..., 2] = np.asarray(theta_vals[:N], dtype=np.float32)[:, None, None]
        X[..., 3] = self.depth_crop[None, :, :]

        # Crop target spatial field
        if hm0_fields.ndim == 3:
            Y = np.asarray(hm0_fields[:N][:, self.crop_slices[0], self.crop_slices[1]], dtype=np.float32)[..., None]
        else:
            Y = np.asarray(hm0_fields[:N], dtype=np.float32)

        return X, Y

    @staticmethod
    def compute_norm_stats(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute scalar mean and std across N, H, W for each channel (matching s01/s16 protocol)."""
        # X: (N, H, W, C), Y: (N, H, W, Cy)
        c_in = X.shape[-1]
        c_out = Y.shape[-1]

        mean_in = np.zeros((c_in,), dtype=np.float64)
        std_in = np.zeros((c_in,), dtype=np.float64)
        mean_out = np.zeros((c_out,), dtype=np.float64)
        std_out = np.zeros((c_out,), dtype=np.float64)

        for c in range(c_in):
            vals = X[..., c].astype(np.float64)
            mean_in[c] = float(np.mean(vals))
            std_in[c] = float(np.std(vals))

        for c in range(c_out):
            vals = Y[..., c].astype(np.float64)
            mean_out[c] = float(np.mean(vals))
            std_out[c] = float(np.std(vals))

        return {
            "mean_in": mean_in.astype(np.float32),
            "std_in": std_in.astype(np.float32),
            "mean_out": mean_out.astype(np.float32),
            "std_out": std_out.astype(np.float32),
        }

    @staticmethod
    def partition_subsets(
        valid_indices: np.ndarray,
        n_val: int = 300,
        n_test: int = 300,
        seed_val: int = 7,
        seed_test: int = 21
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate strictly disjoint validation and quick-test index subsets with fixed random seeds."""
        rng_val = np.random.default_rng(seed_val)
        val_idx = np.sort(rng_val.choice(valid_indices, n_val, replace=False))

        remaining = np.setdiff1d(valid_indices, val_idx)
        rng_test = np.random.default_rng(seed_test)
        test_idx = np.sort(rng_test.choice(remaining, n_test, replace=False))

        assert len(np.intersect1d(val_idx, test_idx)) == 0, "Validation and test sets must be strictly disjoint!"
        return val_idx, test_idx
