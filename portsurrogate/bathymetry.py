"""Bathymetry handling, domain cropping, and water masks for port domains."""
import os
from typing import Dict, Optional, Tuple
import numpy as np

from .config import PortConfig, CropConfig


class PortBathymetry:
    """Manages depth fields, coordinate transforms, and wet/dry masks."""

    def __init__(self, depth_full: np.ndarray, crop_cfg: Optional[CropConfig] = None, dx: float = 5.0, dy: float = 5.0):
        self.depth_full = depth_full.astype(np.float32)
        self.dx = dx
        self.dy = dy
        self.crop_cfg = crop_cfg

        if crop_cfg is not None:
            self.depth_crop = self.depth_full[crop_cfg.slices].copy()
        else:
            self.depth_crop = self.depth_full.copy()

        # Water mask (depth > 0)
        self.water_mask_full = (self.depth_full > 0.0).astype(np.float32)
        self.water_mask_crop = (self.depth_crop > 0.0).astype(np.float32)

    @classmethod
    def from_file(cls, path: str, crop_cfg: Optional[CropConfig] = None, dx: float = 5.0, dy: float = 5.0) -> "PortBathymetry":
        """Load bathymetry from ASCII text / .dep file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Bathymetry file not found: {path}")
        data = np.loadtxt(path, dtype=np.float32)
        return cls(data, crop_cfg=crop_cfg, dx=dx, dy=dy)

    @classmethod
    def from_config(cls, cfg: PortConfig, bathy_key: str = "bathy_mod3") -> "PortBathymetry":
        """Initialize directly from a PortConfig object."""
        path = cfg.resolve_path(bathy_key)
        return cls.from_file(path, crop_cfg=cfg.grid.crop, dx=cfg.grid.dx, dy=cfg.grid.dy)

    def create_intersection_mask(self, other_bathy: "PortBathymetry") -> np.ndarray:
        """Create a conservative water mask where BOTH bathymetries have water (depth > 0).
        Essential when historical test runs used a slightly different mesh (e.g., Mod2 vs Mod3 breakwater).
        """
        mask = (self.depth_crop > 0.0) & (other_bathy.depth_crop > 0.0)
        return mask.astype(np.float32)

    def validate_sites(self, sites: Dict[str, Tuple[int, int]]) -> Dict[str, Dict[str, float]]:
        """Check if control sites fall in water cells and return their coordinates and depths."""
        report = {}
        for name, coords in sites.items():
            r, c = coords
            h_full, w_full = self.depth_full.shape
            if not (0 <= r < h_full and 0 <= c < w_full):
                raise ValueError(f"Site '{name}' at ({r}, {c}) is outside the full grid bounds ({h_full}, {w_full}).")

            d_val = float(self.depth_full[r, c])
            is_wet = d_val > 0.0

            r_crop, c_crop = None, None
            if self.crop_cfg is not None:
                r_c = r - self.crop_cfg.r0
                c_c = c - self.crop_cfg.c0
                if 0 <= r_c < self.crop_cfg.height and 0 <= c_c < self.crop_cfg.width:
                    r_crop, c_crop = r_c, c_c

            report[name] = {
                "depth_m": d_val,
                "is_wet": is_wet,
                "coords_full": (r, c),
                "coords_crop": (r_crop, c_crop) if r_crop is not None else None,
            }
        return report

    def get_physical_extent(self, in_crop: bool = True) -> Tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax) in physical coordinates (meters)."""
        if in_crop and self.crop_cfg is not None:
            xmin = self.crop_cfg.c0 * self.dx
            xmax = self.crop_cfg.c1 * self.dx
            ymin = self.crop_cfg.r0 * self.dy
            ymax = self.crop_cfg.r1 * self.dy
        else:
            h, w = self.depth_full.shape
            xmin, xmax = 0.0, w * self.dx
            ymin, ymax = 0.0, h * self.dy
        return (xmin, xmax, ymin, ymax)
