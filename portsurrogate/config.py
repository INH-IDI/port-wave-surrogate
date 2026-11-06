"""Configuration management for port surrogate modeling."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import yaml


@dataclass
class CropConfig:
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0

    @property
    def width(self) -> int:
        return self.c1 - self.c0

    @property
    def slices(self) -> Tuple[slice, slice]:
        return (slice(self.r0, self.r1), slice(self.c0, self.c1))


@dataclass
class GridConfig:
    n_rows: int
    n_cols: int
    dx: float
    dy: float
    crop: CropConfig


@dataclass
class BoundaryConfig:
    nautical_to_cartesian_offset: float = 270.0
    depth_wavemaker: float = 26.0
    freq_min: float = 0.02
    freq_max: float = 0.25


@dataclass
class SiteConfig:
    name: str
    coords_full: Tuple[int, int]
    depth_m: float


@dataclass
class PortConfig:
    name: str
    country: str
    description: str
    grid: GridConfig
    boundary: BoundaryConfig
    control_sites: Dict[str, SiteConfig]
    data_paths: Dict[str, str]
    training_defaults: Dict[str, Any]
    config_dir: str = ""

    def resolve_path(self, key: str) -> str:
        """Resolve a path from data_paths relative to the port_wave_surrogate root directory."""
        if key not in self.data_paths:
            raise KeyError(f"Path key '{key}' not found in configuration data_paths.")
        p = self.data_paths[key]
        if os.path.isabs(p):
            return p
        # Base directory for relative paths is port_wave_surrogate/ (parent of configs/)
        base_dir = os.path.dirname(self.config_dir)
        return os.path.normpath(os.path.join(base_dir, p))

    def get_crop_site_coords(self, site_key: str) -> Tuple[int, int]:
        """Convert full-grid site coords to cropped grid coords."""
        site = self.control_sites[site_key]
        r_full, c_full = site.coords_full
        r_crop = r_full - self.grid.crop.r0
        c_crop = c_full - self.grid.crop.c0
        return (r_crop, c_crop)


def load_config(config_path: str) -> PortConfig:
    """Load and validate a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Parse Grid
    g_data = data["grid"]
    c_data = g_data["crop"]
    crop = CropConfig(
        r0=int(c_data["r0"]),
        r1=int(c_data["r1"]),
        c0=int(c_data["c0"]),
        c1=int(c_data["c1"]),
    )
    grid = GridConfig(
        n_rows=int(g_data["n_rows"]),
        n_cols=int(g_data["n_cols"]),
        dx=float(g_data["dx"]),
        dy=float(g_data["dy"]),
        crop=crop,
    )

    # Parse Boundary
    b_data = data.get("boundary_conditions", {})
    boundary = BoundaryConfig(
        nautical_to_cartesian_offset=float(b_data.get("nautical_to_cartesian_offset", 270.0)),
        depth_wavemaker=float(b_data.get("depth_wavemaker", 26.0)),
        freq_min=float(b_data.get("freq_min", 0.02)),
        freq_max=float(b_data.get("freq_max", 0.25)),
    )

    # Parse Control Sites
    sites = {}
    for k, v in data.get("control_sites", {}).items():
        sites[k] = SiteConfig(
            name=v.get("name", k),
            coords_full=(int(v["coords_full"][0]), int(v["coords_full"][1])),
            depth_m=float(v.get("depth_m", 0.0)),
        )

    return PortConfig(
        name=data["port"]["name"],
        country=data["port"].get("country", ""),
        description=data["port"].get("description", ""),
        grid=grid,
        boundary=boundary,
        control_sites=sites,
        data_paths=data.get("data_paths", {}),
        training_defaults=data.get("training_defaults", {}),
        config_dir=config_dir,
    )
