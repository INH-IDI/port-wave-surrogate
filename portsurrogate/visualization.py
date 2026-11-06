"""Publication-quality visualization utilities and ParaView VTK exporter.

All 2D spatial maps enforce origin='lower' to preserve correct North-South geographical orientation
(port breakwater at the South / bottom; submarine canyon at the North / top).
"""
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def export_field_to_vtk(
    filename: str,
    fields: Dict[str, np.ndarray],
    dx: float = 5.0,
    dy: float = 5.0,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    title: str = "PortWaveSurrogate Field",
) -> str:
    """Export 2D arrays to ASCII VTK Structured Points for 3D visualization in ParaView.

    Args:
        filename: Destination .vtk file path.
        fields: Dictionary mapping field name -> 2D numpy array of shape (Ny, Nx).
        dx: Spatial grid step in X (meters).
        dy: Spatial grid step in Y (meters).
        origin: (x0, y0, z0) origin in meters.
        title: Header dataset title.
    """
    first_field = next(iter(fields.values()))
    ny, nx = first_field.shape

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {dx} {dy} 1.0\n")
        f.write(f"POINT_DATA {nx * ny}\n")

        for var_name, arr in fields.items():
            clean_name = var_name.replace(" ", "_").replace("-", "_")
            f.write(f"SCALARS {clean_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")

            # Flatten row-major (y from 0 to ny-1, x from 0 to nx-1)
            # In FUNWAVE / numpy, arr is (ny, nx), row 0 is y=0
            data_flat = arr.astype(np.float32).ravel()
            # Replace NaNs with 0.0 for VTK
            data_flat = np.nan_to_num(data_flat, nan=0.0)

            for i in range(0, len(data_flat), 8):
                chunk = data_flat[i : i + 8]
                f.write(" ".join(f"{v:.4f}" for v in chunk) + "\n")

    return filename


def plot_bathymetry(
    depth: np.ndarray,
    dx: float = 5.0,
    dy: float = 5.0,
    sites: Optional[Dict[str, Tuple[int, int]]] = None,
    out_path: Optional[str] = None,
    title: str = "Port Bathymetry",
) -> plt.Figure:
    """Plot bathymetry with depth contours, sites, and lower-origin orientation."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    extent = (0, depth.shape[1] * dx / 1000.0, 0, depth.shape[0] * dy / 1000.0)

    # Water colormap
    d_masked = np.ma.masked_where(depth <= 0, depth)
    im = ax.imshow(d_masked, origin="lower", extent=extent, cmap="YlGnBu", vmin=0, vmax=np.nanmax(depth))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth [m]", fontsize=11)

    # Land boundary
    ax.contour(depth, levels=[0.0], colors="black", linewidths=1.2, origin="lower", extent=extent)

    # Sites
    if sites:
        for name, (r, c) in sites.items():
            x_km = c * dx / 1000.0
            y_km = r * dy / 1000.0
            ax.plot(x_km, y_km, "ro", markersize=6)
            ax.text(x_km + 0.04, y_km + 0.04, name, fontsize=10, weight="bold", color="darkred")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X [km]", fontsize=11)
    ax.set_ylabel("Y [km]", fontsize=11)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path)
    return fig


def plot_spatial_comparison(
    gt: np.ndarray,
    fno: np.ndarray,
    unet: Optional[np.ndarray] = None,
    water_mask: Optional[np.ndarray] = None,
    vmax: float = 4.0,
    dx: float = 5.0,
    dy: float = 5.0,
    out_path: Optional[str] = None,
    title: str = "Significant Wave Height Comparison",
) -> plt.Figure:
    """Plot 2D comparison panels with discrete colormaps and origin='lower'."""
    n_cols = 3 if unet is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.5), dpi=300)

    extent = (0, gt.shape[1] * dx / 1000.0, 0, gt.shape[0] * dy / 1000.0)
    cmap = plt.get_cmap("viridis", 10)

    def mask_field(arr):
        if water_mask is not None:
            return np.ma.masked_where(water_mask <= 0, arr)
        return arr

    # GT
    im0 = axes[0].imshow(mask_field(gt), origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=vmax)
    axes[0].set_title("FUNWAVE (GT)", fontsize=12)
    axes[0].set_xlabel("X [km]")
    axes[0].set_ylabel("Y [km]")

    # FNO
    axes[1].imshow(mask_field(fno), origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=vmax)
    axes[1].set_title("FNO (Surrogate)", fontsize=12)
    axes[1].set_xlabel("X [km]")
    axes[1].set_yticks([])

    # UNet
    if unet is not None:
        axes[2].imshow(mask_field(unet), origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=vmax)
        axes[2].set_title("U-Net (Surrogate)", fontsize=12)
        axes[2].set_xlabel("X [km]")
        axes[2].set_yticks([])

    fig.subplots_adjust(right=0.88, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label("$H_{m0}$ [m]", fontsize=11)

    fig.suptitle(title, fontsize=13, y=0.96)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path)
    return fig


def plot_site_series(
    timestamps: np.ndarray,
    gt_series: np.ndarray,
    pred_series: Dict[str, np.ndarray],
    site_name: str,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plot time-series tracking at a specific harbor site."""
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=300)
    ax.plot(timestamps, gt_series, "k.", markersize=2.5, alpha=0.6, label="FUNWAVE (GT)", zorder=3)

    colors = {"FNO": "#1f77b4", "U-Net": "#ff7f0e", "RBF": "#2ca02c", "DNN": "#9467bd"}
    for name, series in pred_series.items():
        c = colors.get(name, None)
        ax.plot(timestamps, series, linewidth=1.2, alpha=0.85, label=name, color=c)

    ax.set_title(f"Time Series at {site_name}", fontsize=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("$H_{m0}$ [m]", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path)
    return fig
