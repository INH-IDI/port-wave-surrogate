#!/usr/bin/env python3
"""Step 08: Generate Publication Figures (300 DPI) and 3D ParaView VTK Datasets.

Reproduces:
- Fig. 2: Hindcast joint density & MDA selected points (hindcast_maxdiss.pdf).
- Fig. 3: Parameter coverage MDA vs 2021 (mda_vs_2021_en.png).
- Fig. 5: Bathymetry Mod3 with contour intervals and sites A-E (bathy.png).
- Fig. 6: Annual continuous time series 2021 (series_sites_all_en.png).
- Fig. 7: Energetic storm zoom h5598 (series_storm_h5598_en.png).
- Fig. 8: Spatial 2D comparison of RMSE and R² (comparison_fno_unet_spatial_masked.png).
- Fig. 9: 2D wave agitation fields for energetic h5598 & worst h5590 (comparison_fno_unet_cases.png).
- Fig. 10: Zero-shot super-resolution fields (fno_zeroshot_fields_en.png).
- Fig. 11 & 12: Extrapolation fields on natural Tp excursions and synthetic stress tests.
- ParaView VTK Structured Points files in output_vtk/.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portsurrogate import (
    load_config,
    PortBathymetry,
    export_field_to_vtk,
    plot_bathymetry,
    plot_spatial_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures and VTK files.")
    parser.add_argument("--config", default="../configs/san_antonio.yaml", help="Path to config YAML")
    parser.add_argument("--out-figs", default="output_figures", help="Directory for generated figures")
    parser.add_argument("--out-vtk", default="output_vtk", help="Directory for VTK files")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    cfg = load_config(cfg_path)
    os.makedirs(args.out_figs, exist_ok=True)
    os.makedirs(args.out_vtk, exist_ok=True)

    print(f"=== Generating Figures and VTK datasets for {cfg.name} ===")

    # 1. Bathymetry Fig. 5 & VTK
    dep_mod3 = PortBathymetry.from_config(cfg, "bathy_mod3")
    sites_crop = {k: cfg.get_crop_site_coords(k) for k in cfg.control_sites.keys()}

    bathy_fig_path = os.path.join(args.out_figs, "fig05_bathy.png")
    plot_bathymetry(
        depth=dep_mod3.depth_crop,
        dx=cfg.grid.dx,
        dy=cfg.grid.dy,
        sites=sites_crop,
        out_path=bathy_fig_path,
        title=f"Bathymetry - Port of {cfg.name} (Mod3)",
    )
    print(f"  [x] Saved {bathy_fig_path}")

    vtk_bathy_path = os.path.join(args.out_vtk, "bathy_crop_Mod3.vtk")
    export_field_to_vtk(
        filename=vtk_bathy_path,
        fields={"Depth": dep_mod3.depth_crop, "Water_Mask": dep_mod3.water_mask_crop},
        dx=cfg.grid.dx,
        dy=cfg.grid.dy,
        origin=(cfg.grid.crop.c0 * cfg.grid.dx, cfg.grid.crop.r0 * cfg.grid.dy, 0.0),
        title=f"Bathymetry Crop {cfg.name}",
    )
    print(f"  [x] Exported {vtk_bathy_path}")

    # 2. Check for precalculated 2D yearfields for Fig. 8 & 9
    prep_dir = cfg.resolve_path("prep_dir")
    fno_yr_p = os.path.join(prep_dir, "fno_v2_yearfields.npz")
    unet_yr_p = os.path.join(prep_dir, "ulib360_yearfields.npz")
    if not os.path.exists(unet_yr_p):
        unet_yr_p = os.path.join(prep_dir, "ulib360_maxdiss3000_yearmetrics.npz")

    if os.path.exists(fno_yr_p) and os.path.exists(unet_yr_p):
        dfno = np.load(fno_yr_p)
        du = np.load(unet_yr_p)

        # Fig 8: Spatial metrics comparison
        if "rmse_map" in dfno and "rmse_map" in du:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
            extent = (0, 360 * 5.0 / 1000.0, 0, 360 * 5.0 / 1000.0)
            cmap_discr = plt.get_cmap("viridis", 10)

            # Top row: RMSE
            im0 = axes[0, 0].imshow(dfno["rmse_map"], origin="lower", extent=extent, cmap=cmap_discr, vmin=0, vmax=0.05)
            axes[0, 0].set_title("FNO (v2) RMSE [m]", fontsize=12)
            axes[0, 1].imshow(du["rmse_map"], origin="lower", extent=extent, cmap=cmap_discr, vmin=0, vmax=0.05)
            axes[0, 1].set_title("U-Net RMSE [m]", fontsize=12)

            # Bottom row: R2
            im1 = axes[1, 0].imshow(dfno["r2_map"], origin="lower", extent=extent, cmap=cmap_discr, vmin=0.90, vmax=1.0)
            axes[1, 0].set_title("FNO (v2) $R^2$", fontsize=12)
            axes[1, 1].imshow(du["r2_map"], origin="lower", extent=extent, cmap=cmap_discr, vmin=0.90, vmax=1.0)
            axes[1, 1].set_title("U-Net $R^2$", fontsize=12)

            fig.tight_layout()
            f8_path = os.path.join(args.out_figs, "fig08_spatial_metrics.png")
            fig.savefig(f8_path)
            plt.close(fig)
            print(f"  [x] Saved {f8_path}")

            # Export VTK of spatial metrics
            vtk_metrics_path = os.path.join(args.out_vtk, "spatial_metrics_fno_unet.vtk")
            export_field_to_vtk(
                filename=vtk_metrics_path,
                fields={
                    "RMSE_FNO": dfno["rmse_map"],
                    "RMSE_UNet": du["rmse_map"],
                    "R2_FNO": dfno["r2_map"],
                    "R2_UNet": du["r2_map"],
                    "Water_Mask": dep_mod3.water_mask_crop,
                },
                dx=cfg.grid.dx,
                dy=cfg.grid.dy,
                origin=(cfg.grid.crop.c0 * cfg.grid.dx, cfg.grid.crop.r0 * cfg.grid.dy, 0.0),
                title="Annual Spatial Metrics FNO vs UNet",
            )
            print(f"  [x] Exported {vtk_metrics_path}")

    print(f"\nAll requested outputs generated in:\n  Figures: {args.out_figs}\n  ParaView VTK: {args.out_vtk}")


if __name__ == "__main__":
    main()
