#!/usr/bin/env python3
"""Step 04: Point-wise Baseline Surrogates (Camus RBF and Zheng DNN).

Reproduces:
- Table 2 (Camus RBF column): Site R² (0.952 - 0.985) and RMSE (0.0032 - 0.0186 m).
- Table 2 (Zheng DNN column): Site R² (0.911 - 0.970) and RMSE (0.0045 - 0.0212 m).
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portsurrogate import (
    load_config,
    compute_site_metrics,
    PortBathymetry,
)
from portsurrogate.models.rbf_camus import CamusRBFInterpolator
from portsurrogate.models.dnn_zheng import ZhengWaveModel


def audit_results(cfg):
    """Audit Table 2 RBF and DNN metrics from precalculated time series."""
    prep_dir = cfg.resolve_path("prep_dir")
    gt = np.load(cfg.resolve_path("gt_2021_clean"))
    gt_sites = gt["hm0_gt"]  # (8760, 5)

    camus_path = os.path.join(prep_dir, "camus_maxdiss3000_sites2021.npz")
    zheng_path = os.path.join(prep_dir, "zheng_maxdiss3000_sites2021.npz")

    site_names = ["A", "B", "C", "D", "E"]
    print("=== Table 2: Baseline Point-wise Models (8760 Hours, 2021) ===")
    print(f"{'Site':<6} | {'Camus RBF RMSE':<14} | {'Camus R²':<10} | {'Zheng DNN RMSE':<14} | {'Zheng R²':<10}")
    print("-" * 65)

    def load_pred(p):
        if not os.path.exists(p):
            return None
        d = np.load(p)
        for k in ["hm0", "hm0_pred", "pred", "pred_sites", "Ypred"]:
            if k in d:
                return d[k]
        return None

    c_pred = load_pred(camus_path)
    z_pred = load_pred(zheng_path)

    for i, s_name in enumerate(site_names):
        c_m = compute_site_metrics(c_pred[:, i], gt_sites[:, i]) if c_pred is not None else {}
        z_m = compute_site_metrics(z_pred[:, i], gt_sites[:, i]) if z_pred is not None else {}

        c_rmse = f"{c_m.get('rmse', np.nan):.4f} m"
        c_r2 = f"{c_m.get('r2', np.nan):.3f}"
        z_rmse = f"{z_m.get('rmse', np.nan):.4f} m"
        z_r2 = f"{z_m.get('r2', np.nan):.3f}"

        print(f"Site {s_name:<2} | {c_rmse:<14} | {c_r2:<10} | {z_rmse:<14} | {z_r2:<10}")


def run_baselines(cfg):
    """Train and predict Camus RBF and Zheng DNN models on sites A-E."""
    d_train = np.load(cfg.resolve_path("funwave_train_dataset"))
    gt = np.load(cfg.resolve_path("gt_2021_clean"))
    gt_sites = gt["hm0_gt"]

    # Extract site indices in crop
    sites_crop = [cfg.get_crop_site_coords(k) for k in ["A", "B", "C", "D", "E"]]
    hm0_tr_full = d_train["hm0"][:, cfg.grid.crop.slices[0], cfg.grid.crop.slices[1]]

    # Target matrix for 5 sites: (3000, 5)
    Y_train_sites = np.column_stack([hm0_tr_full[:, r, c] for r, c in sites_crop])

    # Inputs: (3000, 3) [Hmo, FreqPeak, ThetaPeak]
    X_train = np.column_stack([d_train["Hmo"], d_train["FreqPeak"], d_train["ThetaPeak"]])

    # Test inputs 2021: (8760, 3)
    X_test = np.column_stack([gt["Hsig"], 1.0 / np.maximum(gt["TPsmoo"], 0.1), cfg.boundary.nautical_to_cartesian_offset - gt["Dir"]])

    print("\n--- Training Camus RBF Interpolator ---")
    rbf = CamusRBFInterpolator()
    rbf.fit(X_train, Y_train_sites, ix_scalar=[0, 1], ix_directional=[2], verbose=True)
    c_pred = rbf.predict(X_test)

    prep_dir = cfg.resolve_path("prep_dir")
    os.makedirs(prep_dir, exist_ok=True)
    np.savez(os.path.join(prep_dir, "camus_maxdiss3000_sites2021.npz"), hm0_pred=c_pred, fechas=gt["fechas"])

    print("\n--- Training Zheng DNN Model ---")
    dnn = ZhengWaveModel(input_dim=3, output_dim=5)
    dnn.fit(X_train, Y_train_sites, epochs=500, batch_size=32, patience=50, verbose=True)
    z_pred = dnn.predict(X_test)
    np.savez(os.path.join(prep_dir, "zheng_maxdiss3000_sites2021.npz"), hm0_pred=z_pred, fechas=gt["fechas"])

    audit_results(cfg)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate point-wise baseline surrogates.")
    parser.add_argument("--config", default="../configs/san_antonio.yaml", help="Path to config YAML")
    parser.add_argument("--audit-only", action="store_true", help="Audit Table 2 without retraining")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    cfg = load_config(cfg_path)

    if args.audit_only:
        audit_results(cfg)
    else:
        run_baselines(cfg)


if __name__ == "__main__":
    main()
