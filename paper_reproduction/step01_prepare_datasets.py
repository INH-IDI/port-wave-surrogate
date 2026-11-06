#!/usr/bin/env python3
"""Step 01: Dataset Preparation and Normalization Statistics.

Reproduces:
- 2021 validation/test subsets with fixed seeds (Seed 7 for val300; Seed 21 for test300).
- Channel-wise Z-score normalization statistics for nested training subsets N in {500, 1000, 2000, 3000}.
"""
import argparse
import os
import sys
import numpy as np

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portsurrogate import load_config, WaveDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset subsets and normalization stats.")
    parser.add_argument("--config", default="../configs/san_antonio.yaml", help="Path to port configuration YAML")
    parser.add_argument("--audit-only", action="store_true", help="Only inspect existing statistics without rewriting")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    cfg = load_config(cfg_path)
    print(f"Loaded configuration for {cfg.name} ({cfg.country})")

    prep_dir = cfg.resolve_path("prep_dir")
    gt_clean_path = cfg.resolve_path("gt_2021_clean")
    train_npz_path = cfg.resolve_path("funwave_train_dataset")

    sizes = [500, 1000, 2000, 3000]

    if args.audit_only:
        print("\n=== Auditing Existing Normalization Statistics ===")
        for n in sizes:
            st_path = os.path.join(prep_dir, f"norm_stats_maxdiss_{n}.npz")
            if os.path.exists(st_path):
                st = np.load(st_path)
                print(f"n={n:4d}: mean_in={np.round(st['mean_in'], 4)} | std_in={np.round(st['std_in'], 4)} | "
                      f"mean_out={np.round(st['mean_out'], 4)} | std_out={np.round(st['std_out'], 4)}")
            else:
                print(f"n={n:4d}: NOT FOUND at {st_path}")

        val_path = os.path.join(prep_dir, "subset_val300.npz")
        test_path = os.path.join(prep_dir, "subset_test300.npz")
        if os.path.exists(val_path) and os.path.exists(test_path):
            v = np.load(val_path)
            t = np.load(test_path)
            overlap = len(np.intersect1d(v["idx"], t["idx"]))
            print(f"\nSubsets verified: Val300 ({len(v['idx'])} states, seed {v['seed']}), "
                  f"Test300 ({len(t['idx'])} states, seed {t['seed']}). Disjoint overlap = {overlap} states.")
        return

    # Full preparation mode
    print("\n--- Generating Disjoint Subsets (Seed 7 and 21) ---")
    gt = np.load(gt_clean_path)
    hm0_gt = np.asarray(gt["hm0_gt"], dtype=np.float64)
    valid_hours = np.where(~np.isnan(hm0_gt).any(axis=1))[0]
    print(f"Valid operational hours in 2021: {len(valid_hours)} / 8760")

    val_idx, test_idx = WaveDatasetBuilder.partition_subsets(valid_hours, n_val=300, n_test=300, seed_val=7, seed_test=21)
    os.makedirs(prep_dir, exist_ok=True)
    np.savez(os.path.join(prep_dir, "subset_val300.npz"), idx=val_idx, fechas=gt["fechas"][val_idx], seed=7)
    np.savez(os.path.join(prep_dir, "subset_test300.npz"), idx=test_idx, fechas=gt["fechas"][test_idx], seed=21)
    print(f"Saved subset_val300.npz and subset_test300.npz to {prep_dir}")

    print("\n--- Computing Normalization Statistics for Nested Training Sets ---")
    d_train = np.load(train_npz_path, mmap_mode="r")
    dep_path = cfg.resolve_path("bathy_mod3")
    builder = WaveDatasetBuilder.from_config(cfg, bathy_path=dep_path)

    for n in sizes:
        X, Y = builder.build_xy(
            hmo_vals=d_train["Hmo"],
            fp_vals=d_train["FreqPeak"],
            theta_vals=d_train["ThetaPeak"],
            hm0_fields=d_train["hm0"],
            n_samples=n
        )
        stats = WaveDatasetBuilder.compute_norm_stats(X, Y)
        out_stats_path = os.path.join(prep_dir, f"norm_stats_maxdiss_{n}.npz")
        np.savez(out_stats_path, **stats)
        print(f"Saved n={n} stats -> {out_stats_path}")


if __name__ == "__main__":
    main()
