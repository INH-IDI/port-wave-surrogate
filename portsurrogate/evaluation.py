"""Evaluation metrics, point-wise site analysis, and training envelope / OOR screening."""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import Delaunay


def compute_spatial_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    water_mask: np.ndarray,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute 2D field performance metrics strictly over water cells.

    Args:
        pred: (N, H, W) or (H, W) predicted wave fields.
        target: (N, H, W) or (H, W) ground truth numerical fields.
        water_mask: (H, W) binary water mask.
    Returns:
        dict with scalar summary metrics ('rel_l2', 'rmse', 'r2', 'bias') and 2D spatial maps.
    """
    if pred.ndim == 2:
        pred = pred[None, ...]
    if target.ndim == 2:
        target = target[None, ...]

    m = water_mask.astype(bool)
    diff = pred - target

    # Global relative L2
    p_wet = pred[:, m]
    t_wet = target[:, m]
    d_wet = diff[:, m]

    num = np.linalg.norm(d_wet, axis=1)
    den = np.clip(np.linalg.norm(t_wet, axis=1), 1e-8, None)
    rel_l2 = float(np.mean(num / den))

    rmse_global = float(np.sqrt(np.mean(d_wet ** 2)))
    bias_global = float(np.mean(d_wet))

    # Global R2 over wet cells
    var_t = float(np.var(t_wet))
    r2_global = float(1.0 - (rmse_global ** 2) / (var_t + 1e-8)) if var_t > 1e-12 else 0.0

    # Per-cell spatial 2D maps
    rmse_map = np.full(water_mask.shape, np.nan, dtype=np.float32)
    bias_map = np.full(water_mask.shape, np.nan, dtype=np.float32)
    r2_map = np.full(water_mask.shape, np.nan, dtype=np.float32)

    rmse_map[m] = np.sqrt(np.mean(diff[:, m] ** 2, axis=0))
    bias_map[m] = np.mean(diff[:, m], axis=0)

    cell_var = np.var(target[:, m], axis=0)
    valid_var = cell_var > 1e-8
    r2_vals = np.full(cell_var.shape, np.nan, dtype=np.float32)
    r2_vals[valid_var] = 1.0 - (rmse_map[m][valid_var] ** 2) / cell_var[valid_var]
    r2_map[m] = r2_vals

    return {
        "rel_l2": rel_l2,
        "rmse": rmse_global,
        "bias": bias_global,
        "r2": r2_global,
        "rmse_map": rmse_map,
        "bias_map": bias_map,
        "r2_map": r2_map,
    }


def compute_site_metrics(pred_series: np.ndarray, target_series: np.ndarray) -> Dict[str, float]:
    """Compute point-wise time-series statistics (RMSE, R2, Bias, Scatter Index)."""
    p = np.asarray(pred_series, dtype=np.float64)
    t = np.asarray(target_series, dtype=np.float64)

    # Filter NaN if any
    valid = (~np.isnan(p)) & (~np.isnan(t))
    p, t = p[valid], t[valid]

    if len(p) == 0:
        return {"rmse": np.nan, "r2": np.nan, "bias": np.nan, "si": np.nan}

    err = p - t
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    var_t = float(np.var(t))
    r2 = float(1.0 - (rmse ** 2) / (var_t + 1e-8)) if var_t > 1e-12 else 0.0
    mean_t = float(np.mean(t))
    si = float(rmse / (mean_t + 1e-8))

    return {
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "si": si,
    }


def check_training_envelope(
    points: np.ndarray,
    training_points: np.ndarray,
    feature_names: List[str] = ["Hs", "Tp", "Dir"],
) -> Dict[str, Any]:
    """Screen operational forecast forcing against the calibration / MDA training envelope.

    Identifies if sea states operate within the safe interpolation regime or in out-of-distribution (OOR) extrapolation.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    tr = np.asarray(training_points, dtype=np.float64)

    n_feats = pts.shape[1]
    bounds = {}
    is_in_bounds = np.ones(pts.shape[0], dtype=bool)
    marginal_violations = {name: [] for name in feature_names}

    for i in range(min(n_feats, len(feature_names))):
        name = feature_names[i]
        b_min = float(np.amin(tr[:, i]))
        b_max = float(np.amax(tr[:, i]))
        bounds[name] = (b_min, b_max)

        v_low = pts[:, i] < b_min
        v_high = pts[:, i] > b_max
        viol = v_low | v_high

        is_in_bounds &= ~viol
        if np.any(viol):
            indices = np.where(viol)[0].tolist()
            marginal_violations[name] = indices

    # Check convex hull membership using Delaunay triangulation
    in_hull = np.ones(pts.shape[0], dtype=bool)
    try:
        # Use first 3 features (Hs, Tp, Dir) for triangulation
        delaunay = Delaunay(tr[:, :3])
        simplex = delaunay.find_simplex(pts[:, :3])
        in_hull = simplex >= 0
    except Exception:
        # Fallback to bounding box if Delaunay fails
        in_hull = is_in_bounds.copy()

    n_oor = int(np.sum(~is_in_bounds))
    n_outside_hull = int(np.sum(~in_hull))

    status = "SAFE_INTERPOLATION"
    if n_oor > 0 or n_outside_hull > 0:
        status = "EXTRAPOLATION_WARNING"

    return {
        "status": status,
        "bounds": bounds,
        "is_in_bounds": is_in_bounds,
        "in_convex_hull": in_hull,
        "n_total": len(pts),
        "n_marginal_oor": n_oor,
        "n_outside_hull": n_outside_hull,
        "marginal_violations": marginal_violations,
    }
