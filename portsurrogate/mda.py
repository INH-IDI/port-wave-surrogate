"""Maximum Dissimilarity Algorithm (MDA / MaxDiss) for optimal wave parameter space selection.

Based on Camus et al. (2011, 2013) with circular distance metrics for directional wave components.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class MaxDissSelector:
    """Greedy Maximum Dissimilarity Algorithm (MDA) with circular directional distance."""

    def __init__(self, data: np.ndarray, ix_scalar: Optional[List[int]] = None, ix_directional: Optional[List[int]] = None):
        """Args:
            data: 2D array of shape (N_samples, N_features).
            ix_scalar: Column indices for scalar features (e.g. Hs, Tp).
            ix_directional: Column indices for directional features (e.g. Dir in degrees).
        """
        self.data = np.asarray(data, dtype=np.float64)
        if ix_directional is None:
            self.ix_directional = []
        else:
            self.ix_directional = list(ix_directional)

        if ix_scalar is None:
            self.ix_scalar = [i for i in range(self.data.shape[1]) if i not in self.ix_directional]
        else:
            self.ix_scalar = list(ix_scalar)

        self.minis: Optional[np.ndarray] = None
        self.maxis: Optional[np.ndarray] = None
        self.data_norm: Optional[np.ndarray] = None

    def normalize(self, data: Optional[np.ndarray] = None, minis: Optional[np.ndarray] = None, maxis: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize scalar features to [0, 1] and directional features from degrees to radians."""
        X = self.data if data is None else np.asarray(data, dtype=np.float64)
        X_norm = np.zeros_like(X)

        if minis is None or maxis is None:
            min_list, max_list = [], []
            for ix in self.ix_scalar:
                v = X[:, ix]
                mi = float(np.amin(v))
                ma = float(np.amax(v))
                if abs(ma - mi) < 1e-12:
                    ma = mi + 1.0
                X_norm[:, ix] = (v - mi) / (ma - mi)
                min_list.append(mi)
                max_list.append(ma)
            minis = np.array(min_list, dtype=np.float64)
            maxis = np.array(max_list, dtype=np.float64)
        else:
            for c, ix in enumerate(self.ix_scalar):
                v = X[:, ix]
                mi, ma = minis[c], maxis[c]
                if abs(ma - mi) < 1e-12:
                    ma = mi + 1.0
                X_norm[:, ix] = (v - mi) / (ma - mi)

        for ix in self.ix_directional:
            X_norm[:, ix] = X[:, ix] * np.pi / 180.0

        if data is None:
            self.minis = minis
            self.maxis = maxis
            self.data_norm = X_norm

        return X_norm, minis, maxis

    def denormalize(self, X_norm: np.ndarray, minis: Optional[np.ndarray] = None, maxis: Optional[np.ndarray] = None) -> np.ndarray:
        """Denormalize back to physical engineering units."""
        minis = self.minis if minis is None else minis
        maxis = self.maxis if maxis is None else maxis
        X = np.zeros_like(X_norm)

        for c, ix in enumerate(self.ix_scalar):
            X[:, ix] = X_norm[:, ix] * (maxis[c] - minis[c]) + minis[c]

        for ix in self.ix_directional:
            X[:, ix] = X_norm[:, ix] * 180.0 / np.pi

        return X

    def normalized_distance(self, M: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Compute squared normalized distance between point(s) M and reference D with circular metric."""
        dif = np.zeros_like(M)
        for ix in self.ix_scalar:
            dif[:, ix] = D[:, ix] - M[:, ix]
        for ix in self.ix_directional:
            ab = np.abs(D[:, ix] - M[:, ix])
            dif[:, ix] = np.minimum(ab, 2.0 * np.pi - ab) / np.pi
        return np.sum(dif ** 2, axis=1)

    def select(self, num_centers: int, verbose: bool = True) -> Tuple[np.ndarray, List[int]]:
        """Run the Maximum Dissimilarity Algorithm.

        Returns:
            centroids: array of shape (num_centers, num_features) in physical units.
            indices: original row indices of selected centers.
        """
        N = self.data.shape[0]
        if num_centers >= N:
            return self.data.copy(), list(range(N))

        data_norm, minis, maxis = self.normalize()

        # Seed with the maximum Hs (column 0 by default)
        seed_idx = int(np.argmax(data_norm[:, 0]))

        subset_norm = [data_norm[seed_idx]]
        selected_indices = [seed_idx]

        # Remaining candidate points
        remaining_indices = np.delete(np.arange(N), seed_idx)
        train_norm = np.delete(data_norm, seed_idx, axis=0)

        # Distance vector to the current subset
        xx = np.repeat([subset_norm[0]], train_norm.shape[0], axis=0)
        d_last = self.normalized_distance(train_norm, xx)

        while len(subset_norm) < num_centers:
            # Find the candidate that maximizes the minimum distance to the current subset
            bmu = int(np.argmax(d_last))
            qerr = d_last[bmu]

            if np.isnan(qerr):
                break

            subset_norm.append(train_norm[bmu])
            selected_indices.append(int(remaining_indices[bmu]))

            # Update remaining pool
            new_center = np.repeat([train_norm[bmu]], train_norm.shape[0] - 1, axis=0)
            train_norm = np.delete(train_norm, bmu, axis=0)
            remaining_indices = np.delete(remaining_indices, bmu)

            # Update distance vector: d_last = min(d_last, dist_to_new_center)
            d_last = np.delete(d_last, bmu)
            d_new = self.normalized_distance(train_norm, new_center)
            d_last = np.minimum(d_last, d_new)

            if verbose and len(subset_norm) % max(1, (num_centers // 10)) == 0:
                print(f"  MDA selected: {len(subset_norm)} / {num_centers}")

        centroids_norm = np.array(subset_norm, dtype=np.float64)
        centroids = self.denormalize(centroids_norm, minis, maxis)
        return centroids, selected_indices


def run_mda(
    data: Union[np.ndarray, pd.DataFrame],
    num_centers: int = 3000,
    ix_scalar: Optional[List[int]] = None,
    ix_directional: Optional[List[int]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[int]]:
    """Helper function to perform MDA selection on a matrix or DataFrame."""
    if isinstance(data, pd.DataFrame):
        arr = data.to_numpy(dtype=np.float64)
        if ix_directional is None:
            ix_directional = [i for i, c in enumerate(data.columns) if "dir" in c.lower()]
    else:
        arr = np.asarray(data, dtype=np.float64)

    selector = MaxDissSelector(arr, ix_scalar=ix_scalar, ix_directional=ix_directional)
    return selector.select(num_centers, verbose=verbose)
