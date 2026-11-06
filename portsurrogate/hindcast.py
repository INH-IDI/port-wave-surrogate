"""Ingestion and transformation of multidecadal spectral wave hindcasts (SWAN / WW3)."""
import glob
import os
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import PortConfig


class HindcastData:
    """Manages wave hindcast time series and converts boundary parameters to FUNWAVE convention."""

    def __init__(self, df: pd.DataFrame, offset_dir: float = 270.0):
        self.df = df
        self.offset_dir = offset_dir
        self._ensure_columns()

    def _ensure_columns(self):
        """Ensure standard physical and transformed columns exist."""
        # Hs / Hm0
        if "Hs" not in self.df.columns and "Hsig" in self.df.columns:
            self.df["Hs"] = self.df["Hsig"]
        elif "Hs" not in self.df.columns and "Hmo" in self.df.columns:
            self.df["Hs"] = self.df["Hmo"]

        # Tp / TPsmoo
        if "Tp" not in self.df.columns and "TPsmoo" in self.df.columns:
            self.df["Tp"] = self.df["TPsmoo"]

        # Nautical Dir
        if "Dir" not in self.df.columns and "PkDir" in self.df.columns:
            self.df["Dir"] = self.df["PkDir"]

        # Transformed wavemaker inputs
        if "FreqPeak" not in self.df.columns and "Tp" in self.df.columns:
            self.df["FreqPeak"] = 1.0 / np.maximum(self.df["Tp"], 0.1)

        if "ThetaPeak" not in self.df.columns and "Dir" in self.df.columns:
            self.df["ThetaPeak"] = self.offset_dir - self.df["Dir"]

    @classmethod
    def from_swan_file(cls, filepath: str, offset_dir: float = 270.0, usecols: Tuple[int, int, int] = (3, 4, 9)) -> "HindcastData":
        """Read a single SWAN output table (FUNW.txt).
        Standard columns: [Time, Xp, Yp, Hsig, TPsmoo, Tm_10, Tm01, Tm02, PkDir, Dir, Dspr].
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SWAN file not found: {filepath}")

        # Load numerical columns
        data = np.loadtxt(filepath, comments="%", usecols=usecols)
        df = pd.DataFrame(data, columns=["Hs", "Tp", "Dir"])
        return cls(df, offset_dir=offset_dir)

    @classmethod
    def from_swan_directory(cls, dirpath: str, pattern: str = "**/FUNW.txt", offset_dir: float = 270.0) -> "HindcastData":
        """Read and concatenate multi-year SWAN unsteady directories."""
        files = sorted(glob.glob(os.path.join(dirpath, pattern), recursive=True))
        if not files:
            raise FileNotFoundError(f"No SWAN files matching '{pattern}' in {dirpath}")

        frames = []
        for f in files:
            try:
                arr = np.loadtxt(f, comments="%", usecols=(3, 4, 9))
                frames.append(pd.DataFrame(arr, columns=["Hs", "Tp", "Dir"]))
            except Exception as e:
                print(f"Warning: Failed to read {f}: {e}")

        df_all = pd.concat(frames, ignore_index=True)
        return cls(df_all, offset_dir=offset_dir)

    @classmethod
    def from_csv(cls, filepath: str, offset_dir: float = 270.0) -> "HindcastData":
        """Load from standard CSV."""
        df = pd.read_csv(filepath)
        return cls(df, offset_dir=offset_dir)

    @classmethod
    def from_config(cls, cfg: PortConfig, use_directory: bool = True) -> "HindcastData":
        """Load hindcast as defined in PortConfig."""
        offset = cfg.boundary.nautical_to_cartesian_offset
        if use_directory and "hindcast_unsteady" in cfg.data_paths:
            p = cfg.resolve_path("hindcast_unsteady")
            return cls.from_swan_directory(p, offset_dir=offset)
        elif "hindcast_file" in cfg.data_paths:
            p = cfg.resolve_path("hindcast_file")
            return cls.from_swan_file(p, offset_dir=offset)
        raise KeyError("Neither 'hindcast_unsteady' nor 'hindcast_file' specified in configuration.")

    def get_parameter_array(self, variables: Tuple[str, ...] = ("Hs", "Tp", "Dir")) -> np.ndarray:
        """Extract a 2D array of shape (N, len(variables))."""
        return self.df[list(variables)].to_numpy(dtype=np.float64)

    def summary(self) -> pd.DataFrame:
        """Statistical summary of the sea states."""
        cols = [c for c in ["Hs", "Tp", "Dir", "FreqPeak", "ThetaPeak"] if c in self.df.columns]
        return self.df[cols].describe().T
