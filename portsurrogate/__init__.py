"""portsurrogate: High-fidelity neural operator and surrogate modeling pipeline for port wave agitation.

Developed for coastal engineering applications and real-time operational forecasting.
"""

__version__ = "1.0.0"
__author__ = "Luis Zamorano & Yarko Niño"

from .config import PortConfig, load_config
from .bathymetry import PortBathymetry
from .hindcast import HindcastData
from .mda import MaxDissSelector, run_mda
from .funwave_runner import FUNWAVEBatchRunner
from .dataset import WaveDatasetBuilder, coarsen_mean
from .models.fno import FNO2d, CustomNormalizer, MaskedRelL2
from .evaluation import compute_spatial_metrics, compute_site_metrics, check_training_envelope
from .visualization import export_field_to_vtk, plot_bathymetry, plot_spatial_comparison, plot_site_series

__all__ = [
    "__version__",
    "PortConfig",
    "load_config",
    "PortBathymetry",
    "HindcastData",
    "MaxDissSelector",
    "run_mda",
    "FUNWAVEBatchRunner",
    "WaveDatasetBuilder",
    "coarsen_mean",
    "FNO2d",
    "CustomNormalizer",
    "MaskedRelL2",
    "compute_spatial_metrics",
    "compute_site_metrics",
    "check_training_envelope",
    "export_field_to_vtk",
    "plot_bathymetry",
    "plot_spatial_comparison",
    "plot_site_series",
]
