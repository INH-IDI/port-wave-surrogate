# PortWaveSurrogate: High-Fidelity Neural Operators for Real-Time Port Wave Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)

Code repository, modular Python library (`portsurrogate`), peer-review audit scripts, interactive Streamlit GUI, and Jupyter Notebook tutorial accompanying the paper:

> **"Real-time operational wave forecasting in ports using Fourier Neural Operators as high-fidelity surrogates"**  
> *Luis Zamorano and Yarko Niño*  
> *Coastal Engineering Journal (Taylor & Francis)*

---

## Executive Summary (Español)

Este paquete proporciona:
1. **Auditoría transparente del artículo**: Scripts ordenados paso a paso (`paper_reproduction/`) que reproducen todas las tablas cuantitativas (Tablas 1 a 4) y figuras (Figs. 2 a 12) del manuscrito.
2. **Librería modular reutilizable (`portsurrogate`)**: Encapsula el flujo completo de ingeniería para implementar modelos sustitutos neuronales en cualquier puerto del mundo:
   $$\text{Hindcast SWAN} \longrightarrow \text{Selección MDA} \longrightarrow \text{Simulación FUNWAVE-GPU} \longrightarrow \text{Entrenamiento FNO / U-Net / RBF} \longrightarrow \text{Filtro OOR y Pronóstico Operacional}$$
3. **Interfaz gráfica interactiva (GUI Streamlit)**: Dashboard bilingüe (EN/ES) con visualizador 2D, generador de casos SLURM, curvas de entrenamiento en vivo y filtro de seguridad en tiempo real.
4. **Tutorial en Jupyter Notebook**: Guía práctica (`notebooks/Tutorial_Port_Wave_Surrogate_SanAntonio.ipynb`) ejecutada directamente sobre el dataset presente sin duplicar datos pesados.

---

## Repository Structure

```text
port_wave_surrogate/
├── configs/                           # Declarative port configurations (YAML)
│   ├── san_antonio.yaml               # Benchmark case from the paper
│   └── new_port_template.yaml         # Step-by-step template for a new port domain
│
├── portsurrogate/                     # Reusable modular Python library
│   ├── __init__.py                    # Public API
│   ├── config.py                      # Dataclasses & YAML loader
│   ├── bathymetry.py                  # Bathymetry loading, crop & water masks
│   ├── hindcast.py                    # SWAN / WW3 spectral output parser
│   ├── mda.py                         # Maximum Dissimilarity Algorithm (Camus et al.)
│   ├── funwave_runner.py              # FUNWAVE-GPU batch generator & 3-tier CFL cascade
│   ├── dataset.py                     # 4-channel tensor assembly [Hm0, Fp, Theta, Depth]
│   ├── training.py                    # PyTorch FNO & U-Net training engine
│   ├── evaluation.py                  # Spatial/site metrics & Convex Hull OOR screening
│   ├── visualization.py               # Publication plots (origin='lower') & VTK exporter
│   ├── cli.py                         # Command-line entry points
│   └── models/                        # Surrogate architectures
│       ├── fno.py                     # Fourier Neural Operator 2D (SpectralConv2d)
│       ├── unet.py                    # U-Net 360 and TransUNet 360 wrappers
│       ├── rbf_camus.py               # Camus RBF interpolator
│       └── dnn_zheng.py               # Zheng MLP [3-7-8-N]
│
├── paper_reproduction/                # Peer-review reproducibility scripts (1:1 with paper)
│   ├── README_REVIEWERS.md            # Traceability table: Paper Section -> Script
│   ├── step01_prepare_datasets.py     # Subsets val300/test300 and norm_stats
│   ├── step02_train_fno.py            # FNO training size sweep & hyperparameter tuning
│   ├── step03_train_unet_transunet.py # U-Net & TransUNet benchmarking
│   ├── step04_train_point_baselines.py# Point baselines (Camus RBF, Zheng DNN)
│   ├── step05_evaluate_2021_annual.py # Full 8,760 h operational benchmark (Table 2)
│   ├── step06_evaluate_zeroshot.py    # Resolution invariance & zero-shot super-resolution
│   ├── step07_evaluate_outliers.py    # Extrapolation limits: natural Tp & synthetic OOR
│   └── step08_generate_paper_figures_and_vtk.py # Publication figures & 3D ParaView VTK
│
├── gui/                               # Interactive Streamlit Web Dashboard
│   ├── app.py                         # Bilingual dashboard (EN/ES)
│   └── run_gui.sh                     # Launch runner
│
├── notebooks/                         # Tutorial notebook
│   └── Tutorial_Port_Wave_Surrogate_SanAntonio.ipynb
│
├── pyproject.toml                     # Package installation metadata
├── requirements.txt                   # Minimal dependencies
├── environment.yml                    # Conda environment
└── LICENSE                            # MIT License
```

---

## Installation

### 1. Using pip in an existing Python 3.10+ environment

```bash
cd port_wave_surrogate
pip install -e .
```

### 2. Using Conda / Mamba

```bash
conda env create -f environment.yml
conda activate portsurrogate
pip install -e .
```

---

## Quick Start: Three Ways to Explore

### 1. Interactive Streamlit GUI

Launch the dashboard in your web browser:

```bash
bash gui/run_gui.sh
# Or directly:
streamlit run gui/app.py
```

Features:
- **Interactive Bathymetry & Control Sites**: Inspect water depth and gauge locations.
- **MDA Parameter Space**: Explore representative clusters and joint distributions.
- **Live 2D Forecast & OOR Screening**: Enter any incoming sea state $(H_{m0}, T_p, \text{Dir})$ to see instant wave fields and receive automated extrapolation warnings if forcing exceeds training bounds.
- **Paper Tables Explorer**: Audit all numerical tables and figures.

### 2. Jupyter Notebook Tutorial

Run the step-by-step tutorial using the present San Antonio dataset:

```bash
jupyter notebook notebooks/Tutorial_Port_Wave_Surrogate_SanAntonio.ipynb
```

### 3. Auditing the Paper Tables & Figures in Seconds

```bash
cd paper_reproduction

# Verify Table 2 (All 5 models across Sites A to E over 8,760 hours of 2021):
python3 step05_evaluate_2021_annual.py --audit-only

# Verify Table 3 & Resolution Invariance:
python3 step06_evaluate_zeroshot.py --audit-only

# Verify Extrapolation Boundaries (Section 3.6):
python3 step07_evaluate_outliers.py --audit-only

# Regenerate publication figures and ParaView VTK files:
python3 step08_generate_paper_figures_and_vtk.py
```

---

## Deploying the Pipeline in a New Port Domain

To implement this workflow in another port (e.g., Valparaíso, San Vicente, Arica, Iquique, or international ports):

1. **Create Configuration**:
   ```bash
   cp configs/new_port_template.yaml configs/my_new_port.yaml
   ```
   Set your bathymetry file path, domain dimensions, and control berths.

2. **Run MDA on your Regional SWAN Hindcast**:
   ```python
   from portsurrogate import HindcastData, run_mda
   
   hd = HindcastData.from_swan_file("data/my_port_FUNW.txt")
   centroids, indices = run_mda(hd.get_parameter_array(), num_centers=2000)
   ```

3. **Generate Batch Simulations with Automated CFL Stabilization**:
   ```python
   from portsurrogate import FUNWAVEBatchRunner
   
   runner = FUNWAVEBatchRunner.from_config(cfg)
   runner.prepare_cases_directory(cases_df, output_dir="simulations/cases")
   ```

4. **Train and Deploy the FNO Surrogate**:
   ```bash
   python3 paper_reproduction/step02_train_fno.py --config ../configs/my_new_port.yaml --modes 45 --width 32 --epochs 600
   ```

---

## Citation

If you use this library or methodology in your research, please cite:

```bibtex
@article{zamorano2026fno,
  title   = {Real-time operational wave forecasting in ports using Fourier Neural Operators as high-fidelity surrogates},
  author  = {Zamorano, Luis and Ni{\~n}o, Yarko},
  journal = {Coastal Engineering Journal},
  year    = {2026},
  publisher = {Taylor \& Francis}
}
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
