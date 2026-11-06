# Reviewer Guide & Code Traceability

> 🔍 **Notice to Peer Reviewers:**  
> This directory provides structured reproduction scripts for auditing the research manuscript:  
> **"Real-time operational wave forecasting in ports using Fourier Neural Operators as high-fidelity surrogates"**  
> *Authors: Luis Zamorano & Yarko Niño*  
> **Status:** Submitted for publication.  
>
> All quantitative tables, figures, hyperparameter sweeps, and out-of-distribution evaluations can be verified using the step-by-step scripts below. If any additional ablation experiment, parameter variation, or raw simulation is desired during the evaluation process, please contact `i-d-i@inh.cl` / `lzamorano@uchile.cl` and it will be provided promptly.

---

## 1. Mapping Between Paper Elements and Reproduction Scripts

Every quantitative table and figure in the manuscript can be verified and reproduced using the structured scripts below:

| Paper Element | Description in Article | Reproduction Script | Underlying Data / Precalculated Inputs |
|---|---|---|---|
| **Fig. 1** | Operational multi-scale chain WW3 $\rightarrow$ SWAN $\rightarrow$ FUNWAVE | Concept diagram | `Esquema1.pdf` |
| **Fig. 2** | MDA selection over 34-year hindcast (joint densities) | [`step08_generate_paper_figures_and_vtk.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step08_generate_paper_figures_and_vtk.py) | `maxdiss_3000_par.txt`, `hindcast_SanAnto_origen/` |
| **Fig. 3** | Parameter space coverage: MDA vs 2021 operational year | [`step08_generate_paper_figures_and_vtk.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step08_generate_paper_figures_and_vtk.py) | `gt_2021_clean.npz`, `maxdiss_3000_par.txt` |
| **Fig. 4** | Methodological workflow with chronologically independent datasets | Concept diagram | `esquema_metodologia.pdf` |
| **Fig. 5** | Bathymetry of Port of San Antonio and control sites A–E | [`step08_generate_paper_figures_and_vtk.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step08_generate_paper_figures_and_vtk.py) | `5x5_malla_moloBatiMod3.dep` |
| **Table 1** | FNO and U-Net error vs training size $N \in \{500, 1000, 2000, 3000\}$ | [`step02_train_fno.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step02_train_fno.py) & [`step03_train_unet_transunet.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step03_train_unet_transunet.py) | `norm_stats_maxdiss_*.npz`, `subset_val300_fields.npz` |
| **Table 2** | Site $R^2$ and RMSE for $H_{m0}$ over 2021 (8,760 h) for FNO, U-Net, TransUNet, RBF, DNN | [`step05_evaluate_2021_annual.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step05_evaluate_2021_annual.py) | `gt_2021_clean.npz`, checkpoints FNO & U-Net |
| **Fig. 6** | Continuous annual time series 2021 at sites A–E | [`step05_evaluate_2021_annual.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step05_evaluate_2021_annual.py) | `series_sites_all_en.png` |
| **Fig. 7** | Peak storm zoom (hour 5598 $\pm$ 120 h) across control sites | [`step05_evaluate_2021_annual.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step05_evaluate_2021_annual.py) | `series_storm_h5598_en.png` |
| **Fig. 8** | Spatial 2D comparison of annual RMSE and $R^2$ (FNO vs U-Net) | [`step08_generate_paper_figures_and_vtk.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step08_generate_paper_figures_and_vtk.py) | `fno_v2_yearfields.npz`, `ulib360_yearfields.npz` |
| **Fig. 9** | 2D wave agitation fields for peak storm (h5598) and worst case (h5590) | [`step08_generate_paper_figures_and_vtk.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step08_generate_paper_figures_and_vtk.py) | `comparison_fno_unet_cases.png` |
| **Table 3** | Resolution invariance: paired vs zero-shot super-resolution ($360\times360$) | [`step06_evaluate_zeroshot.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step06_evaluate_zeroshot.py) | Coarsened checkoints ($r180, r120, r90$) |
| **Fig. 10** | 2D zero-shot field comparison on validation case #203 | [`step06_evaluate_zeroshot.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step06_evaluate_zeroshot.py) | `fno_zeroshot_fields_en.png` |
| **Table 4** | Hyperparameter sweep: Fourier modes ($k$) and channel width ($d$) | [`step02_train_fno.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step02_train_fno.py) | `fno_maxdiss_3000_{k}w{d}_bench.json` |
| **Table 5 & 6** | Baseline computational benchmarks and 7-day operational forecast scaling | Theoretical / timing logs | Section 3.5 |
| **Fig. 11** | Extrapolation boundary: natural peak period excursions ($T_p > 22.10$ s) | [`step07_evaluate_outliers.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step07_evaluate_outliers.py) | `outliers_Tp22_en.png` |
| **Fig. 12** | Extrapolation breakdown under severe synthetic stress tests | [`step07_evaluate_outliers.py`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/paper_reproduction/step07_evaluate_outliers.py) | `outliers_synth_en.png`, `cases_oor/` |

---

## 2. Quick Audit Instructions (Evaluating Precalculated Results in Seconds)

To verify the numbers without performing the 23-hour GPU re-trainings:

```bash
# 1. Inspect Table 2 (Site determination coefficients R² and RMSE):
python3 step05_evaluate_2021_annual.py --audit-only

# 2. Inspect Resolution Invariance (Table 3 and Fig. 10):
python3 step06_evaluate_zeroshot.py --audit-only

# 3. Inspect Extrapolation Boundaries (Section 3.6 and Figs. 11-12):
python3 step07_evaluate_outliers.py --audit-only

# 4. Regenerate all 300-dpi paper figures and 3D ParaView VTK fields:
python3 step08_generate_paper_figures_and_vtk.py
```

All scripts read parameters directly from [`../configs/san_antonio.yaml`](file:///home/model2/luis/Doctorado/investigaciones/paper01_fno_2d_hm0/port_wave_surrogate/configs/san_antonio.yaml).
