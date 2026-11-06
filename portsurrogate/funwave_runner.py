"""Automation of FUNWAVE-GPU simulation campaigns, stabilization cascades, and dataset assembly."""
import csv
import os
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import h5py

from .config import PortConfig


class FUNWAVEBatchRunner:
    """Manages FUNWAVE-GPU simulation campaigns for generating training or test datasets."""

    def __init__(
        self,
        base_input_template: str,
        depth_file: str,
        breakwater_file: Optional[str] = None,
        total_time: float = 1260.0,
        steady_time: float = 250.0,
        t_intv_mean: float = 1000.0,
        nautical_to_cartesian_offset: float = 270.0,
    ):
        self.base_input_template = base_input_template
        self.depth_file = depth_file
        self.breakwater_file = breakwater_file
        self.total_time = total_time
        self.steady_time = steady_time
        self.t_intv_mean = t_intv_mean
        self.nautical_to_cartesian_offset = nautical_to_cartesian_offset

    @classmethod
    def from_config(cls, cfg: PortConfig, template_path: Optional[str] = None) -> "FUNWAVEBatchRunner":
        """Initialize from PortConfig."""
        depth_path = cfg.resolve_path("bathy_mod3")
        bw_path = cfg.resolve_path("breakwater_file") if "breakwater_file" in cfg.data_paths else None
        if template_path is None and "base_input_template" in cfg.data_paths:
            template_path = cfg.resolve_path("base_input_template")
        elif template_path is None:
            template_path = "template_input.txt"

        return cls(
            base_input_template=template_path,
            depth_file=depth_path,
            breakwater_file=bw_path,
            nautical_to_cartesian_offset=cfg.boundary.nautical_to_cartesian_offset,
        )

    @staticmethod
    def set_param(text: str, name: str, value: Union[str, float, int]) -> str:
        """Replace or set a parameter in FUNWAVE input.txt."""
        pat = re.compile(rf"^{name}\s*=.*$", re.MULTILINE)
        new_line = f"{name} = {value}"
        if pat.search(text):
            return pat.sub(new_line, text)
        return text + f"\n{new_line}"

    def generate_case_input(self, hsig: float, tpsmoo: float, dir_nautical: float) -> str:
        """Create input.txt content for a single sea state."""
        freqpeak = 1.0 / max(tpsmoo, 0.1)
        thetapeak = self.nautical_to_cartesian_offset - dir_nautical

        with open(self.base_input_template, "r", encoding="utf-8") as f:
            txt = f.read()

        # Timing and wavemaker parameters
        txt = self.set_param(txt, "TOTAL_TIME", self.total_time)
        txt = self.set_param(txt, "STEADY_TIME", self.steady_time)
        txt = self.set_param(txt, "T_INTV_mean", self.t_intv_mean)
        txt = self.set_param(txt, "PLOT_INTV", 1e9)  # Do not trigger instant fields
        txt = self.set_param(txt, "Hmo", f"{hsig:.4f}")
        txt = self.set_param(txt, "FreqPeak", f"{freqpeak:.6f}")
        txt = self.set_param(txt, "ThetaPeak", f"{thetapeak:.3f}")
        txt = self.set_param(txt, "DEPTH_FILE", self.depth_file)
        if self.breakwater_file:
            txt = self.set_param(txt, "BREAKWATER_FILE", self.breakwater_file)

        # Output switches: save phase-averaged means only
        for v in ["DEPTH_OUT", "U ", "V ", "ETA ", "Hmax ", "Hmin ", "MFmax ", "Umax ", "VORmax ", "MASK ", "MASK9 "]:
            txt = self.set_param(txt, v.strip(), "F")
        for v in ["Umean ", "Vmean ", "ETAmean ", "WaveHeight "]:
            txt = self.set_param(txt, v.strip(), "T")

        return txt

    def prepare_cases_directory(
        self,
        cases_df: pd.DataFrame,
        output_dir: str,
        case_prefix: str = "case_",
        id_col: Optional[str] = None
    ) -> str:
        """Generate case folders with input.txt and a manifest.csv."""
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "manifest.csv")

        with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            writer.writerow(["caso", "Hsig", "TPsmoo", "Dir", "FreqPeak", "ThetaPeak"])

            for idx, row in cases_df.iterrows():
                hsig = float(row.get("Hs", row.get("Hsig", row.get("Hmo"))))
                tp = float(row.get("Tp", row.get("TPsmoo")))
                d = float(row.get("Dir", row.get("PkDir")))

                c_name = str(row[id_col]) if id_col and id_col in row else f"{case_prefix}{idx+1:05d}"
                cdir = os.path.join(output_dir, c_name)
                os.makedirs(cdir, exist_ok=True)

                txt = self.generate_case_input(hsig, tp, d)
                with open(os.path.join(cdir, "input.txt"), "w", encoding="utf-8") as f_in:
                    f_in.write(txt)

                fp = 1.0 / max(tp, 0.1)
                th = self.nautical_to_cartesian_offset - d
                writer.writerow([c_name, f"{hsig:.4f}", f"{tp:.4f}", f"{d:.3f}", f"{fp:.6f}", f"{th:.3f}"])

        return manifest_path

    @staticmethod
    def generate_slurm_scripts(
        cases_dir: str,
        funwave_exec_path: str,
        script_dir: str,
        n_gpus: int = 4,
        cpus_per_task: int = 32,
    ) -> Tuple[str, str]:
        """Generate run_lote.sbatch driver and run_one.sh with 3-tier CFL stabilization cascade."""
        os.makedirs(script_dir, exist_ok=True)
        run_one_path = os.path.join(script_dir, "run_one.sh")
        sbatch_path = os.path.join(script_dir, "run_lote.sbatch")

        # 1. run_one.sh with 3-tier CFL cascade
        run_one_content = f"""#!/bin/bash
# run_one.sh: Executes one FUNWAVE case with automated CFL cascade stabilization
set -e

CASE_DIR="$1"
GPU_ID="$2"
EXEC="{funwave_exec_path}"

if [ -z "$CASE_DIR" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: run_one.sh <case_dir> <gpu_id>"
    exit 1
fi

cd "$CASE_DIR"
mkdir -p output

# Skip if already successfully simulated
if [ -f output/Hsig_00001 ] && ! grep -qE "NaN|Infinity|\\*\\*\\*\\*" output/Hsig_00001 2>/dev/null; then
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Cascaded stability attempts:
# Level 0: default (CFL=0.4, FroudeCap=1.5)
# Level 1: CFL=0.2, FroudeCap=1.0
# Level 2: CFL=0.1, FroudeCap=1.0

for LEVEL in 0 1 2; do
    INPUT_FILE="input.txt"
    if [ "$LEVEL" -eq 1 ]; then
        sed -e 's/CFL\\s*=.*/CFL = 0.2/' -e 's/FroudeCap\\s*=.*/FroudeCap = 1.0/' input.txt > input_stab1.txt
        INPUT_FILE="input_stab1.txt"
    elif [ "$LEVEL" -eq 2 ]; then
        sed -e 's/CFL\\s*=.*/CFL = 0.1/' -e 's/FroudeCap\\s*=.*/FroudeCap = 1.0/' input.txt > input_stab2.txt
        INPUT_FILE="input_stab2.txt"
    fi

    echo "Running $CASE_DIR level $LEVEL on GPU $GPU_ID..." > "stdout_${{LEVEL}}.log"
    if "$EXEC" "$INPUT_FILE" >> "stdout_${{LEVEL}}.log" 2>&1; then
        if [ -f output/Hsig_00001 ] && ! grep -qE "NaN|Infinity|\\*\\*\\*\\*" output/Hsig_00001; then
            echo "STAB$LEVEL" > stab.txt
            exit 0
        fi
    fi
done

echo "FAILED" > FAILED
exit 1
"""
        with open(run_one_path, "w", encoding="utf-8") as f:
            f.write(run_one_content)
        os.chmod(run_one_path, 0o755)

        # 2. run_lote.sbatch driver
        sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=funwave_lote
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output={script_dir}/lote_%j.out

CASES_DIR="{cases_dir}"
RUN_ONE="{run_one_path}"

# Find pending cases and dispatch in batches across available GPUs
find "$CASES_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r CDIR; do
    if [ -f "$CDIR/output/Hsig_00001" ] || [ -f "$CDIR/FAILED" ]; then
        continue
    fi
    # Dispatch
    bash "$RUN_ONE" "$CDIR" 0 &
    wait
done
"""
        with open(sbatch_path, "w", encoding="utf-8") as f:
            f.write(sbatch_content)

        return run_one_path, sbatch_path

    @staticmethod
    def collect_dataset_npz(
        cases_dir: str,
        manifest_path: str,
        out_npz_path: str,
        n_rows: int = 425,
        n_cols: int = 380,
    ) -> str:
        """Read all simulation output ASCII files and assemble a unified .npz archive."""
        manifest = pd.read_csv(manifest_path)
        casos = manifest["caso"].tolist()
        nc = len(casos)

        hm0 = np.zeros((nc, n_rows, n_cols), dtype=np.float32)
        eta_mean = np.zeros((nc, n_rows, n_cols), dtype=np.float32)
        umean = np.zeros((nc, n_rows, n_cols), dtype=np.float32)
        vmean = np.zeros((nc, n_rows, n_cols), dtype=np.float32)

        for i, c in enumerate(casos):
            od = os.path.join(cases_dir, str(c), "output")
            for fname, target_arr in [
                ("Hsig_00001", hm0),
                ("etamean_00001", eta_mean),
                ("umean_00001", umean),
                ("vmean_00001", vmean),
            ]:
                p = os.path.join(od, fname)
                if os.path.exists(p):
                    target_arr[i] = np.loadtxt(p, dtype=np.float32)
                else:
                    target_arr[i] = np.nan

        np.savez_compressed(
            out_npz_path,
            hm0=hm0,
            eta_mean=eta_mean,
            umean=umean,
            vmean=vmean,
            casos=np.array(casos),
            Hmo=manifest["Hsig"].to_numpy(dtype=np.float32),
            FreqPeak=manifest["FreqPeak"].to_numpy(dtype=np.float32),
            ThetaPeak=manifest["ThetaPeak"].to_numpy(dtype=np.float32),
        )
        return out_npz_path
