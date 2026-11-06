"""Command-line interface (CLI) for portsurrogate."""
import argparse
import os
import subprocess
import sys


def run_gui():
    """Launch the Streamlit graphical interface."""
    gui_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui", "app.py")
    if not os.path.exists(gui_script):
        print(f"Error: GUI script not found at {gui_script}")
        sys.exit(1)

    print("Launching Port Wave Surrogate Streamlit GUI...")
    # Prefer ML venv if available (has streamlit installed)
    ml_python = "/home/model2/ambientes/ML/bin/python3"
    python_bin = ml_python if os.path.exists(ml_python) else sys.executable

    cmd = [python_bin, "-m", "streamlit", "run", gui_script]
    subprocess.run(cmd)


def run_mda():
    """CLI to run MDA on a hindcast file."""
    parser = argparse.ArgumentParser(description="Run Maximum Dissimilarity Algorithm on wave data.")
    parser.add_argument("--input", required=True, help="Input hindcast file (.txt or .csv)")
    parser.add_argument("--n_cases", type=int, default=2000, help="Number of representative cases to select")
    parser.add_argument("--output", default="mda_selected.txt", help="Output file path")
    args = parser.parse_args()

    from .hindcast import HindcastData
    from .mda import run_mda

    print(f"Loading wave data from {args.input}...")
    if args.input.endswith(".csv"):
        hd = HindcastData.from_csv(args.input)
    else:
        hd = HindcastData.from_swan_file(args.input)

    arr = hd.get_parameter_array()
    print(f"Selecting {args.n_cases} cases using MDA...")
    centroids, indices = run_mda(arr, num_centers=args.n_cases)

    import numpy as np
    np.savetxt(args.output, centroids, fmt="%.4f", header="Hs Tp Dir")
    print(f"Saved {len(centroids)} selected states to {args.output}")


if __name__ == "__main__":
    run_gui()
