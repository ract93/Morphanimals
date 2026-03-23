"""Entry point for the Evo Island simulation.

Default (no flags) — full pipeline: simulation → visualization → notebooks:
    python evo_island.py

Run simulation only (writes HDF5, skips viz and notebooks):
    python evo_island.py --sim-only

Re-run visualization on an existing experiment (does not re-run simulation):
    python evo_island.py --visualize Experimental_Results/my_run

Re-run notebooks on an existing experiment (does not re-run simulation or viz):
    python evo_island.py --notebooks Experimental_Results/my_run

Visualize a single HDF5 file directly:
    python evo_island.py --visualize-file Experimental_Results/my_run/Trial_1/simulation.h5
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", ".*missing an id field.*")

# Add py/ to path so all module imports inside py/ resolve without changes.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "py"))

from experiment import run_experiment, run_simulation, run_visualization, run_notebooks
from visualize import visualize


def _load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="Evo Island — evolutionary simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sim-only", action="store_true",
        help="Run simulation only; write HDF5 files, skip visualization and notebooks.",
    )
    group.add_argument(
        "--visualize", metavar="RESULTS_DIR",
        help="Re-run visualization on all trials in an existing experiment directory.",
    )
    group.add_argument(
        "--visualize-file", metavar="H5_PATH",
        help="Visualize a single simulation.h5 file.",
    )
    group.add_argument(
        "--notebooks", metavar="RESULTS_DIR",
        help="Re-run notebooks on all trials in an existing experiment directory.",
    )
    args = parser.parse_args()

    config = _load_config()

    if args.visualize:
        run_visualization(os.path.abspath(args.visualize))

    elif args.visualize_file:
        h5_path = os.path.abspath(args.visualize_file)
        visualize(h5_path, os.path.dirname(h5_path))

    elif args.notebooks:
        run_notebooks(os.path.abspath(args.notebooks), config)

    elif args.sim_only:
        base_dir = "Experimental_Results"
        experiment_name = input("Please enter a name for this experimental run: ")
        results_dir = os.path.abspath(os.path.join(base_dir, experiment_name))
        os.makedirs(results_dir, exist_ok=True)
        run_simulation(config, results_dir)
        print(f"\nSimulation complete. HDF5 files written to: {results_dir}")
        print(f"Visualize with:  python evo_island.py --visualize \"{results_dir}\"")

    else:
        run_experiment(config)


if __name__ == "__main__":
    main()
