"""Experiment orchestration — three independent phases:

    Phase 1 — Simulation : run_simulation(config, results_dir)
              Runs all trials.  Each trial writes one simulation.h5.
              No visualization, no notebooks.  Safe to run on headless machines.

    Phase 2 — Visualization : run_visualization(results_dir)
              Reads every Trial_N/simulation.h5, writes Videos/, Images/,
              Game_World.png, phylogeny.png into that trial directory.
              Can be re-run any time after Phase 1 without touching HDF5 files.

    Phase 3 — Notebooks : run_notebooks(results_dir, config)
              Executes trial_analysis.ipynb and aggregate_analysis.ipynb against
              the HDF5 files.  Can be re-run independently of Phase 2.

run_experiment() chains all three phases — this is the default full-pipeline call.
"""

import glob
import json
import os

from notebooks import create_aggregate_notebook, create_trial_notebook
from simulation import run_game


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def run_simulation(config, results_dir):
    """Run all simulation trials, writing simulation.h5 for each.

    Args:
        config:      Config dict loaded from config.json.
        results_dir: Absolute path to the experiment output directory.
    """
    n = config.get("experimental_trials", 1)
    for trial in range(1, n + 1):
        print(f"Trial {trial}/{n} | Starting...")
        run_game(trial, results_dir, None, config)


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def run_visualization(results_dir):
    """Generate visualizations for all trials found in *results_dir*.

    Discovers trials by globbing Trial_*/simulation.h5.  Safe to re-run:
    writes only into Videos/ and Images/ subdirs, never modifies the HDF5.
    """
    from visualize import visualize

    h5_paths = sorted(glob.glob(os.path.join(results_dir, "Trial_*", "simulation.h5")))
    if not h5_paths:
        print(f"No simulation.h5 files found under {results_dir}")
        return

    for h5_path in h5_paths:
        trial_dir = os.path.dirname(h5_path)
        trial_label = os.path.basename(trial_dir)
        print(f"{trial_label} | Generating visualizations...")
        try:
            visualize(h5_path, trial_dir)
        except Exception as e:
            print(f"{trial_label} | Visualization error (HDF5 preserved): {e}")


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def run_notebooks(results_dir, config):
    """Execute analysis notebooks for all trials in *results_dir*.

    Writes config.json to *results_dir* so the aggregate notebook can read it,
    then executes per-trial and aggregate notebooks.  Safe to re-run.
    """
    trial_dirs = sorted(glob.glob(os.path.join(results_dir, "Trial_*")))
    if not trial_dirs:
        print(f"No Trial_* directories found under {results_dir}")
        return

    # Aggregate notebook needs a top-level config.json
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f)

    for trial_dir in trial_dirs:
        trial_label = os.path.basename(trial_dir)
        notebook_path = os.path.join(trial_dir, "analysis_notebook.ipynb")
        print(f"{trial_label} | Executing notebook...")
        create_trial_notebook(trial_dir, notebook_path)

    for trial_dir in trial_dirs:
        executed = os.path.join(trial_dir, "analysis_notebook_executed.ipynb")
        if os.path.exists(executed):
            from pathlib import Path
            print(f"  {os.path.basename(trial_dir)}: {Path(executed).as_uri()}")

    summary_path = os.path.join(results_dir, "summary_notebook.ipynb")
    create_aggregate_notebook(results_dir, summary_path)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_experiment(config):
    """Prompt for an experiment name then run all three phases in order."""
    base_dir = os.path.join("Experimental_Results")
    experiment_name = input("Please enter a name for this experimental run: ")
    results_dir = os.path.abspath(os.path.join(base_dir, experiment_name))
    os.makedirs(results_dir, exist_ok=True)

    run_simulation(config, results_dir)
    run_visualization(results_dir)
    run_notebooks(results_dir, config)
