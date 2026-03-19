import json
import os

import pandas as pd

from notebooks import create_aggregate_notebook, create_trial_notebook
from simulation import run_game


def run_experiment(config):
    base_results_dir = os.path.join("Experimental_Results")

    experiment_name = input("Please enter a name for this experimental run: ")
    unique_results_dir = os.path.abspath(os.path.join(base_results_dir, experiment_name))
    os.makedirs(unique_results_dir, exist_ok=True)

    num_experimental_trials = config.get("experimental_trials", 1)

    for trial in range(1, num_experimental_trials + 1):
        print(f"Trial {trial}/{num_experimental_trials} | Starting...")
        run_game(trial, unique_results_dir, None, config)

    for trial in range(1, num_experimental_trials + 1):
        trial_dir = os.path.join(unique_results_dir, f"Trial_{trial}")
        notebook_path = os.path.join(trial_dir, "analysis_notebook.ipynb")
        create_trial_notebook(trial_dir, notebook_path)

    for trial in range(1, num_experimental_trials + 1):
        path = os.path.abspath(os.path.join(unique_results_dir, f"Trial_{trial}", "analysis_notebook_executed.ipynb"))
        if os.path.exists(path):
            from pathlib import Path
            print(f"  Trial {trial}: {Path(path).as_uri()}")

    aggregate_results(unique_results_dir, num_experimental_trials, config)


def aggregate_results(unique_results_dir, num_trials, config):
    aggregated_data = []
    for trial in range(1, num_trials + 1):
        csv_file_path = os.path.join(
            unique_results_dir, f"Trial_{trial}", "simulation_metrics.csv"
        )
        trial_data = pd.read_csv(csv_file_path)
        trial_data["Trial"] = trial
        aggregated_data.append(trial_data)

    aggregated_df = pd.concat(aggregated_data, ignore_index=True)

    aggregated_csv_path = os.path.join(unique_results_dir, "aggregated_metrics.csv")
    aggregated_df.to_csv(aggregated_csv_path, index=False)

    with open(os.path.join(unique_results_dir, "config.json"), "w") as f:
        json.dump(config, f)

    summary_notebook_path = os.path.join(unique_results_dir, "summary_notebook.ipynb")
    create_aggregate_notebook(unique_results_dir, summary_notebook_path)
