import json
import multiprocessing
import os
import sys
import time

import pandas as pd

from notebooks import create_aggregate_notebook, create_trial_notebook
from simulation import run_game


def run_experiment(config):
    base_results_dir = os.path.join("Experimental_Results")

    # Prompt the user for the name of the experimental run
    experiment_name = input("Please enter a name for this experimental run: ")
    unique_results_dir = os.path.join(base_results_dir, experiment_name)

    # Ensure the unique results directory exists
    os.makedirs(unique_results_dir, exist_ok=True)

    num_experimental_trials = config.get("experimental_trials", 1)

    manager = multiprocessing.Manager()
    status_queue = manager.Queue()
    args = [(trial, unique_results_dir, status_queue, config) for trial in range(1, num_experimental_trials + 1)]

    # Reserve one line per trial
    for i in range(1, num_experimental_trials + 1):
        print(f"Trial {i} | Starting...")
    sys.stdout.flush()

    def flush_queue():
        while not status_queue.empty():
            trial_num, message = status_queue.get_nowait()
            lines_up = num_experimental_trials - trial_num + 1
            sys.stdout.write(f"\033[{lines_up}A\r\033[2K{message}\033[{lines_up}B")
        sys.stdout.flush()

    pool = multiprocessing.Pool()
    try:
        async_result = pool.starmap_async(run_game, args)
        while not async_result.ready():
            flush_queue()
            time.sleep(0.1)
        flush_queue()
        pool.close()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers...")
        pool.terminate()
    finally:
        pool.join()
    print()
    for trial in range(1, num_experimental_trials + 1):
        trial_dir = os.path.join(unique_results_dir, f"Trial_{trial}")
        notebook_path = os.path.join(trial_dir, "analysis_notebook.ipynb")
        create_trial_notebook(trial_dir, notebook_path)

    for trial in range(1, num_experimental_trials + 1):
        path = os.path.abspath(os.path.join(unique_results_dir, f"Trial_{trial}", "analysis_notebook_executed.ipynb"))
        if os.path.exists(path):
            from pathlib import Path
            print(f"  Trial {trial}: {Path(path).as_uri()}")

    aggregate_results(
        unique_results_dir,
        num_experimental_trials,
        config
    )


def aggregate_results(
    unique_results_dir, num_trials, config
):
    # Aggregate CSV files from all trials
    aggregated_data = []
    for trial in range(1, num_trials + 1):
        csv_file_path = os.path.join(
            unique_results_dir, f"Trial_{trial}", "simulation_metrics.csv"
        )
        trial_data = pd.read_csv(csv_file_path)
        trial_data["Trial"] = trial
        aggregated_data.append(trial_data)

    aggregated_df = pd.concat(aggregated_data, ignore_index=True)

    # Save the aggregated data to a new CSV file
    aggregated_csv_path = os.path.join(unique_results_dir, "aggregated_metrics.csv")
    aggregated_df.to_csv(aggregated_csv_path, index=False)

    # Save config alongside the aggregated CSV so the notebook can load it
    with open(os.path.join(unique_results_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Create a summary Jupyter notebook
    summary_notebook_path = os.path.join(unique_results_dir, "summary_notebook.ipynb")
    create_aggregate_notebook(unique_results_dir, summary_notebook_path)
