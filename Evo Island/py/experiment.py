import ctypes
import json
import multiprocessing
import os
import shutil
import struct
import sys
import time

import pandas as pd

from notebooks import create_aggregate_notebook, create_trial_notebook
from simulation import run_game


def _worker_init(py_dir):
    """Add py/ to sys.path in each spawned worker (Windows spawn doesn't inherit it)."""
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)


def _win_get_cursor_row():
    """Return current cursor row via Windows Console API, or None on failure."""
    if sys.platform != "win32":
        return None
    try:
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        csbi = ctypes.create_string_buffer(22)  # sizeof(CONSOLE_SCREEN_BUFFER_INFO)
        if ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, csbi):
            # dwCursorPosition (COORD) is at offset 4: two SHORTs (x, y)
            return struct.unpack_from("hh", csbi.raw, 4)[1]
    except Exception:
        pass
    return None


def _win_set_cursor_row(row):
    """Move cursor to column 0 of the given row via Windows Console API."""
    try:
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        # COORD packed as DWORD: X in low 16 bits, Y in high 16 bits
        ctypes.windll.kernel32.SetConsoleCursorPosition(
            handle, ctypes.c_uint32(row << 16)
        )
        return True
    except Exception:
        pass
    return False


def run_experiment(config):
    base_results_dir = os.path.join("Experimental_Results")

    # Prompt the user for the name of the experimental run
    experiment_name = input("Please enter a name for this experimental run: ")
    unique_results_dir = os.path.abspath(os.path.join(base_results_dir, experiment_name))

    # Ensure the unique results directory exists
    os.makedirs(unique_results_dir, exist_ok=True)

    num_experimental_trials = config.get("experimental_trials", 1)

    manager = multiprocessing.Manager()
    status_queue = manager.Queue()
    args = [(trial, unique_results_dir, status_queue, config) for trial in range(1, num_experimental_trials + 1)]

    trial_statuses = {i: f"Trial {i} | Starting..." for i in range(1, num_experimental_trials + 1)}

    cols = shutil.get_terminal_size(fallback=(220, 24)).columns

    # Record cursor row before printing the trial block so flush_queue can
    # jump back to the exact row and overwrite in place.
    start_row = _win_get_cursor_row()  # None on non-Windows
    if start_row is None:
        sys.stdout.write("\033[s")  # ANSI save — fallback for non-Windows

    for i in range(1, num_experimental_trials + 1):
        sys.stdout.write(f"\033[2K{trial_statuses[i]}\n")
    sys.stdout.flush()

    def flush_queue():
        updated = False
        while not status_queue.empty():
            trial_num, message = status_queue.get_nowait()
            trial_statuses[trial_num] = message
            updated = True
        if not updated:
            return
        if start_row is not None:
            _win_set_cursor_row(start_row)
        else:
            sys.stdout.write("\033[u")
        for i in range(1, num_experimental_trials + 1):
            msg = trial_statuses[i][:cols - 1]
            sys.stdout.write(f"\033[2K{msg}\n")
        sys.stdout.flush()

    # Workers on Windows (spawn) don't inherit sys.path — re-add py/ explicitly.
    py_dir = os.path.dirname(os.path.abspath(__file__))
    pool = multiprocessing.Pool(initializer=_worker_init, initargs=(py_dir,))
    try:
        async_result = pool.starmap_async(run_game, args)
        while not async_result.ready():
            flush_queue()
            time.sleep(0.1)
        flush_queue()
        async_result.get()  # re-raises any worker exception
        pool.close()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers...")
        pool.terminate()
    except Exception:
        pool.terminate()
        raise
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
