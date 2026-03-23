import json
import os

import numpy as np

from environment import Environment
from genes import GENES, VIDEO_SPECS
from metrics import SimulationMetrics
from output import SimulationOutput

try:
    from evo_core import Simulation as _CppSimulation
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False



def run_game(trial_num, unique_results_dir, status_queue, cfg):
    """Run a single simulation trial and write all outputs to trial_dir.

    Delegates the hot-path step loop to the C++ evo_core extension. Python
    handles I/O and metric logging around each C++ step call.  All output
    is written to a single HDF5 file (simulation.h5); visualization is a
    separate post-processing step.

    Args:
        trial_num:          1-based trial index, used for directory naming.
        unique_results_dir: Root results directory for this experiment run.
        status_queue:       Multiprocessing queue for live console status updates.
        cfg:                Config dict loaded from config.json.
    """
    if not _CPP_AVAILABLE:
        raise RuntimeError(
            "evo_core C++ extension not built.\n"
            "Build it with:  pip install scikit-build-core pybind11  &&  pip install -e ."
        )

    trial_dir = os.path.join(unique_results_dir, f"Trial_{trial_num}")
    os.makedirs(trial_dir, exist_ok=True)

    simulation_steps    = cfg["simulation_steps"]
    frame_save_interval = cfg["frame_save_interval"]

    # Map generation stays in Python; the resulting matrices are passed to C++
    environment = Environment(cfg)

    start_pos = environment.find_easiest_starting_location()
    if start_pos is None:
        start_pos = (0, 0)

    # Perlin maps are numpy arrays; convert to plain lists for pybind11 ingestion
    wm_arr  = np.array(environment.world_matrix)
    wm_list = wm_arr.tolist()

    # Hand terrain and food grid to the C++ simulation core
    sim = _CppSimulation(
        cfg,
        wm_list,
        environment.food_matrix,
        start_pos[0],
        start_pos[1],
    )

    # Open HDF5 output file — all simulation data goes here
    h5_path = os.path.join(trial_dir, "simulation.h5")
    output  = SimulationOutput(h5_path, cfg)
    output.log_terrain(wm_arr, np.array(environment.food_matrix))

    metrics = SimulationMetrics()

    # GIF frames collected every frame_save_interval steps
    gif_steps = set(range(frame_save_interval, simulation_steps + 1, frame_save_interval))
    # PNG snapshots at each eighth of the simulation
    capture_steps = {
        simulation_steps // 8,
        2 * simulation_steps // 8,
        3 * simulation_steps // 8,
        4 * simulation_steps // 8,
        5 * simulation_steps // 8,
        6 * simulation_steps // 8,
        7 * simulation_steps // 8,
        simulation_steps,
    }

    attrs            = [name for name, *_ in VIDEO_SPECS]
    current_sim_step = 0

    try:
        while current_sim_step < simulation_steps:
            # Advance the C++ simulation by one step
            result = sim.step(current_sim_step)

            # Accumulate death deltas from this step into running totals
            metrics.deaths_from_aging      += result.deaths_aging
            metrics.death_from_competition += result.deaths_competition
            metrics.deaths_from_starvation += result.deaths_starvation
            metrics.deaths_from_exposure   += result.deaths_exposure
            metrics.deaths_from_predation  += result.deaths_predation
            metrics.cumulative_deaths      += (
                result.deaths_aging + result.deaths_competition +
                result.deaths_starvation + result.deaths_exposure +
                result.deaths_predation
            )

            # Copy absolute per-step totals (used to compute averages)
            metrics.population_count = result.population_count
            metrics.species_counts   = result.species_counts
            metrics.total_age        = result.total_age
            for attr, *_ in GENES:
                setattr(metrics, f"total_{attr}", getattr(result, f"total_{attr}"))

            metrics.calculate_averages()
            output.log_metrics(metrics, current_sim_step)
            state_str = metrics.get_state_string(trial_num, current_sim_step, simulation_steps)
            if status_queue is not None:
                status_queue.put((trial_num, state_str))
            else:
                print(f"\r{state_str}", end="", flush=True)
            metrics.reset_averages()

            current_sim_step += 1

            if result.extinct:
                extinct_str = f"Trial {trial_num} | All agents died at step {current_sim_step}"
                if status_queue is not None:
                    status_queue.put((trial_num, extinct_str))
                else:
                    print(f"\n{extinct_str}")
                break

            if current_sim_step in gif_steps:
                for attr in attrs:
                    output.log_frame(attr, sim.get_attribute_matrix(attr))

            if current_sim_step in capture_steps:
                for attr in attrs:
                    output.log_capture(attr, sim.get_attribute_matrix(attr))
                output.note_capture_step(current_sim_step)

    finally:
        output.log_phylogeny(sim.get_speciation_log())
        output.close()

    done_str = f"Trial {trial_num} | Done"
    if status_queue is not None:
        status_queue.put((trial_num, done_str))
    else:
        print(f"\n{done_str}")
