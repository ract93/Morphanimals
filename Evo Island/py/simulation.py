import json
import os

from environment import Environment
from genes import GENES, VIDEO_SPECS
from metrics import SimulationMetrics
from visualization import open_frame_writers, save_capture_images, save_matrix_image

try:
    from evo_core import Simulation as _CppSimulation
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False



def run_game(trial_num, unique_results_dir, status_queue, cfg):
    """Run a single simulation trial and write all outputs to trial_dir.

    Delegates the hot-path step loop to the C++ evo_core extension. Python
    handles I/O, frame rendering, and metric logging around each C++ step call.

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

    # Set up CSV metric logging for this trial
    metrics = SimulationMetrics()
    csv_file_path = os.path.join(trial_dir, "simulation_metrics.csv")
    metrics.enable_csv_logging(csv_file_path)

    simulation_steps    = cfg["simulation_steps"]
    frame_save_interval = cfg["frame_save_interval"]
    frame_rate          = cfg["frame_rate"]

    # Map generation stays in Python; the resulting matrices are passed to C++
    environment = Environment(cfg)
    save_matrix_image(environment.world_matrix, os.path.join(trial_dir, "Game_World"))

    start_pos = environment.find_easiest_starting_location()
    if start_pos is None:
        start_pos = (0, 0)

    # Perlin maps are numpy arrays; convert to plain lists for pybind11 ingestion
    wm = environment.world_matrix
    if hasattr(wm, "tolist"):
        wm = wm.tolist()

    # Hand terrain and food grid to the C++ simulation core
    sim = _CppSimulation(
        cfg,
        wm,
        environment.food_matrix,
        start_pos[0],
        start_pos[1],
    )

    videos_dir = os.path.join(trial_dir, "Videos")
    images_dir = os.path.join(trial_dir, "Images")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Evenly-spaced steps at which to save a full-resolution image capture
    capture_intervals = [
        simulation_steps // 8,
        2 * simulation_steps // 8,
        3 * simulation_steps // 8,
        4 * simulation_steps // 8,
        5 * simulation_steps // 8,
        6 * simulation_steps // 8,
        7 * simulation_steps // 8,
        simulation_steps,
    ]

    # Attributes rendered as video layers and image captures — derived from genes registry
    attrs = [name for name, *_ in VIDEO_SPECS]
    writers  = open_frame_writers(videos_dir, frame_rate)
    captures = {attr: [] for attr in attrs}

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
            metrics.log_metrics(current_sim_step)
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

            # Write video frames at the configured interval
            if current_sim_step % frame_save_interval == 0:
                for attr, writer in writers.items():
                    writer.write(sim.get_attribute_matrix(attr))

            # Save full-resolution image captures at evenly-spaced checkpoints
            if current_sim_step in capture_intervals:
                for attr in captures:
                    captures[attr].append((current_sim_step, sim.get_attribute_matrix(attr)))
    finally:
        for writer in writers.values():
            writer.close()

    save_capture_images(captures, images_dir)
    metrics.close_csv_logging()

    # Save a copy of the config used for this trial alongside its outputs
    with open(os.path.join(trial_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    done_str = f"Trial {trial_num} | Done"
    if status_queue is not None:
        status_queue.put((trial_num, done_str))
    else:
        print(f"\n{done_str}")
