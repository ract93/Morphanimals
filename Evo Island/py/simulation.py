import json
import os

import numpy as np

from environment import Environment
from metrics import SimulationMetrics
from visualization import open_frame_writers, save_capture_images, save_matrix_image

try:
    from evo_core import Simulation as _CppSimulation
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


# Kept for backward compatibility — not used by run_game anymore.
def is_coordinate_in_range(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True


def initialize_agent_matrix(n):
    from agent import Agent
    return [[Agent() for j in range(n)] for i in range(n)]


def run_game(trial_num, unique_results_dir, status_queue, cfg):
    if not _CPP_AVAILABLE:
        raise RuntimeError(
            "evo_core C++ extension not built.\n"
            "Build it with:  pip install scikit-build-core pybind11  &&  pip install -e ."
        )

    trial_dir = os.path.join(unique_results_dir, f"Trial_{trial_num}")
    os.makedirs(trial_dir, exist_ok=True)

    metrics = SimulationMetrics()
    csv_file_path = os.path.join(trial_dir, "simulation_metrics.csv")
    metrics.enable_csv_logging(csv_file_path)

    simulation_steps    = cfg["simulation_steps"]
    frame_save_interval = cfg["frame_save_interval"]
    frame_rate          = cfg["frame_rate"]

    # Map generation stays in Python (Environment unchanged)
    environment = Environment(cfg)
    save_matrix_image(environment.world_matrix, os.path.join(trial_dir, "Game_World"))

    start_pos = environment.find_easiest_starting_location()
    if start_pos is None:
        start_pos = (0, 0)

    # Normalise world_matrix to a plain list (perlin maps are numpy arrays)
    wm = environment.world_matrix
    if hasattr(wm, "tolist"):
        wm = wm.tolist()

    # Hand terrain + food matrices to the C++ simulation core
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

    attrs = ["strength", "hardiness", "age", "lifespan",
             "metabolism", "reproduction_threshold", "genetic_distance", "color"]
    writers  = open_frame_writers(videos_dir, frame_rate)
    captures = {attr: [] for attr in attrs}

    current_sim_step = 0

    try:
        while current_sim_step < simulation_steps:
            result = sim.step(current_sim_step)

            # Accumulate cumulative death counters from per-step deltas
            metrics.deaths_from_aging      += result.deaths_aging
            metrics.death_from_competition += result.deaths_competition
            metrics.deaths_from_starvation += result.deaths_starvation
            metrics.deaths_from_exposure   += result.deaths_exposure
            metrics.cumulative_deaths      += (
                result.deaths_aging + result.deaths_competition +
                result.deaths_starvation + result.deaths_exposure
            )

            # Per-step totals for average calculation (reset by reset_averages)
            metrics.population_count                 = result.population_count
            metrics.species_counts                   = result.species_counts
            metrics.total_age                        = result.total_age
            metrics.total_lifespan                   = result.total_lifespan
            metrics.total_strength                   = result.total_strength
            metrics.total_hardiness                  = result.total_hardiness
            metrics.total_metabolism                 = result.total_metabolism
            metrics.total_reproduction_threshold     = result.total_reproduction_threshold

            metrics.calculate_averages()
            metrics.log_metrics(current_sim_step)
            if status_queue is not None:
                status_queue.put(
                    (trial_num, metrics.get_state_string(trial_num, current_sim_step, simulation_steps))
                )
            metrics.reset_averages()

            current_sim_step += 1

            if result.extinct:
                if status_queue is not None:
                    status_queue.put(
                        (trial_num, f"Trial {trial_num} | All agents died at step {current_sim_step}")
                    )
                break

            if current_sim_step % frame_save_interval == 0:
                for attr, writer in writers.items():
                    writer.write(sim.get_attribute_matrix(attr))

            if current_sim_step in capture_intervals:
                for attr in captures:
                    captures[attr].append((current_sim_step, sim.get_attribute_matrix(attr)))
    finally:
        for writer in writers.values():
            writer.close()

    save_capture_images(captures, images_dir)

    metrics.close_csv_logging()

    with open(os.path.join(trial_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    if status_queue is not None:
        status_queue.put((trial_num, f"Trial {trial_num} | Done"))
