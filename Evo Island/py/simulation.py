import json
import os
import random

import numpy as np

from agent import Agent
from environment import Environment
from metrics import SimulationMetrics
from speciation import classify_species
from visualization import render_and_save_videos, save_matrix_image, transform_matrix


# Simulation Logic
def is_coordinate_in_range(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True


def initialize_agent_matrix(n):
    return [[Agent() for j in range(n)] for i in range(n)]


def simulate_agent_time_step(current_step, i, j, environment, agent_matrix, metrics, live_agents, cfg):
    # Check if current cell has a live agent
    current_agent = agent_matrix[i][j]
    if not current_agent.alive:
        return

    # Age Logic
    was_alive_before_aging = current_agent.alive
    current_agent.age_agent(cfg["enable_aging"])
    if not current_agent.alive and was_alive_before_aging:
        metrics.deaths_from_aging += 1
        metrics.cumulative_deaths += 1
        live_agents.discard((i, j))
        return

    # Food Consumption Logic
    if cfg["enable_food"]:

        # First, check if agent survived reproduction last turn.
        if current_agent.energy_reserves <= 0:
            # Not enough food available to meet basic metabolic needs
            agent_matrix[i][j].kill_agent()
            metrics.deaths_from_starvation += 1
            metrics.cumulative_deaths += 1
            live_agents.discard((i, j))
            return

        food_available = environment.calculate_food_available(i, j, current_step)
        current_agent.consume_food(food_available)
        current_agent.metabolize(current_agent.metabolism)
        environment.update_food_matrix(i, j, current_step, food_available)

        # Then, check if the agent survives this turn
        if current_agent.energy_reserves <= 0:
            # Not enough food available to meet basic metabolic needs
            agent_matrix[i][j].kill_agent()
            metrics.deaths_from_starvation += 1
            metrics.cumulative_deaths += 1
            live_agents.discard((i, j))
            return

    if cfg["enable_reproduction_threshold"] and cfg["enable_food"]:

        can_reproduce = (
            current_agent.energy_reserves >= current_agent.reproduction_threshold
        )
        if can_reproduce:

            # At this point, agent has survived current turn and reproduces if it has enough food to meet threshold.
            # Note reproducing takes energy and may kill the parent agent at the beginning of next turn.
            current_agent.metabolize(
                current_agent.reproduction_threshold
            )  # Deduct reproduction cost
            new_individual = Agent.reproduce_asexually(current_agent, cfg["mutation_rate"])
            diceroll = random.randint(1, 9)
            movement_offsets = {
                1: (0, 0),
                2: (1, 0),
                3: (-1, 0),
                4: (0, 1),
                5: (0, -1),
                6: (1, 1),
                7: (1, -1),
                8: (-1, 1),
                9: (-1, -1),
            }
            di, dj = movement_offsets.get(diceroll, (0, 0))
            new_i, new_j = i + di, j + dj

            # Check if movement is within the map
            if is_coordinate_in_range(environment.map_size, new_i, new_j):
                # Check if new agent is hardy enough to occupy the cell.
                if (
                    new_individual.hardiness
                    > environment.level_difficulty[
                        environment.world_matrix[new_i][new_j]
                    ]
                ):
                    # Check if cell is already occupied and competition is allowed
                    if agent_matrix[new_i][new_j].alive and cfg["enable_violence"]:
                        # Aggression here
                        metrics.death_from_competition += 1
                        metrics.cumulative_deaths += 1
                        # One or the other agent in competition always dies regardless of who wins, so its safe to increment here.
                        if (
                            new_individual.strength
                            > agent_matrix[new_i][new_j].strength
                        ):
                            agent_matrix[new_i][new_j] = new_individual
                            # position stays in live_agents — still occupied
                            return
                        # If the new agent is weaker or equal in strength, it dies, and no change is made to the cell
                    else:
                        # The prospective cell is empty, the new agent just occupies it with no competition
                        agent_matrix[new_i][new_j] = new_individual
                        live_agents.add((new_i, new_j))
                        return
                else:
                    # New individual dies from exposure as it wasnt hardy enough
                    metrics.deaths_from_exposure += 1
                    metrics.cumulative_deaths += 1
                    return
        else:
            # Original Agent lives but doesnt reproduce
            return

    else:  # Food and Reproduction_Threshold Logic is off
        # At this point, agent has survived current turn and reproduces if it has enough food to meet threshold.
        new_individual = Agent.reproduce_asexually(current_agent, cfg["mutation_rate"])
        diceroll = random.randint(1, 9)
        movement_offsets = {
            1: (0, 0),
            2: (1, 0),
            3: (-1, 0),
            4: (0, 1),
            5: (0, -1),
            6: (1, 1),
            7: (1, -1),
            8: (-1, 1),
            9: (-1, -1),
        }
        di, dj = movement_offsets.get(diceroll, (0, 0))
        new_i, new_j = i + di, j + dj

        # Check if movement is within the map
        if is_coordinate_in_range(environment.map_size, new_i, new_j):
            # Check if agent is hardy enough to occupy the cell.
            if (
                new_individual.hardiness
                > environment.level_difficulty[environment.world_matrix[new_i][new_j]]
            ):
                if agent_matrix[new_i][new_j].alive and cfg["enable_violence"]:
                    # One or the other agent in comeptition dies, so its safe to increment here.
                    metrics.death_from_competition += 1
                    metrics.cumulative_deaths += 1
                    if new_individual.strength > agent_matrix[new_i][new_j].strength:
                        agent_matrix[new_i][new_j] = new_individual
                        # position stays in live_agents — still occupied
                    # If the new individual is weaker or equal in strength, it dies, and no change is made to the cell
                else:
                    # The prospective cell is empty, the new individual just occupies it with no competition
                    agent_matrix[new_i][new_j] = new_individual
                    live_agents.add((new_i, new_j))
            else:
                # New individual dies from exposure as it wasnt hardy enough
                metrics.deaths_from_exposure += 1
                metrics.cumulative_deaths += 1


def run_game(trial_num, unique_results_dir, status_queue, cfg):
    # Create the trial-specific directory within the unique results directory
    trial_dir = os.path.join(unique_results_dir, f"Trial_{trial_num}")
    os.makedirs(trial_dir, exist_ok=True)

    # Create logger for metrics tracking
    metrics = SimulationMetrics()
    csv_file_path = os.path.join(trial_dir, "simulation_metrics.csv")
    metrics.enable_csv_logging(csv_file_path)

    # Use the map configurations
    simulation_steps = cfg["simulation_steps"]
    frame_save_interval = cfg["frame_save_interval"]
    frame_rate = cfg["frame_rate"]

    # Save the config of the experimental run as a text file
    #config_file_path = os.path.join(trial_dir, "config.txt")
    #with open(config_file_path, "w") as f:
        #for key, value in config.items():
           # f.write(f"{key}: {value}\n")
            #if isinstance(value, dict):
                #for sub_key, sub_value in value.items():
                    #f.write(f"  {sub_key}: {sub_value}\n")
    # Initialize the environment
    environment = Environment(cfg)

    # Save environment
    save_matrix_image(environment.world_matrix, os.path.join(trial_dir, "Game_World"))

    # Initialize the agent matrix
    agent_matrix = initialize_agent_matrix(environment.map_size)

    # Find starting position for the initial agent
    agent_starting_pos = environment.find_easiest_starting_location()

    # Create initial agent
    agent_matrix[agent_starting_pos[0]][
        agent_starting_pos[1]
    ] = Agent.create_live_agent()
    live_agents = {(agent_starting_pos[0], agent_starting_pos[1])}

    # Create directories for videos and images
    videos_dir = os.path.join(trial_dir, "Videos")
    images_dir = os.path.join(trial_dir, "Images")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Calculate intervals for capturing static images
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


    # Raw state snapshots — rendered after simulation completes
    attrs = ["strength", "hardiness", "age", "lifespan",
             "metabolism", "reproduction_threshold", "genetic_distance", "color"]
    frames = {attr: [] for attr in attrs}
    captures = {attr: [] for attr in attrs}



    # Main game loop
    current_sim_step = 0

    while current_sim_step < simulation_steps:
        for (i, j) in list(live_agents):  # snapshot so births this step don't act until next
            simulate_agent_time_step(
                current_sim_step, i, j, environment, agent_matrix, metrics, live_agents, cfg
            )

        # Assign species labels to agents
        classify_species(agent_matrix, live_agents, cfg["speciation_threshold"])

        # Calculate and collect metrics
        metrics.update_agent_metrics(agent_matrix, live_agents)
        metrics.calculate_averages()
        metrics.log_metrics(current_sim_step)
        if status_queue is not None:
            status_queue.put((trial_num, metrics.get_state_string(trial_num, current_sim_step, simulation_steps)))
        metrics.reset_averages()

        current_sim_step += 1

        if not live_agents:
            if status_queue is not None:
                status_queue.put((trial_num, f"Trial {trial_num} | All agents died at step {current_sim_step}"))
            break

        if current_sim_step % frame_save_interval == 0:
            for attr in frames:
                frames[attr].append(np.array(transform_matrix(agent_matrix, attr)))

        if current_sim_step in capture_intervals:
            for attr in captures:
                captures[attr].append((current_sim_step, np.array(transform_matrix(agent_matrix, attr))))

    render_and_save_videos(frames, captures, videos_dir, images_dir, frame_rate)

    # Close metrics file
    metrics.close_csv_logging()

    # Save config alongside the CSV so the notebook can load it
    with open(os.path.join(trial_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    if status_queue is not None:
        status_queue.put((trial_num, f"Trial {trial_num} | Done"))
