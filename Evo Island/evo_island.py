import asyncio
import csv
import warnings
import json
import math
import multiprocessing
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import imageio
import matplotlib.pyplot as plt
import nbformat as nbf
import noise
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from nbconvert.preprocessors import ExecutePreprocessor
from noise import pnoise2
from PIL import Image

warnings.filterwarnings("ignore", ".*missing an id field.*")

# Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("config.json", "r") as config_file:
    config = json.load(config_file)


class Environment:
    level_difficulty = {1: 1, 2: 10, 3: 20, 4: 40, 5: 50}

    def __init__(self, config):
        self.config = config
        self.map_size = config["map_size"]
        self.map_type = config.get("map_type", "perlin")
        self.world_matrix = self.generate_map()
        self.food_matrix = self.initialize_food_matrix()

    def generate_map(self):
        if self.map_type == "perlin":
            return self.generate_island(self.map_size)
        elif self.map_type == "petri_dish":
            return self.generate_petri_dish(self.map_size)
        elif self.map_type == "random":
            return self.generate_random_map(self.map_size)
        else:
            raise ValueError(f"Unknown map type: {self.map_type}")

    @staticmethod
    def generate_petri_dish(n):
        if n <= 0:
            return []

        array = [[0] * n for _ in range(n)]
        increment = 4 / (n - 1) if n > 1 else 0

        for i in range(n):
            for j in range(n):
                array[i][j] = round((j * increment) + 1)

        return array

    def generate_perlin_noise_terrain(self, n):
        if self.config["use_random_perlin_params"]:
            scale = random.uniform(50, 200)
            octaves = random.randint(4, 8)
            persistence = random.uniform(0.4, 0.7)
            lacunarity = random.uniform(1.8, 2.2)
        else:
            scale = 100.0
            octaves = 6
            persistence = 0.5
            lacunarity = 2.0

        world = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                world[i][j] = pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=n,
                    repeaty=n,
                    base=0,
                )

        world_range = np.max(world) - np.min(world)
        world = (world - np.min(world)) * (5 - 1) / (world_range if world_range != 0 else 1) + 1
        world = np.round(world).astype(int)

        return world

    def generate_island(self, n):
        world = self.generate_perlin_noise_terrain(n)
        if self.config.get("use_rivers", False):
            world = self.generate_river(world)
        return world

    def generate_random_map(self, n):
        return [[random.randint(1, 3) for _ in range(n)] for _ in range(n)]

    def generate_river(self, world_matrix):
        n = self.map_size

        scale = 200.0
        octaves = 6
        persistence = 0.4
        lacunarity = 2.0
        grid = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grid[i][j] = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=n,
                    repeaty=n,
                    base=0,
                )

        max_val = np.max(grid)
        min_val = np.min(grid)
        grid_range = max_val - min_val
        scaled_grid = (grid - min_val) / (grid_range if grid_range != 0 else 1)

        river_width = n // 50
        river_path = [(i, scaled_grid[i, n // 2]) for i in range(n)]

        for i in range(n):
            for j in range(n):
                if abs(int(river_path[i][1] * n) - j) <= river_width // 2:
                    world_matrix[i][j] = 1

        return world_matrix

    def initialize_food_matrix(self):
        initial_food_amount = self.config["initial_food"]
        food_matrix = [
            [(initial_food_amount, -1) for _ in range(self.map_size)]
            for _ in range(self.map_size)
        ]
        return food_matrix

    def calculate_food_available(self, i, j, current_step):
        food_generation_rate = self.config["food_generation_rate"]
        max_food_capacity = self.config["max_food_capacity"]

        previous_food_amount, last_accessed = self.food_matrix[i][j]
        if last_accessed == -1:
            food_produced = current_step * food_generation_rate
        else:
            food_produced = (current_step - last_accessed) * food_generation_rate

        new_food_amount = min(previous_food_amount + food_produced, max_food_capacity)

        self.food_matrix[i][j] = (new_food_amount, current_step)
        return new_food_amount

    def update_food_matrix(self, i, j, current_step, food_consumed):
        previous_food_amount, _last_accessed = self.food_matrix[i][j]
        new_food_amount = max(previous_food_amount - food_consumed, 0)
        self.food_matrix[i][j] = (new_food_amount, current_step)

    def find_easiest_starting_location(self):
        for i, row in enumerate(self.world_matrix):
            for j, difficulty in enumerate(row):
                if difficulty == 1:
                    return (i, j)
        return None


# Agent Class
class Agent:
    common_ancestor_genome = np.array([20, 10, 5, 5, 3])  # common ancestor's genome

    def __init__(self):
        self.alive = False
        self.genome = None
        self.color = None  # Agent color is based on its genome
        self.age = 0
        self.energy_reserves = 0
        self.genetic_distance = None
        self.species = 0
        self.reset_traits()

    @classmethod
    def create_live_agent(cls, genome=None):
        live_agent = cls()
        live_agent.alive = True
        live_agent.genome = (
            genome if genome is not None else cls.generate_default_genome()
        )
        live_agent.color = cls.genome_to_color(live_agent.genome)  # Calculate color
        live_agent.decode_genome()
        live_agent.calculate_genetic_distance()  # Calculate genetic distance from ancestor
        live_agent.energy_reserves = 5
        return live_agent

    def decode_genome(self):
        # Explicit mapping of genome indices to traits for clarity
        if self.genome is not None:
            self.lifespan = self.genome[0]
            self.hardiness = self.genome[1]
            self.strength = self.genome[2]
            self.metabolism = self.genome[3]
            self.reproduction_threshold = self.genome[4]

    @staticmethod
    def generate_default_genome():
        # Return the default genome, which could be the common ancestor's genome
        return Agent.common_ancestor_genome.copy()

    @staticmethod
    def mutate_genome(genome):
        mutation_rate = config["mutation_rate"]
        mutation_effects = np.zeros_like(genome)
        for i in range(len(genome)):
            if np.random.rand() < mutation_rate:
                mutation_effects[i] = np.random.normal(loc=0, scale=2)
        mutated_genome = genome + mutation_effects
        mutated_genome = np.clip(mutated_genome, 0, 100)
        return mutated_genome

    def calculate_genetic_distance(self, other_genome=None):
        if other_genome is None:
            other_genome = Agent.common_ancestor_genome
        if self.genome is not None:
            self.genetic_distance = np.linalg.norm(self.genome - other_genome)
        return self.genetic_distance

    @classmethod
    def reproduce_asexually(cls, parent_agent):
        child_genome = cls.mutate_genome(parent_agent.genome)
        return cls.create_live_agent(child_genome)

    def age_agent(self):
        self.age += 1
        if config["enable_aging"]:
            midpoint = self.lifespan / 2
            if midpoint == 0:
                self.kill_agent()
                return
            steepness = 10
            death_probability = 1 / (
                1 + math.exp(-steepness * (self.age - midpoint) / midpoint)
            )

            if random.random() < death_probability:
                self.kill_agent()

    def consume_food(self, food_consumed):
        self.energy_reserves += food_consumed

    def metabolize(self, energy_used):
        self.energy_reserves -= energy_used

    def reset_traits(self):
        self.age = 0
        self.energy_reserves = 0
        self.lifespan = 0
        self.hardiness = 0
        self.strength = 0
        self.metabolism = 0
        self.reproduction_threshold = 0
        self.genetic_distance = 0
        self.species = 0

    def kill_agent(self):
        self.alive = False
        self.genome = None
        self.color = None  # Reset color attribute
        self.reset_traits()

    @staticmethod
    def genome_to_color(genome):
        """ Convert a genome to a color in HSL space """
        # Normalize genome values to [0, 1] range
        genome_range = np.max(genome) - np.min(genome)
        norm_genome = (genome - np.min(genome)) / (genome_range if genome_range != 0 else 1)

        # Map normalized values to HSL space
        hue = (norm_genome[0] * 360) % 360
        saturation = 50 + norm_genome[1] * 50  # Keep saturation between 50% and 100%
        lightness = 50 + norm_genome[2] * 50   # Keep lightness between 50% and 100%

        return mcolors.hsv_to_rgb((hue / 360, saturation / 100, lightness / 100))



class SimulationMetrics:
    def __init__(self):
        self.species_counts = 0
        self.population_count = 0  # Keep track of the population to calculate averages
        self.cumulative_deaths = 0
        self.deaths_from_aging = 0
        self.death_from_competition = 0
        self.deaths_from_starvation = 0
        self.deaths_from_exposure = 0

        self.total_age = 0  # Use this to calculate average_age
        self.total_lifespan = 0  # Use this for average_lifespan
        self.total_strength = 0
        self.total_hardiness = 0
        self.total_metabolism = 0
        self.total_reproduction_threshold = 0

        self.average_age = 0
        self.average_lifespan = 0
        self.average_strength = 0
        self.average_hardiness = 0
        self.average_metabolism = 0
        self.average_reproduction_threshold = 0

        # CSV logging fields
        self.csv_logging_enabled = False

    def enable_csv_logging(self, filepath):
        self.csv_logging_enabled = True
        self.filepath = filepath
        self.fields = [
            "Timestep",
            "Population Count",
            "Cumulative Deaths",
            "Deaths from Aging",
            "Deaths from Competition",
            "Deaths from Starvation",
            "Deaths from Exposure",
            "Average Age",
            "Average Lifespan",
            "Average Strength",
            "Average Hardiness",
            "Average Metabolism",
            "Average Reproduction Threshold",
            "Number of Species",
        ]
        self.csv_file = open(
            self.filepath, "w", newline="", buffering=1
        )  # Line buffering
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fields)
        self.writer.writeheader()

    def log_metrics(self, timestep):
        if not self.csv_logging_enabled:
            return
        row = {
            "Timestep": timestep,
            "Population Count": self.population_count,
            "Cumulative Deaths": self.cumulative_deaths,
            "Deaths from Aging": self.deaths_from_aging,
            "Deaths from Competition": self.death_from_competition,
            "Deaths from Starvation": self.deaths_from_starvation,
            "Deaths from Exposure": self.deaths_from_exposure,
            "Average Age": self.average_age,
            "Average Lifespan": self.average_lifespan,
            "Average Strength": self.average_strength,
            "Average Hardiness": self.average_hardiness,
            "Average Metabolism": self.average_metabolism,
            "Average Reproduction Threshold": self.average_reproduction_threshold,
            "Number of Species": self.species_counts,
        }
        self.writer.writerow(row)

    def close_csv_logging(self):
        if self.csv_logging_enabled:
            self.csv_file.close()

    def update_agent_metrics(self, agent_matrix, live_agents):
        species_set = set()
        for (i, j) in live_agents:
            agent = agent_matrix[i][j]
            self.total_age += agent.age
            self.total_lifespan += agent.lifespan
            self.total_strength += agent.strength
            self.total_hardiness += agent.hardiness
            self.total_metabolism += agent.metabolism
            self.total_reproduction_threshold += agent.reproduction_threshold
            self.population_count += 1
            species_set.add(agent.species)

        self.species_counts = len(species_set)

    def calculate_averages(self):
        # Ensure division by zero is handled
        population = max(self.population_count, 1)
        self.average_age = self.total_age / population
        self.average_lifespan = self.total_lifespan / population
        self.average_strength = self.total_strength / population
        self.average_hardiness = self.total_hardiness / population
        self.average_metabolism = self.total_metabolism / population
        self.average_reproduction_threshold = (
            self.total_reproduction_threshold / population
        )

    def reset_averages(self):
        # Reset total stats (but not cumulative ones) for the next calculation step
        self.total_age = 0
        self.total_lifespan = 0
        self.total_strength = 0
        self.total_hardiness = 0
        self.total_metabolism = 0
        self.total_reproduction_threshold = 0
        self.population_count = 0

    def get_state_string(self, trial_num, step, total_steps):
        return (
            f"Trial {trial_num} | Step {step}/{total_steps} | "
            f"Pop:{self.population_count} | "
            f"Deaths:{self.cumulative_deaths}(Age:{self.deaths_from_aging} Comp:{self.death_from_competition} Starv:{self.deaths_from_starvation} Exp:{self.deaths_from_exposure}) | "
            f"Age:{self.average_age:.1f} Life:{self.average_lifespan:.1f} Str:{self.average_strength:.1f} "
            f"Hard:{self.average_hardiness:.1f} Metab:{self.average_metabolism:.1f} Repr:{self.average_reproduction_threshold:.1f} | "
            f"Species:{self.species_counts}"
        )



# Speciation Functions
def classify_species(agent_matrix, live_agents, threshold=config["speciation_threshold"]):
    species_counter = 1
    rep_genomes = []   # parallel arrays — avoids attribute lookup inside hot loop
    rep_labels = []

    for (i, j) in live_agents:
        agent = agent_matrix[i][j]
        assigned = False
        if rep_genomes:
            rep_arr = np.stack(rep_genomes)                          # (S, 5)
            dists = np.linalg.norm(rep_arr - agent.genome, axis=1)  # (S,) — one call
            idx = int(np.argmin(dists))
            if dists[idx] < threshold:
                agent.species = rep_labels[idx]
                assigned = True

        if not assigned:
            agent.species = species_counter
            rep_genomes.append(agent.genome)
            rep_labels.append(species_counter)
            species_counter += 1



# Simulation Logic
def is_coordinate_in_range(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True


def initialize_agent_matrix(n):
    return [[Agent() for j in range(n)] for i in range(n)]


def simulate_agent_time_step(current_step, i, j, environment, agent_matrix, metrics, live_agents):
    # Check if current cell has a live agent
    current_agent = agent_matrix[i][j]
    if not current_agent.alive:
        return

    # Age Logic
    was_alive_before_aging = current_agent.alive
    current_agent.age_agent()
    if not current_agent.alive and was_alive_before_aging:
        metrics.deaths_from_aging += 1
        metrics.cumulative_deaths += 1
        live_agents.discard((i, j))
        return

    # Food Consumption Logic
    if config["enable_food"]:

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

    if config["enable_reproduction_threshold"] and config["enable_food"]:

        can_reproduce = (
            current_agent.energy_reserves >= current_agent.reproduction_threshold
        )
        if can_reproduce:

            # At this point, agent has survived current turn and reproduces if it has enough food to meet threshold.
            # Note reproducing takes energy and may kill the parent agent at the beginning of next turn.
            current_agent.metabolize(
                current_agent.reproduction_threshold
            )  # Deduct reproduction cost
            new_individual = Agent.reproduce_asexually(current_agent)
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
                    if agent_matrix[new_i][new_j].alive and config["enable_violence"]:
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
        new_individual = Agent.reproduce_asexually(current_agent)
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
                if agent_matrix[new_i][new_j].alive and config["enable_violence"]:
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


# Matrix Creation, Visualization, Saving
def transform_matrix(agent_matrix, attribute):
    if attribute == "color":
        return [[(getattr(agent, attribute, (0, 0, 0)) if agent.alive else (0, 0, 0)) for agent in row] for row in agent_matrix]
    else:
        return [[getattr(agent, attribute, 0) for agent in row] for row in agent_matrix]


def save_matrix_image(matrix, file_name):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Game World")
    cbar = plt.colorbar(im, ax=ax)
    plt.savefig(str(file_name))



def render_and_save_videos(frames, captures, videos_dir, images_dir, frame_rate):
    video_specs = [
        ("strength",               "viridis", 0,   100, "strength_map.mp4"),
        ("hardiness",              "viridis", 0,   100, "hardiness_map.mp4"),
        ("age",                    "viridis", 0,    50, "age_map.mp4"),
        ("lifespan",               "inferno", 0,   100, "lifespan_map.mp4"),
        ("metabolism",             "inferno", 0,   100, "metabolism_map.mp4"),
        ("reproduction_threshold", "magma",   0,    50, "reproduction_threshold_map.mp4"),
        ("genetic_distance",       "magma",   0,    50, "genetic_drift_map.mp4"),
        ("color",                  None,      None, None, "species_map.mp4"),
    ]

    def to_rgb(raw, cmap_name, vmin, vmax):
        arr = np.array(raw)
        if cmap_name is None:
            return (arr * 255).astype(np.uint8)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return (plt.get_cmap(cmap_name)(norm(arr))[:, :, :3] * 255).astype(np.uint8)

    def save_one(spec):
        attr, cmap_name, vmin, vmax, filename = spec
        rgb_frames = [to_rgb(f, cmap_name, vmin, vmax) for f in frames[attr]]
        imageio.mimsave(os.path.join(videos_dir, filename), rgb_frames, fps=frame_rate, macro_block_size=1)
        for step, raw in captures[attr]:
            imageio.imwrite(
                os.path.join(images_dir, f"{attr}_step_{step}.png"),
                to_rgb(raw, cmap_name, vmin, vmax)
            )

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_one, spec) for spec in video_specs]
        for f in futures:
            f.result()


def create_trial_notebook(trial_dir, notebook_path, status_queue=None, trial_num=None):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "trial_analysis.ipynb")
    shutil.copy(template_path, notebook_path)

    executed_notebook_path = notebook_path.replace(".ipynb", "_executed.ipynb")
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        with open(notebook_path) as f:
            nb = nbf.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": trial_dir}})
        with open(executed_notebook_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        os.remove(notebook_path)
    except Exception as e:
        if status_queue is not None and trial_num is not None:
            status_queue.put((trial_num, f"Trial {trial_num} | Notebook error: {e}"))
        else:
            print("Error during notebook execution:", e)




def run_game(trial_num, unique_results_dir, status_queue=None):
    # Create the trial-specific directory within the unique results directory
    trial_dir = os.path.join(unique_results_dir, f"Trial_{trial_num}")
    os.makedirs(trial_dir, exist_ok=True)

    # Create logger for metrics tracking
    metrics = SimulationMetrics()
    csv_file_path = os.path.join(trial_dir, "simulation_metrics.csv")
    metrics.enable_csv_logging(csv_file_path)

    # Use the map configurations
    simulation_steps = config["simulation_steps"]
    frame_save_interval = config["frame_save_interval"]
    frame_rate = config["frame_rate"]

    # Save the config of the experimental run as a text file
    #config_file_path = os.path.join(trial_dir, "config.txt")
    #with open(config_file_path, "w") as f:
        #for key, value in config.items():
           # f.write(f"{key}: {value}\n")
            #if isinstance(value, dict):
                #for sub_key, sub_value in value.items():
                    #f.write(f"  {sub_key}: {sub_value}\n")
    # Initialize the environment
    environment = Environment(config)

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
                current_sim_step, i, j, environment, agent_matrix, metrics, live_agents
            )

        # Assign species labels to agents
        classify_species(agent_matrix, live_agents)

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
        json.dump(config, f)

    # Create metrics notebook
    notebook_path = os.path.join(trial_dir, "analysis_notebook.ipynb")
    create_trial_notebook(trial_dir, notebook_path, status_queue, trial_num)

    if status_queue is not None:
        status_queue.put((trial_num, f"Trial {trial_num} | Done"))


def run_experiment():
    base_results_dir = os.path.join("Experimental_Results")

    # Prompt the user for the name of the experimental run
    experiment_name = input("Please enter a name for this experimental run: ")
    unique_results_dir = os.path.join(base_results_dir, experiment_name)

    # Ensure the unique results directory exists
    os.makedirs(unique_results_dir, exist_ok=True)

    num_experimental_trials = config.get("experimental_trials", 1)

    manager = multiprocessing.Manager()
    status_queue = manager.Queue()
    args = [(trial, unique_results_dir, status_queue) for trial in range(1, num_experimental_trials + 1)]

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


def create_aggregate_notebook(unique_results_dir, notebook_path):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "aggregate_analysis.ipynb")
    shutil.copy(template_path, notebook_path)

    executed_notebook_path = notebook_path.replace(".ipynb", "_executed.ipynb")
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        with open(notebook_path) as f:
            nb = nbf.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": unique_results_dir}})
        with open(executed_notebook_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("Executed summary notebook saved at", executed_notebook_path)
        os.remove(notebook_path)
    except Exception as e:
        print("Error during notebook execution:", e)

    print("Summary notebook creation process complete.")




def main():
    run_experiment()


if __name__ == "__main__":
    main()
