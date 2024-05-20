import csv
import json
import math
import os
import random
from datetime import datetime
from itertools import combinations

import imageio
import matplotlib.pyplot as plt
import nbformat as nbf
import noise
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor
from noise import pnoise2
from PIL import Image

# Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("config.json", "r") as config_file:
    config = json.load(config_file)


class Environment:
    level_difficulty = {1: 1, 2: 10, 3: 30, 4: 40, 5: 70}

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
        increment = 4 / (n - 1)

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

        world = (world - np.min(world)) * (5 - 1) / (np.max(world) - np.min(world)) + 1
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
        scaled_grid = (grid - min_val) / (max_val - min_val)

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
    common_ancestor_genome = np.array(
        [20, 10, 5, 5, 3]
    )  #common ancestor's genome

    def __init__(self):
        self.alive = False
        self.genome = None
        self.age = 0
        self.energy_reserves = 0
        self.genetic_distance = None
        self.reset_traits()

    @classmethod
    def create_live_agent(cls, genome=None):
        live_agent = cls()
        live_agent.alive = True
        live_agent.genome = (
            genome if genome is not None else cls.generate_default_genome()
        )
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

    def calculate_genetic_distance(self):
        if self.genome is not None:
            self.genetic_distance = np.linalg.norm(
                self.genome - Agent.common_ancestor_genome
            )

    @classmethod
    def reproduce_asexually(cls, parent_agent):
        child_genome = cls.mutate_genome(parent_agent.genome)
        return cls.create_live_agent(child_genome)

    def age_agent(self):
        self.age += 1
        if config["enable_aging"]:
            # Use a sigmoid function to calculate death probability
            midpoint = (
                self.lifespan / 2
            )  # The age at which the death probability is 0.5
            steepness = 10  # Adjust this value to control the steepness of the curve
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

    def kill_agent(self):
        self.alive = False
        self.genome = None
        self.reset_traits()

class SimulationMetrics:
    def __init__(self):
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
            "Average Reproduction Threshold"
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
            "Average Reproduction Threshold": self.average_reproduction_threshold
        }
        self.writer.writerow(row)

    def close_csv_logging(self):
        if self.csv_logging_enabled:
            self.csv_file.close()

    def update_agent_metrics(self, agent_matrix):
        # Iterate over each agent in the matrix
        for row in agent_matrix:
            for agent in row:
                if agent.alive:  # Only consider agents that are alive
                    self.total_age += agent.age
                    self.total_lifespan += agent.lifespan
                    self.total_strength += agent.strength
                    self.total_hardiness += agent.hardiness
                    self.total_metabolism += agent.metabolism
                    self.total_reproduction_threshold += agent.reproduction_threshold
                    self.population_count += 1

    def calculate_averages(self):
        # Ensure division by zero is handled
        population = max(self.population_count, 1)
        self.average_age = self.total_age / population
        self.average_lifespan = self.total_lifespan / population
        self.average_strength = self.total_strength / population
        self.average_hardiness = self.total_hardiness / population
        self.average_metabolism = self.total_metabolism / population
        self.average_reproduction_threshold = self.total_reproduction_threshold / population

    def reset_averages(self):
        # Reset total stats (but not cumulative ones) for the next calculation step
        self.total_age = 0
        self.total_lifespan = 0
        self.total_strength = 0
        self.total_hardiness = 0
        self.total_metabolism = 0
        self.total_reproduction_threshold = 0
        self.population_count = 0

    def print_current_state(self):
            # Format and print the current state in a single line
            print(
                f"Population: {self.population_count}, "
                f"Cumulative Deaths: {self.cumulative_deaths} (Deaths by Aging: {self.deaths_from_aging}, "
                f"Deaths by Competition: {self.death_from_competition}, Deaths by Starvation: {self.deaths_from_starvation}, "
                f"Deaths by Exposure: {self.deaths_from_exposure}), "
                f"Avg Age: {self.average_age:.2f}, Avg Lifespan: {self.average_lifespan:.2f}, "
                f"Avg Strength: {self.average_strength:.2f}, "
                f"Avg Hardiness: {self.average_hardiness:.2f}, "
                f"Avg Metabolism: {self.average_metabolism:.2f} "
                f"Avg Reproduction Threshold: {self.average_reproduction_threshold:.2f} "
            )

    def calculate_genetic_similarity(self, agent_matrix, world_matrix):
            def genetic_similarity(agents):
                if len(agents) < 2:
                    return 1.0  # Maximum similarity if there's only one agent or none

                total_distance = 0.0
                num_comparisons = 0
                for agent1, agent2 in combinations(agents, 2):
                    total_distance += np.linalg.norm(agent1.genome - agent2.genome)
                    num_comparisons += 1

                average_distance = (
                    total_distance / num_comparisons if num_comparisons > 0 else 0
                )
                return 1 / (1 + average_distance)  # Inverse of the average distance

            # Filter out dead agents
            living_agents = [agent for row in agent_matrix for agent in row if agent.alive]
            global_similarity = genetic_similarity(living_agents)

            # Group agents by biome (difficulty level)
            biomes = {}
            for i, row in enumerate(agent_matrix):
                for j, agent in enumerate(row):
                    if agent.alive:
                        difficulty = world_matrix[i][j]
                        if difficulty not in biomes:
                            biomes[difficulty] = []
                        biomes[difficulty].append(agent)

            biome_similarities = {
                difficulty: genetic_similarity(agents)
                for difficulty, agents in biomes.items()
            }

            return global_similarity, biome_similarities

    def print_genetic_similarities(self, global_similarity, biome_similarities):
            print(f"Global Genetic Similarity: {global_similarity:.4f}")
            for difficulty, similarity in biome_similarities.items():
                print(f"Genetic Similarity in Biome {difficulty}: {similarity:.4f}")

# Simulation Logic
def isCoordinateInRange(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True


def initialize_agent_matrix(n):
    return [[Agent() for j in range(n)] for i in range(n)]


def simulateAgentTimeStep(current_step, i, j, environment, agent_matrix, metrics):
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
        return

    # Food Consumption Logic
    if config["enable_food"]:

        # First, check if agent survived reproduction last turn.
        if current_agent.energy_reserves <= 0:
            # Not enough food available to meet basic metabolic needs
            agent_matrix[i][j].kill_agent()
            metrics.deaths_from_starvation += 1
            metrics.cumulative_deaths += 1
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
            if isCoordinateInRange(environment.map_size, new_i, new_j):
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
                            return
                        # If the new agent is weaker or equal in strength, it dies, and no change is made to the cell
                    else:
                        # The prospective cell is empty, the new agent just occupies it with no competition
                        agent_matrix[new_i][new_j] = new_individual
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
        if isCoordinateInRange(environment.map_size, new_i, new_j):
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
                    # If the new individual is weaker or equal in strength, it dies, and no change is made to the cell
                else:
                    # The prospective cell is empty, the new individual just occupies it with no competition
                    agent_matrix[new_i][new_j] = new_individual
            else:
                # New individual dies from exposure as it wasnt hardy enough
                metrics.deaths_from_exposure += 1
                metrics.cumulative_deaths += 1


# Matrix Creation, Visualization, Saving
def transform_matrix(agent_matrix, attribute):
    return [[getattr(agent, attribute, 0) for agent in row] for row in agent_matrix]


def save_matrix_image(matrix, file_name):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Game World")
    cbar = plt.colorbar(im, ax=ax)
    plt.savefig(str(file_name))


# Function to overlay data on a background
def overlay_on_background(background_img, data_matrix, cmap="viridis"):
    # Generate an image from the data matrix
    fig, ax = plt.subplots(
        figsize=(background_img.size[0] / 100, background_img.size[1] / 100), dpi=100
    )
    ax.imshow(background_img, extent=(0, data_matrix.shape[1], data_matrix.shape[0], 0))
    im = ax.imshow(
        data_matrix,
        cmap=cmap,
        alpha=0.5,
        extent=(0, data_matrix.shape[1], data_matrix.shape[0], 0),
    )  # semi-transparent
    ax.axis("off")  # Turn off axis labels and ticks
    fig.canvas.draw()

    # Convert the Matplotlib figure to a PIL Image and return it
    data_img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)  # Close the figure to free up memory
    return data_img


def save_agent_matrix_image(matrix, file_name, attribute):
    # Check if the attribute is valid for the objects in the matrix
    if not matrix or not hasattr(matrix[0][0], attribute):
        raise ValueError(f"Attribute '{attribute}' not found in matrix elements")

    # Transform the matrix into a 2D array of the specified attribute
    transformed_matrix = [[getattr(cell, attribute) for cell in row] for row in matrix]

    fig, ax = plt.subplots()
    im = ax.imshow(transformed_matrix, cmap="viridis")
    plt.colorbar(im, ax=ax)  # Optional: Adds a colorbar to the plot
    plt.savefig(str(file_name))
    plt.close(fig)


def get_image_from_fig(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def create_analysis_notebook(
    csv_file_path, notebook_path, global_similarity, biome_similarities
):
    # Create a new notebook object
    nb = nbf.v4.new_notebook()

    # Add cells with the necessary code
    cells = []

    # Cell to import libraries
    cells.append(
        nbf.v4.new_code_cell(
            """\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')"""
        )
    )

    # Cell to load data
    cells.append(
        nbf.v4.new_code_cell(
            f"""\
data = pd.read_csv('{csv_file_path}')"""
        )
    )

    # Cell to calculate and display summary statistics
    cells.append(
        nbf.v4.new_code_cell(
            """\
# Columns of interest
columns_of_interest = [
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
    "Average Reproduction Threshold"
]

# Calculate and display meaningful statistics
summary_stats = data[columns_of_interest].describe().loc[["mean", "min", "50%", "max"]]
summary_stats.rename(index={"50%": "median"}, inplace=True)
summary_stats"""
        )
    )

    # Cell to plot gene value averages over time
    cells.append(
        nbf.v4.new_code_cell(
            """\
plt.figure(figsize=(12, 6))
plt.plot(data['Timestep'], data['Average Lifespan'], label='Average Maximum Lifespan')
plt.plot(data['Timestep'], data['Average Strength'], label='Average Strength')
plt.plot(data['Timestep'], data['Average Hardiness'], label='Average Hardiness')
plt.plot(data['Timestep'], data['Average Metabolism'], label='Average Metabolism')
plt.plot(data['Timestep'], data['Average Reproduction Threshold'], label='Average Reproduction Threshold')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.title('Time Series of Average Gene Values')
plt.legend()
plt.show()"""
        )
    )

    # Cell to plot deaths over time
    cells.append(
        nbf.v4.new_code_cell(
            """\
plt.figure(figsize=(12, 6))
plt.plot(data['Timestep'], data['Deaths from Aging'], label='Deaths from Aging')
plt.plot(data['Timestep'], data['Deaths from Competition'], label='Deaths from Competition')
plt.plot(data['Timestep'], data['Deaths from Starvation'], label='Deaths from Starvation')
plt.plot(data['Timestep'], data['Deaths from Exposure'], label='Deaths from Exposure')

plt.xlabel('Timestep')
plt.ylabel('Value')
plt.title('Agent Deaths Over Time')
plt.legend()
plt.show()"""
        )
    )

    # Cell to display genetic similarities
    cells.append(
        nbf.v4.new_code_cell(
            f"""\
# Display genetic similarities
global_similarity = {global_similarity}
biome_similarities = {biome_similarities}

print(f"Global Genetic Similarity: {{global_similarity:.4f}}")
for biome, similarity in biome_similarities.items():
    print(f"Genetic Similarity in Biome {{biome}}: {{similarity:.4f}}")
"""
        )
    )

    # Add cells to the notebook
    nb["cells"] = cells

    # Write the notebook to a new file
    with open(notebook_path, "w") as f:
        nbf.write(nb, f)

    print(f"Notebook created at {notebook_path}")

    # Try to execute the notebook and handle exceptions
    executed_notebook_path = notebook_path.replace(".ipynb", "_executed.ipynb")
    try:
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": "./"}})
        with open(executed_notebook_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("Executed notebook saved at", executed_notebook_path)
        # Delete the original unexecuted notebook if execution is successful
        os.remove(notebook_path)

    except Exception as e:
        print("Error during notebook execution:", e)

    print("Notebook creation process complete.")


def run_game():
    # Generate a timestamped folder name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("Experimental_Results", timestamp)

    # Ensure the base directory exists
    if not os.path.exists("Experimental_Results"):
        os.makedirs("Experimental_Results")

    # Create the timestamped directory for this run
    os.makedirs(results_dir)

    # Create logger for metrics tracking
    metrics = SimulationMetrics()
    csv_file_path = os.path.join(results_dir, "simulation_metrics.csv")
    metrics.enable_csv_logging(csv_file_path)

    # Use the map configurations
    simulation_steps = config["simulation_steps"]
    frame_save_interval = config["frame_save_interval"]
    frame_rate = config["frame_rate"]

    # Save the config of the experimental run as a text file
    config_file_path = os.path.join(results_dir, "config.txt")
    with open(config_file_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")

    # Initialize the environment
    environment = Environment(config)

    # Save environment
    save_matrix_image(environment.world_matrix, os.path.join(results_dir, "Game_World"))

    # Initialize the agent matrix
    agent_matrix = initialize_agent_matrix(environment.map_size)

    # Find starting position for the initial agent
    agent_starting_pos = environment.find_easiest_starting_location()

    # Create initial agent
    agent_matrix[agent_starting_pos[0]][
        agent_starting_pos[1]
    ] = Agent.create_live_agent()

    # Create directories for GIFs and images
    gifs_dir = os.path.join(results_dir, "Gifs")
    images_dir = os.path.join(results_dir, "Images")
    os.makedirs(gifs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Calculate intervals for capturing images
    capture_intervals = [
        simulation_steps // 4,
        simulation_steps // 2,
        3 * simulation_steps // 4,
        simulation_steps,
    ]

    # Declare data frames for gif generation
    strength_frames = []
    hardiness_frames = []
    age_frames = []
    lifespan_frames = []
    metabolism_frames = []
    reproduction_threshold_frames = []
    genetic_distance_frames = []

    # Declare plots for visualization
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(
        transform_matrix(agent_matrix, "strength"), cmap="viridis", vmin=0, vmax=100
    )
    ax1.set_title("Agent Strength")
    plt.colorbar(im1, ax=ax1)

    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(
        transform_matrix(agent_matrix, "hardiness"), cmap="inferno", vmin=0, vmax=100
    )
    ax2.set_title("Agent Hardiness")
    plt.colorbar(im2, ax=ax2)

    fig3, ax3 = plt.subplots()
    im3 = ax3.imshow(
        transform_matrix(agent_matrix, "age"), cmap="plasma", vmin=0, vmax=100
    )
    ax3.set_title("Agent Age")
    plt.colorbar(im3, ax=ax3)

    fig4, ax4 = plt.subplots()
    im4 = ax4.imshow(
        transform_matrix(agent_matrix, "lifespan"), cmap="viridis", vmin=0, vmax=100
    )
    ax4.set_title("Agent Max Lifespan")
    plt.colorbar(im4, ax=ax4)

    fig5, ax5 = plt.subplots()
    im5 = ax5.imshow(
        transform_matrix(agent_matrix, "metabolism"), cmap="inferno", vmin=0, vmax=100
    )
    ax5.set_title("Agent Metabolism")
    plt.colorbar(im5, ax=ax5)

    fig6, ax6 = plt.subplots()
    im6 = ax6.imshow(
        transform_matrix(agent_matrix, "reproduction_threshold"),
        cmap="plasma",
        vmin=0,
        vmax=100,
    )
    ax6.set_title("Agent Reproduction Threshold")
    plt.colorbar(im6, ax=ax6)

    fig7, ax7 = plt.subplots()
    im7 = ax7.imshow(
        transform_matrix(agent_matrix, "genetic_distance"),
        cmap="viridis",
        vmin=0,
        vmax=200,
    )
    ax7.set_title("Genetic Distance From Ancestor")
    plt.colorbar(im7, ax=ax7)

    # Main game loop
    print("Running Simulation...\n")
    current_sim_step = 0

    while current_sim_step < simulation_steps:
        living_agents_count = 0

        for i in range(len(agent_matrix)):
            for j in range(len(agent_matrix[0])):
                if agent_matrix[i][j].alive:
                    living_agents_count += 1
                    simulateAgentTimeStep(
                        current_sim_step, i, j, environment, agent_matrix, metrics
                    )

        # Calculate and Collect Metrics
        metrics.update_agent_metrics(agent_matrix)
        metrics.calculate_averages()
        metrics.log_metrics(current_sim_step)
        metrics.print_current_state()
        print()
        metrics.reset_averages()

        current_sim_step += 1
        print(f"\rSimulation Step {current_sim_step}/{simulation_steps}", end="")
        print()

        if living_agents_count == 0:
            print()
            print("All agents have died. Ending simulation at step", current_sim_step)
            break

        if current_sim_step % frame_save_interval == 0:
            # Draw canvas and convert attributes to an image array
            im1.set_data(transform_matrix(agent_matrix, "strength"))
            im2.set_data(transform_matrix(agent_matrix, "hardiness"))
            im3.set_data(transform_matrix(agent_matrix, "age"))
            im4.set_data(transform_matrix(agent_matrix, "lifespan"))
            im5.set_data(transform_matrix(agent_matrix, "metabolism"))
            im6.set_data(transform_matrix(agent_matrix, "reproduction_threshold"))
            im7.set_data(transform_matrix(agent_matrix, "genetic_distance"))

            # Draw the updated data on the canvas
            fig1.canvas.draw()
            fig2.canvas.draw()
            fig3.canvas.draw()
            fig4.canvas.draw()
            fig5.canvas.draw()
            fig6.canvas.draw()
            fig7.canvas.draw()

            # Convert the updated canvas to an image and append to respective frame lists
            strength_frames.append(get_image_from_fig(fig1))
            hardiness_frames.append(get_image_from_fig(fig2))
            age_frames.append(get_image_from_fig(fig3))
            lifespan_frames.append(get_image_from_fig(fig4))
            metabolism_frames.append(get_image_from_fig(fig5))
            reproduction_threshold_frames.append(get_image_from_fig(fig6))
            genetic_distance_frames.append(get_image_from_fig(fig7))

        # Capture images at specified intervals
        if current_sim_step in capture_intervals:
            fig1.savefig(
                os.path.join(images_dir, f"strength_step_{current_sim_step}.png")
            )
            fig2.savefig(
                os.path.join(images_dir, f"hardiness_step_{current_sim_step}.png")
            )
            fig3.savefig(os.path.join(images_dir, f"age_step_{current_sim_step}.png"))
            fig4.savefig(
                os.path.join(images_dir, f"lifespan_step_{current_sim_step}.png")
            )
            fig5.savefig(
                os.path.join(images_dir, f"metabolism_step_{current_sim_step}.png")
            )
            fig6.savefig(
                os.path.join(
                    images_dir, f"reproduction_threshold_step_{current_sim_step}.png"
                )
            )
            fig7.savefig(
                os.path.join(
                    images_dir, f"genetic_distance_step_{current_sim_step}.png"
                )
            )

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
    plt.close(fig7)

    # Save GIFs
    print()
    print("Generating GIFs...")
    imageio.mimsave(
        os.path.join(gifs_dir, "strength_map.gif"), strength_frames, fps=frame_rate
    )
    imageio.mimsave(
        os.path.join(gifs_dir, "hardiness_map.gif"), hardiness_frames, fps=frame_rate
    )
    imageio.mimsave(os.path.join(gifs_dir, "age_map.gif"), age_frames, fps=frame_rate)
    imageio.mimsave(
        os.path.join(gifs_dir, "lifespan_map.gif"), lifespan_frames, fps=frame_rate
    )
    imageio.mimsave(
        os.path.join(gifs_dir, "metabolism_map.gif"), metabolism_frames, fps=frame_rate
    )
    imageio.mimsave(
        os.path.join(gifs_dir, "reproduction_threshold_map.gif"),
        reproduction_threshold_frames,
        fps=frame_rate,
    )
    imageio.mimsave(
        os.path.join(gifs_dir, "genetic_drift_map.gif"),
        genetic_distance_frames,
        fps=frame_rate,
    )

    # Calculate genetic similarities at the final step
    print("Calculating genetic similarity....\n")
    global_similarity, biome_similarities = metrics.calculate_genetic_similarity(
        agent_matrix, environment.world_matrix
    )
    metrics.print_genetic_similarities(global_similarity, biome_similarities)

    # Close metrics file
    metrics.close_csv_logging()

    # Create metrics notebook
    csv_file_path = os.path.join(results_dir, "simulation_metrics.csv")
    notebook_path = os.path.join(results_dir, "analysis_notebook.ipynb")
    create_analysis_notebook(
        csv_file_path, notebook_path, global_similarity, biome_similarities
    )

    print("Simulation complete.\n")


def main():
    run_game()


if __name__ == "__main__":
    main()
