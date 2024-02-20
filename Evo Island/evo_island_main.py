import noise
from noise import pnoise2
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import json
import os

# Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("config.json", "r") as config_file:
    config = json.load(config_file)


# Helper Methods
def isCoordinateInRange(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True


def find_easiest_starting_location(world_matrix):
    for i, row in enumerate(world_matrix):
        for j, difficulty in enumerate(row):
            if difficulty == 1:
                return (i, j)  # Return the coordinates as a tuple (row, column)
    return None  # Return None if no cell with that difficulty is found


def get_image_from_fig(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

class Environment:
    level_difficulty = {1: 1, 2: 10, 3: 30, 4: 50, 5: 80}

    def __init__(self, config):
        self.config = config
        self.map_size = config["map_size"]
        self.world_matrix = self.generate_island(
            use_perlin_noise=config["use_perlin_noise"],
            use_random_params=config["use_random_params"],
            use_rivers=config["use_rivers"]
        )
        self.food_matrix = self.initialize_food_matrix()

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
    
    def generate_terrain(self, use_perlin_noise=True, randomize_params=True):
        if use_perlin_noise:
            if randomize_params:
                scale = random.uniform(50, 200)
                octaves = random.randint(4, 8)
                persistence = random.uniform(0.4, 0.7)
                lacunarity = random.uniform(1.8, 2.2)
            else:
                scale = 100.0
                octaves = 6
                persistence = 0.5
                lacunarity = 2.0

            world = np.zeros((self.map_size, self.map_size))
            for i in range(self.map_size):
                for j in range(self.map_size):
                    world[i][j] = pnoise2(
                        i / scale,
                        j / scale,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=self.map_size,
                        repeaty=self.map_size,
                        base=0,
                    )

            world = (world - np.min(world)) * (5 - 1) / (np.max(world) - np.min(world)) + 1
            world = np.round(world)
            return world.astype(int)

        else:
            return [[random.randint(1, 3) for _ in range(self.map_size)] for _ in range(self.map_size)]

    def generate_island(self, use_perlin_noise=True, use_random_params=True, use_rivers=True):
        world = self.generate_terrain(use_perlin_noise, use_random_params)
        if use_rivers:
            world = self.generate_river(world)  # Passing world matrix as argument
        return world

    def generate_river(self, world_matrix):
        n = self.map_size

        # Generate 2D Perlin noise
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

        # Find the maximum and minimum values in the grid
        max_val = np.max(grid)
        min_val = np.min(grid)

        # Rescale the values in the grid to be between 0 and 1
        scaled_grid = (grid - min_val) / (max_val - min_val)

        # Define the river path as a series of points with y-values along the center of the grid
        river_width = n // 50
        river_path = [(i, scaled_grid[i, n // 2]) for i in range(n)]

        # Write the river pattern directly to the world array
        for i in range(n):
            for j in range(n):
                if abs(int(river_path[i][1] * n) - j) <= river_width // 2:
                    world_matrix[i][j] = 1

        # Return the updated world array
        return world_matrix



    def initialize_food_matrix(self):
        initial_food_amount = self.config["initial_food"]
        food_matrix = [[(initial_food_amount, -1) for _ in range(self.map_size)] for _ in range(self.map_size)]
        return food_matrix

    def calculate_food(self, i, j, current_step, agent_metabolism):
        food_generation_rate = self.config["food_generation_rate"]
        max_food_capacity = self.config["max_food_capacity"]
        food_amount, last_accessed = self.food_matrix[i][j]
        if last_accessed == -1:
            food_produced = current_step * food_generation_rate
        else:
            food_produced = (current_step - last_accessed) * food_generation_rate
        new_food_amount = min(food_amount + food_produced, max_food_capacity)
        if new_food_amount >= agent_metabolism:
            self.food_matrix[i][j] = (new_food_amount - agent_metabolism, current_step)
            return True
        else:
            self.food_matrix[i][j] = (0, current_step)
            return False
        

# Agent Class
class Agent:
    GENOME_LENGTH = 32  # Adjust as needed for genome length

    def __init__(self):
        # Default constructor for dead agents
        self.alive = False
        self.genome = None
        self.reset_traits()

    @classmethod
    def create_live_agent(cls, genome=None):
        # Alternative constructor for live agents
        instance = cls()
        instance.alive = True
        instance.genome = genome if genome is not None else ''.join(random.choice('01') for _ in range(cls.GENOME_LENGTH))
        instance.decode_traits()
        return instance

    def decode_traits(self):
        # Implement the logic to decode traits from the genome
        # Example:
        self.age = 0
        self.maxage = self.decode_trait(0, 8, 0, 100)
        self.hardiness = self.decode_trait(8, 16, 0, 100)
        self.strength = self.decode_trait(16, 24, 0, 100)
        self.metabolism = self.decode_trait(24, 32, 0, 100)

    def reset_traits(self):
        # Reset traits for dead agents
        self.age = 0
        self.maxage = 0
        self.hardiness = 0
        self.strength = 0
        self.metabolism = 0

    @classmethod
    def generate_default_genome(cls):
        # Assuming max_age=100, hardiness=10, strength=5, metabolism=5
        return cls.generate_genome(max_age=100, hardiness=10, strength=5, metabolism=5)

    @staticmethod
    def encode_trait(value, min_value, max_value, start, end):
        # Scale the value from its range to a binary number
        binary_segment = format(int((value - min_value) / (max_value - min_value) * (2**(end-start) - 1)), f'0{end-start}b')
        return binary_segment

    def decode_trait(self, start, end, min_value, max_value):
        binary_segment = self.genome[start:end]
        trait_value = int(binary_segment, 2)
        scaled_value = min_value + (trait_value / (2**(end-start) - 1)) * (max_value - min_value)
        return int(scaled_value)

    @staticmethod
    def mutate_genome(genome):
        mutation_rate = config['mutation_rate']  # Assuming config is accessible
        mutated_genome = genome
        for i in range(len(genome)):
            if random.random() < mutation_rate:
                mutated_bit = '1' if genome[i] == '0' else '0'
                mutated_genome = mutated_genome[:i] + mutated_bit + mutated_genome[i+1:]
        return mutated_genome

    @classmethod
    def generate_genome(cls, max_age, hardiness, strength, metabolism):
        max_age_genome = cls.encode_trait(max_age, 0, 100, 0, 8)
        hardiness_genome = cls.encode_trait(hardiness, 0, 100, 8, 16)
        strength_genome = cls.encode_trait(strength, 0, 100, 16, 24)
        metabolism_genome = cls.encode_trait(metabolism, 0, 100, 24, 32)
        # Concatenate all the genome segments
        full_genome = max_age_genome + hardiness_genome + strength_genome + metabolism_genome
        return full_genome

    @classmethod
    def reproduce_asexually(cls, parent_agent):
        child_genome = cls.mutate_genome(parent_agent.genome)
        return cls.create_live_agent(genome=child_genome)

    def age_agent(self):
        self.age += 1
        if self.age >= self.maxage:
            self.kill_agent()

    def kill_agent(self):
        self.alive = False
        self.reset_traits()  # Reset traits for consistency


# Agent Logic
def initialize_agent_matrix(n):
    return [[Agent() for j in range(n)] for i in range(n)]


def calc_move(current_step, i, j, environment, agent_matrix):
    current_agent = agent_matrix[i][j]
    if not current_agent.alive:
        return

    if config["enable_aging"] == True:
        current_agent.age_agent()
        if not current_agent.alive:
            return
    else:
        current_agent.age += 1

    # Attempt to consume food and survive
    if config["enable_food"] == True:
        survived = environment.calculate_food(i, j, current_step, current_agent.metabolism)
        if not survived:
            agent_matrix[i][j].kill_agent()
            return

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

    if isCoordinateInRange(environment.map_size, new_i, new_j) and \
       new_individual.hardiness > environment.level_difficulty[environment.world_matrix[new_i][new_j]]:
        if new_individual.strength > agent_matrix[new_i][new_j].strength:
            agent_matrix[new_i][new_j] = new_individual



# Matrix Creation, Visualization, Saving
def visualize_world(matrix):
    fig, ax = plt.subplots()
    ax.set_title("Game World")

    # Generate the image data based on the specified attribute
    im = ax.imshow(matrix, cmap="viridis")

    # Create and configure the colorbar
    cbar = plt.colorbar(im, ax=ax)

    plt.show()


def transform_matrix(agent_matrix, attribute):
    return [[getattr(agent, attribute, 0) for agent in row] for row in agent_matrix]


def save_matrix_image(matrix, file_name):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Game World")
    cbar = plt.colorbar(im, ax=ax)
    plt.savefig(str(file_name))


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


# main game loop
def run_game():
    # Use the map configurations
    simulation_steps = config["simulation_steps"]
    frame_save_interval = config["frame_save_interval"]
    frame_rate = config["frame_rate"]

     # Print initial state
    # Print initial state and game parameters
    print("Game Parameters:")
    print("Map size =", config["map_size"])
    print("Perlin Noise:", config["use_perlin_noise"])
    print("Random Params:", config["use_random_params"])
    print("Rivers:", config["use_rivers"])
    print("Food Simulation:", config["enable_food"])
    print("Aging Simulation:", config["enable_aging"])
    print("Simulation Steps:", config["simulation_steps"])
    print()

    # Initialize the environment
    environment = Environment(config)

    # Initialize the agent matrix
    agent_matrix = initialize_agent_matrix(environment.map_size)

    # Find starting position for the initial agent
    agent_starting_pos = find_easiest_starting_location(environment.world_matrix)

    # Create initial agent
    default_genome = Agent.generate_default_genome()
    agent_matrix[agent_starting_pos[0]][agent_starting_pos[1]] = Agent.create_live_agent(genome=default_genome)

    visualize_world(environment.world_matrix)
    save_matrix_image(environment.world_matrix, "Game_World")

    # Main game loop
    print("Running Simulation...\n")
    strength_frames = []  
    hardiness_frames = []  
    age_frames = [] 
    maxage_frames = [] 
    metabolism_frames = [] 

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
        transform_matrix(agent_matrix, "maxage"), cmap="plasma", vmin=0, vmax=100
    )
    ax4.set_title("Agent Max Lifespan")
    plt.colorbar(im3, ax=ax4)

    fig5, ax5 = plt.subplots()
    im5 = ax5.imshow(
        transform_matrix(agent_matrix, "metabolism"), cmap="plasma", vmin=0, vmax=100
    )
    ax5.set_title("Agent Metabolism")
    plt.colorbar(im5, ax=ax5)


    current_sim_step = 0

    while current_sim_step < simulation_steps:
        living_agents_count = 0  

        for i in range(len(agent_matrix)):
            for j in range(len(agent_matrix[0])):
                if agent_matrix[i][j].alive:
                    living_agents_count += 1
                    calc_move(current_sim_step, i, j, environment, agent_matrix)

        if living_agents_count == 0:
            print("All agents have died. Ending simulation at step", current_sim_step)
            break  

        current_sim_step += 1

        if current_sim_step % frame_save_interval == 0:
            # Draw canvas and convert attributes to an image array
            im1.set_data(transform_matrix(agent_matrix, "strength"))
            im2.set_data(transform_matrix(agent_matrix, "hardiness"))
            im3.set_data(transform_matrix(agent_matrix, "age"))
            im4.set_data(transform_matrix(agent_matrix, "maxage"))
            im5.set_data(transform_matrix(agent_matrix, "metabolism"))

            # Draw the updated data on the canvas
            fig1.canvas.draw()
            fig2.canvas.draw()
            fig3.canvas.draw()
            fig4.canvas.draw()
            fig5.canvas.draw()

            # Convert the updated canvas to an image and append to respective frame lists
            strength_frames.append(get_image_from_fig(fig1))
            hardiness_frames.append(get_image_from_fig(fig2))
            age_frames.append(get_image_from_fig(fig3))
            maxage_frames.append(get_image_from_fig(fig4))
            metabolism_frames.append(get_image_from_fig(fig5))

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    # Save final state
    print("Generating gifs...")
    imageio.mimsave("strength_map.gif", strength_frames, fps=frame_rate)
    imageio.mimsave("hardiness_map.gif", hardiness_frames, fps=frame_rate)
    imageio.mimsave("age_map.gif", age_frames, fps=frame_rate)
    imageio.mimsave("max_age_map.gif", maxage_frames, fps=frame_rate)
    imageio.mimsave("metabolism.gif", metabolism_frames, fps=frame_rate)

    print("Simulation complete.\n")

# main
def main():
    run_game()


if __name__ == "__main__":
    main()
