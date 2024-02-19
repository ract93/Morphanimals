import noise
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import json
import os
from noise import pnoise2

# Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("config.json", "r") as config_file:
    config = json.load(config_file)

enable_food = config["enable_food"]
enable_aging = config["enable_aging"]

level_difficulty = {1: 1, 2: 10, 3: 30, 4: 50, 5: 80}


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


# Agent Class
class Agent:
    def __init__(self, hardiness=0, strength=0, metabolism=5, alive=False):
        self.hardiness = hardiness  # Defines agent's environmental hardiness
        self.strength = strength  # Defines agent's fighting ability with other agents
        self.alive = alive
        self.age = 0
        self.maxage = 100
        self.metabolism = metabolism

    def isStronger(self, other):
        return self.strength > other.strength

    def reproduce_asexually(parent_agent):
        mutation_probability = 0.00084  # True mutation rate of E. Coli.
        # Mutate hardiness
        mutation_effect = (
            np.random.normal(loc=0, scale=8)
            if random.random() < mutation_probability
            else 0
        )
        hardiness = parent_agent.hardiness + mutation_effect
        hardiness = max(0, min(100, hardiness))

        # Mutate strength
        mutation_effect = (
            np.random.normal(loc=0, scale=2)
            if random.random() < mutation_probability
            else 0
        )
        strength = parent_agent.strength + mutation_effect
        strength = max(0, min(100, strength))

        return Agent(int(hardiness), int(strength), alive=True)

    def age_agent(self):
        # Increases the agent's age and kills it if it reaches max age.
        self.age += 1

    def kill_agent(self):
        # Kills the agent, setting its attributes to their default 'dead' state.
        self.alive = False
        self.hardiness = 0
        self.strength = 0
        self.age = 0
        return self  # Return the 'dead' agent


# Agent Logic


def initialize_agent_matrix(n):
    return [[Agent() for j in range(n)] for i in range(n)]


def calc_move(current_step, i, j, n, world_matrix, agent_matrix, food_matrix):

    current_agent = agent_matrix[i][j]  # Get agent at current location.
    if current_agent.alive == False:
        return  # Check if current cell is populated. if is not, skip.

    current_agent.age_agent()  # Increment current agent age.
    if enable_aging:
        if (
            current_agent.age >= current_agent.maxage
        ):  # Check if agent has reached its maximum lifespan
            agent_matrix[i][j] = current_agent.kill_agent()
            return

        # Attempt to consume food and survive
        if enable_food:
            survived = calculate_food(
                i, j, food_matrix, current_step, current_agent.metabolism
            )
            if not survived:
                agent_matrix[i][j] = current_agent.kill_agent()  # Handle starvation
                return

            else:
                pass

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
    if isCoordinateInRange(n, new_i, new_j):
        if new_individual.hardiness > level_difficulty[world_matrix[new_i][new_j]]:
            if new_individual.isStronger(agent_matrix[new_i][new_j]):
                agent_matrix[new_i][new_j] = new_individual


# Food Logic
def initialize_food_matrix(n):
    # Initialize all cells with (initial food amount, -1 as last accessed step)
    initial_food_amount = config["initial_food"]
    return [[(initial_food_amount, -1) for _ in range(n)] for _ in range(n)]


def calculate_food(i, j, food_matrix, current_step, agent_metabolism):

    food_generation_rate = config["food_generation_rate"]
    max_food_capacity = config["max_food_capacity"]

    food_amount, last_accessed = food_matrix[i][j]

    # Calculate food production since last accessed
    if last_accessed == -1:  # Never been accessed
        food_produced = current_step * food_generation_rate
    else:
        food_produced = (current_step - last_accessed) * food_generation_rate

    # Update food amount with new production, capped at max capacity
    new_food_amount = min(food_amount + food_produced, max_food_capacity)

    # Consume food based on agent's metabolism, update last accessed step
    if new_food_amount >= agent_metabolism:
        food_matrix[i][j] = (
            new_food_amount - agent_metabolism,
            current_step,
        )  # Update food amount and access step
        return True  # Agent survives
    else:
        food_matrix[i][j] = (0, current_step)  # Reset food to 0 and update access step
        return False  # Agent starves


# Terrain Generation


def generate_petri_dish(n):
    if n <= 0:
        return []

    array = [[0] * n for _ in range(n)]
    increment = 4 / (n - 1)

    for i in range(n):
        for j in range(n):
            array[i][j] = round((j * increment) + 1)

    return array


def generate_terrain(n, use_perlin_noise=True, randomize_params=True):
    if use_perlin_noise:
        if randomize_params:
            scale = random.uniform(50, 200)
            octaves = random.randint(4, 8)
            persistence = random.uniform(0.4, 0.7)
            lacunarity = random.uniform(1.8, 2.2)
        else:
            scale = (
                100.0  # adjust this parameter to change the "roughness" of the terrain
            )
            octaves = (
                6  # adjust this parameter to change the level of detail in the terrain
            )
            persistence = 0.5  # adjust this parameter to change the balance of high and low terrain
            lacunarity = 2.0  # adjust this parameter to change the level of detail in the terrain

        # Generate Perlin noise values for each cell in the array
        world = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                world[i][j] = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=n,
                    repeaty=n,
                    base=0,
                )

        # Scale and shift the values to the range [1, 5]
        world = (world - np.min(world)) * (5 - 1) / (np.max(world) - np.min(world)) + 1

        # Round the values to integers
        world = np.round(world)

        # Set the top left cell to biome 1, starting cell is always a "safe" zone.
        # world[0][0] = 1

        return world.astype(int).tolist()

    else:
        # Map is randomly generated with no structure.
        return [[random.randint(1, 3) for i in range(n)] for j in range(n)]



def generate_river(n, world):
    # Generate 2D Perlin noise
    scale = 200.0
    octaves = 6
    persistence = 0.4
    lacunarity = 2.0
    grid = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            grid[i][j] = pnoise2(
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
    max_val = max(map(max, grid))
    min_val = min(map(min, grid))

    # Rescale the values in the grid to be between 0 and 1
    scaled_grid = [
        [(val - min_val) / (max_val - min_val) for val in row] for row in grid
    ]

    # Define the river path as a series of points with y-values along the center of the grid
    river_width = n // 50
    river_path = [[i, scaled_grid[i][n // 2]] for i in range(n)]

    # Write the river pattern directly to the world array
    for i in range(n):
        for j in range(n):
            if abs(int(river_path[i][1] * n) - j) <= river_width // 2:
                world[i][j] = 1

    # Return the world array
    return world


def generate_island(
    n, use_perlin_noise=True, randomize_params=True, use_migration_route=True
):
    world = generate_terrain(n, use_perlin_noise, randomize_params)
    if use_migration_route:
        world = generate_river(n, world)

    return world


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

    # Use the configurations
    map_size = config["map_size"]
    simulation_steps = config["simulation_steps"]
    use_perlin_noise = config["use_perlin_noise"]
    use_random_params = config["use_random_params"]
    use_rivers = config["use_rivers"]
    frame_save_interval = config["frame_save_interval"]
    frame_rate = config["frame_rate"]

    # game world initialization
    game_world = generate_island(
        map_size, use_perlin_noise, use_random_params, use_rivers
    )
    # game_world = generate_petri_dish(map_size)

    # food initialization
    food_matrix = initialize_food_matrix(map_size)

    # agent initialization
    agent_matrix = initialize_agent_matrix(map_size)
    # agent_starting_pos = 0,0
    agent_starting_pos = find_easiest_starting_location(game_world)
    agent_matrix[agent_starting_pos[0]][agent_starting_pos[1]] = Agent(
        hardiness=10, strength=5, alive=True
    )  # Create initial agent

    # print initial state
    print("Game Parameters: ")
    print("Map size = " + str(map_size))
    print("Perlin Noise:" + str(use_perlin_noise))
    print("Random_Params:" + str(use_random_params))
    print("Rivers:" + str(use_rivers))
    print( "Food Simulation: " + str(enable_food))
    print( "Aging Simulation: " + str(enable_aging))
    print("Simulation Steps:" + str(simulation_steps))
    print("\n")
    # print ("Displaying Game_World in console.")
    visualize_world(game_world)
    save_matrix_image(game_world, "Game_World")

    # main game loop here
    print("Running Simulation...")
    print("\n")

    strength_frames = []  # array to save images of agent strength for heatmap
    hardiness_frames = []  # array to save images of agent hardiness for heatmap
    age_frames = []  # array to save images of agent age for heatmap

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

    # For additional attributes, follow pattern

    current_sim_step = 0

    while current_sim_step < simulation_steps:
        living_agents_count = 0  # Reset count for each simulation step
        for i in range(len(agent_matrix)):
            for j in range(len(agent_matrix[0])):
                if agent_matrix[i][j].alive:
                    living_agents_count += 1
                    calc_move(
                        current_sim_step,
                        i,
                        j,
                        map_size,
                        game_world,
                        agent_matrix,
                        food_matrix,
                    )

        if living_agents_count == 0:
            print("All agents have died. Ending simulation at step", current_sim_step)
            break  # Exit the loop if no living agents are left

        current_sim_step += 1

        if current_sim_step % frame_save_interval == 0:

            # Draw canvas and convert attributes to an image array
            im1.set_data(transform_matrix(agent_matrix, "strength"))
            im2.set_data(transform_matrix(agent_matrix, "hardiness"))
            im3.set_data(transform_matrix(agent_matrix, "age"))

            # Draw the updated data on the canvas
            fig1.canvas.draw()
            fig2.canvas.draw()
            fig3.canvas.draw()

            # Convert the updated canvas to an image and append to respective frame lists
            strength_frames.append(get_image_from_fig(fig1))
            hardiness_frames.append(get_image_from_fig(fig2))
            age_frames.append(get_image_from_fig(fig3))

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    # save final state
    # save_agent_matrix_image(agent_matrix, 'Final_Game_State: Hardiness', 'hardiness')
    # save_agent_matrix_image(agent_matrix, 'Final_Game_State: Strength', 'strength')
    # save_agent_matrix_image(agent_matrix, 'Final_Game_State: Age', 'age')

    # Create gif of execution
    print("Generating gifs...")
    imageio.mimsave("strength_map.gif", strength_frames, fps=frame_rate)
    imageio.mimsave("hardiness_map.gif", hardiness_frames, fps=frame_rate)
    imageio.mimsave("age_map.gif", age_frames, fps=frame_rate)

    print("Simulation complete.")
    print("\n")


# main
def main():
    run_game()


if __name__ == "__main__":
    main()
