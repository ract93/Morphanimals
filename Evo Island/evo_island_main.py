import noise
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import json
import os

#Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('config.json', 'r') as config_file:
        config = json.load(config_file)

level_difficulty = {1: 1, 2: 20, 3: 50, 4: 70, 5: 90}

#Helper Methods
def isCoordinateInRange(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True

def find_easiest_difficulty_location(world_matrix):
    for i, row in enumerate(world_matrix):
        for j, difficulty in enumerate(row):
            if difficulty == 1:
                return (i, j)  # Return the coordinates as a tuple (row, column)
    return None  # Return None if no cell with that difficulty is found


#Agent Class
class Agent:
    def __init__(self, fitness=0, alive=False):
        self.fitness = fitness
        self.alive = alive
        self.age = 0
        self.maxage = 100

    def is_more_fit_than(self, other):
        return self.fitness > other.fitness
    
#Agent Logic
def isMoreFit(challenger, occupant):
    return challenger.is_more_fit_than(occupant)

def generate_agent_matrix(n):
    return [[Agent() for j in range(n)] for i in range(n)]

def generate_new_individual():
    # generate a random number on a normal distribution
    fitness = np.random.normal(loc=50, scale=10)
    # scale the number so that it falls within the range of 0 to 100
    fitness = max(0, min(100, fitness))
    # invert the number so that larger numbers are more rare
    fitness = 100 - fitness
    return Agent(int(fitness), alive=True)

def generate_new_individual_from_parent(parent_agent):
    mutation_probability = 0.00084 #True mutation rate of E. Coli.
    mutation_effect = np.random.normal(loc=0, scale=2) if random.random() < mutation_probability else 0
    fitness = parent_agent.fitness + mutation_effect
    fitness = max(0, min(100, fitness))

    # Implement fitness cost or reproduction threshold if needed
    # ...

    return Agent(int(fitness), alive=True)

def calc_move(i,j,n, world_matrix, agent_matrix):
   
   current_agent = agent_matrix[i][j]
   if current_agent.alive == False: return # Check if current cell is populated. if is not, skip.

   current_agent.age = current_agent.age + 1# Increment current agent age.
   if current_agent.age >= current_agent.maxage: #Check if agent has reached its maximum lifespan
      agent_matrix[i][j] = Agent()
      return 

   #newIndividual = generate_new_individual()
   newIndividual = generate_new_individual_from_parent(current_agent)

   diceroll = random.randint(1, 9) #roll for which action the child will take
   #the offspring can move to an adacent valid cell, or challenge the parent for the current cell

    #Stay at current cell and challenge parent
   if diceroll == 1:  
    if newIndividual.fitness > level_difficulty[world_matrix[i][j]]: #Does the new invidual meet the baseline fitness to occupy the cell?
        if newIndividual.is_more_fit_than(current_agent): #Is the new individual's fitness higher than its parent? 
            agent_matrix[i][j] = newIndividual #new individual takes the niche, old agent is killed.
            #or nothing happens and it dies. 

    #go north
   if diceroll == 2: 
       if isCoordinateInRange(n,i+1,j): #Is the move in range? 
          if newIndividual.fitness > level_difficulty[world_matrix[i+1][j]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i+1][j]): #Does the new individual's fitness beat the fitness of the current occupant?
            agent_matrix[i+1][j] = newIndividual #new individual takes the niche. 
            #or nothing happens and it dies. 
       
    #go south
   if diceroll == 3: 
       if isCoordinateInRange(n,i-1,j): #Is the move in range?
        if newIndividual.fitness > level_difficulty[world_matrix[i-1][j]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i-1][j]):
            agent_matrix[i-1][j] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 
       
    #go east
   if diceroll == 4:  
       if isCoordinateInRange(n,i,j+1): #Is the move in range?
        if newIndividual.fitness > level_difficulty[world_matrix[i][j+1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i][j+1]):
            agent_matrix[i][j+1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go west
   if diceroll == 5:  
       if isCoordinateInRange(n,i,j-1): #Is the move in range?
        if newIndividual.fitness > level_difficulty[world_matrix[i][j-1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i][j-1]):
            agent_matrix[i][j-1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go northeast
   if diceroll == 6:  
       if isCoordinateInRange(n,i+1,j+1): #Is the move in range?
        if newIndividual.fitness > level_difficulty[world_matrix[i+1][j+1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i+1][j+1]):
            agent_matrix[i+1][j+1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go northwest
   if diceroll == 7:  
       if isCoordinateInRange(n,i+1,j-1): #Is the move in range?
        if newIndividual.fitness > level_difficulty[world_matrix[i+1][j-1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i+1][j-1]):
            agent_matrix[i+1][j-1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go southeast
   if diceroll == 8:  
       if isCoordinateInRange(n,i-1,j+1): #Is the move in range?
        if newIndividual.fitness > level_difficulty[world_matrix[i-1][j+1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i-1][j+1]):
            agent_matrix[i-1][j+1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go southwest
   if diceroll == 9:  
       if isCoordinateInRange(n,i-1,j-1): #Is the move in range?
        if newIndividual.fitness > level_difficulty[world_matrix[i-1][j-1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i-1][j-1]):
            agent_matrix[i-1][j-1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies.       


  #Terrain Generation

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
            scale = 100.0  # adjust this parameter to change the "roughness" of the terrain
            octaves = 6    # adjust this parameter to change the level of detail in the terrain
            persistence = 0.5  # adjust this parameter to change the balance of high and low terrain
            lacunarity = 2.0    # adjust this parameter to change the level of detail in the terrain

        # Generate Perlin noise values for each cell in the array
        world = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                world[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, 
                                            lacunarity=lacunarity, repeatx=n, repeaty=n, base=0)

        # Scale and shift the values to the range [1, 5]
        world = (world - np.min(world)) * (5-1)/(np.max(world)-np.min(world)) + 1

        # Round the values to integers
        world = np.round(world)

        # Set the top left cell to biome 1, starting cell is always a "safe" zone. 
        #world[0][0] = 1

        return world.astype(int).tolist()
    
    else:
        #Map is randomly generated with no structure. 
        return [[random.randint(1, 3) for i in range(n)] for j in range(n)]

from noise import pnoise2

def generate_river(n, world):
    # Generate 2D Perlin noise
    scale = 200.0
    octaves = 6
    persistence = 0.4
    lacunarity = 2.0
    grid = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            grid[i][j] = pnoise2(i / scale, j / scale, octaves=octaves,
                                 persistence=persistence, lacunarity=lacunarity,
                                 repeatx=n, repeaty=n, base=0)

    # Find the maximum and minimum values in the grid
    max_val = max(map(max, grid))
    min_val = min(map(min, grid))

    # Rescale the values in the grid to be between 0 and 1
    scaled_grid = [[(val - min_val) / (max_val - min_val) for val in row] for row in grid]

    # Define the river path as a series of points with y-values along the center of the grid
    river_width = n // 50
    river_path = [[i, scaled_grid[i][n//2]] for i in range(n)]

    # Write the river pattern directly to the world array
    for i in range(n):
        for j in range(n):
            if abs(int(river_path[i][1]*n) - j) <= river_width // 2:
                world[i][j] = 1

    # Return the world array
    return world

def generate_island(n, use_perlin_noise=True, randomize_params=True, use_migration_route=True):
    world = generate_terrain(n, use_perlin_noise, randomize_params)
    if use_migration_route:
       world = generate_river(n, world)

    return world

#Matrix Creation, Visualization, Saving
def visualize_world(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='viridis')
    plt.show()

def visualize_agent_attribute(matrix, attribute):
   fig, ax = plt.subplots()
   im = ax.imshow([[cell.attribute for cell in row] for row in matrix], cmap='viridis')
   plt.show() 

def transform_matrix(agent_matrix, attribute):
    return [[getattr(agent, attribute, 0) for agent in row] for row in agent_matrix]

def save_matrix_image(matrix, file_name):
   fig, ax = plt.subplots()
   im = ax.imshow(matrix, cmap='viridis')
   plt.savefig(str(file_name))

def save_agent_matrix_image(matrix, file_name, attribute):
    # Check if the attribute is valid for the objects in the matrix
    if not matrix or not hasattr(matrix[0][0], attribute):
        raise ValueError(f"Attribute '{attribute}' not found in matrix elements")

    # Transform the matrix into a 2D array of the specified attribute
    transformed_matrix = [[getattr(cell, attribute) for cell in row] for row in matrix]

    fig, ax = plt.subplots()
    im = ax.imshow(transformed_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax)  # Optional: Adds a colorbar to the plot
    plt.savefig(str(file_name))
    plt.close(fig)

#main game loop
def run_game():

    # Use the configurations
    map_size = config['map_size']
    simulation_steps = config['simulation_steps']
    use_perlin_noise = config['use_perlin_noise']
    use_random_params = config['use_random_params']
    use_rivers = config['use_rivers']
    frame_save_interval = config['frame_save_interval']
    frame_rate = config['frame_rate']

    #game world initialization
    game_world = generate_island (map_size, use_perlin_noise, use_random_params, use_rivers)
    #game_world = generate_petri_dish(map_size)
    
    #agent initialization
    agent_matrix = generate_agent_matrix(map_size)
    #agent_starting_pos = 0,0
    agent_starting_pos = find_easiest_difficulty_location(game_world)
    agent_matrix[agent_starting_pos[0]][agent_starting_pos[1]] = Agent(fitness=10, alive = True) #Create initial
    #agent_matrix[random.randint(0, len(agent_matrix)-1)][random.randint(0, len(agent_matrix[0])-1)] = generate_new_individual()


    #print initial state
    print ("Game Parameters: ")
    print ("Map size = " + str(map_size))
    print ("Perlin Noise:" + str(use_perlin_noise))
    print ("Random_Params:" + str(use_random_params))
    print ("Rivers:" + str(use_rivers))
    print ("Simulation Steps:" + str(simulation_steps))
    print ("\n")
    #print ("Displaying Game_World in console.")
    visualize_world(game_world)
    save_matrix_image(game_world, 'Game_World')


    #main game loop here
    print ("Running Simulation...")
    print ("\n")
    
    fitness_frames = [] # array to save images of agent fitness for heatmap
    age_frames = [] # array to save images of agent age for heatmap  

    fitness_matrix = transform_matrix(agent_matrix, 'fitness')
    age_matrix = transform_matrix(agent_matrix, 'age')

    current_sim_step = 0
    
    while (current_sim_step < simulation_steps):
        for i in range(len(agent_matrix)):
            for j in range(len(agent_matrix[0])):
                calc_move(i,j,map_size,game_world,agent_matrix)   

        current_sim_step += 1

        if current_sim_step % frame_save_interval == 0:
            # Transform the agent_matrix into a numerical matrices for different attributes
            fitness_matrix = transform_matrix(agent_matrix, 'fitness')
            age_matrix = transform_matrix(agent_matrix, 'age')

             # Create frames for visualization
            fig1, ax1 = plt.subplots()
            im1 = ax1.imshow(fitness_matrix, cmap='viridis')
            ax1.axis('off')  # Optional: Turn off the axis
                    
            fig2, ax2 = plt.subplots()
            im2 = ax2.imshow(age_matrix, cmap='plasma')
            ax2.axis('off')  # Optional: Turn off the axis

            # Draw canvas and convert attributes to an image array
            fig1.canvas.draw()
            image1 = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8')
            image1 = image1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))

            fitness_frames.append(image1)
            

            fig2.canvas.draw()
            image2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8')
            image2 = image2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))

            age_frames.append(image2)

            plt.close(fig1)
            plt.close(fig2)
   

    #save final state
    save_agent_matrix_image(agent_matrix, 'Final_Game_State', 'fitness')
    
    #Create gif of execution
    print ("Generating gifs...")
    imageio.mimsave('fitnessmap.gif', fitness_frames, fps=frame_rate)
    imageio.mimsave('agemap.gif', age_frames, fps=frame_rate)

    print ("Simulation complete.") 
    print ("\n")
       
#main
def main():
    run_game()

if __name__ == "__main__":
    main()
