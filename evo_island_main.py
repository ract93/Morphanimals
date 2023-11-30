import noise
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio

level_difficulty = {1: 10, 2: 40, 3: 50, 4: 70, 5: 90}

class Agent:
    def __init__(self, fitness=0, alive=False):
        self.fitness = fitness
        self.alive = alive if fitness > 0 else False

    def is_more_fit_than(self, other):
        return self.fitness > other.fitness


#Matrix Creation, Visualization, Saving
def visualize_world(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='viridis')
    plt.show()

def visualize_agent_matrix(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow([[cell.fitness for cell in row] for row in matrix], cmap='inferno')
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
   

def print_matrix(array_2d):
    for row in array_2d:
        print(' '.join(str(cell.fitness) for cell in row))


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
        world[0][0] = 1

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

def isCoordinateInRange(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True

#Agent Logic
def isMoreFit(challenger, occupant):
    return challenger.is_more_fit_than(occupant)

def generate_agent_matrix(n):
    return [[Agent() for j in range(n)] for i in range(n)]

def train_new_individual():
    # generate a random number on a normal distribution
    fitness = np.random.normal(loc=50, scale=10)
    # scale the number so that it falls within the range of 0 to 100
    fitness = max(0, min(100, fitness))
    # invert the number so that larger numbers are more rare
    fitness = 100 - fitness
    return Agent(int(fitness))

def calc_move(i,j,n, world_matrix, agent_matrix):
   
   current_agent = agent_matrix[i][j]
   if current_agent.fitness == 0: return # Check if current cell is populated. if is not, skip. Add living or dead bool and update. 
   
   newIndividual = train_new_individual()


   diceroll = random.randint(1, 9) #roll for which action the child will take
   #the offspring can move to an adacent valid cell, or challenge the parent for its niche

    #Stay at current cell and challenge parent
   if diceroll == 1:  
    if newIndividual.fitness > level_difficulty[world_matrix[i][j]]: #Does the new invidual meet the baseline fitness to occupy the cell?
        if newIndividual.is_more_fit_than(current_agent): #Is the new individual's fitness higher than its parent? 
            agent_matrix[i][j] = newIndividual #new individual takes the niche.
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

#main game loop
def run_game():

    
    # configurable parameters, change as needed
    map_size = 500
    simulation_steps = 1000
    use_perlin_noise = True  # set to False to use random integer generation instead
    use_random_params = False # set to False to use preset perlin parameters
    use_rivers = True # set to False to remove rivers

    #game initialization
    agent_starting_pos = 0,0
    game_world = generate_island (map_size, use_perlin_noise, use_random_params, use_rivers)
    #game_world = generate_petri_dish(map_size)
    agent_matrix = generate_agent_matrix(map_size)

    #place initial agent
    agent_matrix[agent_starting_pos[0]][agent_starting_pos[1]] = train_new_individual()
    #agent_matrix[random.randint(0, len(agent_matrix)-1)][random.randint(0, len(agent_matrix[0])-1)] = train_new_individual()


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
    
    frames = [] # array to save images of the heatmap

    print ("Running Simulation...")
    print ("\n")
    #main game loop here
    current_sim_step = 0
    while (current_sim_step < simulation_steps):
        for i in range(len(agent_matrix)):
            for j in range(len(agent_matrix[0])):
                calc_move(i,j,map_size,game_world,agent_matrix)   

        current_sim_step += 1
        # Transform the agent_matrix into a numerical matrix for the 'fitness' attribute
        fitness_matrix = transform_matrix(agent_matrix, 'fitness')

        # Create the frame
        fig, ax = plt.subplots()
        im = ax.imshow(fitness_matrix, cmap='inferno')
        ax.axis('off')  # Optional: Turn off the axis

        # Draw canvas and convert to an image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(image)
        plt.close(fig)
   

    #print final state
    #print ("Displaying Final_Map_State in console.")
    #visualize_matrix(agent_matrix)
    save_agent_matrix_image(agent_matrix, 'Final_Game_State', 'fitness')
    
    #Create gif of execution
    print ("Generating gif...")
    imageio.mimsave('heatmap.gif', frames, fps=30)

    print ("Simulation complete.") 
    print ("\n")
       
#main
def main():
    run_game()

if __name__ == "__main__":
    main()
