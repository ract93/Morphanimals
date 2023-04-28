import noise
import numpy as np
import random
import matplotlib.pyplot as plt

level_difficulty = {1: 10, 2: 50, 3: 80}

def visualize_matrix(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='viridis')
    ax.set_xticks(range(len(matrix[0])))
    ax.set_yticks(range(len(matrix)))
    ax.set_xticklabels(range(1, len(matrix[0])+1))
    ax.set_yticklabels(range(1, len(matrix)+1))
    plt.show()

def generate_agent_matrix(n):#Initialize empty agent matrix
 
    return [[0 for j in range(n)] for i in range(n)]

def generate_random_island(n, use_perlin_noise=True): #Generate a random island map, with or without structure
    if use_perlin_noise:
        scale = 100.0  # adjust this parameter to change the "roughness" of the terrain
        octaves = 6    # adjust this parameter to change the level of detail in the terrain
        persistence = 0.5  # adjust this parameter to change the balance of high and low terrain
        lacunarity = 2.0    # adjust this parameter to change the level of detail in the terrain

        # Generate Perlin noise values for each cell in the array
        world = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                world[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=n, repeaty=n, base=0)

        # Scale and shift the values to the range [1, 3]
        world = (world - np.min(world)) * (3-1)/(np.max(world)-np.min(world)) + 1

        # Round the values to integers
        world = np.round(world)

        # Set the top left cell to biome 1, starting cell is always a "safe" zone. 
        world[0][0] = 1

        # Generate a random but connected path of the easiest biome starting from the top left cell
        # This is a migrational route
        curr_x, curr_y = 0, 0
        while True:
            # Move either right or down randomly
            if random.choice([True, False]):
                curr_x += 1
            else:
                curr_y += 1

            # Check if the new position is in range of the array
            if 0 <= curr_x < n and 0 <= curr_y < n:
                # Set the cell to 1 and continue the path
                world[curr_x][curr_y] = 1
            else:
                # End the path when the end of the array is reached
                break

        return world.astype(int).tolist()
    
    else:
        #Map is randomly generated with no structure. 
        return [[random.randint(1, 3) for i in range(n)] for j in range(n)]


def print_matrix(array_2d):
    for row in array_2d:
        print(' '.join(str(cell) for cell in row))

def isMoreFit(challenger, Occupant):
    return challenger > Occupant
    
def isCoordinateInRange(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True
    
def train_new_individual():
    # generate a random number on a normal distribution
    x = np.random.normal(loc=50, scale=10)
    # scale the number so that it falls within the range of 0 to 100
    x = max(0, min(100, x))
    # invert the number so that larger numbers are more rare
    x = 100 - x
    return int(x)

def calc_move(i,j,n, world_matrix, agent_matrix):
   
   if agent_matrix[i][j]== 0: # check if current cell is populated (max fitness is > 0), if not, do nothing. 
    return 

   #if occupied, the occupant spawns an offspring
   newIndividual = train_new_individual()

   directionroll = random.randint(1, 9) #roll for which action the child will take
   #the offspring can move to an adacent valid cell, or challenge the parent for its niche

    #Stay at current cell and challenge parent
   if directionroll == 1:  
    if newIndividual > level_difficulty[world_matrix[i][j]]: #Does the new invidual meet the baseline fitness to occupy the cell?
        if isMoreFit(newIndividual,agent_matrix[i][j]): #Is the new individual's fitness higher than its parent? 
            agent_matrix[i][j] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go north
   if directionroll == 2: 
       if isCoordinateInRange(n,i+1,j): #Is the move in range? 
          if newIndividual > level_difficulty[world_matrix[i+1][j]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i+1][j]): #Does the new individual's fitness beat the fitness of the current occupant?
            agent_matrix[i+1][j] = newIndividual #new individual takes the niche. 
            #or nothing happens and it dies. 
       
    #go south
   if directionroll == 3: 
       if isCoordinateInRange(n,i-1,j): #Is the move in range?
        if newIndividual > level_difficulty[world_matrix[i-1][j]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i-1][j]):
            agent_matrix[i-1][j] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 
       
    #go east
   if directionroll == 4:  
       if isCoordinateInRange(n,i,j+1): #Is the move in range?
        if newIndividual > level_difficulty[world_matrix[i][j+1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i][j+1]):
            agent_matrix[i][j+1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go west
   if directionroll == 5:  
       if isCoordinateInRange(n,i,j-1): #Is the move in range?
        if newIndividual > level_difficulty[world_matrix[i][j-1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i][j-1]):
            agent_matrix[i][j-1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go northeast
   if directionroll == 6:  
       if isCoordinateInRange(n,i+1,j+1): #Is the move in range?
        if newIndividual > level_difficulty[world_matrix[i+1][j+1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i+1][j+1]):
            agent_matrix[i+1][j+1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go northwest
   if directionroll == 7:  
       if isCoordinateInRange(n,i+1,j-1): #Is the move in range?
        if newIndividual > level_difficulty[world_matrix[i+1][j-1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i+1][j-1]):
            agent_matrix[i+1][j-1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go southeast
   if directionroll == 8:  
       if isCoordinateInRange(n,i-1,j+1): #Is the move in range?
        if newIndividual > level_difficulty[world_matrix[i-1][j+1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i-1][j+1]):
            agent_matrix[i-1][j+1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies. 

    #go southwest
   if directionroll == 9:  
       if isCoordinateInRange(n,i-1,j-1): #Is the move in range?
        if newIndividual > level_difficulty[world_matrix[i-1][j-1]]: #Does the new invidual meet the baseline fitness to occupy the empty cell?
           if isMoreFit(newIndividual,agent_matrix[i-1][j-1]):
            agent_matrix[i-1][j-1] = newIndividual #new individual takes the niche.
            #or nothing happens and it dies.       
def run_game():

    #game initialization
    # configurable parameters, change as needed
    map_size = 500 
    simulation_steps = 500
    use_perlin_noise = True  # set to False to use random integer generation instead
    agent_starting_pos = 0,0
    game_world = generate_random_island(map_size, use_perlin_noise)
    agent_matrix = generate_agent_matrix(map_size)
    print ("\n")

    #place initial agent
    agent_matrix[agent_starting_pos[0]][agent_starting_pos[1]] = train_new_individual()

     #print initial state
    print ("Game Parameters: ")
    print ("Map size = " + str(map_size))
    print ("Perlin Noise:" + str(use_perlin_noise))
    print ("Simulation Steps:" + str(simulation_steps))
    print ("\n")
    print ("Game World:")
    #print_matrix(game_world)
    visualize_matrix(game_world)
    print ("\n")
    print ("Initial Agent State")
    #print_matrix(agent_matrix)
    visualize_matrix(agent_matrix)
    print ("\n")

    #game loop here
    current_sim_step = 0
    while (current_sim_step < simulation_steps):
        for i in range(len(agent_matrix)):
            for j in range(len(agent_matrix[0])):
                calc_move(i,j,map_size,game_world,agent_matrix)
        #print ("Simulation Step: " + str(current_sim_step))   
        #print_matrix(agent_matrix)     
        #print ("\n")
        current_sim_step += 1


    #print final state
    print ("Final Map State")
    #print_matrix(agent_matrix)
    visualize_matrix(agent_matrix)
       
#main
def main():
    run_game()

if __name__ == "__main__":
    main()
