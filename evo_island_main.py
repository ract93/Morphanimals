import noise
import numpy as np
import random


def generate_robot_matrix(n):
 
    return [[0 for i in range(n)] for j in range(n)]


import random
import numpy as np
import noise

def generate_random_island(n, use_perlin_noise=True):
    if use_perlin_noise:
        print ("Perlin Noise On")
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

        # Set the top left cell to 1
        world[0][0] = 1

        # Generate a random but connected path of 1s starting from the top left cell
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
        print ("Perlin Noise Off")
        return [[random.randint(1, 3) for i in range(n)] for j in range(n)]


def print_matrix(array_2d):
    for row in array_2d:
        print(' '.join(str(cell) for cell in row))

def is_coordinate_in_range(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True

def generate_fake_fitness():
    return random.randint(0, 100)

def run_game(game_steps):

    #game initialization
    print ("Game Parameters: ")
    map_size = 10  # configurable parameter, change as needed
    simulation_steps = game_steps
    use_perlin_noise = True  # set to False to use random integer generation instead
    game_world = generate_random_island(map_size, use_perlin_noise)
    fitness_matrix = generate_robot_matrix (map_size)
    print ("\n")

    #print initial state
    print ("Map size = " + str(map_size))
    print ("Simulation Steps:" + str(simulation_steps))
    print ("Game World:")
    print_matrix(game_world)
    print ("\n")
    print ("Initial Map State")
    print_matrix(fitness_matrix)
    print ("\n")

    #game loop here

    #print final state
    print ("Final Map State")
    print_matrix(fitness_matrix)
       
#main
def main():
    run_game(100)

if __name__ == "__main__":
    main()
