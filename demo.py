import noise
import numpy as np
import random

def generate_robot_matrix(n):
 
    return [[0 for i in range(n)] for j in range(n)]


def generate_random_island(n, use_perlin_noise=True):
    if use_perlin_noise:
        print ("Perlin Noise On")
        scale = 100.0  # adjust this parameter to change the "roughness" of the terrain
        octaves = 6    # adjust this parameter to change the level of detail in the terrain
        persistence = 0.5  # adjust this parameter to change the balance of high and low terrain
        lacunarity = 2.0    # adjust this parameter to change the level of detail in the terrain

        world = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                world[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=n, repeaty=n, base=0)

        # scale and shift the values to the range [1, 3]
        world = (world - np.min(world)) * (3-1)/(np.max(world)-np.min(world)) + 1

        # round the values to integers
        world = np.round(world)

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

def initialize_game():
    map_size = 3  # configurable parameter, change as needed
    simulation_steps = 100
    use_perlin_noise = True  # set to False to use random integer generation instead
    game_world = generate_random_island(map_size, use_perlin_noise)
    fitness_matrix = generate_robot_matrix (map_size)

    print ("Map size = " + str(map_size))
    print ("Simulation Steps:" + str(simulation_steps))
    print ("Game World:")
    print_matrix(game_world)
    print ("\n")
    print ("Final Map State")
    print_matrix(fitness_matrix)


def main():
    initialize_game()

if __name__ == "__main__":
    main()
