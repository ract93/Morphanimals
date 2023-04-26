import noise
import numpy as np
import random

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


def print_island(array_2d):
    for row in array_2d:
        print(' '.join(str(cell) for cell in row))

def is_coordinate_in_range(n, x, y):
    if x < 0 or x >= n:
        return False
    if y < 0 or y >= n:
        return False
    return True

def main():
    n = 10  # configurable parameter, change as needed
    use_perlin_noise = True  # set to False to use random integer generation instead
    array_2d = generate_random_island(n, use_perlin_noise)
    print_island(array_2d)

if __name__ == "__main__":
    main()
