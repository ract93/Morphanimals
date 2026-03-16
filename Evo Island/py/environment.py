import random

import noise
import numpy as np
from noise import pnoise2


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
