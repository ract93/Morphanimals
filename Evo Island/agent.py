import math
import random

import matplotlib.colors as mcolors
import numpy as np


# Agent Class
class Agent:
    common_ancestor_genome = np.array([20, 10, 5, 5, 3])  # common ancestor's genome

    def __init__(self):
        self.alive = False
        self.genome = None
        self.color = None  # Agent color is based on its genome
        self.age = 0
        self.energy_reserves = 0
        self.genetic_distance = None
        self.species = 0
        self.reset_traits()

    @classmethod
    def create_live_agent(cls, genome=None):
        live_agent = cls()
        live_agent.alive = True
        live_agent.genome = (
            genome if genome is not None else cls.generate_default_genome()
        )
        live_agent.color = cls.genome_to_color(live_agent.genome)  # Calculate color
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
    def mutate_genome(genome, mutation_rate):
        mutation_effects = np.zeros_like(genome)
        for i in range(len(genome)):
            if np.random.rand() < mutation_rate:
                mutation_effects[i] = np.random.normal(loc=0, scale=2)
        mutated_genome = genome + mutation_effects
        mutated_genome = np.clip(mutated_genome, 0, 100)
        return mutated_genome

    def calculate_genetic_distance(self, other_genome=None):
        if other_genome is None:
            other_genome = Agent.common_ancestor_genome
        if self.genome is not None:
            self.genetic_distance = np.linalg.norm(self.genome - other_genome)
        return self.genetic_distance

    @classmethod
    def reproduce_asexually(cls, parent_agent, mutation_rate):
        child_genome = cls.mutate_genome(parent_agent.genome, mutation_rate)
        return cls.create_live_agent(child_genome)

    def age_agent(self, enable_aging):
        self.age += 1
        if enable_aging:
            midpoint = self.lifespan / 2
            if midpoint == 0:
                self.kill_agent()
                return
            steepness = 10
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
        self.species = 0

    def kill_agent(self):
        self.alive = False
        self.genome = None
        self.color = None  # Reset color attribute
        self.reset_traits()

    @staticmethod
    def genome_to_color(genome):
        """ Convert a genome to a color in HSL space """
        # Normalize genome values to [0, 1] range
        genome_range = np.max(genome) - np.min(genome)
        norm_genome = (genome - np.min(genome)) / (genome_range if genome_range != 0 else 1)

        # Map normalized values to HSL space
        hue = (norm_genome[0] * 360) % 360
        saturation = 50 + norm_genome[1] * 50  # Keep saturation between 50% and 100%
        lightness = 50 + norm_genome[2] * 50   # Keep lightness between 50% and 100%

        return mcolors.hsv_to_rgb((hue / 360, saturation / 100, lightness / 100))
