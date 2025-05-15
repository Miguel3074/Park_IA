# genetic_algorithm.py
import random
import numpy as np
from neural_network import NeuralNetwork
from constants import POPULATION_SIZE, MUTATION_RATE, SELECTION_PERCENTAGE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

class GeneticAlgorithm:
    def __init__(self):
        self.population_size = POPULATION_SIZE
        self.mutation_rate = MUTATION_RATE
        self.selection_size = max(1, int(POPULATION_SIZE * SELECTION_PERCENTAGE)) # At least 1 parent
        self.population = self._create_initial_population() # List of NeuralNetwork brains

    def _create_initial_population(self):
        """Creates a list of new, random neural networks."""
        return [NeuralNetwork() for _ in range(self.population_size)]

    def create_next_generation(self, sorted_population_with_rewards):
        """
        Creates the next generation of brains based on the rewards of the current generation.

        Args:
            sorted_population_with_rewards: A list of (brain, reward) tuples,
                                            sorted from highest reward to lowest.
        """
        next_generation_brains = []

        # 1. Selection: Select the top brains (elitism + parents for crossover)
        # Elitism: Keep the best brain(s) without modification
        best_brain = sorted_population_with_rewards[0][0].clone()
        next_generation_brains.append(best_brain)

        # Select parents from the top percentage
        parents = [brain for brain, reward in sorted_population_with_rewards[:self.selection_size]]

        # Ensure we have at least two parents for crossover if population_size > 1
        while len(parents) < 2 and self.population_size > 1:
             parents.append(random.choice(parents).clone()) # Duplicate the best if needed

        # 2. Reproduction: Create new brains through crossover and mutation
        # We need to create population_size - num_elites children
        num_elites = 1 # We kept the single best brain
        num_children_needed = self.population_size - num_elites

        for _ in range(num_children_needed):
            # Choose two parents randomly from the selected parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Perform crossover
            child_brain = parent1.crossover(parent2)

            # Perform mutation
            child_brain.mutate()

            next_generation_brains.append(child_brain)

        # Update the population with the new generation
        self.population = next_generation_brains

        return self.population