# genetic_algorithm.py
import random
import numpy as np
from neural_network import NeuralNetwork
from constants import POPULATION_SIZE, MUTATION_RATE, SELECTION_PERCENTAGE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

class GeneticAlgorithm:
    def __init__(self):
        self.population_size = POPULATION_SIZE
        self.mutation_rate = MUTATION_RATE
        self.selection_size = max(1, int(POPULATION_SIZE * SELECTION_PERCENTAGE))
        self.population = self._create_initial_population() 

    def _create_initial_population(self):
        return [NeuralNetwork() for _ in range(self.population_size)]

    def create_next_generation(self, sorted_population_with_rewards):
    
        next_generation_brains = []

        best_brain = sorted_population_with_rewards[0][0].clone()
        next_generation_brains.append(best_brain)

        parents = [brain for brain, reward in sorted_population_with_rewards[:self.selection_size]]

        while len(parents) < 2 and self.population_size > 1:
             parents.append(random.choice(parents).clone()) 
        num_elites = 1
        num_children_needed = self.population_size - num_elites

        for _ in range(num_children_needed):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child_brain = parent1.crossover(parent2)
            child_brain.mutate()

            next_generation_brains.append(child_brain)

        self.population = next_generation_brains

        return self.population