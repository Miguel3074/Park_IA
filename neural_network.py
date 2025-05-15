# neural_network.py
import numpy as np
import random
from constants import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, MUTATION_RATE

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clipping to avoid overflow

def tanh(x):
    return np.tanh(x)

class NeuralNetwork:
    def __init__(self):
        # Weights and biases for the network
        # Input layer (INPUT_SIZE) -> Hidden layer (HIDDEN_SIZE)
        self.weights_input_hidden = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0 / INPUT_SIZE) # He initialization
        self.bias_hidden = np.zeros((1, HIDDEN_SIZE))

        # Hidden layer (HIDDEN_SIZE) -> Output layer (OUTPUT_SIZE)
        self.weights_hidden_output = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / HIDDEN_SIZE) # He initialization
        self.bias_output = np.zeros((1, OUTPUT_SIZE))

    def feedforward(self, inputs):
        # Ensure inputs are a numpy array with shape (1, INPUT_SIZE)
        inputs = np.array(inputs).reshape(1, INPUT_SIZE)

        # Input to Hidden layer
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = tanh(hidden_layer_input) # Using tanh activation

        # Hidden to Output layer
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        # For actions, we want distinct outputs, no activation or perhaps softmax if actions were probabilities
        # Using argmax later, so raw outputs are fine.

        return output_layer_input[0] # Return a 1D array of outputs

    def crossover(self, other_brain):
        # Creates a new brain by combining parts of this brain and another
        new_brain = NeuralNetwork()

        # Crossover weights and biases using uniform crossover
        mask_wih = np.random.rand(*self.weights_input_hidden.shape) > 0.5
        mask_bh = np.random.rand(*self.bias_hidden.shape) > 0.5
        mask_who = np.random.rand(*self.weights_hidden_output.shape) > 0.5
        mask_bo = np.random.rand(*self.bias_output.shape) > 0.5

        new_brain.weights_input_hidden = np.where(mask_wih, self.weights_input_hidden, other_brain.weights_input_hidden)
        new_brain.bias_hidden = np.where(mask_bh, self.bias_hidden, other_brain.bias_hidden)
        new_brain.weights_hidden_output = np.where(mask_who, self.weights_hidden_output, other_brain.weights_hidden_output)
        new_brain.bias_output = np.where(mask_bo, self.bias_output, other_brain.bias_output)

        return new_brain

    def mutate(self):
        # Mutates the brain by adding random noise to weights and biases
        def mutate_matrix(matrix):
            mutation_mask = np.random.rand(*matrix.shape) < MUTATION_RATE
            # Add random noise scaled by a small factor
            noise = np.random.randn(*matrix.shape) * 0.1
            return matrix + mutation_mask * noise

        self.weights_input_hidden = mutate_matrix(self.weights_input_hidden)
        self.bias_hidden = mutate_matrix(self.bias_hidden)
        self.weights_hidden_output = mutate_matrix(self.weights_hidden_output)
        self.bias_output = mutate_matrix(self.bias_output)

    def clone(self):
        # Creates a copy of the brain
        clone = NeuralNetwork()
        clone.weights_input_hidden = self.weights_input_hidden.copy()
        clone.bias_hidden = self.bias_hidden.copy()
        clone.weights_hidden_output = self.weights_hidden_output.copy()
        clone.bias_output = self.bias_output.copy()
        return clone