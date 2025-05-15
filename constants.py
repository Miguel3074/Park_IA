# constants.py
import pygame

# --- Screen Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1500, 1000

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
OBSTACLE_COLOR = (100, 100, 100)
CAR_COLOR = BLUE
OBSTACLE_CAR_COLOR = OBSTACLE_COLOR
SPOT_COLOR = GREEN + (100,) # Green with transparency

# --- Game Settings ---
FPS = 60

# --- Car Dimensions ---
CAR_WIDTH = 30
CAR_HEIGHT = 60

# --- Parking Spot Dimensions ---
PARK_WIDTH = 50
PARK_HEIGHT = 80
PARK_ANGLE = 90 # Default parking spot angle

# --- Obstacle Dimensions ---
OBSTACLE_WIDTH = 40
OBSTACLE_HEIGHT = 80

# --- UI Buttons ---
ADD_OBS_BUTTON = pygame.Rect(50, SCREEN_HEIGHT - 60, 150, 40)
DEL_OBS_BUTTON = pygame.Rect(250, SCREEN_HEIGHT - 60, 150, 40)

# --- Car Physics Parameters ---
MAX_VELOCITY = 3.0
ACCELERATION_RATE = 0.1
BRAKING_RATE = 0.2
TURN_RATE = 2.0
FRICTION = 0.05
MAX_STEERING_ANGLE = 40

# --- Parking Success Criteria ---
PARKING_MIN_VELOCITY = 0.5
PARKING_MAX_ANGLE_DIFF = 15
PARKING_MAX_DIST_TO_CENTER = CAR_WIDTH * 0.75 # Adjusted based on car size

# --- Genetic Algorithm Parameters ---
POPULATION_SIZE = 100 # Number of cars per generation
GENERATIONS = 200 # Total number of generations to run
SIMULATION_DURATION_SECONDS = 10 # How long each car attempts to park (in seconds)
SIMULATION_DURATION_TICKS = SIMULATION_DURATION_SECONDS * FPS

MUTATION_RATE = 0.15 # Probability of a gene mutating
SELECTION_PERCENTAGE = 0.2 # Percentage of top cars selected for reproduction

# --- Neural Network Parameters ---
INPUT_SIZE = 10 # Number of inputs to the neural network (see car.py get_state)
HIDDEN_SIZE = 32 # Number of neurons in the hidden layer
OUTPUT_SIZE = 5 # Number of possible actions (forward, reverse, steer_left, steer_right, brake)

# --- Reward Parameters ---
COLLISION_PENALTY = -100000
PARKING_BONUS = 1000000
DISTANCE_REWARD_MULTIPLIER = 5 # Reward for getting closer (initial - current)
ANGLE_PENALTY_MULTIPLIER = -5 # Penalty for angle difference to spot heading
ALMOST_PARKED_BONUS = 30 # Bonus for being in spot and close to angle, even if not perfectly parked
SPEED_PENALTY_MULTIPLIER = -0.05 # Small penalty for high speed when close to spot
STEP_PENALTY = -100 # Small penalty per time step to encourage faster parking
STOPPED_PENALTY = -4000
ALIGNMENT_REWARD = 5