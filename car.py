# car.py
import pygame
import math
import numpy as np
from constants import (
    CAR_WIDTH, CAR_HEIGHT, BLUE, RED, OBSTACLE_COLOR, MAX_VELOCITY,
    ACCELERATION_RATE, BRAKING_RATE, TURN_RATE, FRICTION, MAX_STEERING_ANGLE,
    SCREEN_WIDTH, SCREEN_HEIGHT, COLLISION_PENALTY, PARKING_BONUS,ALIGNMENT_REWARD,
    DISTANCE_REWARD_MULTIPLIER, ANGLE_PENALTY_MULTIPLIER, STOPPED_PENALTY,
    PARKING_MIN_VELOCITY, PARKING_MAX_ANGLE_DIFF, PARKING_MAX_DIST_TO_CENTER,
    ALMOST_PARKED_BONUS, SPEED_PENALTY_MULTIPLIER, STEP_PENALTY,OBSTACLE_HEIGHT,OBSTACLE_WIDTH,FPS
)
from neural_network import NeuralNetwork # Import the NN class
from utils import do_polygons_collide, normalize_angle, point_to_line_distance # Import utilities

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, initial_angle=0, color=BLUE, car_id="player", brain=None):
        super().__init__()
        self.id = car_id
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        try:
            self.original_image = pygame.image.load("car.png").convert_alpha()
            # Ajustar a escala da imagem se necessário para corresponder ao tamanho anterior
            self.original_image = pygame.transform.scale(self.original_image, (CAR_WIDTH, CAR_HEIGHT))
        except pygame.error as e:
            print(f"Erro ao carregar a imagem do carro: {e}")
            # Se a imagem não carregar, use o retângulo como fallback
            self.original_image = pygame.Surface([CAR_WIDTH, CAR_HEIGHT], pygame.SRCALPHA)
            pygame.draw.rect(self.original_image, color, (0, 0, CAR_WIDTH, CAR_HEIGHT))
            pygame.draw.line(self.original_image, RED, (CAR_WIDTH / 2, 0), (CAR_WIDTH / 2, 10), 3)

        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.position = pygame.math.Vector2(x, y)
        self.velocity = 0.0
        self.angle = float(initial_angle) # Angle in degrees
        self.steering_angle = 0.0 # Steering angle in degrees

        # Physics parameters
        self.max_velocity = MAX_VELOCITY
        self.max_steering_angle = MAX_STEERING_ANGLE
        self.acceleration_rate = ACCELERATION_RATE
        self.braking_rate = BRAKING_RATE
        self.turn_rate = TURN_RATE
        self.friction = FRICTION
        self.L = self.height # Wheelbase length (simplified to car height)
        self.stopped_timer = 0
        # Initial state for resetting
        self.initial_pos = pygame.math.Vector2(x, y)
        self.initial_angle = float(initial_angle)

        self.mask = pygame.mask.from_surface(self.image)

        # AI attributes
        self.brain = brain if brain else NeuralNetwork() # Give the car a brain
        self.reward = 0.0
        self.active = True # Is the car still trying to park in this generation?
        self.collided = False
        self.parked_successfully = False

        # State tracking for reward calculation
        self.initial_distance_to_spot = None
        self.stopped_threshold = FPS * 2 # 2 segundos parado (ajuste conforme necessário)
        self.last_position = pygame.math.Vector2(x, y) # Para calcular a distância percorrida
        self.time_alive = 0 # Counter for how long the car has been active
        self.sensor_readings = [] # To store sensor data


    def reset(self):
        """Resets the car to its initial state for a new generation."""
        self.position = pygame.math.Vector2(self.initial_pos.x, self.initial_pos.y)
        self.angle = float(self.initial_angle)
        self.velocity = 0.0
        self.steering_angle = 0.0
        self.reward = 0.0
        self.active = True
        self.collided = False
        self.parked_successfully = False
        self.initial_distance_to_spot = None # Will be set at the start of the simulation
        self._update_image_and_rect()

    def _update_image_and_rect(self):
        """Updates the car's image rotation and rectangle based on its angle and position."""
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image) # Update mask after rotation

    def get_state(self, target_spot):

        # State vector includes:
        # [0] car_velocity
        # [1] car_angle (normalized, e.g., -1 to 1)
        # [2] car_steering_angle (normalized, e.g., -1 to 1)
        # [3] dx_to_spot (relative x distance to spot center)
        # [4] dy_to_spot (relative y distance to spot center)
        # [5] angle_diff_to_spot_heading (normalized angle difference)
        # [6] dist_to_left_boundary (normalized)
        # [7] dist_to_right_boundary (normalized)
        # [8] dist_to_top_boundary (normalized)
        # [9] dist_to_bottom_boundary (normalized)

        state = []

        # Car state
        state.append(self.velocity / self.max_velocity) # Normalize velocity
        state.append(normalize_angle(self.angle) / 180.0) # Normalize angle to -1 to 1
        state.append(self.steering_angle / self.max_steering_angle) # Normalize steering angle

        # Relative position and angle to target spot
        dx_to_spot = target_spot.center.x - self.position.x
        dy_to_spot = target_spot.center.y - self.position.y
        state.append(dx_to_spot / SCREEN_WIDTH) # Normalize by screen dimensions
        state.append(dy_to_spot / SCREEN_HEIGHT)

        angle_diff_to_spot_heading = normalize_angle(target_spot.angle - self.angle)
        state.append(angle_diff_to_spot_heading / 180.0) # Normalize angle difference

        # Distances to boundaries (simple sensing)
        state.append(self.position.x / SCREEN_WIDTH)
        state.append((SCREEN_WIDTH - self.position.x) / SCREEN_WIDTH)
        state.append(self.position.y / SCREEN_HEIGHT)
        state.append((SCREEN_HEIGHT - self.position.y) / SCREEN_HEIGHT)

        # Add more complex sensor inputs here if needed (e.g., ray casting to obstacles)

        return state

    def calculate_reward(self, target_spot, obstacles):
        """Calculates the reward for the current state."""
        if not self.active:
            return 0 

        reward_change = 0

        if self.collided:
            reward_change += COLLISION_PENALTY
            self.active = False 

        current_distance = self.position.distance_to(target_spot.center)
        if self.initial_distance_to_spot is not None:
            reward_change += (self.initial_distance_to_spot - current_distance) * DISTANCE_REWARD_MULTIPLIER
            pass 

        angle_difference = abs(normalize_angle(self.angle - target_spot.angle))
        reward_change += angle_difference * ANGLE_PENALTY_MULTIPLIER

        alignment_reward = 1.0 - (angle_difference / 180.0)
        reward_change += alignment_reward * ALIGNMENT_REWARD


        if current_distance < max(target_spot.width, target_spot.height):
             reward_change += abs(self.velocity) * SPEED_PENALTY_MULTIPLIER

        dist_to_spot_center = self.position.distance_to(target_spot.center)
        angle_diff_spot_heading = abs(normalize_angle(self.angle - target_spot.angle))
        is_velocity_low = abs(self.velocity) < PARKING_MIN_VELOCITY

        is_car_in_spot_area = dist_to_spot_center < max(target_spot.width, target_spot.height) * 0.75
        is_aligned_with_spot_pos = dist_to_spot_center < PARKING_MAX_DIST_TO_CENTER


        successfully_parked = False
        if is_aligned_with_spot_pos and \
           angle_diff_spot_heading < PARKING_MAX_ANGLE_DIFF and \
           is_velocity_low and \
           do_polygons_collide(self.get_corners(), target_spot.get_corners()):
                successfully_parked = True

        if successfully_parked:
            reward_change += PARKING_BONUS
            self.parked_successfully = True
            self.active = False

        if not successfully_parked and is_car_in_spot_area and angle_diff_spot_heading < PARKING_MAX_ANGLE_DIFF + 10 and is_velocity_low:
             reward_change += ALMOST_PARKED_BONUS * (1 - (angle_diff_spot_heading / (PARKING_MAX_ANGLE_DIFF + 10))) * (1 - (dist_to_spot_center / (max(target_spot.width, target_spot.height) * 0.75)))

        if self.stopped_timer > self.stopped_threshold:
            reward_change += STOPPED_PENALTY

        reward_change += STEP_PENALTY

        self.reward += reward_change
        return reward_change


    def update(self, target_spot, obstacles):
        """Updates the car's state based on AI input (action) and physics."""
        if not self.active:
            return # Don't update inactive cars

        # Get state and decide action using brain
        state = self.get_state(target_spot)
        output = self.brain.feedforward(state)
        action = np.argmax(output) # Choose the action with the highest output

        # Apply action (based on original code's action mapping)
        # 0: Accelerate Forward
        # 1: Accelerate Backward
        # 2: Steer Left (while moving)
        # 3: Steer Right (while moving)
        # 4: Brake (Reduce velocity to zero)

        if action == 0: # Accelerate Forward
            self.velocity = min(self.velocity + self.acceleration_rate, self.max_velocity)
        elif action == 1: # Accelerate Backward
            self.velocity = max(self.velocity - self.acceleration_rate, -self.max_velocity / 2) # Reverse speed limit
        elif action == 4: # Brake
            if abs(self.velocity) > self.braking_rate:
                self.velocity -= math.copysign(self.braking_rate, self.velocity)
            else:
                self.velocity = 0
        else: # Apply friction if not accelerating or braking
            if abs(self.velocity) > self.friction:
                self.velocity -= math.copysign(self.friction, self.velocity)
            else:
                self.velocity = 0

        # Steering is independent of acceleration input, but only affects angle if moving
        # Let the AI control steering directly
        if action == 2: # Steer Left
            self.steering_angle = max(self.steering_angle - self.turn_rate, -self.max_steering_angle)
        elif action == 3: # Steer Right
            self.steering_angle = min(self.steering_angle + self.turn_rate, self.max_steering_angle)
        else: # Center steering if no steering action is taken
            if abs(self.steering_angle) > self.turn_rate / 2:
                self.steering_angle -= math.copysign(self.turn_rate / 2, self.steering_angle)
            else:
                self.steering_angle = 0

        current_speed = self.velocity
        stopped_velocity_threshold = 0.1 # Limiar de velocidade para considerar parado

        if current_speed < stopped_velocity_threshold:
            self.stopped_timer += 1
        else:
            self.stopped_timer = 0 # Reset the timer if the car moves

        # Kinematic Model (Bicycle Model)
        # Update position and angle based on velocity and steering
        if abs(self.velocity) > 0.1: # Only update angle if moving significantly
             # radians_per_degree = math.pi / 180.0
             # if abs(self.steering_angle) > 0.1:
             #     turn_radius = self.L / math.tan(math.radians(self.steering_angle))
             #     angular_velocity_rad = (self.velocity / turn_radius) # This is the angular velocity of the vehicle's rotation about the ICC
             #     # The change in heading (angle) is velocity / L * tan(steering_angle)
             #     self.angle += math.degrees(self.velocity / self.L * math.tan(math.radians(self.steering_angle)))
             # else:
             #    pass # Straight movement, angle doesn't change

            # Simplified update from original code (works but might not be strict bicycle model)
            # The original implementation's angle update is only dependent on steering angle, not velocity, which is incorrect for the formula used.
            # Let's use a corrected integration:
            rad_angle = math.radians(self.angle)
            self.position.x += self.velocity * math.sin(rad_angle)
            self.position.y -= self.velocity * math.cos(rad_angle)

            # Update angle based on steering and velocity (approximation)
            # A better approximation for angle update: angle_change = (velocity / L) * tan(steering_angle) * dt
            # Since dt is absorbed into velocity in the update rules, we can simplify or
            # use the rate-based update from the original code which *feels* okay visually.
            # Let's revert to the original angle update logic for consistency with initial code feel
            # but link it to velocity > 0.1:
            # The original angle update was:
            # if self.steering_angle != 0 and self.velocity != 0:
            #    turning_radius = self.L / math.tan(math.radians(self.steering_angle))
            #    angular_velocity_rad = self.velocity / turning_radius
            #    self.angle = (self.angle + math.degrees(angular_velocity_rad)) % 360

            # Let's stick to the original code's approach for angle update linked to steering AND velocity > 0.1
            if abs(self.steering_angle) > 0.1: # and abs(self.velocity) > 0.1: # Original also required velocity
                 # The original angle update was simplified and not strictly bicycle model,
                 # it adds angular_velocity based on velocity/turning_radius.
                 # This is conceptually closer to how the car *rotates* around an instantaneous center.
                 # Let's use this for now as it was in the original code and might be sufficient.
                 if abs(self.velocity) > 0.1: # Re-add velocity check for angle change
                    turning_radius = self.L / math.tan(math.radians(self.steering_angle))
                     # This angular velocity is for the *center of rotation*, not the car's heading directly.
                     # A common kinematic model updates x, y, and angle based on velocity and steering simultaneously.
                     # Example: x += v * cos(angle), y += v * sin(angle), angle += v / L * tan(steer)
                     # Our original code uses: x += v * sin(angle), y -= v * cos(angle) -- Y axis inverted in pygame
                     # and angle = (angle + degrees(v / (L/tan(steer))))
                     # Let's stick to the original math structure as it was working visually,
                     # but ensure angle update only happens if velocity is non-zero.
                    self.angle = (self.angle + math.degrees(self.velocity / (self.L / math.tan(math.radians(self.steering_angle))))) % 360


        # Update the car's image and rectangle for drawing and collision detection
        self._update_image_and_rect()

    def check_collisions(self, obstacles):
        """Checks for collisions with boundaries and obstacles."""
        if not self.active:
            return # No collision check if inactive

        # Boundary collision
        car_corners = self.get_corners()
        for corner in car_corners:
            if not (0 <= corner.x <= SCREEN_WIDTH and 0 <= corner.y <= SCREEN_HEIGHT):
                self.collided = True
                # print(f"Car {self.id} collided with boundary at {corner}")
                return

        # Obstacle collision
        for obs_car in obstacles:
            if do_polygons_collide(car_corners, obs_car.get_corners()):
                self.collided = True
                # print(f"Car {self.id} collided with obstacle")
                return

    def get_corners(self):
        """Calculates the world coordinates of the car's corners."""
        points = []
        half_width = self.width / 2
        half_height = self.height / 2
        # Corners relative to the car's center (front is -y, back is +y)
        corners_local = [
            pygame.math.Vector2(-half_width, -half_height), # Top Left
            pygame.math.Vector2(half_width, -half_height),  # Top Right
            pygame.math.Vector2(half_width, half_height),   # Bottom Right
            pygame.math.Vector2(-half_width, half_height)   # Bottom Left
        ]
        rad_angle = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)
        for p_local in corners_local:
            # Apply rotation and translation
            x_rot = p_local.x * cos_a - p_local.y * sin_a
            y_rot = p_local.x * sin_a + p_local.y * cos_a
            points.append(pygame.math.Vector2(self.position.x + x_rot, self.position.y + y_rot))
        return points


class ObstacleCar(Car):
    """A static obstacle represented as a car."""
    def __init__(self, x, y, initial_angle=0):
        # Obstacles don't need brains or AI attributes
        super().__init__(x, y, initial_angle, color=OBSTACLE_COLOR, car_id="obstacle", brain=None)
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT

        try:
            self.original_image = pygame.image.load("car2.png").convert_alpha()
            # Ajustar a escala da imagem se necessário para corresponder ao tamanho anterior
            self.original_image = pygame.transform.scale(self.original_image, (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        except pygame.error as e:
            print(f"Erro ao carregar a imagem do carro: {e}")
            self.original_image = pygame.Surface([self.width, self.height], pygame.SRCALPHA)
            pygame.draw.rect(self.original_image, OBSTACLE_COLOR, (0, 0, self.width, self.height))

        self.image = self.original_image # Start with the original image
        self.velocity = 0 # Obstacles are static
        self.steering_angle = 0 # Obstacles don't steer
        # Update the image and rect based on initial angle and custom size
        self._update_image_and_rect()

    def update(self, *args, **kwargs):
        """Obstacles do not update their state."""
        pass

    # Obstacles don't need to calculate reward or get state
    def calculate_reward(self, *args, **kwargs):
        pass

    def get_state(self, *args, **kwargs):
        pass