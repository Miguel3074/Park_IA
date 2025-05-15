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
from neural_network import NeuralNetwork 
from utils import do_polygons_collide, normalize_angle, point_to_line_distance 

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, initial_angle=0, color=BLUE, car_id="player", brain=None):
        super().__init__()
        self.id = car_id
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        try:
            self.original_image = pygame.image.load("car.png").convert_alpha()
            
            self.original_image = pygame.transform.scale(self.original_image, (CAR_WIDTH, CAR_HEIGHT))
        except pygame.error as e:
            print(f"Error: {e}")
            self.original_image = pygame.Surface([CAR_WIDTH, CAR_HEIGHT], pygame.SRCALPHA)
            pygame.draw.rect(self.original_image, color, (0, 0, CAR_WIDTH, CAR_HEIGHT))
            pygame.draw.line(self.original_image, RED, (CAR_WIDTH / 2, 0), (CAR_WIDTH / 2, 10), 3)

        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.position = pygame.math.Vector2(x, y)
        self.velocity = 0.0
        self.angle = float(initial_angle)
        self.steering_angle = 0.0 

        self.max_velocity = MAX_VELOCITY
        self.max_steering_angle = MAX_STEERING_ANGLE
        self.acceleration_rate = ACCELERATION_RATE
        self.braking_rate = BRAKING_RATE
        self.turn_rate = TURN_RATE
        self.friction = FRICTION
        self.L = self.height 
        self.stopped_timer = 0
        self.initial_pos = pygame.math.Vector2(x, y)
        self.initial_angle = float(initial_angle)

        self.mask = pygame.mask.from_surface(self.image)

        self.brain = brain if brain else NeuralNetwork() 
        self.reward = 0.0
        self.active = True 
        self.collided = False
        self.parked_successfully = False

        self.initial_distance_to_spot = None
        self.stopped_threshold = FPS * 2 
        self.last_position = pygame.math.Vector2(x, y)
        self.time_alive = 0 
        self.sensor_readings = []


    def reset(self):
        self.position = pygame.math.Vector2(self.initial_pos.x, self.initial_pos.y)
        self.angle = float(self.initial_angle)
        self.velocity = 0.0
        self.steering_angle = 0.0
        self.reward = 0.0
        self.active = True
        self.collided = False
        self.parked_successfully = False
        self.initial_distance_to_spot = None 
        self._update_image_and_rect()

    def _update_image_and_rect(self):
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image) 

    def get_state(self, target_spot):

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

        state.append(self.velocity / self.max_velocity) 
        state.append(normalize_angle(self.angle) / 180.0) 
        state.append(self.steering_angle / self.max_steering_angle) 
        dx_to_spot = target_spot.center.x - self.position.x
        dy_to_spot = target_spot.center.y - self.position.y
        state.append(dx_to_spot / SCREEN_WIDTH) 
        state.append(dy_to_spot / SCREEN_HEIGHT)

        angle_diff_to_spot_heading = normalize_angle(target_spot.angle - self.angle)
        state.append(angle_diff_to_spot_heading / 180.0) 
        state.append(self.position.x / SCREEN_WIDTH)
        state.append((SCREEN_WIDTH - self.position.x) / SCREEN_WIDTH)
        state.append(self.position.y / SCREEN_HEIGHT)
        state.append((SCREEN_HEIGHT - self.position.y) / SCREEN_HEIGHT)

        return state

    def calculate_reward(self, target_spot, obstacles):
        if not self.active:
            return 0 

        reward_change = 0

        if self.collided:
            reward_change += COLLISION_PENALTY
            self.active = False 

        current_distance = self.position.distance_to(target_spot.center)
        if (self.initial_distance_to_spot - current_distance > 0):
            distance_change = self.initial_distance_to_spot - current_distance
            reward_change += distance_change * DISTANCE_REWARD_MULTIPLIER
            self.initial_distance_to_spot = current_distance
        else:
            distance_change = self.initial_distance_to_spot - current_distance
            reward_change -= distance_change * DISTANCE_REWARD_MULTIPLIER
            self.initial_distance_to_spot = current_distance
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
        if not self.active:
            return 
        state = self.get_state(target_spot)
        output = self.brain.feedforward(state)
        action = np.argmax(output) 

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
        stopped_velocity_threshold = 0.1

        if current_speed < stopped_velocity_threshold:
            self.stopped_timer += 1
        else:
            self.stopped_timer = 0
        if abs(self.velocity) > 0.1: 
             # if abs(self.steering_angle) > 0.1:
             #     turn_radius = self.L / math.tan(math.radians(self.steering_angle))
             #     angular_velocity_rad = (self.velocity / turn_radius) 
             #     # The change in heading (angle) is velocity / L * tan(steering_angle)
             #     self.angle += math.degrees(self.velocity / self.L * math.tan(math.radians(self.steering_angle)))
             # else:
             #    pass 

            rad_angle = math.radians(self.angle)
            self.position.x += self.velocity * math.sin(rad_angle)
            self.position.y -= self.velocity * math.cos(rad_angle)

            # if self.steering_angle != 0 and self.velocity != 0:
            #    turning_radius = self.L / math.tan(math.radians(self.steering_angle))
            #    angular_velocity_rad = self.velocity / turning_radius
            #    self.angle = (self.angle + math.degrees(angular_velocity_rad)) % 360

            if abs(self.steering_angle) > 0.1: 
                 if abs(self.velocity) > 0.1:
                    turning_radius = self.L / math.tan(math.radians(self.steering_angle))
                
                    self.angle = (self.angle + math.degrees(self.velocity / (self.L / math.tan(math.radians(self.steering_angle))))) % 360

        self._update_image_and_rect()

    def check_collisions(self, obstacles):
        if not self.active:
            return 

        car_corners = self.get_corners()
        for corner in car_corners:
            if not (0 <= corner.x <= SCREEN_WIDTH and 0 <= corner.y <= SCREEN_HEIGHT):
                self.collided = True
                return

        # Obstacle collision
        for obs_car in obstacles:
            if do_polygons_collide(car_corners, obs_car.get_corners()):
                self.collided = True
                return

    def get_corners(self):
        points = []
        half_width = self.width / 2
        half_height = self.height / 2
        corners_local = [
            pygame.math.Vector2(-half_width, -half_height), # Top Left
            pygame.math.Vector2(half_width, -half_height),  # Top Right
            pygame.math.Vector2(half_width, half_height),   # Bottom Right
            pygame.math.Vector2(-half_width, half_height)   # Bottom Left
        ]
        rad_angle = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)
        for p_local in corners_local:
            x_rot = p_local.x * cos_a - p_local.y * sin_a
            y_rot = p_local.x * sin_a + p_local.y * cos_a
            points.append(pygame.math.Vector2(self.position.x + x_rot, self.position.y + y_rot))
        return points


class ObstacleCar(Car):
    def __init__(self, x, y, initial_angle=0):
        super().__init__(x, y, initial_angle, color=OBSTACLE_COLOR, car_id="obstacle", brain=None)
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT

        try:
            self.original_image = pygame.image.load("car2.png").convert_alpha()
            self.original_image = pygame.transform.scale(self.original_image, (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        except pygame.error as e:
            print(f"Error {e}")
            self.original_image = pygame.Surface([self.width, self.height], pygame.SRCALPHA)
            pygame.draw.rect(self.original_image, OBSTACLE_COLOR, (0, 0, self.width, self.height))

        self.image = self.original_image
        self.velocity = 0
        self.steering_angle = 0
        self._update_image_and_rect()

    def update(self, *args, **kwargs):
        pass

    def calculate_reward(self, *args, **kwargs):
        pass

    def get_state(self, *args, **kwargs):
        pass