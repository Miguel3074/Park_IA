import pygame
import sys
import math

pygame.init()

# --- Screen Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 900
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("IA Parking")

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
OBSTACLE_COLOR = (100, 100, 100)

# --- Clock and Font ---
clock = pygame.time.Clock()
FPS = 60
font = pygame.font.SysFont(None, 30)

# --- Car Dimensions ---
CAR_WIDTH = 30
CAR_HEIGHT = 60

PARK_WIDTH = 50
PARK_HEIGHT = 80

OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 80

ADD_OBS_BUTTON = pygame.Rect(50, SCREEN_HEIGHT - 60, 150, 40)
DEL_OBS_BUTTON = pygame.Rect(250, SCREEN_HEIGHT - 60, 150, 40)

player_car_start_pos = (SCREEN_HEIGHT - 100, 100)
angle_park = 90


class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, initial_angle=0, color=BLUE, car_id="player"):
        super().__init__()
        self.id = car_id
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.original_image = pygame.Surface([self.width, self.height], pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, color, (0, 0, self.width, self.height))
        if color == BLUE:
            pygame.draw.line(self.original_image, RED, (self.width / 2, 0), (self.width / 2, 10), 3)

        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.position = pygame.math.Vector2(x, y)
        self.velocity = 0.0
        self.angle = float(initial_angle)
        self.steering_angle = 0.0

        self.max_velocity = 3.0
        self.max_steering_angle = 40
        self.acceleration_rate = 0.1
        self.braking_rate = 0.2
        self.turn_rate = 2.0
        self.friction = 0.05

        self.L = self.height

        self.initial_pos = pygame.math.Vector2(x, y)
        self.initial_angle = float(initial_angle)

        self.mask = pygame.mask.from_surface(self.image)

    def reset(self):
        self.position = pygame.math.Vector2(self.initial_pos.x, self.initial_pos.y)
        self.angle = float(self.initial_angle)
        self.velocity = 0.0
        self.steering_angle = 0.0
        self._update_image_and_rect()

    def _update_image_and_rect(self):
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, pressed_keys=None, action=None):
        # AI Control
        if action is not None:
            if action == 0:
                self.velocity = min(self.velocity + self.acceleration_rate, self.max_velocity)
            elif action == 1:
                self.velocity = max(self.velocity - self.acceleration_rate, -self.max_velocity / 2)
            elif action == 2:
                if self.velocity != 0:
                    self.steering_angle = max(self.steering_angle - self.turn_rate, -self.max_steering_angle)
            elif action == 3:
                if self.velocity != 0:
                    self.steering_angle = min(self.steering_angle + self.turn_rate, self.max_steering_angle)
            elif action == 4:
                if abs(self.velocity) > self.braking_rate:
                    self.velocity -= math.copysign(self.braking_rate, self.velocity)
                else:
                    self.velocity = 0

            if action not in [2, 3] and self.velocity != 0:
                if abs(self.steering_angle) > self.turn_rate / 2:
                    self.steering_angle -= math.copysign(self.turn_rate / 2, self.steering_angle)
                else:
                    self.steering_angle = 0

            if action not in [0, 1, 4]:
                if abs(self.velocity) > self.friction:
                    self.velocity -= math.copysign(self.friction, self.velocity)
                else:
                    self.velocity = 0

        # Manual Control
        elif pressed_keys:
            if pressed_keys[pygame.K_w]:
                self.velocity = min(self.velocity + self.acceleration_rate, self.max_velocity)
            elif pressed_keys[pygame.K_s]:
                self.velocity = max(self.velocity - self.acceleration_rate, -self.max_velocity / 2)
            else:
                if abs(self.velocity) > self.friction:
                    self.velocity -= math.copysign(self.friction, self.velocity)
                else:
                    self.velocity = 0

            if self.velocity != 0:
                if pressed_keys[pygame.K_a]:
                    self.steering_angle = max(self.steering_angle - self.turn_rate, -self.max_steering_angle)
                elif pressed_keys[pygame.K_d]:
                    self.steering_angle = min(self.steering_angle + self.turn_rate, self.max_steering_angle)
                else:
                    if abs(self.steering_angle) > self.turn_rate / 2:
                        self.steering_angle -= math.copysign(self.turn_rate / 2, self.steering_angle)
                    else:
                        self.steering_angle = 0
            else:
                if pressed_keys[pygame.K_a]:
                    self.steering_angle = max(self.steering_angle - self.turn_rate, -self.max_steering_angle)
                elif pressed_keys[pygame.K_d]:
                    self.steering_angle = min(self.steering_angle + self.turn_rate, self.max_steering_angle)

        # Kinematic Model
        if self.steering_angle != 0 and self.velocity != 0:
            turning_radius = self.L / math.tan(math.radians(self.steering_angle))
            angular_velocity_rad = self.velocity / turning_radius
            self.angle = (self.angle + math.degrees(angular_velocity_rad)) % 360

        self.position.x += self.velocity * math.sin(math.radians(self.angle))
        self.position.y -= self.velocity * math.cos(math.radians(self.angle))

        self._update_image_and_rect()

    def get_corners(self):
        points = []
        half_width = self.width / 2
        half_height = self.height / 2
        corners_local = [
            pygame.math.Vector2(-half_width, -half_height), pygame.math.Vector2(half_width, -half_height),
            pygame.math.Vector2(half_width, half_height), pygame.math.Vector2(-half_width, half_height)
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
        super().__init__(x, y, initial_angle, car_id="obstacle")
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT
        self.original_image = pygame.Surface([self.width, self.height], pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, OBSTACLE_COLOR, (0, 0, self.width, self.height))
        self.image = self.original_image
        self.velocity = 0
        self.steering_angle = 0
        self._update_image_and_rect()

    def update(self, *args, **kwargs):
        pass



class ParkingSpot(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, angle=0):
        super().__init__()
        self.width = width
        self.height = height
        self.center = pygame.math.Vector2(x, y)
        self.angle = float(angle)

        self.original_image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.original_image.fill(GREEN + (100,))
        pygame.draw.rect(self.original_image, WHITE, (0, 0, self.width, self.height), 2)

        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.center)
        self.mask = pygame.mask.from_surface(self.image)

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)

    def set_center(self, pos, angle=None):
        self.center = pygame.math.Vector2(pos)
        if angle is not None:
            self.angle = float(angle)
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.center)
        self.mask = pygame.mask.from_surface(self.image)

    def get_corners(self):
        points = []
        half_width = self.width / 2
        half_height = self.height / 2
        corners_local = [
            pygame.math.Vector2(-half_width, -half_height), pygame.math.Vector2(half_width, -half_height),
            pygame.math.Vector2(half_width, half_height), pygame.math.Vector2(-half_width, half_height)
        ]
        rad_angle = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)
        for p_local in corners_local:
            x_rot = p_local.x * cos_a - p_local.y * sin_a
            y_rot = p_local.x * sin_a + p_local.y * cos_a
            points.append(pygame.math.Vector2(self.center.x + x_rot, self.center.y + y_rot))
        return points


def show_message_on_screen(text, color=YELLOW, y_offset=0, size=30):
    msg_font = pygame.font.SysFont(None, size)
    txt_surf = msg_font.render(text, True, color)
    txt_rect = txt_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + y_offset))
    screen.blit(txt_surf, txt_rect)


def check_collision_car_corners_rect(car_obj, rect_obj):
    car_mask = car_obj.mask
    rect_sprite_temp = pygame.sprite.Sprite()
    rect_sprite_temp.image = pygame.Surface((rect_obj.width, rect_obj.height), pygame.SRCALPHA)
    pygame.draw.rect(rect_sprite_temp.image, (255, 0, 0, 100)), (0, 0, rect_obj.width, rect_obj.height)
    rect_sprite_temp.rect = rect_obj
    rect_sprite_temp.mask = pygame.mask.from_surface(rect_sprite_temp.image)

    offset_x = rect_sprite_temp.rect.x - car_obj.rect.x
    offset_y = rect_sprite_temp.rect.y - car_obj.rect.y

    return car_mask.overlap(rect_sprite_temp.mask, (offset_x, offset_y)) is not None


def do_polygons_collide(poly1_corners, poly2_corners):
    polygons = [poly1_corners, poly2_corners]
    for polygon in polygons:
        for i1 in range(len(polygon)):
            i2 = (i1 + 1) % len(polygon)
            p1 = polygon[i1]
            p2 = polygon[i2]

            normal = pygame.math.Vector2(p2.y - p1.y, p1.x - p2.x).normalize()

            minA, maxA = None, None
            for p_polyA in poly1_corners:
                projected = normal.dot(p_polyA)
                if minA is None or projected < minA:
                    minA = projected
                if maxA is None or projected > maxA:
                    maxA = projected

            minB, maxB = None, None
            for p_polyB in poly2_corners:
                projected = normal.dot(p_polyB)
                if minB is None or projected < minB:
                    minB = projected
                if maxB is None or projected > maxB:
                    maxB = projected

            if maxA < minB or maxB < minA:
                return False
    return True


# --- Initial Setup ---
player_car = Car(player_car_start_pos[0], player_car_start_pos[1], initial_angle=-90)
target_spot = ParkingSpot(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width=PARK_WIDTH, height=PARK_HEIGHT, angle=angle_park)

all_sprites_group = pygame.sprite.Group()
all_sprites_group.add(player_car)

obstacle_cars_list = []
obstacle_cars_group = pygame.sprite.Group()

obs1 = ObstacleCar(target_spot.center.x + PARK_HEIGHT, target_spot.center.y, initial_angle=target_spot.angle)
obs2 = ObstacleCar(target_spot.center.x - PARK_HEIGHT, target_spot.center.y, initial_angle=target_spot.angle)


obstacle_cars_list.extend([obs1, obs2])
for obs in obstacle_cars_list:
    obstacle_cars_group.add(obs)
    all_sprites_group.add(obs)

game_message = ""
message_display_timer = 0
MESSAGE_DURATION = 1500

# --- Main Game Loop ---
dragging_object = None
drag_offset = pygame.math.Vector2(0, 0)

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    current_tick_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.math.Vector2(event.pos)

            if event.button == 1:
                for obs in obstacle_cars_list:
                    if obs.rect.collidepoint(mouse_pos):
                        dragging_object = obs
                        drag_offset = mouse_pos - obs.position
                        break
                else:
                    if target_spot.rect.collidepoint(mouse_pos):
                        dragging_object = target_spot
                        drag_offset = mouse_pos - pygame.math.Vector2(target_spot.center)
                    if ADD_OBS_BUTTON.collidepoint(mouse_pos):
                        obstacle_cars_list.append(ObstacleCar(len(obstacle_cars_list) *100,100))
                        obstacle_cars_group.add(obstacle_cars_list[-1])
                        all_sprites_group.add(obstacle_cars_list[-1])
                    elif DEL_OBS_BUTTON.collidepoint(mouse_pos):
                        if obstacle_cars_list:
                            removed_obs = obstacle_cars_list.pop()
                            obstacle_cars_group.remove(removed_obs)
                            all_sprites_group.remove(removed_obs)
                    else:
                        current_spot_angle = target_spot.angle
                        target_spot.set_center(event.pos, angle=current_spot_angle)
                        print(f"New spot center: {target_spot.center}")
                        player_car.reset()
                        game_message = ""

            elif event.button == 3:
                for obj in [*obstacle_cars_list, target_spot]:
                    if obj.rect.collidepoint(mouse_pos):
                        obj.angle = (obj.angle + 90) % 180
                        if isinstance(obj, ObstacleCar):
                            obj._update_image_and_rect()
                        elif isinstance(obj, ParkingSpot):
                            obj.set_center(obj.center, angle=obj.angle)
                        break

        elif event.type == pygame.MOUSEMOTION:
            if dragging_object:
                mouse_pos = pygame.math.Vector2(event.pos)
                new_pos = mouse_pos - drag_offset

                if dragging_object == target_spot:
                    target_spot.set_center(new_pos, angle=target_spot.angle)
                else:
                    dragging_object.position = new_pos
                    dragging_object._update_image_and_rect()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging_object = None

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                player_car.reset()
                game_message = ""

    # --- Input and Car Update ---
    keys_pressed = pygame.key.get_pressed()
    player_car.update(pressed_keys=keys_pressed)

    # --- Collision Detection and Game State ---
    boundary_collision = False
    player_car_corners = player_car.get_corners()
    for corner in player_car_corners:
        if not (0 <= corner.x <= SCREEN_WIDTH and 0 <= corner.y <= SCREEN_HEIGHT):
            boundary_collision = True
            break

    if boundary_collision:
        game_message = "Boundary Collision!"
        message_display_timer = current_tick_time + MESSAGE_DURATION
        player_car.reset()

    collided_with_obstacle = False
    for obs_car in obstacle_cars_list:
        if do_polygons_collide(player_car_corners, obs_car.get_corners()):
            collided_with_obstacle = True
            break

    if collided_with_obstacle:
        game_message = "Obstacle Collision!"
        message_display_timer = current_tick_time + MESSAGE_DURATION
        player_car.reset()
    
    for obs in obstacle_cars_list:
        if do_polygons_collide(obs.get_corners(), target_spot.get_corners()):
            game_message = "Obstacle colliding with spot!"
            message_display_timer = current_tick_time + MESSAGE_DURATION
            break

    # --- Parked Successfully Check ---
    dist_to_spot_center = player_car.position.distance_to(target_spot.center)

    norm_player_angle = (player_car.angle % 360)
    if norm_player_angle > 180: norm_player_angle -= 360
    norm_spot_angle = (target_spot.angle % 360)
    if norm_spot_angle > 180: norm_spot_angle -= 360

    angle_difference = abs(norm_player_angle - norm_spot_angle)
    if angle_difference > 170 and angle_difference < 190:
        angle_difference = abs(angle_difference - 180)
    elif angle_difference > 350 and angle_difference < 370:
        angle_difference = abs(angle_difference - 360)

    is_velocity_low = abs(player_car.velocity) < 0.5

    is_car_in_spot = False
    if dist_to_spot_center < max(target_spot.width, target_spot.height) * 0.7:
        if do_polygons_collide(player_car_corners, target_spot.get_corners()):
            is_car_in_spot = True

    successfully_parked = False
    if is_car_in_spot and \
            angle_difference < 15 and \
            is_velocity_low and \
            dist_to_spot_center < CAR_WIDTH * 0.75:
        successfully_parked = True

    if successfully_parked and not game_message:
        game_message = "Parked Successfully!"
        message_display_timer = current_tick_time + MESSAGE_DURATION

    # --- Drawing ---
    screen.fill(BLACK)
    target_spot.draw(screen)
    all_sprites_group.draw(screen)

    pygame.draw.rect(screen, GREEN, ADD_OBS_BUTTON)
    screen.blit(font.render("Add Obstacle", True, WHITE), (ADD_OBS_BUTTON.x + 10, ADD_OBS_BUTTON.y + 10))

    pygame.draw.rect(screen, RED, DEL_OBS_BUTTON)
    screen.blit(font.render("Remove Obstacle", True, WHITE), (DEL_OBS_BUTTON.x + 10, DEL_OBS_BUTTON.y + 10))


    if game_message and current_tick_time < message_display_timer:
        show_message_on_screen(game_message)
    elif current_tick_time >= message_display_timer:
        game_message = ""

    pygame.display.flip()

pygame.quit()
sys.exit()
