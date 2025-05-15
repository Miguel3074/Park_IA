# simulation.py
import pygame
import sys
import math
import random
import numpy as np # Needed for sorting and argmax

from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, BLACK, WHITE, YELLOW, RED, GREEN, BLUE,
    FPS,PARK_WIDTH, PARK_HEIGHT, PARK_ANGLE,OBSTACLE_COLOR,ADD_OBS_BUTTON, DEL_OBS_BUTTON,
    POPULATION_SIZE, GENERATIONS, SIMULATION_DURATION_TICKS
)
from car import Car, ObstacleCar
from parking_spot import ParkingSpot
from genetic_algorithm import GeneticAlgorithm
from utils import show_message_on_screen, do_polygons_collide, draw_polygon

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("IA Parking - Genetic Algorithm")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# --- Initial Setup ---
target_spot = ParkingSpot(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width=PARK_WIDTH, height=PARK_HEIGHT, angle=PARK_ANGLE)

# Initial obstacles (can be moved/added/deleted in the manual setup phase)
obstacle_cars_list = [
    ObstacleCar(target_spot.center.x + PARK_HEIGHT, target_spot.center.y, initial_angle=target_spot.angle),
    ObstacleCar(target_spot.center.x - PARK_HEIGHT, target_spot.center.y, initial_angle=target_spot.angle)
]


# --- Game State Variables ---
game_message = ""
message_display_timer = 0
MESSAGE_DURATION = 1500 # Milliseconds

# --- Manual Setup State ---
manual_setup_mode = True
dragging_object = None
drag_offset = pygame.math.Vector2(0, 0)
initial_car_position = pygame.math.Vector2(100, SCREEN_HEIGHT - 100) # Starting pos for all AI cars
initial_car_angle = -90 # Starting angle for all AI cars
dragging_start_point = False # Nova variável

# --- AI Simulation State ---
current_generation = 0
genetic_algorithm = GeneticAlgorithm()
population_cars = [] # List of Car objects for the current generation
simulation_tick_counter = 0
best_reward_current_gen = -float('inf')
parked_count = 0
collision_count = 0


def setup_new_generation():
    """Sets up the environment and cars for a new generation."""
    global population_cars, simulation_tick_counter, best_reward_current_gen, parked_count, collision_count

    print(f"--- Starting Generation {current_generation + 1} ---")

    population_cars = []
    brains = genetic_algorithm.population # Get brains from the GA manager

    for i in range(POPULATION_SIZE):
        # Create a new car with the brain from the population
        car = Car(
            initial_car_position.x,
            initial_car_position.y,
            initial_angle=initial_car_angle,
            color=BLUE,
            car_id=f"car_{current_generation}_{i}",
            brain=brains[i]
        )
        car.initial_distance_to_spot = car.position.distance_to(target_spot.center) # Record initial distance
        population_cars.append(car)

    simulation_tick_counter = 0
    best_reward_current_gen = -float('inf')
    parked_count = 0
    collision_count = 0


def run_manual_setup():
    """Handles input and updates for manual setup mode."""
    global running, dragging_object, drag_offset, manual_setup_mode, game_message, message_display_timer,dragging_start_point,initial_car_position

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.math.Vector2(event.pos)

            if event.button == 1: # Left click
                # Check obstacle dragging
                for obs in obstacle_cars_list:
                    if obs.rect.collidepoint(mouse_pos):
                        dragging_object = obs
                        drag_offset = mouse_pos - obs.position
                        break
                # Check parking spot dragging
                else:
                    if target_spot.rect.collidepoint(mouse_pos):
                        dragging_object = target_spot
                        drag_offset = mouse_pos - pygame.math.Vector2(target_spot.center)
                    # Check buttons
                    elif initial_car_position.distance_to(mouse_pos) < 15: # Raio de detecção
                        dragging_start_point = True
                        drag_offset = mouse_pos - initial_car_position
                    elif ADD_OBS_BUTTON.collidepoint(mouse_pos):
                        obstacle_cars_list.append(ObstacleCar(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)) # Add obstacle at center initially
                        game_message = "Obstacle Added! Drag to position."
                        message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                    elif DEL_OBS_BUTTON.collidepoint(mouse_pos):
                        if obstacle_cars_list:
                            removed_obs = obstacle_cars_list.pop()
                            # No need to remove from sprite groups yet as AI cars aren't created
                            game_message = "Obstacle Removed!"
                            message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION


            elif event.button == 3: # Right click to rotate
                mouse_pos = pygame.math.Vector2(event.pos)
                # Rotate obstacles or parking spot
                for obj in [*obstacle_cars_list, target_spot]:
                    if obj.rect.collidepoint(mouse_pos):
                        obj.angle = (obj.angle + 90) % 360 # Rotate by 90 degrees
                        if isinstance(obj, ObstacleCar):
                             obj._update_image_and_rect() # Need to manually update obstacle image/rect after angle change
                        elif isinstance(obj, ParkingSpot):
                             print("teste")
                             obj.set_center(obj.center, angle=obj.angle) # Parking spot has a setter
                        break

        elif event.type == pygame.MOUSEMOTION:
            if dragging_object:
                mouse_pos = pygame.math.Vector2(event.pos)
                new_pos = mouse_pos - drag_offset

                if isinstance(dragging_object, ParkingSpot):
                    dragging_object.set_center(new_pos, angle=dragging_object.angle)
                elif isinstance(dragging_object, ObstacleCar):
                    dragging_object.position = new_pos
                    dragging_object._update_image_and_rect() # Update obstacle image/rect after position change
            elif dragging_start_point:
                mouse_pos = pygame.math.Vector2(event.pos)
                initial_car_position = mouse_pos - drag_offset


        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging_object = None
                dragging_start_point = False
                # Check if any obstacles are colliding with the parking spot after drag
                for obs in obstacle_cars_list:
                     if do_polygons_collide(obs.get_corners(), target_spot.get_corners()):
                         game_message = "Obstacle collision with spot! Please move."
                         message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                         break # Only need to show one message


        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN: # Press Enter to start simulation
                manual_setup_mode = False
                setup_new_generation() # Start the first generation


    # --- Drawing in Manual Setup Mode ---
    screen.fill(BLACK)
    target_spot.draw(screen)

    # Draw obstacles
    for obs in obstacle_cars_list:
         screen.blit(obs.image, obs.rect.topleft) # Obstacles don't have update/draw methods like Car


    pygame.draw.circle(screen, BLUE, (int(initial_car_position.x), int(initial_car_position.y)), 10)
    start_point_text = font.render("Start Point", True, WHITE)
    start_point_rect = start_point_text.get_rect(center=(int(initial_car_position.x), int(initial_car_position.y) - 20))
    screen.blit(start_point_text, start_point_rect)

    pygame.draw.rect(screen, GREEN, ADD_OBS_BUTTON)
    screen.blit(font.render("Add Obstacle", True, WHITE), (ADD_OBS_BUTTON.x + 10, ADD_OBS_BUTTON.y + 10))

    pygame.draw.rect(screen, RED, DEL_OBS_BUTTON)
    screen.blit(font.render("Remove Obstacle", True, WHITE), (DEL_OBS_BUTTON.x + 10, DEL_OBS_BUTTON.y + 10))

    show_message_on_screen(screen, "Manual Setup Mode: Drag objects, Right-click to rotate. Press ENTER to start IA.", y_offset=SCREEN_HEIGHT/2 - 50, size=25)

    # Display messages
    current_tick_time = pygame.time.get_ticks()
    if game_message and current_tick_time < message_display_timer:
         show_message_on_screen(screen, game_message, y_offset=50)
    elif current_tick_time >= message_display_timer:
         game_message = "" # Clear message


def run_ia_simulation():
    """Handles updates and drawing for the AI simulation mode."""
    global running, simulation_tick_counter, current_generation, population_cars
    global best_reward_current_gen, parked_count, collision_count, game_message, message_display_timer

    # --- Event Handling (only quit in AI mode) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
             if event.key == pygame.K_ESCAPE: # Press ESC to return to manual setup
                 global manual_setup_mode, dragging_object
                 manual_setup_mode = True
                 dragging_object = None # Clear any potential dragging state
                 population_cars = [] # Clear AI cars
                 game_message = "Returned to Manual Setup."
                 message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION

    # Inicialize active_cars aqui
    active_cars = []

    # --- Simulation Update ---
    if simulation_tick_counter < SIMULATION_DURATION_TICKS:
        active_cars = [car for car in population_cars if car.active]
        if not active_cars:
            # All cars are inactive (parked or crashed) before duration ends
            simulation_tick_counter = SIMULATION_DURATION_TICKS # End generation early

        for car in active_cars:
            car.update(target_spot, obstacle_cars_list) # Pass target and obstacles for state/collision
            car.check_collisions(obstacle_cars_list) # Check collisions after update
            car.calculate_reward(target_spot, obstacle_cars_list) # Calculate reward after collisions

            # Update stats
            if not car.active: # If car just became inactive
                 if car.parked_successfully:
                      parked_count += 1
                 elif car.collided:
                      collision_count += 1

    # --- End of Generation ---
    if simulation_tick_counter >= SIMULATION_DURATION_TICKS:
        # Collect brains and rewards
        population_with_rewards = [(car.brain, car.reward) for car in population_cars]
        # Sort by reward in descending order
        population_with_rewards.sort(key=lambda item: item[1], reverse=True)

        # Get the best reward of the generation
        if population_with_rewards:
             best_reward_current_gen = population_with_rewards[0][1]
             print(f"Generation {current_generation + 1} finished. Best Reward: {best_reward_current_gen:.2f}, Parked: {parked_count}/{POPULATION_SIZE}, Collided: {collision_count}/{POPULATION_SIZE}")

        # Create the next generation using the GA
        if current_generation < GENERATIONS - 1:
            genetic_algorithm.create_next_generation(population_with_rewards)
            current_generation += 1
            setup_new_generation() # Prepare for the next generation
        else:
            print(f"--- Simulation Finished after {GENERATIONS} Generations ---")
            game_message = "Simulation Complete!"
            message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION # Keep message on screen

    # --- Drawing in AI Simulation Mode ---
    screen.fill(BLACK)
    target_spot.draw(screen)

    # Draw obstacles
    for obs in obstacle_cars_list:
         screen.blit(obs.image, obs.rect.topleft)

    # Draw all cars in the population
    for car in population_cars:
        # Optionally draw crashed/parked cars differently
        if not car.active:
            if car.parked_successfully:
                draw_color = GREEN # Parked cars green
            elif car.collided:
                draw_color = RED # Crashed cars red
            else:
                draw_color = OBSTACLE_COLOR # Inactive for other reasons (e.g., time ran out)
            # Draw polygon outline for inactive cars
            draw_polygon(screen, draw_color, car.get_corners(), 2)
        screen.blit(car.image, car.rect.topleft)

    # Display UI and info
    pygame.draw.rect(screen, GREEN, ADD_OBS_BUTTON) # Still draw buttons but they don't work
    screen.blit(font.render("Add Obstacle", True, WHITE), (ADD_OBS_BUTTON.x + 10, ADD_OBS_BUTTON.y + 10))

    pygame.draw.rect(screen, RED, DEL_OBS_BUTTON)
    screen.blit(font.render("Remove Obstacle", True, WHITE), (DEL_OBS_BUTTON.x + 10, DEL_OBS_BUTTON.y + 10))

    # Display generation info
    gen_text = font.render(f"Generation: {current_generation + 1}/{GENERATIONS}", True, WHITE)
    screen.blit(gen_text, (10, 10))

    tick_text = font.render(f"Tick: {simulation_tick_counter}/{SIMULATION_DURATION_TICKS}", True, WHITE)
    screen.blit(tick_text, (10, 40))

    reward_text = font.render(f"Best Reward (Gen): {best_reward_current_gen:.2f}", True, YELLOW)
    screen.blit(reward_text, (10, 70))

    stats_text = font.render(f"Parked: {parked_count} | Collided: {collision_count} | Active: {len(active_cars)}", True, WHITE)
    screen.blit(stats_text, (10, 100))

    # Display messages
    current_tick_time = pygame.time.get_ticks()
    if game_message and current_tick_time < message_display_timer:
         show_message_on_screen(screen, game_message, y_offset=50)
    elif current_tick_time >= message_display_timer:
         game_message = "" # Clear message

    # Increment simulation tick counter
    simulation_tick_counter += 1


# --- Main Loop ---
running = True
while running:
    dt = clock.tick(FPS) / 1000.0 # dt is not strictly used in physics but useful for clock

    if manual_setup_mode:
        run_manual_setup()
    else:
        run_ia_simulation()

    pygame.display.flip()

pygame.quit()
sys.exit()