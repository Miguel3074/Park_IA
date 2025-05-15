# simulation.py
import pygame
import sys
import math
import random
import numpy as np 

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

target_spot = ParkingSpot(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width=PARK_WIDTH, height=PARK_HEIGHT, angle=PARK_ANGLE)

obstacle_cars_list = [
    ObstacleCar(target_spot.center.x + PARK_HEIGHT, target_spot.center.y, initial_angle=target_spot.angle),
    ObstacleCar(target_spot.center.x - PARK_HEIGHT, target_spot.center.y, initial_angle=target_spot.angle)
]


game_message = ""
message_display_timer = 0
MESSAGE_DURATION = 1500 
manual_setup_mode = True
dragging_object = None
drag_offset = pygame.math.Vector2(0, 0)
initial_car_position = pygame.math.Vector2(100, SCREEN_HEIGHT - 100) 
initial_car_angle = -90 
dragging_start_point = False 

current_generation = 0
genetic_algorithm = GeneticAlgorithm()
population_cars = [] 
simulation_tick_counter = 0
best_reward_current_gen = -float('inf')
parked_count = 0
collision_count = 0


def setup_new_generation():
    global population_cars, simulation_tick_counter, best_reward_current_gen, parked_count, collision_count

    print(f"--- Starting Generation {current_generation + 1} ---")

    population_cars = []
    brains = genetic_algorithm.population 
    for i in range(POPULATION_SIZE):
        car = Car(
            initial_car_position.x,
            initial_car_position.y,
            initial_angle=initial_car_angle,
            color=BLUE,
            car_id=f"car_{current_generation}_{i}",
            brain=brains[i]
        )
        car.initial_distance_to_spot = car.position.distance_to(target_spot.center) 
        population_cars.append(car)

    simulation_tick_counter = 0
    best_reward_current_gen = -float('inf')
    parked_count = 0
    collision_count = 0


def run_manual_setup():
    global running, dragging_object, drag_offset, manual_setup_mode, game_message, message_display_timer,dragging_start_point,initial_car_position

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.math.Vector2(event.pos)

            if event.button == 1: # Left click
                for obs in obstacle_cars_list:
                    if obs.rect.collidepoint(mouse_pos):
                        dragging_object = obs
                        drag_offset = mouse_pos - obs.position
                        break
                else:
                    if target_spot.rect.collidepoint(mouse_pos):
                        dragging_object = target_spot
                        drag_offset = mouse_pos - pygame.math.Vector2(target_spot.center)
                    elif initial_car_position.distance_to(mouse_pos) < 15: 
                        dragging_start_point = True
                        drag_offset = mouse_pos - initial_car_position
                    elif ADD_OBS_BUTTON.collidepoint(mouse_pos):
                        obstacle_cars_list.append(ObstacleCar(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)) 
                        game_message = "Obstacle Added! Drag to position."
                        message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                    elif DEL_OBS_BUTTON.collidepoint(mouse_pos):
                        if obstacle_cars_list:
                            removed_obs = obstacle_cars_list.pop()
                            game_message = "Obstacle Removed!"
                            message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION


            elif event.button == 3: # Right click to rotate
                mouse_pos = pygame.math.Vector2(event.pos)
                for obj in [*obstacle_cars_list, target_spot]:
                    if obj.rect.collidepoint(mouse_pos):
                        obj.angle = (obj.angle + 90) % 360
                        if isinstance(obj, ObstacleCar):
                             obj._update_image_and_rect()
                        elif isinstance(obj, ParkingSpot):
                             print("teste")
                             obj.set_center(obj.center, angle=obj.angle) 
                        break

        elif event.type == pygame.MOUSEMOTION:
            if dragging_object:
                mouse_pos = pygame.math.Vector2(event.pos)
                new_pos = mouse_pos - drag_offset

                if isinstance(dragging_object, ParkingSpot):
                    dragging_object.set_center(new_pos, angle=dragging_object.angle)
                elif isinstance(dragging_object, ObstacleCar):
                    dragging_object.position = new_pos
                    dragging_object._update_image_and_rect() 
            elif dragging_start_point:
                mouse_pos = pygame.math.Vector2(event.pos)
                initial_car_position = mouse_pos - drag_offset


        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging_object = None
                dragging_start_point = False
                for obs in obstacle_cars_list:
                     if do_polygons_collide(obs.get_corners(), target_spot.get_corners()):
                         game_message = "Obstacle collision with spot! Please move."
                         message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                         break 


        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN: # Enter to start
                manual_setup_mode = False
                setup_new_generation() 


    screen.fill(BLACK)
    target_spot.draw(screen)

    for obs in obstacle_cars_list:
         screen.blit(obs.image, obs.rect.topleft) 

    pygame.draw.circle(screen, BLUE, (int(initial_car_position.x), int(initial_car_position.y)), 10)
    start_point_text = font.render("Start Point", True, WHITE)
    start_point_rect = start_point_text.get_rect(center=(int(initial_car_position.x), int(initial_car_position.y) - 20))
    screen.blit(start_point_text, start_point_rect)

    pygame.draw.rect(screen, GREEN, ADD_OBS_BUTTON)
    screen.blit(font.render("Add Obstacle", True, WHITE), (ADD_OBS_BUTTON.x + 10, ADD_OBS_BUTTON.y + 10))

    pygame.draw.rect(screen, RED, DEL_OBS_BUTTON)
    screen.blit(font.render("Remove Obstacle", True, WHITE), (DEL_OBS_BUTTON.x + 10, DEL_OBS_BUTTON.y + 10))

    show_message_on_screen(screen, "Manual Setup Mode: Drag objects, Right-click to rotate. Press ENTER to start IA.", y_offset=SCREEN_HEIGHT/2 - 50, size=25)

    current_tick_time = pygame.time.get_ticks()
    if game_message and current_tick_time < message_display_timer:
         show_message_on_screen(screen, game_message, y_offset=50)
    elif current_tick_time >= message_display_timer:
         game_message = "" 

def run_ia_simulation():
    global running, simulation_tick_counter, current_generation, population_cars
    global best_reward_current_gen, parked_count, collision_count, game_message, message_display_timer

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
             if event.key == pygame.K_ESCAPE: # Press ESC to return to manual setup
                 global manual_setup_mode, dragging_object
                 manual_setup_mode = True
                 dragging_object = None 
                 population_cars = [] 
                 game_message = "Returned to Manual Setup."
                 message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION

    active_cars = []

    if simulation_tick_counter < SIMULATION_DURATION_TICKS:
        active_cars = [car for car in population_cars if car.active]
        if not active_cars:
            simulation_tick_counter = SIMULATION_DURATION_TICKS 
        for car in active_cars:
            car.update(target_spot, obstacle_cars_list)
            car.check_collisions(obstacle_cars_list)
            car.calculate_reward(target_spot, obstacle_cars_list) 


            if not car.active: 
                 if car.parked_successfully:
                      parked_count += 1
                 elif car.collided:
                      collision_count += 1

    # --- End of Generation ---
    if simulation_tick_counter >= SIMULATION_DURATION_TICKS:
        population_with_rewards = [(car.brain, car.reward) for car in population_cars]
        population_with_rewards.sort(key=lambda item: item[1], reverse=True)

        if population_with_rewards:
             best_reward_current_gen = population_with_rewards[0][1]
             print(f"Generation {current_generation + 1} finished. Best Reward: {best_reward_current_gen:.2f}, Parked: {parked_count}/{POPULATION_SIZE}, Collided: {collision_count}/{POPULATION_SIZE}")


        if current_generation < GENERATIONS - 1:
            genetic_algorithm.create_next_generation(population_with_rewards)
            current_generation += 1
            setup_new_generation()
        else:
            print(f"--- Simulation Finished after {GENERATIONS} Generations ---")
            game_message = "Simulation Complete!"
            message_display_timer = pygame.time.get_ticks() + MESSAGE_DURATION

    screen.fill(BLACK)
    target_spot.draw(screen)

    for obs in obstacle_cars_list:
         screen.blit(obs.image, obs.rect.topleft)

    for car in population_cars:
        if not car.active:
            if car.parked_successfully:
                draw_color = GREEN 
            elif car.collided:
                draw_color = RED 
            else:
                draw_color = OBSTACLE_COLOR 
            draw_polygon(screen, draw_color, car.get_corners(), 2)
        screen.blit(car.image, car.rect.topleft)

    pygame.draw.rect(screen, GREEN, ADD_OBS_BUTTON) 
    screen.blit(font.render("Add Obstacle", True, WHITE), (ADD_OBS_BUTTON.x + 10, ADD_OBS_BUTTON.y + 10))

    pygame.draw.rect(screen, RED, DEL_OBS_BUTTON)
    screen.blit(font.render("Remove Obstacle", True, WHITE), (DEL_OBS_BUTTON.x + 10, DEL_OBS_BUTTON.y + 10))

    gen_text = font.render(f"Generation: {current_generation + 1}/{GENERATIONS}", True, WHITE)
    screen.blit(gen_text, (10, 10))

    tick_text = font.render(f"Tick: {simulation_tick_counter}/{SIMULATION_DURATION_TICKS}", True, WHITE)
    screen.blit(tick_text, (10, 40))

    reward_text = font.render(f"Best Reward (Gen): {best_reward_current_gen:.2f}", True, YELLOW)
    screen.blit(reward_text, (10, 70))

    stats_text = font.render(f"Parked: {parked_count} | Collided: {collision_count} | Active: {len(active_cars)}", True, WHITE)
    screen.blit(stats_text, (10, 100))

    current_tick_time = pygame.time.get_ticks()
    if game_message and current_tick_time < message_display_timer:
         show_message_on_screen(screen, game_message, y_offset=50)
    elif current_tick_time >= message_display_timer:
         game_message = ""
    simulation_tick_counter += 1


running = True
while running:
    dt = clock.tick(FPS) / 1000.0

    if manual_setup_mode:
        run_manual_setup()
    else:
        run_ia_simulation()

    pygame.display.flip()

pygame.quit()
sys.exit()