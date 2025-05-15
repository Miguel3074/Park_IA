# utils.py
import pygame
import math
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, YELLOW, BLACK, WHITE

def show_message_on_screen(surface, text, color=YELLOW, y_offset=0, size=30):
    msg_font = pygame.font.SysFont(None, size)
    txt_surf = msg_font.render(text, True, color)
    txt_rect = txt_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + y_offset))
    surface.blit(txt_surf, txt_rect)

def do_polygons_collide(poly1_corners, poly2_corners):
    polygons = [poly1_corners, poly2_corners]

    for polygon in polygons:
        for i1 in range(len(polygon)):
            i2 = (i1 + 1) % len(polygon)
            p1 = polygon[i1]
            p2 = polygon[i2]

            edge = p2 - p1
            axis = pygame.math.Vector2(-edge.y, edge.x)

            if axis.length_squared() == 0:
                continue

            axis = axis.normalize()

            minA, maxA = None, None
            for p_polyA in poly1_corners:
                projected = axis.dot(p_polyA)
                if minA is None or projected < minA:
                    minA = projected
                if maxA is None or projected > maxA:
                    maxA = projected

            minB, maxB = None, None
            for p_polyB in poly2_corners:
                projected = axis.dot(p_polyB)
                if minB is None or projected < minB:
                    minB = projected
                if maxB is None or projected > maxB:
                    maxB = projected

            if maxA < minB or maxB < minA:
                return False 
    return True

def draw_polygon(surface, color, points, width=0):
    if len(points) >= 3:
        pygame.draw.polygon(surface, color, points, width)

def normalize_angle(angle):
    angle = angle % 360
    if angle > 180:
        angle -= 360
    elif angle < -180:
         angle += 360
    return angle

def point_to_line_distance(point, a, b):
    if a == b:
        return point.distance_to(a)
    segment_length_sq = (b - a).length_squared()
    t = max(0, min(1, (point - a).dot(b - a) / segment_length_sq))
    projection = a + t * (b - a)
    return point.distance_to(projection)