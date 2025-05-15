# utils.py
import pygame
import math
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, YELLOW, BLACK, WHITE

def show_message_on_screen(surface, text, color=YELLOW, y_offset=0, size=30):
    """Helper function to display text messages on the screen."""
    msg_font = pygame.font.SysFont(None, size)
    txt_surf = msg_font.render(text, True, color)
    txt_rect = txt_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + y_offset))
    surface.blit(txt_surf, txt_rect)

def do_polygons_collide(poly1_corners, poly2_corners):
    """
    Checks for collision between two convex polygons using the Separating Axis Theorem (SAT).
    Args:
        poly1_corners: A list of pygame.math.Vector2 representing the vertices of the first polygon.
        poly2_corners: A list of pygame.math.Vector2 representing the vertices of the second polygon.
    Returns:
        True if the polygons collide, False otherwise.
    """
    polygons = [poly1_corners, poly2_corners]

    for polygon in polygons:
        # Check each edge of the polygon as a potential separating axis
        for i1 in range(len(polygon)):
            i2 = (i1 + 1) % len(polygon)
            p1 = polygon[i1]
            p2 = polygon[i2]

            # Get the edge vector
            edge = p2 - p1
            # The perpendicular vector is the axis (normal to the edge)
            # Use (y, -x) or (-y, x) for the perpendicular vector
            axis = pygame.math.Vector2(-edge.y, edge.x)

            # Handle zero-length edges (shouldn't happen with rectangles, but good practice)
            if axis.length_squared() == 0:
                continue

            axis = axis.normalize()

            # Project all points of both polygons onto the axis
            minA, maxA = None, None
            for p_polyA in poly1_corners:
                # The projection of a point onto a vector is (point . vector) / ||vector||
                # Since our axis is normalized, the projection is just the dot product
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

            # If there is a gap between the projected intervals, the polygons do not overlap
            # The intervals [minA, maxA] and [minB, maxB] overlap if !(maxA < minB or maxB < minA)
            if maxA < minB or maxB < minA:
                return False # Found a separating axis

    # If no separating axis was found after checking all edges, the polygons overlap
    return True

def draw_polygon(surface, color, points, width=0):
    """Helper function to draw a polygon on the surface."""
    if len(points) >= 3:
        pygame.draw.polygon(surface, color, points, width)

def normalize_angle(angle):
    """Normalizes an angle to be within the range [-180, 180]."""
    angle = angle % 360
    if angle > 180:
        angle -= 360
    elif angle < -180:
         angle += 360
    return angle

def point_to_line_distance(point, a, b):
    """Calculates the shortest distance from a point to a line segment defined by two points a and b."""
    if a == b:
        return point.distance_to(a)
    segment_length_sq = (b - a).length_squared()
    t = max(0, min(1, (point - a).dot(b - a) / segment_length_sq))
    projection = a + t * (b - a)
    return point.distance_to(projection)