# parking_spot.py
import pygame
import math
from constants import PARK_WIDTH, PARK_HEIGHT, SPOT_COLOR, WHITE

class ParkingSpot(pygame.sprite.Sprite):
    def __init__(self, x, y, width=PARK_WIDTH, height=PARK_HEIGHT, angle=None):
        super().__init__()
        self.width = width
        self.height = height
        self.center = pygame.math.Vector2(x, y)
        self.angle = float(angle) # Angle in degrees

        self.original_image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.original_image.fill(SPOT_COLOR) # Semi-transparent green
        pygame.draw.rect(self.original_image, WHITE, (0, 0, self.width, self.height), 2) # White outline

        # Rotate the image based on the initial angle
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.center)
        self.mask = pygame.mask.from_surface(self.image) # Create mask for collision (polygon check is also used)


    def set_center(self, pos, angle=None):
        """Sets the center position and optionally the angle of the parking spot."""
        self.center = pygame.math.Vector2(pos)
        if angle is not None:
            self.angle = float(angle)
        # Re-rotate the original image and update the rect
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.center)
        self.mask = pygame.mask.from_surface(self.image)

    def draw(self, surface):
        """Draws the parking spot on the given surface."""
        surface.blit(self.image, self.rect.topleft)

    def get_corners(self):
        """Calculates the world coordinates of the parking spot's corners."""
        points = []
        half_width = self.width / 2
        half_height = self.height / 2
        # Corners relative to the spot's center
        corners_local = [
            pygame.math.Vector2(-half_width, -half_height),
            pygame.math.Vector2(half_width, -half_height),
            pygame.math.Vector2(half_width, half_height),
            pygame.math.Vector2(-half_width, half_height)
        ]
        rad_angle = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)
        for p_local in corners_local:
            # Apply rotation and translation
            x_rot = p_local.x * cos_a - p_local.y * sin_a
            y_rot = p_local.x * sin_a + p_local.y * cos_a
            points.append(pygame.math.Vector2(self.center.x + x_rot, self.center.y + y_rot))
        return points