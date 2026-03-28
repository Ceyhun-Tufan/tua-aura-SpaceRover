import pygame

class Rover:
    def __init__(self, x, y):
        self.gx, self.gy = float(x), float(y)
        self.angle = 0
        self.speed = 0.15

    def update(self, keys, map_w, map_h):
        # Movement logic
        dx, dy = 0, 0
        if keys[pygame.K_w]: dy -= self.speed; dx -= self.speed
        if keys[pygame.K_s]: dy += self.speed; dx += self.speed
        if keys[pygame.K_a]: dy += self.speed; dx -= self.speed
        if keys[pygame.K_d]: dy -= self.speed; dx += self.speed

        # Collision / Map Boundary check
        new_x, new_y = self.gx + dx, self.gy + dy
        if 0 <= new_x < map_w and 0 <= new_y < map_h:
            self.gx, self.gy = new_x, new_y