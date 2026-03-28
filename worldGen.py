import pygame
import numpy as np
import sys
import math
from random import randint
# ── Config ───────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080
TILE_W,   TILE_H   = 24, 12       # Smaller tiles to fit the "bigger" world
TILE_DEPTH         = 5
MAP_W,    MAP_H    = 400, 400     # Much larger map
FPS                = 60

# Moon colour palette
COL_SKY        = (5, 8, 20)
# Updated palette for style_idx 3 (Lowlands/Maria)
COL_TILE_TOP   = [(80, 85, 100), (95, 100, 115), (120, 125, 140), (40, 42, 50)]
COL_TILE_LEFT  = [(45, 48, 60), (38, 42, 55), (50, 54, 65), (35, 40, 52)]
COL_TILE_RIGHT = [(30, 33, 45), (25, 28, 38), (35, 38, 48), (22, 26, 35)]

# ══════════════════════════════════════════════════════════════════════════════
# WORLD GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class WorldGenerator:
    HEIGHT_LEVELS = 30  # Increased for more verticality
    
    def __init__(self, width, height, seed=None):
        self.width  = width
        self.height = height
        self.seed   = seed if seed is not None else randint(0, 100000)
        self.rng    = np.random.default_rng(self.seed)

        self.gx, self.gy = np.meshgrid(np.arange(width), np.arange(height))
        
        # 1. Start with base rough terrain
        hmap = self._build_heightmap()
        
        # 2. Add Craters
        hmap = self._generate_craters(hmap, num_craters=150)
        
        # 3. Normalize and quantize
        hmap = np.clip(hmap, 0, 1)
        self.heightmap = hmap
        self.height_steps = (self.heightmap * self.HEIGHT_LEVELS).astype(np.int32)
        self.style_idx = self._build_style_map()

    def _build_heightmap(self):
            # 1. Initialize empty map
            hmap = np.zeros((self.height, self.width), dtype=np.float32)
            
            # 2. Layer multiple "Octaves" of noise
            # Lower octaves = big mountains; Higher octaves = small rocks/craters
            octaves = 8
            persistence = 0.5  # How much detail is added each layer
            lacunarity = 2.0   # How much the frequency increases each layer
            
            amplitude = 1.0
            frequency = 0.005 # Base scale
            
            for i in range(octaves):
                # We use a randomized sine-wave composition to simulate Perlin noise
                # Adding a random phase shift makes every seed unique
                phase_x = self.rng.uniform(0, 1000)
                phase_y = self.rng.uniform(0, 1000)
                
                layer = np.sin((self.gx * frequency) + phase_x) * \
                        np.cos((self.gy * frequency) + phase_y)
                
                hmap += layer * amplitude
                
                # Prepare next octave
                amplitude *= persistence
                frequency *= lacunarity

            # 3. Normalize to 0.0 - 1.0 range
            hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())

            # 4. THE LUNAR TOUCH: Apply a power function
            # This pushes middle-ground values down, creating wide flat plains 
            # and leaving only the highest points as sharp peaks.
            hmap = np.power(hmap, 2.0) 

            return hmap

    def _generate_craters(self, hmap, num_craters):
        """Carves circular depressions with raised rims into the map."""
        for _ in range(num_craters):
            # Random crater stats
            cx = self.rng.integers(0, self.width)
            cy = self.rng.integers(0, self.height)
            radius = self.rng.uniform(2, 15)
            depth = self.rng.uniform(0.1, 0.3)

            # Calculate distance from crater center for every point in the map
            # (Optimized via NumPy)
            dist_sq = (self.gx - cx)**2 + (self.gy - cy)**2
            dist = np.sqrt(dist_sq)
            
            # The Crater Formula:
            # Inside the radius, it dips. At the radius, it peaks (the rim).
            # We use a smoothstep-like function for the bowl and a Gaussian for the rim.
            mask = dist <= radius * 2
            if not np.any(mask): continue
            
            # Normalize distance relative to radius
            r_dist = dist[mask] / radius
            
            # Crater Bowl (parabolic dip)
            bowl = np.where(r_dist < 1.0, -depth * (1 - r_dist**2), 0)
            
            # Crater Rim (sharp peak at r=1.0)
            rim = 0.15 * depth * np.exp(-5 * (r_dist - 1.0)**2)
            
            hmap[mask] += (bowl + rim)
            
        return hmap

    def _build_style_map(self):
        h = self.heightmap
        idx = np.zeros((self.height, self.width), dtype=np.int32)
        # 0: Dust (Mid), 1: Highland (High), 2: Peak (Very High), 3: Basalt/Maria (Low)
        idx[h > 0.55] = 1
        idx[h > 0.80] = 2
        idx[h < 0.30] = 3
        return idx

# ══════════════════════════════════════════════════════════════════════════════
# WORLD RENDERER (Numpy Accelerated)
# ══════════════════════════════════════════════════════════════════════════════
class WorldRenderer:
    def __init__(self, world, camera):
        self.world   = world
        self.camera  = camera
        self._tiles  = self._build_flat_tiles()
        self._star_bg = self._build_star_background(SCREEN_W, SCREEN_H)
        self._overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)

    def _build_flat_tiles(self):
        return [self._make_tile_surface(COL_TILE_TOP[i], COL_TILE_LEFT[i], COL_TILE_RIGHT[i]) 
                for i in range(len(COL_TILE_TOP))]

    @staticmethod
    def _make_tile_surface(top_col, left_col, right_col):
        surf = pygame.Surface((TILE_W, TILE_H + TILE_DEPTH), pygame.SRCALPHA)
        tw, th = TILE_W, TILE_H
        pygame.draw.polygon(surf, top_col, [(tw//2, 0), (tw, th//2), (tw//2, th), (0, th//2)])
        pygame.draw.polygon(surf, left_col, [(0, th//2), (tw//2, th), (tw//2, th+TILE_DEPTH), (0, th//2+TILE_DEPTH)])
        pygame.draw.polygon(surf, right_col, [(tw//2, th), (tw, th//2), (tw, th//2+TILE_DEPTH), (tw//2, th+TILE_DEPTH)])
        return surf

    @staticmethod
    def _build_star_background(w, h):
        surf = pygame.Surface((w, h))
        surf.fill(COL_SKY)
        rng = np.random.default_rng(99)
        for _ in range(400):
            x, y = rng.integers(0, w), rng.integers(0, h)
            col = rng.integers(150, 255)
            surf.set_at((x, y), COL_SKY)
        return surf

    def render(self, screen, hovered):
        screen.blit(self._star_bg, (0, 0))
        self._overlay.fill((0, 0, 0, 0))

        # Vectorized Coordinate Projection
        # (gx - gy) * half_w + offset
        sx = (self.world.gx - self.world.gy) * (TILE_W // 2) + self.camera.ox
        # (gx + gy) * half_h - depth + offset
        sy = (self.world.gx + self.world.gy) * (TILE_H // 2) - (self.world.height_steps * TILE_DEPTH) + self.camera.oy

        # Frustum Culling: Create a boolean mask of tiles inside the screen
        margin = TILE_W * 2
        visible_mask = (sx > -margin) & (sx < SCREEN_W + margin) & \
                       (sy > -margin) & (sy < SCREEN_H + margin)

        # Draw only visible tiles (Painter's algorithm: iterate Y then X)
        visible_indices = np.argwhere(visible_mask)
        
        for y, x in visible_indices:
            sidx = self.world.style_idx[y, x]
            # Offset tile by half width to center it on the coordinate
            screen.blit(self._tiles[sidx], (sx[y, x] - TILE_W // 2, sy[y, x]))

            if (x, y) == hovered:
                self._draw_hover(sx[y, x], sy[y, x], self.world.height_steps[y, x])

        screen.blit(self._overlay, (0, 0))

    def _draw_hover(self, sx, sy, hz):
        pts = [(sx, sy + TILE_H//2), (sx + TILE_W//2, sy), (sx, sy - TILE_H//2), (sx - TILE_W//2, sy)]
        pygame.draw.polygon(self._overlay, (200, 230, 255, 150), pts, 2)

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA & INPUT
# ══════════════════════════════════════════════════════════════════════════════
class Camera:
    def __init__(self):
        self.ox, self.oy = SCREEN_W // 2, SCREEN_H // 4
        self.speed = 10

    def move(self, dx, dy):
        self.ox += dx
        self.oy += dy

    def screen_to_world(self, sx, sy):
        # Inverse isometric projection
        rel_x, rel_y = sx - self.ox, sy - self.oy
        gx = (rel_x / (TILE_W // 2) + rel_y / (TILE_H // 2)) / 2
        gy = (rel_y / (TILE_H // 2) - rel_x / (TILE_W // 2)) / 2
        return int(round(gx)), int(round(gy))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 14, bold=True)

    world    = WorldGenerator(MAP_W, MAP_H)
    camera   = Camera()
    renderer = WorldRenderer(world, camera)

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # Camera Pan
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: camera.move(camera.speed, 0)
        if keys[pygame.K_d]: camera.move(-camera.speed, 0)
        if keys[pygame.K_w]: camera.move(0, camera.speed)
        if keys[pygame.K_s]: camera.move(0, -camera.speed)

        # Mouse Interaction
        mx, my = pygame.mouse.get_pos()
        hovered = camera.screen_to_world(mx, my)

        renderer.render(screen, hovered)
        
        # UI
        fps_text = font.render(f"FPS: {int(clock.get_fps())} | Pos: {hovered}", True, (0, 255, 0))
        screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()