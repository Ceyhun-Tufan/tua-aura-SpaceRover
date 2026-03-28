import sys
import os

# This adds the root directory of your project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
from random import randint
from pathFinding.pathFinding import astar

# --- Config (Same as yours) ---
SCREEN_W, SCREEN_H = 1920, 1080
TILE_W,   TILE_H   = 48, 24
TILE_DEPTH         = 5
MAP_W,    MAP_H    = 400, 400 
FPS                = 60

COL_SKY        = (5, 8, 20)
COL_TILE_TOP   = [(80, 85, 100), (95, 100, 115), (120, 125, 140), (40, 42, 50)]
COL_TILE_LEFT  = [(45, 48, 60), (38, 42, 55), (50, 54, 65), (35, 40, 52)]
COL_TILE_RIGHT = [(30, 33, 45), (25, 28, 38), (35, 38, 48), (22, 26, 35)]

OBJ_EMPTY, OBJ_ROCK_SMALL, OBJ_ROCK_LARGE = 0, 1, 2

# 

class WorldGenerator:
    HEIGHT_LEVELS = 50 
    def __init__(self, width, height, seed=None):
        self.width, self.height = width, height
        self.seed = seed if seed is not None else randint(0, 100000)
        self.rng = np.random.default_rng(self.seed)
        self.gx, self.gy = np.meshgrid(np.arange(width), np.arange(height))
        hmap = self._build_heightmap()
        hmap = self._add_craters(hmap, num_craters=300)
        self.heightmap = np.clip(hmap, 0, 1)
        self.height_steps = (self.heightmap * self.HEIGHT_LEVELS).astype(np.int32)
        self.style_idx = self._build_style_map()
        self.object_map = self._generate_objects()

    def _build_heightmap(self):
        hmap = np.zeros((self.height, self.width), dtype=np.float32)
        oct, pers, lac, amp, freq = 8, 0.5, 2.0, 1.0, 0.005
        for _ in range(oct):
            px, py = self.rng.uniform(0, 1000), self.rng.uniform(0, 1000)
            hmap += (np.sin((self.gx * freq) + px) * np.cos((self.gy * freq) + py)) * amp
            amp *= pers; freq *= lac
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
        return np.power(hmap, 2.0)

    def _add_craters(self, hmap, num_craters):
        for _ in range(num_craters):
            cx, cy = self.rng.integers(0, self.width), self.rng.integers(0, self.height)
            rad, dep = self.rng.uniform(3, 18), self.rng.uniform(0.1, 0.3)
            x0, x1 = max(0, int(cx-rad*2)), min(self.width, int(cx+rad*2))
            y0, y1 = max(0, int(cy-rad*2)), min(self.height, int(cy+rad*2))
            lx, ly = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            dist = np.sqrt((lx-cx)**2 + (ly-cy)**2) / rad
            hmap[y0:y1, x0:x1] += np.where(dist < 1.0, -dep * (1 - dist**2), 0) + (dep*0.4)*np.exp(-10*(dist-1.0)**2)
        return hmap

    def _generate_objects(self):
        obj_map = np.zeros((self.height, self.width), dtype=np.int32)
        for _ in range(3000):
            rx, ry = self.rng.integers(0, self.width), self.rng.integers(0, self.height)
            if 0.3 < self.heightmap[ry, rx] < 0.8:
                obj_map[ry, rx] = OBJ_ROCK_SMALL if self.rng.random() > 0.15 else OBJ_ROCK_LARGE
        return obj_map

    def _build_style_map(self):
        idx = np.zeros((self.height, self.width), dtype=np.int32)
        idx[self.heightmap > 0.40], idx[self.heightmap > 0.75], idx[self.heightmap < 0.15] = 1, 2, 3
        return idx

class Camera:
    def __init__(self):
        self.ox, self.oy = SCREEN_W // 2, SCREEN_H // 4
        self.speed, self.zoom = 12, 1.0
        self.min_zoom, self.max_zoom = 0.3, 4.0

    def move(self, dx, dy): self.ox += dx; self.oy += dy
    def adjust_zoom(self, delta, center):
        old = self.zoom
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom + delta))
        ratio = self.zoom / old
        self.ox = center[0] - (center[0] - self.ox) * ratio
        self.oy = center[1] - (center[1] - self.oy) * ratio

    def screen_to_world(self, sx, sy):
        zw, zh = TILE_W * self.zoom, TILE_H * self.zoom
        rx, ry = sx - self.ox, sy - self.oy
        gx = (rx / (zw/2) + ry / (zh/2)) / 2
        gy = (ry / (zh/2) - rx / (zw/2)) / 2
        return int(round(gx)), int(round(gy))

class WorldRenderer:
    def __init__(self, world, camera):
        self.world, self.camera = world, camera
        self._base_tiles = [self._make_tile(COL_TILE_TOP[i], COL_TILE_LEFT[i], COL_TILE_RIGHT[i]) for i in range(4)]
        self._scaled_tiles, self._last_zoom = [], -1.0
        self._star_bg = pygame.Surface((SCREEN_W, SCREEN_H)); self._star_bg.fill(COL_SKY)
        self._overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)

    def _make_tile(self, t, l, r):
        s = pygame.Surface((TILE_W, TILE_H + TILE_DEPTH), pygame.SRCALPHA)
        pygame.draw.polygon(s, t, [(TILE_W//2, 0), (TILE_W, TILE_H//2), (TILE_W//2, TILE_H), (0, TILE_H//2)])
        pygame.draw.polygon(s, l, [(0, TILE_H//2), (TILE_W//2, TILE_H), (TILE_W//2, TILE_H+5), (0, TILE_H//2+5)])
        pygame.draw.polygon(s, r, [(TILE_W//2, TILE_H), (TILE_W, TILE_H//2), (TILE_W, TILE_H//2+5), (TILE_W//2, TILE_H+5)])
        return s

    def render(self, screen, hovered, path, start_node, end_node):
        screen.blit(self._star_bg, (0, 0))
        self._overlay.fill((0, 0, 0, 0))
        
        z = self.camera.zoom
        zw, zh, zd = int(TILE_W*z), int(TILE_H*z), int(TILE_DEPTH*z)
        
        if z != self._last_zoom:
            self._scaled_tiles = [pygame.transform.scale(t, (zw, zh+zd)) for t in self._base_tiles]
            self._last_zoom = z

        # Calculate all screen positions at once
        sx = (self.world.gx - self.world.gy) * (zw//2) + self.camera.ox
        sy = (self.world.gx + self.world.gy) * (zh//2) - (self.world.height_steps * zd) + self.camera.oy
        
        vis = (sx > -zw*2) & (sx < SCREEN_W+zw*2) & (sy > -zh*2) & (sy < SCREEN_H+zh*2)
        path_set = set(path) if path else set()

        for y, x in np.argwhere(vis):
            # 1. Draw Tile
            screen.blit(self._scaled_tiles[self.world.style_idx[y,x]], (sx[y,x]-zw//2, sy[y,x]))
            
            # 2. Draw Path Trail (Subtle)
            if (x, y) in path_set and (x, y) != start_node and (x, y) != end_node:
                self._draw_path_trail(sx[y,x], sy[y,x], zw, zh)

            # 3. Draw Objects (Rocks)
            if self.world.object_map[y,x] != OBJ_EMPTY:
                self._draw_rock(screen, sx[y,x], sy[y,x], zw, zh, self.world.object_map[y,x])

            # 4. Draw Start/End Holograms (Drawn last so they appear "on top")
            if (x, y) == start_node:
                self._draw_marker(sx[y,x], sy[y,x], zw, zh, (0, 255, 150), "START")
            elif (x, y) == end_node:
                self._draw_marker(sx[y,x], sy[y,x], zw, zh, (255, 50, 50), "TARGET")

        # 5. Draw Connected Path Line
        if path and len(path) > 1:
            self._draw_connected_path(path, zw, zh, zd)

        screen.blit(self._overlay, (0, 0))

    def _draw_rock(self, screen, sx, sy, zw, zh, obj_type):
        color = (100,105,120) if obj_type==1 else (70,75,90)
        sz = 0.2 if obj_type==1 else 0.45
        rw, rh = zw*sz, zh*sz
        bx, by = sx, sy + zh//4
        pts = [(bx-rw//2, by), (bx, by-rh), (bx+rw//2, by+rh//4), (bx, by+rh//2)]
        pygame.draw.polygon(screen, color, pts); pygame.draw.polygon(screen, (10,10,20), pts, 1)

    def _draw_node(self, sx, sy, zw, zh, color, label=""):
        # Create a pulsing effect using pygame.time
        pulse = (np.sin(pygame.time.get_ticks() * 0.005) + 1) * 5
        
        # 1. Draw a vertical beacon line
        pygame.draw.line(self._overlay, (*color, 150), (sx, sy), (sx, sy - 40 - pulse), 2)
        
        # 2. Draw the base diamond
        pts = [(sx, sy + zh//2), (sx + zw//2, sy), (sx, sy - zh//2), (sx - zw//2, sy)]
        pygame.draw.polygon(self._overlay, (*color, 100), pts) # Semi-transparent fill
        pygame.draw.polygon(self._overlay, (*color, 255), pts, 3) # Bright border
        
        # 3. Floating diamond top
        offset = 20 + pulse
        pts_float = [(sx, sy + zh//2 - offset), (sx + zw//2, sy - offset), 
                     (sx, sy - zh//2 - offset), (sx - zw//2, sy - offset)]
        pygame.draw.polygon(self._overlay, (*color, 200), pts_float, 2)

    def _draw_marker(self, sx, sy, zw, zh, color, label):
        """Draws a pulsing holographic diamond with a vertical beacon."""
        t = pygame.time.get_ticks() * 0.005
        pulse = (np.sin(t) + 1) * 4  # Gentle vertical bobbing
        
        # Glow base
        pts = [(sx, sy + zh//2), (sx + zw//2, sy), (sx, sy - zh//2), (sx - zw//2, sy)]
        pygame.draw.polygon(self._overlay, (*color, 60), pts)
        pygame.draw.polygon(self._overlay, (*color, 200), pts, 2)
        
        # Floating "Hologram" Diamond
        off = 15 + pulse
        float_pts = [(sx, sy + zh//2 - off), (sx + zw//2, sy - off), 
                     (sx, sy - zh//2 - off), (sx - zw//2, sy - off)]
        pygame.draw.polygon(self._overlay, (*color, 255), float_pts, 2)
        
        # Vertical Beacon Line
        pygame.draw.line(self._overlay, (*color, 100), (sx, sy), (sx, sy - 30 - pulse), 1)

    def _draw_path_trail(self, sx, sy, zw, zh):
        """Draws a small glowing dot for the path trail."""
        pygame.draw.circle(self._overlay, (255, 255, 0, 80), (sx, sy), 3)

    def _draw_connected_path(self, path, zw, zh, zd):
        """Draws a continuous line through the path coordinates."""
        points = []
        for px, py in path:
            psx = (px - py) * (zw // 2) + self.camera.ox
            psy = (px + py) * (zh // 2) - (self.world.height_steps[py, px] * zd) + self.camera.oy
            points.append((psx, psy))
        
        if len(points) > 1:
            # Draw a thick semi-transparent line
            pygame.draw.lines(self._overlay, (255, 255, 0, 150), False, points, 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock, font = pygame.time.Clock(), pygame.font.SysFont("monospace", 14, bold=True)

    world = WorldGenerator(MAP_W, MAP_H)
    camera = Camera()
    renderer = WorldRenderer(world, camera)

    start_node = None
    end_node = None
    path = []

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                grid_pos = camera.screen_to_world(mx, my)
                
                if event.button == 1: # Left Click: Start
                    start_node = grid_pos
                    path = [] 
                if event.button == 3: # Right Click: End
                    end_node = grid_pos
                    # Calculate Path
                    if start_node and end_node:
                        # Convert numpy grid to list for the astar function
                        grid_data = world.height_steps.tolist()
                        path = astar(grid_data, start_node, end_node)
                
                if event.button == 4: camera.adjust_zoom(0.1, (mx, my))
                if event.button == 5: camera.adjust_zoom(-0.1, (mx, my))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: camera.move(camera.speed, 0)
        if keys[pygame.K_d]: camera.move(-camera.speed, 0)
        if keys[pygame.K_w]: camera.move(0, camera.speed)
        if keys[pygame.K_s]: camera.move(0, -camera.speed)

        mx, my = pygame.mouse.get_pos()
        hovered = camera.screen_to_world(mx, my)

        renderer.render(screen, hovered, path, start_node, end_node)
        
        # UI
        path_len = len(path) if path is not None else 0

        ui = [
            f"FPS: {int(clock.get_fps())}", 
            f"Zoom: {camera.zoom:.1f}", 
            f"Start: {start_node}", 
            f"End: {end_node}", 
            f"Path Length: {path_len}"
        ]
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()