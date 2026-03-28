import pygame
import numpy as np
from random import randint

# ── Config ───────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1280, 720
TILE_W,   TILE_H   = 48, 24       # Standard tile size
TILE_DEPTH         = 5
MAP_W,    MAP_H    = 400, 400     # 400x400 Grid
FPS                = 60

# Moon colour palette
COL_SKY        = (5, 8, 20)
# Index 0: Dust, 1: Highlands, 2: Peaks, 3: Maria (Basalt)
COL_TILE_TOP   = [(80, 85, 100), (95, 100, 115), (120, 125, 140), (40, 42, 50)]
COL_TILE_LEFT  = [(45, 48, 60), (38, 42, 55), (50, 54, 65), (35, 40, 52)]
COL_TILE_RIGHT = [(30, 33, 45), (25, 28, 38), (35, 38, 48), (22, 26, 35)]

# Object Types
OBJ_EMPTY = 0
OBJ_ROCK_SMALL = 1  # Rover can drive over
OBJ_ROCK_LARGE = 2  # Obstacle

# ══════════════════════════════════════════════════════════════════════════════
# WORLD GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class WorldGenerator:
    HEIGHT_LEVELS = 50 

    def __init__(self, width, height, seed=None):
        self.width  = width
        self.height = height
        self.seed   = seed if seed is not None else randint(0, 100000)
        self.rng    = np.random.default_rng(self.seed)

        self.gx, self.gy = np.meshgrid(np.arange(width), np.arange(height))
        
        # 1. Base Perlin Noise
        hmap = self._build_heightmap()
        
        # 2. Add Craters
        hmap = self._add_craters(hmap, num_craters=300)
        
        # 3. Finalize Height
        hmap = np.clip(hmap, 0, 1)
        self.heightmap = hmap
        self.height_steps = (self.heightmap * self.HEIGHT_LEVELS).astype(np.int32)
        
        # 4. Generate Styles and Object Matrix
        self.style_idx = self._build_style_map()
        self.object_map = self._generate_objects()

    def _build_heightmap(self):
        hmap = np.zeros((self.height, self.width), dtype=np.float32)
        octaves = 8
        persistence = 0.5
        lacunarity = 2.0
        amplitude = 1.0
        frequency = 0.005 
        
        for i in range(octaves):
            phase_x = self.rng.uniform(0, 1000)
            phase_y = self.rng.uniform(0, 1000)
            layer = np.sin((self.gx * frequency) + phase_x) * \
                    np.cos((self.gy * frequency) + phase_y)
            hmap += layer * amplitude
            amplitude *= persistence
            frequency *= lacunarity

        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
        return np.power(hmap, 2.0) # Creates the flat Maria basins

    def _add_craters(self, hmap, num_craters):
        for _ in range(num_craters):
            cx, cy = self.rng.integers(0, self.width), self.rng.integers(0, self.height)
            radius = self.rng.uniform(3, 18)
            depth = self.rng.uniform(0.1, 0.3)

            x0, x1 = max(0, int(cx - radius*2)), min(self.width, int(cx + radius*2))
            y0, y1 = max(0, int(cy - radius*2)), min(self.height, int(cy + radius*2))
            
            lx, ly = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            dist = np.sqrt((lx - cx)**2 + (ly - cy)**2)
            r_dist = dist / radius

            floor = np.where(r_dist < 1.0, -depth * (1 - r_dist**2), 0)
            rim = (depth * 0.4) * np.exp(-10 * (r_dist - 1.0)**2)
            hmap[y0:y1, x0:x1] += (floor + rim)
        return hmap

    def _generate_objects(self):
        obj_map = np.zeros((self.height, self.width), dtype=np.int32)
        # Scatter rocks based on terrain height
        for _ in range(3000):
            rx, ry = self.rng.integers(0, self.width), self.rng.integers(0, self.height)
            h = self.heightmap[ry, rx]
            if 0.3 < h < 0.8: # Likely highland dust areas
                obj_map[ry, rx] = OBJ_ROCK_SMALL if self.rng.random() > 0.15 else OBJ_ROCK_LARGE
        return obj_map

    def _build_style_map(self):
        h = self.heightmap
        idx = np.zeros((self.height, self.width), dtype=np.int32)
        idx[h > 0.40] = 1 # Dust
        idx[h > 0.75] = 2 # Highlands
        idx[h < 0.15] = 3 # Maria
        return idx

# ══════════════════════════════════════════════════════════════════════════════
# WORLD RENDERER
# ══════════════════════════════════════════════════════════════════════════════
class WorldRenderer:
    def __init__(self, world, camera):
        self.world = world
        self.camera = camera
        self._base_tiles = self._build_flat_tiles() 
        self._scaled_tiles = []
        self._last_zoom = -1.0
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

    def _build_star_background(self, w, h):
        surf = pygame.Surface((w, h))
        surf.fill(COL_SKY)
        rng = np.random.default_rng(99)
        for _ in range(400):
            x, y = rng.integers(0, w), rng.integers(0, h)
            val = rng.integers(150, 255)
            surf.set_at((x, y), COL_SKY)
        return surf

    def render(self, screen, hovered):
        screen.blit(self._star_bg, (0, 0))
        self._overlay.fill((0, 0, 0, 0))

        curr_zoom = self.camera.zoom
        zw, zh, zd = int(TILE_W * curr_zoom), int(TILE_H * curr_zoom), int(TILE_DEPTH * curr_zoom)

        if curr_zoom != self._last_zoom:
            self._scaled_tiles = [pygame.transform.scale(t, (zw, zh + zd)) for t in self._base_tiles]
            self._last_zoom = curr_zoom

        sx = (self.world.gx - self.world.gy) * (zw // 2) + self.camera.ox
        sy = (self.world.gx + self.world.gy) * (zh // 2) - (self.world.height_steps * zd) + self.camera.oy

        margin = zw * 2
        visible_mask = (sx > -margin) & (sx < SCREEN_W + margin) & (sy > -margin) & (sy < SCREEN_H + margin)
        indices = np.argwhere(visible_mask)
        
        for y, x in indices:
            # 1. Draw Tile
            sidx = self.world.style_idx[y, x]
            screen.blit(self._scaled_tiles[sidx], (sx[y, x] - zw // 2, sy[y, x]))

            # 2. Draw Rocks
            obj = self.world.object_map[y, x]
            if obj != OBJ_EMPTY:
                self._draw_rock(screen, sx[y, x], sy[y, x], zw, zh, obj)

            # 3. Hover
            if (x, y) == hovered:
                self._draw_hover(sx[y, x], sy[y, x], zw, zh)

        screen.blit(self._overlay, (0, 0))

    def _draw_rock(self, screen, sx, sy, zw, zh, obj_type):
        color = (100, 105, 120) if obj_type == OBJ_ROCK_SMALL else (70, 75, 90)
        size = 0.2 if obj_type == OBJ_ROCK_SMALL else 0.45
        rw, rh = zw * size, zh * size
        bx, by = sx, sy + (zh // 4)
        pts = [(bx - rw//2, by), (bx, by - rh), (bx + rw//2, by + rh//4), (bx, by + rh//2)]
        pygame.draw.polygon(screen, color, pts)
        pygame.draw.polygon(screen, (10, 10, 20), pts, 1)

    def _draw_hover(self, sx, sy, zw, zh):
        pts = [(sx, sy + zh//2), (sx + zw//2, sy), (sx, sy - zh//2), (sx - zw//2, sy)]
        pygame.draw.polygon(self._overlay, (200, 230, 255, 150), pts, 2)

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA
# ══════════════════════════════════════════════════════════════════════════════
class Camera:
    def __init__(self):
        self.ox, self.oy = SCREEN_W // 2, SCREEN_H // 4
        self.speed = 12
        self.zoom = 1.0
        self.min_zoom, self.max_zoom = 0.3, 4.0

    def move(self, dx, dy):
        self.ox += dx
        self.oy += dy

    def adjust_zoom(self, delta, center):
        old_zoom = self.zoom
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom + delta))
        ratio = self.zoom / old_zoom
        self.ox = center[0] - (center[0] - self.ox) * ratio
        self.oy = center[1] - (center[1] - self.oy) * ratio

    def screen_to_world(self, sx, sy):
        zw, zh = TILE_W * self.zoom, TILE_H * self.zoom
        rel_x, rel_y = sx - self.ox, sy - self.oy
        gx = (rel_x / (zw / 2) + rel_y / (zh / 2)) / 2
        gy = (rel_y / (zh / 2) - rel_x / (zw / 2)) / 2
        return int(round(gx)), int(round(gy))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def getWorld():
    return WorldGenerator(MAP_W,MAP_H)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Lunar Rover Simulation World")
    clock, font = pygame.time.Clock(), pygame.font.SysFont("monospace", 14, bold=True)

    world = WorldGenerator(MAP_W, MAP_H)
    camera = Camera()
    renderer = WorldRenderer(world, camera)

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4: camera.adjust_zoom(0.1, pygame.mouse.get_pos())
                if event.button == 5: camera.adjust_zoom(-0.1, pygame.mouse.get_pos())

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: camera.move(camera.speed, 0)
        if keys[pygame.K_d]: camera.move(-camera.speed, 0)
        if keys[pygame.K_w]: camera.move(0, camera.speed)
        if keys[pygame.K_s]: camera.move(0, -camera.speed)

        mx, my = pygame.mouse.get_pos()
        hovered = camera.screen_to_world(mx, my)

        renderer.render(screen, hovered)
        
        # Simple UI
        ui_bg = pygame.Surface((300, 60), pygame.SRCALPHA)
        ui_bg.fill((0, 0, 0, 150))
        screen.blit(ui_bg, (0, 0))
        
        info = f"FPS: {int(clock.get_fps())} | Zoom: {camera.zoom:.1f}"
        pos_info = f"Map Pos: {hovered}"
        screen.blit(font.render(info, True, (0, 255, 0)), (10, 10))
        screen.blit(font.render(pos_info, True, (0, 255, 0)), (10, 30))
        
        pygame.display.flip()
    pygame.quit()



if __name__ == "__main__":
    main()