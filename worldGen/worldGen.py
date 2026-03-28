import sys
import os

# This adds the root directory of your project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
from random import randint
from pathFinding.pathFinding import astar, dijkstra, get_straight_line, calculate_path_cost
from rover.rover import Rover

# --- Config (Same as yours) ---
SCREEN_W, SCREEN_H = 1920, 1080
TILE_W,   TILE_H   = 48, 24
TILE_DEPTH         = 5
MAP_W,    MAP_H    = 400, 400 
NUM_ROCKS          = 10000
FPS                = 60

COL_SKY        = (5, 8, 20)
COL_TILE_TOP   = [(80, 85, 100), (95, 100, 115), (120, 125, 140), (40, 42, 50)]
COL_TILE_LEFT  = [(45, 48, 60), (38, 42, 55), (50, 54, 65), (35, 40, 52)]
COL_TILE_RIGHT = [(30, 33, 45), (25, 28, 38), (35, 38, 48), (22, 26, 35)]

OBJ_EMPTY, OBJ_ROCK_SMALL, OBJ_ROCK_LARGE = 0, 1, 2

# 

class WorldGenerator:
    HEIGHT_LEVELS = 70
    def __init__(self, width, height, seed=None):
        self.width, self.height = width, height
        self.seed = seed if seed is not None else randint(0, 100000)
        self.rng = np.random.default_rng(self.seed)
        self.gx, self.gy = np.meshgrid(np.arange(width), np.arange(height))
        hmap = self._build_heightmap()
        hmap = self._add_craters(hmap, num_craters=800)
        self.heightmap = np.clip(hmap, 0, 1)
        self.height_steps = (self.heightmap * self.HEIGHT_LEVELS).astype(np.int32)
        self.roughness_map = self._build_roughness_map()
        self.style_idx = self._build_style_map()
        self.object_map = self._generate_objects()
        self.known_object_map = np.zeros((self.height, self.width), dtype=np.int32)

    def _build_heightmap(self):
        hmap = np.zeros((self.height, self.width), dtype=np.float32)
        oct, pers, lac, amp, freq = 8, 0.5, 2.0, 1.0, 0.004
        for _ in range(oct):
            px, py = self.rng.uniform(0, 1000), self.rng.uniform(0, 1000)
            hmap += (np.sin((self.gx * freq) + px) * np.cos((self.gy * freq) + py)) * amp
            amp *= pers; freq *= lac
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
        # Power of 2.2 creates rolling lunar highlands and flatter maria
        return np.power(hmap, 2.2)

    def _build_roughness_map(self):
        rmap = np.zeros((self.height, self.width), dtype=np.float32)
        # Higher frequency creates distinct localized patches of soft/rough terrain
        oct, pers, lac, amp, freq = 6, 0.5, 2.0, 1.0, 0.015
        for _ in range(oct):
            px, py = self.rng.uniform(0, 1000), self.rng.uniform(0, 1000)
            rmap += (np.sin((self.gx * freq) + px) * np.cos((self.gy * freq) + py)) * amp
            amp *= pers; freq *= lac
        rmap = (rmap - rmap.min()) / (rmap.max() - rmap.min())
        return rmap

    def _add_craters(self, hmap, num_craters):
        for _ in range(num_craters):
            cx, cy = self.rng.integers(0, self.width), self.rng.integers(0, self.height)
            # Use exponential distribution: lots of small craters, very few massive ones
            rad = np.clip(self.rng.exponential(scale=5.0), 3, 30)
            dep = rad * self.rng.uniform(0.015, 0.035) 
            
            x0, x1 = max(0, int(cx-rad*2)), min(self.width, int(cx+rad*2))
            y0, y1 = max(0, int(cy-rad*2)), min(self.height, int(cy+rad*2))
            lx, ly = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            dist = np.sqrt((lx-cx)**2 + (ly-cy)**2) / rad
            # Excavate crater bowl and add a localized raised rim
            hmap[y0:y1, x0:x1] += np.where(dist < 1.0, -dep * (1 - dist**2), 0) + (dep*0.5)*np.exp(-10*(dist-1.0)**2)
        return hmap

    def _generate_objects(self):
        obj_map = np.zeros((self.height, self.width), dtype=np.int32)
        for _ in range(NUM_ROCKS):
            rx, ry = self.rng.integers(0, self.width), self.rng.integers(0, self.height)
            if 0.3 < self.heightmap[ry, rx] < 0.8:
                obj_map[ry, rx] = OBJ_ROCK_SMALL if self.rng.random() > 0.15 else OBJ_ROCK_LARGE
        return obj_map

    def _build_style_map(self):
        idx = np.zeros((self.height, self.width), dtype=np.int32)
        idx[self.heightmap > 0.40] = 1
        idx[self.heightmap > 0.75] = 2
        idx[self.heightmap < 0.15] = 3
        
        # If roughness > 0.6, shift to rough texture variations (+4 shift)
        idx[self.roughness_map > 0.6] += 4
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

    def screen_to_world(self, mx, my, world):
        z = self.zoom
        zw, zh, zd = int(TILE_W*z), int(TILE_H*z), int(TILE_DEPTH*z)
        
        # Calculate screen center coordinates for all tiles
        sx = (world.gx - world.gy) * (zw//2) + self.ox
        sy = (world.gx + world.gy) * (zh//2) - (world.height_steps * zd) + self.oy
        
        # The true top face center is offset by zh//2
        cx = sx
        cy = sy + zh//2
        
        # Check inside the isometric diamond mathematically
        dx = np.abs(mx - cx) / (zw/2)
        dy = np.abs(my - cy) / (zh/2)
        inside = (dx + dy) <= 1.0
        
        valid_indices = np.argwhere(inside)
        if len(valid_indices) > 0:
            # Pick the tile rendered last (front-most)
            best = valid_indices[-1]
            return int(best[1]), int(best[0]) # (x, y)
            
        # Fallback to flat Z=0 plane if clicking outside any valid raised face
        rx, ry = mx - self.ox, my - self.oy
        gx = (rx / (zw/2) + ry / (zh/2)) / 2
        gy = (ry / (zh/2) - rx / (zw/2)) / 2
        return int(round(gx)), int(round(gy))

class WorldRenderer:
    def __init__(self, world, camera):
        self.world, self.camera = world, camera
        
        self._base_tiles = []
        # Index 0-3: Smooth Textures
        for i in range(4):
            self._base_tiles.append(self._make_tile(COL_TILE_TOP[i], COL_TILE_LEFT[i], COL_TILE_RIGHT[i], is_rough=False))
        # Index 4-7: Rough/Gritty Textures
        for i in range(4):
            self._base_tiles.append(self._make_tile(COL_TILE_TOP[i], COL_TILE_LEFT[i], COL_TILE_RIGHT[i], is_rough=True))
            
        self._scaled_tiles, self._last_zoom = [], -1.0
        self._scaled_rocks = {}
        self._star_bg = pygame.Surface((SCREEN_W, SCREEN_H)); self._star_bg.fill(COL_SKY)
        self._overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("monospace", 14, bold=True)

    def _make_tile(self, t, l, r, is_rough=False):
        s = pygame.Surface((TILE_W, TILE_H + TILE_DEPTH), pygame.SRCALPHA)
        pygame.draw.polygon(s, t, [(TILE_W//2, 0), (TILE_W, TILE_H//2), (TILE_W//2, TILE_H), (0, TILE_H//2)])
        
        if is_rough:
            import random
            for _ in range(15):
                dx = random.randint(TILE_W//4, TILE_W - TILE_W//4)
                dy = random.randint(TILE_H//4, TILE_H - TILE_H//4)
                pygame.draw.line(s, (20, 20, 25, 120), (dx, dy), (dx+1, dy))
                
        pygame.draw.polygon(s, l, [(0, TILE_H//2), (TILE_W//2, TILE_H), (TILE_W//2, TILE_H+5), (0, TILE_H//2+5)])
        pygame.draw.polygon(s, r, [(TILE_W//2, TILE_H), (TILE_W, TILE_H//2), (TILE_W, TILE_H//2+5), (TILE_W//2, TILE_H+5)])
        return s

    def _make_rock_surf(self, zw, zh, obj_type):
        surf = pygame.Surface((zw, zh), pygame.SRCALPHA)
        color = (100,105,120) if obj_type==1 else (70,75,90)
        sz = 0.2 if obj_type==1 else 0.45
        rw, rh = zw*sz, zh*sz
        bx, by = zw//2, zh//2 + zh//4
        pts = [(bx-rw//2, by), (bx, by-rh), (bx+rw//2, by+rh//4), (bx, by+rh//2)]
        pygame.draw.polygon(surf, color, pts); pygame.draw.polygon(surf, (10,10,20), pts, 1)
        return surf

    def render(self, screen, hovered, path, start_node, end_node, rover=None, straight_path=None, dijkstra_path=None, path_cost=0.0, straight_cost=0.0, dijkstra_cost=0.0):
        screen.blit(self._star_bg, (0, 0))
        self._overlay.fill((0, 0, 0, 0))
        
        z = self.camera.zoom
        zw, zh, zd = int(TILE_W*z), int(TILE_H*z), int(TILE_DEPTH*z)
        
        if z != self._last_zoom:
            self._scaled_tiles = [pygame.transform.scale(t, (zw, zh+zd)) for t in self._base_tiles]
            self._scaled_rocks = {
                1: self._make_rock_surf(zw, zh, 1),
                2: self._make_rock_surf(zw, zh, 2)
            }
            self._last_zoom = z

        # Calculate all screen positions at once
        sx = (self.world.gx - self.world.gy) * (zw//2) + self.camera.ox
        sy = (self.world.gx + self.world.gy) * (zh//2) - (self.world.height_steps * zd) + self.camera.oy
        
        vis = (sx > -zw*2) & (sx < SCREEN_W+zw*2) & (sy > -zh*2) & (sy < SCREEN_H+zh*2)
        path_set = set(path) if path else set()

        blit_seq = []

        # 1 & 2: Build bulk sequence for Tiles and Objects to draw simultaneously respecting Depth 
        for y, x in np.argwhere(vis):
            blit_seq.append((self._scaled_tiles[self.world.style_idx[y,x]], (sx[y,x]-zw//2, sy[y,x])))
            
            obj = self.world.known_object_map[y,x]
            if obj != 0: # OBJ_EMPTY is 0
                blit_seq.append((self._scaled_rocks[obj], (sx[y,x]-zw//2, sy[y,x]-zh//2)))
                
        # Fire bulk draw command (C-Accelerated, exponentially faster than Python loop blits)
        screen.blits(blit_seq)

        # Draw overlays (Paths, Holograms) separately
        for y, x in np.argwhere(vis):
            if (x, y) in path_set and (x, y) != start_node and (x, y) != end_node:
                self._draw_path_trail(sx[y,x], sy[y,x], zw, zh)

            if (x, y) == start_node:
                self._draw_marker(sx[y,x], sy[y,x], zw, zh, (0, 255, 150), "START")
            elif (x, y) == end_node:
                self._draw_marker(sx[y,x], sy[y,x], zw, zh, (255, 50, 50), "TARGET")

        # 5. Draw Connected Path Line (A*)
        if path and len(path) > 1:
            self._draw_connected_path(path, zw, zh, zd, color=(255, 255, 0, 150))
            
        # Draw Dijkstra Path
        if dijkstra_path and len(dijkstra_path) > 1:
            self._draw_connected_path(dijkstra_path, zw, zh, zd, color=(180, 50, 255, 180), thick=2)
            mid = dijkstra_path[max(0, len(dijkstra_path)//2 - 2)]
            msx = (mid[0] - mid[1]) * (zw//2) + self.camera.ox
            msy = (mid[0] + mid[1]) * (zh//2) - (self.world.height_steps[mid[1], mid[0]] * zd) + self.camera.oy
            self._draw_text(screen, f"Dijkstra: {dijkstra_cost:.0f}", msx, msy + 35, (180, 100, 255))

        # Draw Straight Line
        if straight_path and len(straight_path) > 1:
            self._draw_connected_path(straight_path, zw, zh, zd, color=(255, 50, 50, 200), thick=3)
            # Find midpoint for text
            mid = straight_path[len(straight_path)//2]
            msx = (mid[0] - mid[1]) * (zw//2) + self.camera.ox
            msy = (mid[0] + mid[1]) * (zh//2) - (self.world.height_steps[mid[1], mid[0]] * zd) + self.camera.oy
            self._draw_text(screen, f"Str Cost: {straight_cost:.0f}", msx, msy - 40, (255, 100, 100))
            
        # Draw Danger Signs for Dodged Obstacles
        if rover and hasattr(rover, 'dodged_obstacles'):
            for (dx, dy) in rover.dodged_obstacles:
                if 0 <= dy < self.world.height and 0 <= dx < self.world.width:
                    dsx = (dx - dy) * (zw//2) + self.camera.ox
                    dsy = (dx + dy) * (zh//2) - (self.world.height_steps[dy, dx] * zd) + self.camera.oy
                    self._draw_marker(dsx, dsy, zw, zh, (255, 100, 0), "!")

        if start_node and end_node:
            ex, ey = end_node
            esx = (ex - ey) * (zw//2) + self.camera.ox
            esy = (ex + ey) * (zh//2) - (self.world.height_steps[ey, ex] * zd) + self.camera.oy
            if path_cost > 0:
                self._draw_text(screen, f"Cost: {path_cost:.0f}", esx, esy - 65, (0, 255, 150))

        # 6. Draw Rover
        if rover:
            rsx = (rover.gx - rover.gy) * (zw // 2) + self.camera.ox
            rx_int, ry_int = int(round(rover.gx)), int(round(rover.gy))
            rover_zd = 0
            if 0 <= ry_int < self.world.height and 0 <= rx_int < self.world.width:
                rover_zd = self.world.height_steps[ry_int, rx_int] * zd
            
            rsy = (rover.gx + rover.gy) * (zh // 2) - rover_zd + self.camera.oy
            
            # Subtle radar cone visualization
            if rover.state == "MOVING" and rover.current_path:
                cone_pts = [(rsx, rsy)]
                look_limit = min(rover.target_index + rover.radar_range, len(rover.current_path))
                for i in range(rover.target_index, look_limit):
                    tx, ty = rover.current_path[i]
                    tsx = (tx - ty) * (zw // 2) + self.camera.ox
                    tsy = (tx + ty) * (zh // 2) - (self.world.height_steps[ty, tx] * zd) + self.camera.oy
                    cone_pts.append((tsx, tsy))
                if len(cone_pts) > 1:
                    pygame.draw.lines(self._overlay, (0, 255, 255, 60), False, cone_pts, 4)

            self._draw_rover(screen, rsx, rsy, zw, zh)

        screen.blit(self._overlay, (0, 0))

    # Replaced _draw_rock manual polygon rasterization into cached _make_rock_surf

    def _draw_rover(self, screen, rsx, rsy, zw, zh):
        """Draws the rover as a distinct golden object."""
        rw, rh = zw * 0.4, zh * 0.4
        bx, by = rsx, rsy - rh//2
        pts = [(bx - rw // 2, by), (bx, by - rh), (bx + rw // 2, by), (bx, by + rh)]
        pygame.draw.polygon(screen, (255, 200, 50), pts)
        pygame.draw.polygon(screen, (200, 150, 0), pts, 2)
        
        # Draw a tiny solar panel / antenna
        pygame.draw.line(screen, (150, 150, 150), (bx, by - rh), (bx, by - rh - 10), 2)
        pygame.draw.circle(screen, (0, 255, 255), (bx, int(by - rh - 10)), 3)

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

    def _draw_connected_path(self, path, zw, zh, zd, color=(255, 255, 0, 150), thick=2):
        """Draws a continuous line through the path coordinates."""
        points = []
        for px, py in path:
            psx = (px - py) * (zw // 2) + self.camera.ox
            psy = (px + py) * (zh // 2) - (self.world.height_steps[py, px] * zd) + self.camera.oy
            points.append((psx, psy))
        
        if len(points) > 1:
            pygame.draw.lines(self._overlay, color, False, points, thick)

    def _draw_text(self, screen, text, sx, sy, color):
        surface = self.font.render(text, True, color)
        rect = surface.get_rect(center=(sx, sy))
        bg_rect = rect.inflate(8, 4)
        pygame.draw.rect(screen, (0, 0, 0, 150), bg_rect, border_radius=4)
        screen.blit(surface, rect)

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
    dijkstra_path = []
    straight_path = []
    path_cost = 0.0
    dijkstra_cost = 0.0
    straight_cost = 0.0
    rover = None

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                grid_pos = camera.screen_to_world(mx, my, world)
                
                if event.button == 1: # Left Click: Start
                    start_node = grid_pos
                    path = [] 
                if event.button == 3: # Right Click: End
                    end_node = grid_pos
                    # Calculate Path
                    if start_node and end_node:
                        world_w, world_h = world.width, world.height
                        if (0 <= start_node[0] < world_w and 0 <= start_node[1] < world_h and 
                            0 <= end_node[0] < world_w and 0 <= end_node[1] < world_h):
                            
                            # First path generated using known world memory
                            grid_data = world.height_steps.tolist()
                            known_grid = world.known_object_map.tolist()
                            rough_grid = world.roughness_map.tolist()
                            
                            path = astar(grid_data, start_node, end_node, object_grid=known_grid, roughness_grid=rough_grid)
                            dijkstra_path = dijkstra(grid_data, start_node, end_node, object_grid=known_grid, roughness_grid=rough_grid)
                            
                            if path or dijkstra_path:
                                path_cost = calculate_path_cost(grid_data, path, roughness_grid=rough_grid) if path else float('inf')
                                dijkstra_cost = calculate_path_cost(grid_data, dijkstra_path, roughness_grid=rough_grid) if dijkstra_path else float('inf')
                                straight_path = get_straight_line(start_node[0], start_node[1], end_node[0], end_node[1])
                                straight_cost = calculate_path_cost(grid_data, straight_path, roughness_grid=rough_grid)
                                
                                rover = Rover(start_node)
                                if path_cost <= dijkstra_cost and path:
                                    rover.set_path(path)
                                    print(f"A* chosen. A*: {path_cost:.1f}, Dijkstra: {dijkstra_cost:.1f}")
                                elif dijkstra_path:
                                    rover.set_path(dijkstra_path)
                                    print(f"Dijkstra chosen. A*: {path_cost:.1f}, Dijkstra: {dijkstra_cost:.1f}")
                
                if event.button == 4: camera.adjust_zoom(0.1, (mx, my))
                if event.button == 5: camera.adjust_zoom(-0.1, (mx, my))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: camera.move(camera.speed, 0)
        if keys[pygame.K_d]: camera.move(-camera.speed, 0)
        if keys[pygame.K_w]: camera.move(0, camera.speed)
        if keys[pygame.K_s]: camera.move(0, -camera.speed)

        mx, my = pygame.mouse.get_pos()
        hovered = camera.screen_to_world(mx, my, world)

        # Update Rover
        if rover:
            grid_data = world.height_steps.tolist()
            # Pass ground truth, the mutable known_map memory, and the roughness grid directly
            rover.update(grid_data, world.object_map, world.known_object_map, world.roughness_map.tolist())
            
            # Recalculate active path cost dynamically
            if rover.state == "MOVING" and rover.current_path:
                # Use the actual logical departure node to prevent float rounding errors triggering artificial cliff penalties
                remainder_path = rover.current_path[max(0, rover.target_index - 1):]
                path_cost = rover.accumulated_cost + calculate_path_cost(grid_data, remainder_path, roughness_grid=world.roughness_map.tolist())

        renderer.render(screen, hovered, path, start_node, end_node, rover, straight_path, dijkstra_path, path_cost, straight_cost, dijkstra_cost)
        
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