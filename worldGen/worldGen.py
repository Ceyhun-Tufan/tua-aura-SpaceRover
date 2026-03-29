import sys
import os

# This adds the root directory of your project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
import time
from random import randint
from pathFinding.pathFinding import astar, dijkstra, get_straight_line, calculate_path_cost
from rover.rover import Rover

# --- Config (Same as yours) ---
SCREEN_W, SCREEN_H = 1280, 720
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
    def __init__(self, width, height, seed=80368):
        self.width, self.height = width, height
        self.seed = seed if seed is not None else randint(0, 100000)

        # ── Check for an existing save file BEFORE doing any heavy generation ──
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _save_path = os.path.join(_root, f"{self.seed}.txt")
        if os.path.exists(_save_path):
            self._load_data_from_file(_save_path)
            return   # skip generation entirely

        # ── Generate world from scratch ─────────────────────────────────
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

        # ── Performance caches (static for the lifetime of the world) ────────
        self._build_caches()

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
        idx[self.roughness_map > 0.6] += 4
        return idx

    def _build_caches(self):
        """Pre-compute performance-critical static arrays (call after all grids are ready)."""
        self._iso_dx = (self.gx - self.gy).astype(np.int32)
        self._iso_dy = (self.gx + self.gy).astype(np.int32)
        self.height_steps_list  = self.height_steps.tolist()
        self.roughness_map_list = self.roughness_map.tolist()

    def _load_data_from_file(self, filepath):
        """Populate all world arrays from a previously saved .txt file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # ── Parse header ─────────────────────────────────────────────────
        for line in lines:
            if line.startswith("Width:"):         self.width  = int(line.split(":")[1].strip())
            elif line.startswith("Height:") and "Levels" not in line:
                                                  self.height = int(line.split(":")[1].strip())

        self.gx, self.gy = np.meshgrid(np.arange(self.width), np.arange(self.height))

        # ── Parse sections ───────────────────────────────────────────────
        section = None
        hmap_rows, rmap_rows, objects = [], [], []
        for line in lines:
            stripped = line.rstrip()
            if stripped.startswith("[HEIGHTMAP]"):   section = "h"
            elif stripped.startswith("[ROUGHNESS]") and "STATS" not in stripped: section = "r"
            elif stripped.startswith("[OBJECTS]"):   section = "o"
            elif stripped.startswith("["):           section = None
            elif not stripped or stripped.startswith("#"): continue
            elif section == "h":
                hmap_rows.append(list(map(int, stripped.split())))
            elif section == "r":
                rmap_rows.append([v / 10000.0 for v in map(int, stripped.split())])
            elif section == "o":
                parts = stripped.split()
                if len(parts) == 3 and parts[0].isdigit():
                    objects.append((int(parts[0]), int(parts[1]), int(parts[2])))

        # ── Reconstruct numpy arrays ─────────────────────────────────────────
        self.height_steps  = np.array(hmap_rows, dtype=np.int32)
        self.heightmap     = (self.height_steps / self.HEIGHT_LEVELS).astype(np.float32)

        if rmap_rows and len(rmap_rows) == self.height:
            self.roughness_map = np.array(rmap_rows, dtype=np.float32)
        else:
            # Old save file (no [ROUGHNESS] section) — regenerate from seed so the
            # style map and pathfinding costs are still correct, then re-save.
            print("  (old format — regenerating roughness from seed and re-saving)")
            self.rng = np.random.default_rng(self.seed)
            # Advance RNG past heightmap (8 oct × 2) and craters (800 × 4) calls
            _ = [self.rng.uniform(0, 1000) for _ in range(16)]
            _ = [(self.rng.integers(0, 2), self.rng.exponential(), self.rng.uniform())
                 for _ in range(800)]
            self.roughness_map = self._build_roughness_map()
            self._regen_roughness = True   # flag save_to_file to overwrite

        self.style_idx = self._build_style_map()

        self.object_map = np.zeros((self.height, self.width), dtype=np.int32)
        for x, y, t in objects:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.object_map[y, x] = t

        self.known_object_map = np.zeros((self.height, self.width), dtype=np.int32)
        self._build_caches()
        self._loaded_from_file = True
        print(f"World loaded  <- {filepath}")


    def save_to_file(self, save_dir=None):
        """Save world data to '{seed}.txt' in the project root (or save_dir)."""
        if save_dir is None:
            save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{self.seed}.txt")

        ys, xs = np.where(self.object_map > 0)
        rocks = list(zip(xs.tolist(), ys.tolist(), self.object_map[ys, xs].tolist()))
        rough_int = (self.roughness_map * 10000).astype(np.int32)

        with open(filepath, 'w') as f:
            f.write("=== SpaceRover World Save ===\n")
            f.write(f"Seed:          {self.seed}\n")
            f.write(f"Width:         {self.width}\n")
            f.write(f"Height:        {self.height}\n")
            f.write(f"Height Levels: {self.HEIGHT_LEVELS}\n")
            f.write(f"Rock Count:    {len(rocks)}\n\n")

            f.write("[HEIGHTMAP]  # rows of space-separated int height values (0-70)\n")
            for row in self.height_steps.tolist():
                f.write(' '.join(map(str, row)) + '\n')

            f.write("\n[ROUGHNESS]  # values ×10000 as ints for lossless reload\n")
            for row in rough_int.tolist():
                f.write(' '.join(map(str, row)) + '\n')

            f.write("\n[OBJECTS]  # x y type  (1=small rock, 2=large rock)\n")
            for x, y, t in rocks:
                f.write(f"{x} {y} {t}\n")

        print(f"World saved   -> {filepath}")
        return filepath

class Camera:
    def __init__(self):
        self.ox, self.oy = SCREEN_W // 2, SCREEN_H // 4
        self.speed, self.zoom = 12, 1.0
        self.min_zoom, self.max_zoom = 0.04, 4.0

    def move(self, dx, dy): self.ox += dx; self.oy += dy
    def adjust_zoom(self, delta, center):
        """Multiplicative zoom: each tick scales by a fixed ratio so it feels
        consistent at every zoom level. Pivot is kept under the mouse cursor."""
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        # Use the same integer-truncated tile sizes the renderer will use, so
        # the pivot point (tile under cursor) doesn't drift.
        old_half_zw = int(TILE_W * self.zoom) // 2
        new_half_zw = int(TILE_W * new_zoom)  // 2
        old_half_zh = int(TILE_H * self.zoom) // 2
        new_half_zh = int(TILE_H * new_zoom)  // 2
        ratio_w = new_half_zw / max(old_half_zw, 1)
        ratio_h = new_half_zh / max(old_half_zh, 1)
        self.ox = center[0] - (center[0] - self.ox) * ratio_w
        self.oy = center[1] - (center[1] - self.oy) * ratio_h
        self.zoom = new_zoom

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

        # Background image — load felfena.jpeg from the project root
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _bg_path = os.path.join(_root, "felfena.jpeg")
        try:
            _raw = pygame.image.load(_bg_path).convert()
            self._star_bg = pygame.transform.scale(_raw, (SCREEN_W, SCREEN_H))
        except Exception:
            self._star_bg = pygame.Surface((SCREEN_W, SCREEN_H))
            self._star_bg.fill(COL_SKY)

        self._overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("monospace", 14, bold=True)
        # Per-zoom projection cache — recomputed only when zoom changes
        self._sx_base: np.ndarray | None = None  # iso_dx * (zw//2)
        self._sy_base: np.ndarray | None = None  # iso_dy * (zh//2) - height_steps * zd
        # Dark curtain drawn between the JPEG background and the tiles so that
        # transparent SRCALPHA tile corners show dark space, not the photo.
        self._bg_curtain = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        self._bg_curtain.fill((5, 8, 20, 210))   # very dark, ~82 % opaque

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
        # Dark curtain: tile transparent corners show this dark layer, not the JPEG
        screen.blit(self._bg_curtain, (0, 0))
        self._overlay.fill((0, 0, 0, 0))
        
        z = self.camera.zoom
        zw  = max(2, int(TILE_W * z))   # floor at 2 so surfaces are never 0-sized
        zh  = max(1, int(TILE_H * z))   # floor at 1
        zd  = max(0, int(TILE_DEPTH * z))
        half_zw, half_zh = zw >> 1, zh >> 1

        if z != self._last_zoom:
            self._scaled_tiles = [pygame.transform.scale(t, (zw, zh+zd)) for t in self._base_tiles]
            self._scaled_rocks = {
                1: self._make_rock_surf(zw, zh, 1),
                2: self._make_rock_surf(zw, zh, 2)
            }
            self._last_zoom = z
            # Recompute static projection bases (camera-offset is added per-frame below)
            self._sx_base = self.world._iso_dx * half_zw
            self._sy_base = self.world._iso_dy * (half_zh) - self.world.height_steps * zd

        # Per-frame: add integer camera offset to static bases (2 numpy ops vs 4 before)
        ox_i = int(self.camera.ox)
        oy_i = int(self.camera.oy)
        sx = self._sx_base + ox_i
        sy = self._sy_base + oy_i

        # Visibility cull
        vis = (sx > -zw*2) & (sx < SCREEN_W+zw*2) & (sy > -zh*2) & (sy < SCREEN_H+zh*2)

        # ── BATCH index all needed data for visible tiles in one numpy call each ──
        vis_yx   = np.argwhere(vis)
        if len(vis_yx):
            vy, vx           = vis_yx[:, 0], vis_yx[:, 1]
            vis_sx           = sx[vy, vx].tolist()       # batch → Python ints
            vis_sy           = sy[vy, vx].tolist()
            vis_style        = self.world.style_idx[vy, vx].tolist()
            vis_obj          = self.world.known_object_map[vy, vx].tolist()

            scaled_tiles = self._scaled_tiles
            scaled_rocks = self._scaled_rocks
            blit_seq     = []
            for i in range(len(vis_sx)):
                blit_seq.append((scaled_tiles[vis_style[i]], (vis_sx[i] - half_zw, vis_sy[i])))
                if vis_obj[i]:
                    blit_seq.append((scaled_rocks[vis_obj[i]], (vis_sx[i] - half_zw, vis_sy[i] - half_zh)))
            screen.blits(blit_seq)

        # ── Path trail dots: iterate path nodes, not all vis tiles ──────────
        path_set = set(path) if path else set()
        for (px, py) in path_set:
            if (px, py) != start_node and (px, py) != end_node:
                if vis[py, px]:
                    self._draw_path_trail(sx[py, px], sy[py, px], zw, zh)

        # ── Markers: look up screen pos directly, no vis scan needed ────────
        def _node_screen(node):
            nx, ny = node
            return sx[ny, nx], sy[ny, nx]

        if start_node:
            nsx, nsy = _node_screen(start_node)
            self._draw_marker(nsx, nsy, zw, zh, (0, 255, 150), "START")
        if end_node:
            nsx, nsy = _node_screen(end_node)
            self._draw_marker(nsx, nsy, zw, zh, (255, 50, 50), "TARGET")

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

    def _draw_text_left(self, screen, text, x, y, color):
        surface = self.font.render(text, True, color)
        rect = surface.get_rect(topleft=(x, y))
        bg_rect = rect.inflate(8, 4)
        bg_rect.topleft = (x - 4, y - 2)
        pygame.draw.rect(screen, (0, 0, 0, 150), bg_rect, border_radius=4)
        screen.blit(surface, rect)

    def _draw_stats_table(self, screen, x, base_y, headers, rows, highlight_idx=-1):
        col_widths = []
        for c in range(len(headers)):
            w = max([self.font.size(headers[c])[0]] + [self.font.size(str(row[c]))[0] for row in rows])
            col_widths.append(w + 30)

        row_h = 25
        total_w = sum(col_widths)
        total_h = (len(rows) + 1) * row_h
        y = base_y - total_h
        
        s = pygame.Surface((total_w, total_h), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 200), (0, 0, total_w, total_h), border_radius=6)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, (100, 100, 150), (x, y, total_w, total_h), width=2, border_radius=6)
        
        cx = x
        for c, header in enumerate(headers):
            surf = self.font.render(header, True, (150, 200, 255))
            screen.blit(surf, (cx + 15, y + 5))
            cx += col_widths[c]
            
        pygame.draw.line(screen, (100, 100, 150), (x, y + row_h), (x + total_w, y + row_h), 2)
        
        for r_idx, row in enumerate(rows):
            ry = y + (r_idx + 1) * row_h
            cx = x
            if highlight_idx == r_idx:
                s_hl = pygame.Surface((total_w - 4, row_h), pygame.SRCALPHA)
                s_hl.fill((0, 255, 100, 40))
                screen.blit(s_hl, (x + 2, ry))
                
            for c, cell in enumerate(row):
                color = (255, 255, 255)
                if highlight_idx == r_idx: color = (100, 255, 100)
                surf = self.font.render(str(cell), True, color)
                screen.blit(surf, (cx + 15, ry + 5))
                cx += col_widths[c]

def main():
    pygame.init()
    try:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.SCALED | pygame.RESIZABLE)
    except:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock, font = pygame.time.Clock(), pygame.font.SysFont("monospace", 14, bold=True)

    world = WorldGenerator(MAP_W, MAP_H)
    # Save only when freshly generated or when an old-format file was upgraded
    if getattr(world, '_regen_roughness', False) or not getattr(world, '_loaded_from_file', False):
        world.save_to_file()
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
    astar_time = 0.0
    dijkstra_time = 0.0
    straight_time = 0.0
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
                            
                            t0 = time.perf_counter()
                            path = astar(grid_data, start_node, end_node, object_grid=known_grid, roughness_grid=rough_grid)
                            t1 = time.perf_counter()
                            dijkstra_path = dijkstra(grid_data, start_node, end_node, object_grid=known_grid, roughness_grid=rough_grid)
                            t2 = time.perf_counter()
                            
                            astar_time = (t1 - t0) * 1000
                            dijkstra_time = (t2 - t1) * 1000
                            
                            if path or dijkstra_path:
                                path_cost = calculate_path_cost(grid_data, path, roughness_grid=rough_grid) if path else float('inf')
                                dijkstra_cost = calculate_path_cost(grid_data, dijkstra_path, roughness_grid=rough_grid) if dijkstra_path else float('inf')
                                
                                t3 = time.perf_counter()
                                straight_path = get_straight_line(start_node[0], start_node[1], end_node[0], end_node[1])
                                t4 = time.perf_counter()
                                straight_time = (t4 - t3) * 1000
                                
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

            if event.type == pygame.MOUSEMOTION:
                if event.buttons[1]: # Middle mouse button dragged
                    camera.move(event.rel[0], event.rel[1])

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: camera.move(camera.speed, 0)
        if keys[pygame.K_d]: camera.move(-camera.speed, 0)
        if keys[pygame.K_w]: camera.move(0, camera.speed)
        if keys[pygame.K_s]: camera.move(0, -camera.speed)

        # Edge panning
        mx, my = pygame.mouse.get_pos()
        if pygame.mouse.get_focused():
            edge = 20
            if mx <= edge: camera.move(camera.speed, 0)
            elif mx >= SCREEN_W - edge: camera.move(-camera.speed, 0)
            if my <= edge: camera.move(0, camera.speed)
            elif my >= SCREEN_H - edge: camera.move(0, -camera.speed)

        hovered = camera.screen_to_world(mx, my, world)

        # Update Rover
        if rover:
            # Use the cached Python-list copies — no .tolist() allocation every frame
            grid_data  = world.height_steps_list
            rough_data = world.roughness_map_list
            rover.update(grid_data, world.object_map, world.known_object_map, rough_data)

            # Recalculate active path cost dynamically
            if rover.state == "MOVING" and rover.current_path:
                remainder_path = rover.current_path[max(0, rover.target_index - 1):]
                path_cost = rover.accumulated_cost + calculate_path_cost(grid_data, remainder_path, roughness_grid=rough_data)

        # Compute active full path to render preserving the traversal history
        active_path = path
        if rover and rover.current_path:
            active_path = rover.traversed_path + rover.current_path[rover.target_index:]

        renderer.render(screen, hovered, active_path, start_node, end_node, rover, straight_path, dijkstra_path, path_cost, straight_cost, dijkstra_cost)
        
        # UI
        path_len = len(path) if path is not None else 0

        ui = [
            f"FPS: {int(clock.get_fps())}", 
            f"Zoom: {camera.zoom:.1f}", 
            f"Start: {start_node}", 
            f"End: {end_node}", 
            f"Path Length: {path_len}"
        ]
        
        for i, text in enumerate(ui):
            renderer._draw_text_left(screen, text, 10, 10 + i * 25, (255, 255, 255))
            
        if start_node and end_node:
            headers = ["Algorithm", "Cost", "Time(ms)", "Length", "Nodes/ms"]
            
            def calc_nodes_ms(length, ms):
                if ms <= 0.01: return "N/A"
                return f"{int(length / ms)}"

            rows = [
                ["A*", f"{path_cost:.1f}", f"{astar_time:.2f}", f"{len(path) if path else 0}", calc_nodes_ms(len(path) if path else 0, astar_time)],
                ["Dijkstra", f"{dijkstra_cost:.1f}", f"{dijkstra_time:.2f}", f"{len(dijkstra_path) if dijkstra_path else 0}", calc_nodes_ms(len(dijkstra_path) if dijkstra_path else 0, dijkstra_time)],
                ["Straight", f"{straight_cost:.1f}", f"{straight_time:.2f}", f"{len(straight_path) if straight_path else 0}", "N/A"]
            ]
            
            highlight_idx = -1
            if path_cost <= dijkstra_cost and path:
                highlight_idx = 0
            elif dijkstra_path:
                highlight_idx = 1
                
            renderer._draw_stats_table(screen, 15, SCREEN_H - 50, headers, rows, highlight_idx)

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()