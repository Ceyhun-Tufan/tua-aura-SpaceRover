import math
from pathFinding.pathFinding import astar

class Rover:
    def __init__(self, start_node):
        self.gx, self.gy = float(start_node[0]), float(start_node[1])
        self.speed = 0.08  # Movement speed across tiles
        
        self.global_path = [] # Full satellite path
        self.local_path = []  # Bypass path if obstacle found
        
        self.current_path = [] # Merged or active path
        self.target_index = 0
        
        self.radar_range = 4 # How many tiles ahead to check
        self.state = "IDLE"

    def set_path(self, path):
        if not path:
            self.state = "IDLE"
            self.current_path = []
            return
            
        self.global_path = path
        self.current_path = path
        self.target_index = 1 if len(path) > 1 else 0
        self.state = "MOVING"
        self.local_path = []

    def scan_radar(self, object_map, known_object_map):
        """Sweeps a radius, reveals terrain to known memory, then checks immediate path."""
        if self.state != "MOVING" or not self.current_path:
            return None

        # 1. Fog of War Sweep (360 degrees)
        cx, cy = int(round(self.gx)), int(round(self.gy))
        h, w = len(object_map), len(object_map[0])
        r = self.radar_range
        
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy <= r*r:
                    tx, ty = cx + dx, cy + dy
                    if 0 <= ty < h and 0 <= tx < w:
                        # Reveal ground truth to memory!
                        known_object_map[ty][tx] = object_map[ty][tx]

        # 2. Check the active path for obstacles USING THE KNOWN MAP
        look_limit = min(self.target_index + self.radar_range, len(self.current_path))
        for i in range(self.target_index, look_limit):
            tx, ty = self.current_path[i]
            if 0 <= ty < h and 0 <= tx < w:
                if known_object_map[ty][tx] != 0:
                    return self.current_path[i] # Obstacle found!
        return None

    def calculate_bypass(self, height_grid, known_object_map):
        """Calculates a totally new path to the goal using updated memory."""
        print("Obstacle detected! Calculating bypass...")
        self.state = "RECALCULATING"
        
        curr_node = (int(round(self.gx)), int(round(self.gy)))
        final_goal = self.global_path[-1]
        
        # Use A* to find path from current directly to the final goal!
        # PASSING known_object_map so it routes around everything we know.
        bypass_path = astar(height_grid, curr_node, final_goal, object_grid=known_object_map.tolist() if hasattr(known_object_map, 'tolist') else known_object_map)
        
        if bypass_path:
            self.current_path = bypass_path
            self.global_path = bypass_path  # Update global to our new route
            self.target_index = 1 if len(bypass_path) > 1 else 0
            self.state = "MOVING"
            print("Bypass path established. Resuming.")
        else:
            print("No viable bypass could be computed. Halting.")
            self.state = "IDLE"

    def update(self, height_grid, object_map, known_object_map):
        if self.state == "IDLE" or not self.current_path:
            return
            
        if self.state == "MOVING":
            # 1. Radar sweep ground truth, update memory, check if blocked.
            obstacle_node = self.scan_radar(object_map, known_object_map)
            if obstacle_node:
                self.calculate_bypass(height_grid, known_object_map)
                if self.state == "IDLE": return

            # 2. Move
            if self.target_index >= len(self.current_path):
                self.state = "IDLE"
                return

            tx, ty = self.current_path[self.target_index]
            dx = tx - self.gx
            dy = ty - self.gy
            dist = math.hypot(dx, dy)
            
            if dist < self.speed:
                self.gx, self.gy = float(tx), float(ty)
                self.target_index += 1
            else:
                self.gx += (dx / dist) * self.speed
                self.gy += (dy / dist) * self.speed