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

    def scan_radar(self, object_map):
        """Checks the route immediately ahead for obstacles."""
        if self.state != "MOVING" or not self.current_path:
            return None

        # Look ahead 'radar_range' steps in current path
        look_limit = min(self.target_index + self.radar_range, len(self.current_path))
        for i in range(self.target_index, look_limit):
            tx, ty = self.current_path[i]
            # Avoid bounds errors
            if 0 <= ty < len(object_map) and 0 <= tx < len(object_map[0]):
                if object_map[ty][tx] != 0:
                    return self.current_path[i] # Obstacle found!
        return None

    def calculate_bypass(self, height_grid, object_map):
        """A simple local bypass logic to reconnect to the global path."""
        print("Obstacle detected! Calculating local bypass...")
        self.state = "RECALCULATING"
        
        curr_node = (int(round(self.gx)), int(round(self.gy)))
        
        # 1. Find closest index in global_path to current position
        closest_dist = float('inf')
        closest_idx = 0
        for i, (gx, gy) in enumerate(self.global_path):
            dist = (gx - curr_node[0])**2 + (gy - curr_node[1])**2
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        # 2. Search forward from closest_idx to find a clear node
        # Give it a small buffer (+2) to ensure we jump PAST the rock
        search_start = min(closest_idx + 2, len(self.global_path) - 1)
        reconnect_node = None
        reconnect_idx = -1
        
        for i in range(search_start, len(self.global_path)):
            tx, ty = self.global_path[i]
            if 0 <= ty < len(object_map) and 0 <= tx < len(object_map[0]):
                if object_map[ty][tx] == 0:
                    reconnect_node = (tx, ty)
                    reconnect_idx = i
                    break
                    
        if not reconnect_node:
            print("Cannot find a reconnect point. Stopping.")
            self.state = "IDLE"
            return
            
        # Use A* to find path from current to reconnect_node PASSING object_map
        bypass_path = astar(height_grid, curr_node, reconnect_node, object_grid=object_map)
        
        if bypass_path:
            self.current_path = bypass_path + self.global_path[reconnect_idx+1:]
            self.target_index = 1 if len(bypass_path) > 1 else 0
            self.state = "MOVING"
            print("Bypass path established. Resuming.")
        else:
            print("No viable bypass could be computed. Halting.")
            self.state = "IDLE"

    def update(self, height_grid, object_map):
        if self.state == "IDLE" or not self.current_path:
            return
            
        if self.state == "MOVING":
            # 1. Radar check against object_map (numpy array so tolist is not needed, array indexing works for 2d)
            obstacle_node = self.scan_radar(object_map)
            if obstacle_node:
                self.calculate_bypass(height_grid, object_map)
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