import math
from pathFinding.pathFinding import astar, dijkstra, calculate_path_cost

class Rover:
    def __init__(self, start_node):
        self.gx, self.gy = float(start_node[0]), float(start_node[1])
        self.speed = 0.5  # Movement speed across tiles
        
        self.global_path = [] # Full satellite path
        self.local_path = []  # Bypass path if obstacle found
        
        self.current_path = [] # Merged or active path
        self.target_index = 0
        
        self.radar_range = 4 # How many tiles ahead to check
        self.state = "IDLE"
        self.current_speed = 0.0 # For UI telemetrics
        
        self.dodged_obstacles = set()
        self.accumulated_cost = 0.0
        self.traversed_path = [(int(round(self.gx)), int(round(self.gy)))]
        self.previous_node = (int(round(self.gx)), int(round(self.gy)))

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
        
        self.dodged_obstacles = set()
        self.accumulated_cost = 0.0
        self.traversed_path = [(int(round(self.gx)), int(round(self.gy)))]
        self.previous_node = (int(round(self.gx)), int(round(self.gy)))

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

    def calculate_bypass(self, height_grid, known_object_map, roughness_grid=None):
        """Calculates a totally new path to the goal using updated memory."""
        print("Obstacle detected! Calculating bypass...")
        self.state = "RECALCULATING"
        
        curr_node = (int(round(self.gx)), int(round(self.gy)))
        final_goal = self.global_path[-1]
        
        obstacle_grid = known_object_map.tolist() if hasattr(known_object_map, 'tolist') else known_object_map
        bypass_path_astar = astar(height_grid, curr_node, final_goal, object_grid=obstacle_grid, roughness_grid=roughness_grid)
        bypass_path_dijkstra = dijkstra(height_grid, curr_node, final_goal, object_grid=obstacle_grid, roughness_grid=roughness_grid)
        
        cost_a = calculate_path_cost(height_grid, bypass_path_astar, roughness_grid=roughness_grid) if bypass_path_astar else float('inf')
        cost_d = calculate_path_cost(height_grid, bypass_path_dijkstra, roughness_grid=roughness_grid) if bypass_path_dijkstra else float('inf')
        
        if cost_a <= cost_d and bypass_path_astar:
            bypass_path = bypass_path_astar
            print(f"A* chosen. A*: {cost_a:.1f}, Dijkstra: {cost_d:.1f}")
        else:
            bypass_path = bypass_path_dijkstra
            print(f"Dijkstra chosen. A*: {cost_a:.1f}, Dijkstra: {cost_d:.1f}")
        
        if bypass_path:
            self.current_path = bypass_path
            self.global_path = bypass_path  # Update global to our new route
            self.target_index = 1 if len(bypass_path) > 1 else 0
            self.state = "MOVING"
            
            # Snap previous node to prevent INF cost gaps when resuming physical movement
            if self.previous_node != curr_node:
                cost = calculate_path_cost(height_grid, [self.previous_node, curr_node], roughness_grid=roughness_grid)
                if cost == float('inf'): cost = 1.0
                self.accumulated_cost += cost
                self.previous_node = curr_node
                
            print("Bypass path established. Resuming.")
        else:
            print("No viable bypass could be computed. Halting.")
            self.state = "IDLE"

    def update(self, height_grid, object_map, known_object_map, roughness_grid=None):
        if self.state == "IDLE" or not self.current_path:
            return
            
        if self.state == "MOVING":
            # 1. Radar sweep ground truth, update memory, check if blocked.
            obstacle_node = self.scan_radar(object_map, known_object_map)
            if obstacle_node:
                self.dodged_obstacles.add(obstacle_node)
                self.calculate_bypass(height_grid, known_object_map, roughness_grid=roughness_grid)
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
                self.accumulated_cost += calculate_path_cost(height_grid, [self.previous_node, (tx, ty)], roughness_grid=roughness_grid)
                self.previous_node = (tx, ty)
                self.traversed_path.append((int(tx), int(ty)))
                
                self.gx, self.gy = float(tx), float(ty)
                self.target_index += 1
            else:
                # Calculate movement penalty based on local roughness
                rx_int, ry_int = int(round(self.gx)), int(round(self.gy))
                h, w = len(height_grid), len(height_grid[0])
                
                # Fetch roughness (0.0 to 1.0)
                local_roughness = 0.0
                if roughness_grid and 0 <= ry_int < h and 0 <= rx_int < w:
                    local_roughness = roughness_grid[ry_int][rx_int]
                
                # Dynamic speed: moves up to 4x slower on max roughness
                effective_speed = self.speed / (1.0 + local_roughness * 3.0)
                self.current_speed = effective_speed
                
                self.gx += (dx / dist) * effective_speed
                self.gy += (dy / dist) * effective_speed
        else:
            self.current_speed = 0.0

    def restore_state(self, data):
        """Restores the rover's physical state from a dictionary."""
        self.gx = float(data.get('gx', self.gx))
        self.gy = float(data.get('gy', self.gy))
        self.accumulated_cost = float(data.get('cost', self.accumulated_cost))
        self.traversed_path = data.get('traversed', self.traversed_path)
        self.current_path = data.get('path', self.current_path)
        self.global_path = data.get('path', self.global_path)
        self.target_index = int(data.get('target_idx', self.target_index))
        self.state = data.get('state', self.state)
        self.previous_node = (int(round(self.gx)), int(round(self.gy)))