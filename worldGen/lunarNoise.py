import numpy as np

class LunarNoise:
    def __init__(self, width, height, seed=42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        
    def generate(self, octaves=6, persistence=0.5, lacunarity=2.0):
        """
        Generates a fractal noise map optimized for lunar surfaces.
        """
        final_map = np.zeros((self.height, self.width))
        amplitude = 1.0
        frequency = 1.0
        
        # Create a grid of coordinates
        lin_x = np.linspace(0, 1, self.width)
        lin_y = np.linspace(0, 1, self.height)
        x_grid, y_grid = np.meshgrid(lin_x, lin_y)

        for _ in range(octaves):
            # Generate a pseudo-random gradient noise layer
            layer = self._simple_noise(x_grid * frequency, y_grid * frequency)
            final_map += layer * amplitude
            
            amplitude *= persistence
            frequency *= lacunarity

        # Normalize to 0.0 - 1.0
        final_map = (final_map - final_map.min()) / (final_map.max() - final_map.min())
        
        # --- LUNAR TRANSFORMATION ---
        # Raising to a power creates flat 'seas' (Maria) and sharp highlands
        final_map = np.power(final_map, 2.5) 
        
        return final_map

    def _simple_noise(self, x, y):
        """Vectorized sine-based noise (fast approximation of Perlin)"""
        return (np.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1.0