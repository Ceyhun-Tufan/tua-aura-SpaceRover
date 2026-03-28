"""
pathfinder.py
─────────────────────────────────────────────────────────────────────────────
Saf A* arama algoritması.

Kullanım:
    from pathfinder import astar

    path = astar(height_grid, start=(x1, y1), goal=(x2, y2))
"""

import math
import heapq

# ── Ayarlar ───────────────────────────────────────────────────────────────────
SLOPE_PENALTY = 3.0          # Yokuş yukarı her birim için ek maliyet
DIAGONAL_COST = math.sqrt(2) # Çapraz hareketin taban maliyeti
MAX_CLIMB = 3                # Maksimum tırmanılabilir yükseklik adımı (yaklaşık 30 derecelik eşik)
ROUGHNESS_PENALTY_MAX = 8.0  # Pürüzlü / gevşek zeminlerde maksimum gecikme cezası

_NEIGHBOURS = [
    ( 1,  0, 1.0),
    (-1,  0, 1.0),
    ( 0,  1, 1.0),
    ( 0, -1, 1.0),
    ( 1,  1, DIAGONAL_COST),
    ( 1, -1, DIAGONAL_COST),
    (-1,  1, DIAGONAL_COST),
    (-1, -1, DIAGONAL_COST),
]


def astar(
    height_grid: list[list[int]],
    start: tuple[int, int],
    goal:  tuple[int, int],
    object_grid: list[list[int]] | None = None,
    roughness_grid: list[list[float]] | None = None
) -> list[tuple[int, int]] | None:
    """
    Yükseklik haritası üzerinde en düşük maliyetli yolu bulur. Opsiyonel olarak engelleri (0'dan farklı değerleri) önleyebilir.

    Parametreler
    ────────────
    height_grid : 2D liste — height_grid[y][x] (int yükseklik değerleri)
    start       : (x, y)  başlangıç koordinatı
    goal        : (x, y)  hedef koordinatı
    object_grid : 2D liste (isteğe bağlı) — object_grid[y][x], engelleri temsil eder (0 boş)

    Döndürür
    ────────
    Başlangıçtan hedefe (x, y) tuple listesi; yol yoksa None.
    """
    rows = len(height_grid)
    cols = len(height_grid[0])
    sx, sy = start
    gx, gy = goal

    for label, (x, y) in [("start", start), ("goal", goal)]:
        if not (0 <= x < cols and 0 <= y < rows):
            raise ValueError(f"{label} {(x, y)} sınır dışı ({cols}×{rows})")

    # (f_score, g_score, x, y)
    heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(heap, (0.0, 0.0, sx, sy))

    came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    g_score:   dict[tuple[int, int], float]                  = {(sx, sy): 0.0}

    while heap:
        f, g, cx, cy = heapq.heappop(heap)

        if (cx, cy) == (gx, gy):
            # Yolu geri izle
            path, cur = [], (gx, gy)
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        if g > g_score.get((cx, cy), math.inf):
            continue

        ch = height_grid[cy][cx]

        for dx, dy, base_cost in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue

            if object_grid is not None and object_grid[ny][nx] != 0:
                continue

            dh = height_grid[ny][nx] - ch                 # pozitif = yokuş yukarı
            
            # Fiziksel eğim sınırı (Rover çok dik yerlere çıkamaz/inemez)
            if abs(dh) > MAX_CLIMB:
                continue
                
            is_diag = (dx != 0 and dy != 0)
            base_cost = DIAGONAL_COST if is_diag else 1.0
            
            r_cost = 0.0
            if roughness_grid is not None:
                r_cost = roughness_grid[ny][nx] * ROUGHNESS_PENALTY_MAX
                
            tg = g + base_cost + max(0.0, dh) * SLOPE_PENALTY + r_cost

            if tg < g_score.get((nx, ny), math.inf):
                g_score[(nx, ny)]   = tg
                came_from[(nx, ny)] = (cx, cy)
                f_new = tg + _heuristic(nx, ny, gx, gy)
                heapq.heappush(heap, (f_new, tg, nx, ny))

    return None  # Yol bulunamadı

def dijkstra(
    height_grid: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    object_grid: list[list[int]] = None,
    roughness_grid: list[list[float]] = None
) -> list[tuple[int, int]]:
    """Dijkstra pathfinding algoritması (A* ile aynı, sadece sezgisel/heuristic kullanmaz)."""
    rows = len(height_grid)
    cols = len(height_grid[0])
    
    open_set = []
    heapq.heappush(open_set, (0.0, start[0], start[1]))
    
    came_from = {}
    g_score = {start: 0.0}
    
    while open_set:
        current_g, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
            
        if current_g > g_score.get((cx, cy), float('inf')):
            continue
            
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= ny < rows and 0 <= nx < cols:
                if object_grid is not None and object_grid[ny][nx] != 0:
                    continue  

                dh = height_grid[ny][nx] - height_grid[cy][cx]
                
                # Fiziksel eğim sınırı
                if abs(dh) > MAX_CLIMB:
                    continue
                
                is_diag = (dx != 0 and dy != 0)
                base_cost = DIAGONAL_COST if is_diag else 1.0
                
                r_cost = 0.0
                if roughness_grid is not None:
                    r_cost = roughness_grid[ny][nx] * ROUGHNESS_PENALTY_MAX
                    
                tg = current_g + base_cost + max(0.0, dh) * SLOPE_PENALTY + r_cost

                if tg < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)]   = tg
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_set, (tg, nx, ny))

    return None

def _heuristic(ax: int, ay: int, bx: int, by: int) -> float:
    """8 yönlü grid için kabul edilebilir oktil mesafe tahmini."""
    dx, dy = abs(bx - ax), abs(by - ay)
    return max(dx, dy) + (DIAGONAL_COST - 1) * min(dx, dy)

def get_straight_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham's Line Algorithm for grid."""
    path = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            path.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            path.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    path.append((x, y))
    return path

def calculate_path_cost(height_grid: list[list[int]], path: list[tuple[int, int]], roughness_grid: list[list[float]] = None) -> float:
    """Calculates the total travel cost of a generated path."""
    if not path or len(path) < 2:
        return 0.0
    cost = 0.0
    for i in range(len(path) - 1):
        cx, cy = path[i]
        nx, ny = path[i+1]
        
        is_diag = (cx != nx) and (cy != ny)
        base_cost = DIAGONAL_COST if is_diag else 1.0
        
        dh = height_grid[ny][nx] - height_grid[cy][cx]
        if abs(dh) > MAX_CLIMB:
            cost += float('inf')
            
        r_cost = 0.0
        if roughness_grid is not None:
            r_cost = roughness_grid[ny][nx] * ROUGHNESS_PENALTY_MAX
            
        cost += base_cost + max(0.0, dh) * SLOPE_PENALTY + r_cost
    return cost