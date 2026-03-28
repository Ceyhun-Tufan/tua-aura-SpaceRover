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
) -> list[tuple[int, int]] | None:
    """
    Yükseklik haritası üzerinde en düşük maliyetli yolu bulur.

    Parametreler
    ────────────
    height_grid : 2D liste — height_grid[y][x] (int yükseklik değerleri)
    start       : (x, y)  başlangıç koordinatı
    goal        : (x, y)  hedef koordinatı

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

            dh = height_grid[ny][nx] - ch                 # pozitif = yokuş yukarı
            tg = g + base_cost + max(0.0, dh) * SLOPE_PENALTY

            if tg < g_score.get((nx, ny), math.inf):
                g_score[(nx, ny)]   = tg
                came_from[(nx, ny)] = (cx, cy)
                f_new = tg + _heuristic(nx, ny, gx, gy)
                heapq.heappush(heap, (f_new, tg, nx, ny))

    return None  # Yol bulunamadı


def _heuristic(ax: int, ay: int, bx: int, by: int) -> float:
    """8 yönlü grid için kabul edilebilir oktil mesafe tahmini."""
    dx, dy = abs(bx - ax), abs(by - ay)
    return max(dx, dy) + (DIAGONAL_COST - 1) * min(dx, dy)