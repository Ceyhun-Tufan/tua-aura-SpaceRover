#  LunAura Lunar Pathfinding Simulator

An isometric lunar terrain explorer with real-time A\* and Dijkstra pathfinding, fog-of-war radar, dynamic obstacle avoidance, and procedurally generated terrain.

---

## Features

- **Procedural Lunar World** — Multi-octave fractal noise with craters, lunar highlands (maria), and roughness variation
- **Dual Pathfinding** — A\* and Dijkstra run in parallel; the rover automatically picks the lower-cost route
- **Straight-Line Comparison** — Bresenham line drawn alongside computed paths for visual cost comparison
- **Dynamic Obstacle Avoidance** — Rover scans ahead with a radar cone; on detecting a rock, it recalculates a bypass in real time using updated map memory
- **Fog of War** — The rover only knows what it has scanned; the world map is progressively revealed
- **Isometric Renderer** — Smooth zooming, panning, edge scrolling, and per-tile height extrusion
- **Persistent World** — World state (terrain, objects, rover position, path) auto-saves every 30 seconds and on exit; reloaded automatically on next launch
- **Algorithm Stats Table** — Live HUD displaying cost, computation time, path length, and nodes/ms for each algorithm

---

## Project Structure

```
.
├── worldGen/
│   ├── worldGen.py        # World generation, rendering, camera, and main game loop
│   └── lunarNoise.py      # Fractal noise generator for terrain
├── pathFinding/
│   └── pathFinding.py     # A*, Dijkstra, Bresenham line, and path cost calculator
├── rover/
│   └── rover.py           # Rover agent: movement, radar, bypass logic, state restore
├── saved/                 # Auto-generated world save files (gitignored)
├── felfena.jpeg           # Background star field image
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- pygame >= 2.5.0
- numpy >= 1.24.0

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Simulator

```bash
python worldGen/worldGen.py
```

The world generates on first run (may take a few seconds) and is saved automatically. Subsequent launches load from the save file instantly.

---

## Controls

| Input | Action |
|---|---|
| **Left Click** | Set start node |
| **Right Click** | Set target node & compute paths |
| **Scroll Wheel** | Zoom in / out (pivots under cursor) |
| **Middle Mouse Drag** | Pan camera |
| **W / A / S / D** | Pan camera (keyboard) |
| **Mouse at Screen Edge** | Auto-pan |

---

## Pathfinding Details

### Cost Model

All three algorithms share the same edge cost formula:

```
cost = base_cost + max(0, Δheight) × SLOPE_PENALTY + roughness × ROUGHNESS_PENALTY_MAX
```

| Parameter | Value | Description |
|---|---|---|
| `SLOPE_PENALTY` | 3.0 | Extra cost per unit of uphill elevation |
| `DIAGONAL_COST` | √2 ≈ 1.414 | Base cost of diagonal movement |
| `MAX_CLIMB` | 3 | Max elevation step per tile (steeper = impassable) |
| `ROUGHNESS_PENALTY_MAX` | 8.0 | Max extra cost on loose/rough terrain |

### Algorithm Comparison

| | A\* | Dijkstra | Straight Line |
|---|---|---|---|
| **Heuristic** | Octile distance | None | N/A |
| **Optimality** | Optimal (admissible heuristic) | Optimal | Not guaranteed |
| **Speed** | Faster (guided search) | Slower (exhaustive) | Instant |
| **Use case** | Primary pathfinding | Verification / fallback | Cost baseline |

The rover selects whichever of A\* or Dijkstra produces the lower travel cost. In practice A\* is faster and equally optimal; Dijkstra occasionally finds cheaper routes in dense obstacle fields where the heuristic misleads.

### Obstacle Avoidance

The rover's radar sweeps a configurable radius (`radar_range = 4` tiles) every frame:

1. **Reveal** — All tiles within radar range are copied from the ground-truth map into the rover's memory map
2. **Check** — The active path ahead is scanned for newly revealed obstacles
3. **Bypass** — If blocked, both A\* and Dijkstra replan from the current position using the updated memory; the cheaper result is taken

---

## World Generation

Terrain is built in layers:

1. **Base heightmap** — 8-octave sine-based fractal noise (approximates Perlin)
2. **Crater overlay** — 800 craters with exponential size distribution; each excavates a bowl and raises a rim
3. **Roughness map** — Independent 6-octave noise layer controlling terrain friction
4. **Style map** — Tile visual variant assigned by height + roughness threshold
5. **Object scatter** — 10,000 rocks placed stochastically in mid-elevation bands

### World Seeds

The default seed is `80368`. Change the `seed` parameter in `WorldGenerator.__init__()` to explore different terrains. Each seed produces a deterministic, fully reproducible world.

---

## Save Format

Worlds are saved to `saved/<seed>.txt` as plain text with labeled sections:

```
[HEIGHTMAP]     — integer height values (0–70) per tile
[ROUGHNESS]     — roughness values × 10000 (lossless integer encoding)
[OBJECTS]       — x y type for each placed rock
[KNOWN_OBJECTS] — x y type for tiles the rover has scanned
[ROVER_STATE]   — position, accumulated cost, active path, traversal history
```

---

## Architecture Notes

- **Renderer** pre-computes isometric screen coordinates as NumPy arrays and updates them only on zoom changes. Per-frame rendering uses `screen.blits()` for batched tile drawing.
- **Rover update loop** uses pre-converted Python lists (`height_steps_list`, `roughness_map_list`) to avoid repeated `.tolist()` allocations every frame.
- **Camera zoom** uses multiplicative scaling with mouse-pivot correction to prevent drift.
- **Path cost** is calculated incrementally during movement; the HUD displays live remaining cost rather than re-computing the full path each frame.
