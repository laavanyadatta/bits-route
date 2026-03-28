# Multi-criteria Route Planning System - BITS Pilani

A pathfinding system for navigating the BITS Pilani campus using classical AI search algorithms, with a primary focus on **Time-Dependent A\*** (TD-A\*) that adapts routes based on real-world congestion patterns.

---

## Overview

The navigator models the BITS Pilani campus as a weighted, undirected graph of **40 nodes** (landmarks, hostels, academic buildings, amenities) connected by **GPS-verified edges** measured in metres. Five search algorithms are implemented and benchmarked against each other, with TD-A\* as the default for interactive routing.

The key feature is **time-dependent edge weighting**: edge costs are multiplied by a congestion factor `k(t)` that varies across 15-minute slots throughout the day. This means the planner automatically routes around crowded corridors during rush hours, even if the alternative path is physically longer.

---

## Project Structure

```
bits-route-main/
├── main.py                  # Entry point — interactive planner + benchmarks
├── algorithms.py            # BFS, UCS, Greedy, A*, TD-A* implementations
├── campus_graph.py          # Graph construction, Haversine heuristic, congestion model
├── graph_data.py            # Node coordinates, edges (metres), congestion schedule
├── empirical_validation.py  # Benchmark queries + theoretical complexity comparison
├── visualisation.py         # Graph drawing, path animation, comparison charts
└── outputs/                 # Auto-created — PNGs saved here
```

---

## Algorithms

| Algorithm | Strategy | Optimal | Complete | Notes |
|---|---|---|---|---|
| **BFS** | Uniform level-by-level expansion | No | Yes | Minimises hops, ignores edge weights |
| **UCS** | Lowest cumulative cost first | Yes | Yes | Equivalent to Dijkstra for a single target |
| **Greedy (haversine)** | Lowest `h(n)` first | No | Yes | Fast but can miss cheaper routes |
| **Greedy (euclidean)** | Lowest `h(n)` first | No | Yes | Flat-Earth variant; near-identical to haversine at campus scale |
| **A\* (haversine)** | Lowest `f(n) = g(n) + h(n)` | Yes | Yes | Admissible & consistent Haversine heuristic |
| **A\* (euclidean)** | Lowest `f(n) = g(n) + h(n)` | Yes | Yes | Flat-Earth projection; admissible & consistent |
| **TD-A\*** | A\* with time-dependent edge weights | Yes\* | Yes | Default algorithm; congestion-aware |

\* Optimal within the time-dependent cost model.

### Heuristics

Two admissible, consistent heuristics are available for Greedy and A\*:

**Haversine** — great-circle distance between two GPS coordinates using the spherical Earth model:

```
a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
h = 2R · arcsin(√a)        where R = 6,371,000 m
```

**Euclidean** — flat-Earth planar projection centred on the campus reference latitude (~28.36°N):

```
dy = Δlat × 111,195 m/°
dx = Δlon × 111,195 × cos(ref_lat) m/°
h  = √(dx² + dy²)
```

Both are admissible (never overestimate true walking distance) and consistent, guaranteeing optimality for A\*. On the BITS campus (~1 km²) the difference between them is under 0.01 %, so any variation in node expansion counts between the two reflects floating-point tie-breaking rather than a meaningful quality gap.

### Time-Dependent Edge Weights

```
w_eff(u, v, t) = max(k(u, t), k(v, t)) × w_base
```

- `t` is a **time slot** — an integer in `[0, 95]` representing 15-minute intervals since midnight (slot 0 = 00:00, slot 35 = 08:45, etc.)
- `k ≥ 1.0` is looked up from `CONGESTION_SCHEDULE` in `graph_data.py`
- Taking `max()` over both endpoints conservatively applies the stricter congestion factor when crossing a zone boundary, preserving admissibility

---

## Installation

**Requirements:** Python 3.10+

```bash
pip install matplotlib numpy
```

No other dependencies are needed — all search algorithms use only the Python standard library (`heapq`, `collections`).

---

## Usage

### Interactive Route Planner

```bash
python main.py
```

The interactive session walks you through three steps:

1. **Start location** — pick by number or type a name (partial matches work)
2. **Intermediate stops** — add optional waypoints; press Enter when done
3. **Destination** — pick your end point
4. **Departure time** — enter in `HH:MM` format (default `09:00`)

The planner then runs TD-A\* on each leg, advances the clock between legs so congestion is evaluated at the correct time, and prints path details alongside a per-algorithm comparison.

### Example Session

```
  Step 1 - Start location
  > Meera Bhawan

  Step 2 - Intermediate stops (press Enter to skip)
  Add stop 1 (or Enter to finish):

  Step 3 - Destination
  > Library

  Departure time HH:MM [Enter = 09:00]: 08:30

  Route     : Meera Bhawan  ->  Library
  Algorithm : TD-A* (Haversine, time-dependent weights)

  Run? [Y/N]: Y
```

**Output:**
- Console summary: path, distance (metres), estimated arrival time, nodes expanded
- `outputs/user_leg<N>_<time>.png` — graph visualisation with highlighted path
- `outputs/comparison_user.png` — bar chart comparing all 5 algorithms
- `outputs/empirical_vs_theory_user.png` — empirical vs theoretical node expansions

---

## Campus Graph

### Nodes (40 locations)

Hostels, academic buildings, canteens, shops, and key intersections across the BITS Pilani campus — including Clock Tower, Library, Lecture Theatre Complex, SAC, Main Gate, Food Ministry, All Night Canteen, and all major bhawans.

### Edges

Each edge stores a **base weight in metres** derived from Haversine-verified GPS coordinates. The graph is undirected — all paths are bidirectional. Edges are defined once in `graph_data.py` under the alphabetically first endpoint.

### Adding New Locations

1. Add a `"Name": (lat, lon)` entry to `COORDINATES` in `graph_data.py`
2. Add edges to `EDGES` under the new node's alphabetical position
3. Optionally add congestion entries to `CONGESTION_SCHEDULE`

---

## Empirical Validation

`empirical_validation.py` runs all algorithms against 7 benchmark query pairs and compares measured node expansions to theoretical complexity predictions:

| Algorithm | Theoretical Bound | Basis |
|---|---|---|
| BFS | 34 | `min(b^d, N)` bounded by graph size |
| UCS | 34 | `O((V+E) log V)` bounded by `N = 34` |
| Greedy (haversine) | ~20 | Typical `O(b·d)`, mid estimate |
| Greedy (euclidean) | ~20 | Same complexity class as haversine |
| A\* (haversine) | ~18 | Typical `O(b·d)` with good heuristic |
| A\* (euclidean) | ~18 | Same complexity class as haversine |

Results are saved to `empirical_results.csv` and plotted as `empirical_vs_theory.png`.

---

## Output Files

All output files are written to the `outputs/` directory (created automatically):

| File | Description |
|---|---|
| `user_leg<N>_<HHMM>.png` | Campus graph with TD-A\* path for leg N |
| `comparison_user.png` | Algorithm comparison bar chart (nodes expanded, cost) |
| `empirical_vs_theory_user.png` | Predicted vs actual node expansion scatter plot |
| `empirical_results.csv` | Raw benchmark data across all query pairs |

---

## SearchResult Fields

Every algorithm returns a `SearchResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `algorithm` | `str` | Algorithm name and parameters |
| `source` / `goal` | `str` | Start and end node names |
| `path` | `list[str]` | Ordered list of nodes from source to goal |
| `cost` | `float` | Total path cost in metres (0 for BFS hop-based) |
| `nodes_expanded` | `int` | Nodes popped from the frontier |
| `nodes_generated` | `int` | Nodes added to the frontier |
| `frontier_log` | `list` | Per-step `{step, node, g, h, f}` records for visualisation |
| `found` | `bool` | `True` if a path was found |

---

## Acknowledgements

Built as part of an AI course project at BITS Pilani. Node coordinates are real GPS positions; edge weights are computed from the Haversine formula between adjacent landmark pairs.
