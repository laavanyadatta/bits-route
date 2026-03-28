#campus_graph.py
#Builds the BITS Pilani campus graph.
#1. Build an adjacency-list graph from EDGES in graph_data.py
#2. Compute Haversine distances on demand (used as the heuristic)
#3. Apply time-dependent congestion: w_eff(e, t) = max(k_u, k_v) * w_base
#4. Provide helper methods used by all four search algorithms

import math
from collections import defaultdict
from graph_data import COORDINATES, EDGES, CONGESTION_SCHEDULE

# Haversine formula
EARTH_RADIUS_M = 6_371_000  # metres

def haversine(node_a: str, node_b: str,
              coords: dict = COORDINATES) -> float:

# This is the heuristic h(n) used in Greedy and A* search.
    lat1, lon1 = coords[node_a]
    lat2, lon2 = coords[node_b]

    # Convert degrees -> radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

LAT_M   = 111_195.0                      # metres per degree of latitude
REF_LAT = math.radians(28.362)           # campus centre latitude for lon scaling
 
def euclidean(node_a: str, node_b: str,
              coords: dict = COORDINATES) -> float:
                
    lat1, lon1 = coords[node_a]
    lat2, lon2 = coords[node_b]
    dy = (lat2 - lat1) * LAT_M
    dx = (lon2 - lon1) * LAT_M * math.cos(REF_LAT)
                
    return math.sqrt(dx * dx + dy * dy)
 

class CampusGraph:
#Sparse Weighted Undirected Graph of the BITS Pilani campus.
#Attributes:
#  nodes : list of node names (str)
# adj  : dict  {node: [(neighbour, base_weight_metres), ...]}
# coords: dict  {node: (lat, lon)}

    def __init__(self):
        self.coords = COORDINATES.copy()
        self.adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._base_weights: dict[tuple[str, str], float] = {}

        # Add edges in both directions
        for u, v, w in EDGES:
            self.adj[u].append((v, w))
            self.adj[v].append((u, w))
            # Store canonical weight (min name first for easy lookup)
            key = (min(u, v), max(u, v))
            self._base_weights[key] = w

        self.nodes = list(self.coords.keys())

    # Heuristics
    def h_haversine(self, node: str, goal: str) -> float:
        return haversine(node, goal, self.coords)

    def h_euclidean(self, node: str, goal: str) -> float:
        return euclidean(node, goal, self.coords)

    # Time-dependent edge weight
    def get_congestion(self, node: str, time_slot: int) -> float:
       #Look up the congestion factor k for a node at a given time slot. Returns 1.0 (free-flow) if no congestion is scheduled.
       #time_slot : integer in [0, 95]  (floor(minutes_since_midnight / 15))
      
        slot_data = CONGESTION_SCHEDULE.get(time_slot, {})
        return slot_data.get(node, 1.0)

    def w_eff(self, u: str, v: str, time_slot: int) -> float:
        """
        Return the effective (time-dependent) weight for edge (u, v)
        at the given time slot.

        w_eff = max(k_u, k_v) * w_base

        The max() ensures that crossing a congested zone boundary uses the stricter (higher) factor — conservative but admissibility-safe."""
        key = (min(u, v), max(u, v))
        w_base = self._base_weights.get(key, 0.0)
        k = max(self.get_congestion(u, time_slot), self.get_congestion(v, time_slot))
        return k * w_base

    def neighbours(self, node: str) -> list[tuple[str, float]]:
        """Return list of (neighbour, base_weight) for a node."""
        return self.adj.get(node, [])

    def neighbours_timed(self, node: str, time_slot: int) -> list[tuple[str, float]]:
        """Return list of (neighbour, effective_weight) at a given time slot."""
        return [(nb, self.w_eff(node, nb, time_slot))
                for nb, _ in self.adj.get(node, [])]

    # Utility
    @staticmethod
    def minutes_to_slot(minutes_since_midnight: int) -> int:
        """Convert minutes-since-midnight (0–1439) to slot index (0–95)."""
        return minutes_since_midnight // 15

    @staticmethod
    def hhmm_to_slot(hhmm: str) -> int:
        """Convert 'HH:MM' string to time slot index."""
        h, m = map(int, hhmm.split(":"))
        return (h * 60 + m) // 15

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(EDGES)

    def degree(self, node: str) -> int:
        return len(self.adj.get(node, []))
