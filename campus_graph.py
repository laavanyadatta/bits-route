#campus_graph.py
#Builds the BITS Pilani campus graph.
#1. Build an adjacency-list graph from EDGES in graph_data.py
#2. Compute Haversine distances on demand 

import math
from collections import defaultdict
from graph_data import COORDINATES, EDGES, CONGESTION_SCHEDULE
EARTH_RADIUS_M = 6_371_000  # metres

def haversine(node_a: str, node_b: str, coords: dict = COORDINATES) -> float:
    lat1, lon1 = coords[node_a]
    lat2, lon2 = coords[node_b]
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def euclidean(node_a: str, node_b: str,coords: dict = COORDINATES) -> float:
    x1, y1 = coords[node_a]
    x2, y2 = coords[node_b]
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)

class CampusGraph:
    def __init__(self):
        self.coords = COORDINATES.copy()
        self.adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._base_weights: dict[tuple[str, str], float] = {}

        for u, v, w in EDGES:
            self.adj[u].append((v, w))
            self.adj[v].append((u, w))
            key = (min(u, v), max(u, v))
            self._base_weights[key] = w

        self.nodes = list(self.coords.keys())

    def h_haversine(self, node: str, goal: str) -> float:
        return haversine(node, goal, self.coords)

    def h_euclidean(self, node: str, goal: str) -> float:
        return euclidean(node, goal, self.coords) 
    
    def get_congestion(self, node: str, time_slot: int) -> float:
        slot_data = CONGESTION_SCHEDULE.get(time_slot, {})
        return slot_data.get(node, 1.0)

    def w_eff(self, u: str, v: str, time_slot: int) -> float:
        key = (min(u, v), max(u, v))
        w_base = self._base_weights.get(key, 0.0)
        k = max(self.get_congestion(u, time_slot), self.get_congestion(v, time_slot))
        return k * w_base

    def neighbours(self, node: str) -> list[tuple[str, float]]:
        return self.adj.get(node, [])

    def neighbours_timed(self, node: str, time_slot: int) -> list[tuple[str, float]]:
        return [(nb, self.w_eff(node, nb, time_slot))
                for nb, _ in self.adj.get(node, [])]

    @staticmethod
    def minutes_to_slot(minutes_since_midnight: int) -> int:
        return minutes_since_midnight // 15

    @staticmethod
    def hhmm_to_slot(hhmm: str) -> int:
        h, m = map(int, hhmm.split(":"))
        return (h * 60 + m) // 15

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(EDGES)

    def degree(self, node: str) -> int:
        return len(self.adj.get(node, []))
