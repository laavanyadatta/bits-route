"""algorithms.py
1. BFS         
2. UCS           
3. Greedy      
4. A*                   

Each function returns a SearchResult namedtuple containing:
- path: list of node names from source to goal
.- cost : total path cost in metres (0 for BFS)
- nodes_expanded: count of nodes popped from the frontier
- nodes_generated: count of nodes added to the frontier
- frontier_log: list of (step, node, g, h, f) for visualisation

All search algorithms are hand-coded; only heapq is used for priority queue

Time-Dependent variants (TD-UCS, TD-A*) are included as separate functions that accept a departure time slot (0–95).
"""

import heapq
from collections import deque
from typing import Callable, Optional
from dataclasses import dataclass, field

from campus_graph import CampusGraph


# Result container
@dataclass
class SearchResult:
    algorithm: str
    source: str
    goal:str
    path:list
    cost:float # total metres (0 for BFS = hop-based)
    nodes_expanded: int # nodes popped from frontier
    nodes_generated: int # nodes added to frontier
    frontier_log:list = field(default_factory=list)
    # Each entry: {"step": int, "node": str, "g": float, "h": float, "f": float}
    found:bool = True


# 1. BFS — Breadth-First Search
def bfs(graph: CampusGraph, source: str, goal: str) -> SearchResult:
    """BFS — expands nodes level by level (FIFO queue).

    - Uses no heuristic and no edge weights
    - Finds the minimum-HOP path
    - Complete: Yes (finite connected graph)
    - Optimal: No (ignores edge weights)

    Tracks:
        nodes_expanded  = nodes popped from the queue
        nodes_generated = nodes added to the queue  """
    
    # frontier holds (node, path_so_far)
    frontier = deque([(source, [source])])
    visited = {source} # explored set prevents cycles

    nodes_expanded = 0
    nodes_generated = 1 # source is the first generated node
    frontier_log= []

    while frontier:
        node, path = frontier.popleft()
        nodes_expanded += 1

        # Log this expansion (g = hop count, h = 0 for BFS)
        frontier_log.append({
            "step": nodes_expanded,
            "node": node,
            "g": len(path) - 1,   # number of hops so far
            "h":0.0,
            "f": float(len(path) - 1),
        })

        if node == goal:
            # Compute actual distance for the found path
            cost = _path_cost(graph, path)
            return SearchResult(algorithm="BFS", source=source, goal=goal,
                path=path, cost=cost, nodes_expanded=nodes_expanded, nodes_generated=nodes_generated, frontier_log=frontier_log,)

        # Expand: add unvisited neighbours
        for neighbour, _ in graph.neighbours(node):
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append((neighbour, path + [neighbour]))
                nodes_generated += 1

    # If we exhaust the frontier without finding the goal
    return SearchResult( algorithm="BFS", source=source, goal=goal,path=[], cost=float("inf"),
        nodes_expanded=nodes_expanded,nodes_generated=nodes_generated, frontier_log=frontier_log,found=False, )


# 2. UCS
def ucs(graph: CampusGraph, source: str, goal: str) -> SearchResult:
    """
    Uniform Cost Search — expands the lowest cumulative-cost node first.
    Equivalent to Dijkstra's algorithm for a single target. Uses a min-heap (priority queue) ordered by g(n).

    - Complete: Yes (all weights > 0).
    - Optimal:  Yes (expands in cost order).
    """
    # heap entries: (cumulative_cost, tie_breaker, node, path)
    counter   = 0
    heap      = [(0.0, counter, source, [source])]
    visited   = set()

    nodes_expanded  = 0
    nodes_generated = 1
    frontier_log    = []

    while heap:
        g, _, node, path = heapq.heappop(heap)

        # Skip if already expanded with a cheaper cost
        if node in visited:
            continue

        visited.add(node)
        nodes_expanded += 1

        frontier_log.append({
            "step": nodes_expanded,
            "node": node,
            "g": round(g, 2),
            "h": 0.0,
            "f": round(g, 2),
        })

        if node == goal:
            return SearchResult(algorithm="UCS", source=source, goal=goal,path=path, cost=g,
                nodes_expanded=nodes_expanded, nodes_generated=nodes_generated, frontier_log=frontier_log, )

        for neighbour, weight in graph.neighbours(node):
            if neighbour not in visited:
                counter += 1
                heapq.heappush(heap, (g + weight, counter,
                                      neighbour, path + [neighbour]))
                nodes_generated += 1

    return SearchResult(
        algorithm="UCS", source=source, goal=goal,
        path=[], cost=float("inf"),
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        frontier_log=frontier_log,
        found=False,
    )


# 3. Greedy Best-First Search
def greedy(graph: CampusGraph, source: str, goal: str,
           heuristic: str = "haversine") -> SearchResult:
    """
    Greedy Best-First Search — expands the node closest to the goal according to heuristic h(n) alone.  Ignores cumulative cost g(n).

    - Complete: Yes (with explored set on a finite graph).
    - Optimal:  No  (can find longer paths if h is misleading).
    """
    h_fn = _get_heuristic(graph, heuristic)

    counter  = 0
    heap     = [(h_fn(source, goal), counter, source, [source])]
    visited  = set()

    nodes_expanded  = 0
    nodes_generated = 1
    frontier_log    = []

    while heap:
        h_val, _, node, path = heapq.heappop(heap)

        if node in visited:
            continue

        visited.add(node)
        nodes_expanded += 1

        g_val = _path_cost(graph, path)   # actual cost (for logging only)
        frontier_log.append({
            "step": nodes_expanded,
            "node": node,
            "g": round(g_val, 2),
            "h": round(h_val, 2),
            "f": round(h_val, 2),   # f = h only for Greedy
        })

        if node == goal:
            return SearchResult(
                algorithm=f"Greedy ({heuristic})",
                source=source, goal=goal,
                path=path, cost=g_val,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                frontier_log=frontier_log,
            )

        for neighbour, _ in graph.neighbours(node):
            if neighbour not in visited:
                h_nb = h_fn(neighbour, goal)
                counter += 1
                heapq.heappush(heap, (h_nb, counter,
                                      neighbour, path + [neighbour]))
                nodes_generated += 1

    return SearchResult(
        algorithm=f"Greedy ({heuristic})",
        source=source, goal=goal,
        path=[], cost=float("inf"),
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        frontier_log=frontier_log,
        found=False,
    )


# 4. A* Search
def astar(graph: CampusGraph, source: str, goal: str,
          heuristic: str = "haversine") -> SearchResult:
    # A* Search — expands the node with the lowest f(n) = g(n) + h(n).
    h_fn = _get_heuristic(graph, heuristic)

    counter   = 0
    start_h   = h_fn(source, goal)
    heap      = [(start_h, counter, 0.0, source, [source])]
    # heap entry: (f, tie_breaker, g, node, path)
    visited   = {}    # {node: best_g_seen}  — closed list

    nodes_expanded  = 0
    nodes_generated = 1
    frontier_log    = []

    while heap:
        f, _, g, node, path = heapq.heappop(heap)

        # Skip if we have already found a cheaper path to this node
        if node in visited and visited[node] <= g:
            continue

        visited[node] = g
        nodes_expanded += 1

        h_val = h_fn(node, goal)
        frontier_log.append({
            "step": nodes_expanded,
            "node": node,
            "g": round(g, 2),
            "h": round(h_val, 2),
            "f": round(f, 2),
        })

        if node == goal:
            return SearchResult(
                algorithm=f"A* ({heuristic})",
                source=source, goal=goal,
                path=path, cost=g,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                frontier_log=frontier_log,
            )

        for neighbour, weight in graph.neighbours(node):
            g_new = g + weight
            if neighbour not in visited or visited[neighbour] > g_new:
                h_nb  = h_fn(neighbour, goal)
                f_new = g_new + h_nb
                counter += 1
                heapq.heappush(heap, (f_new, counter,
                                      g_new, neighbour,
                                      path + [neighbour]))
                nodes_generated += 1

    return SearchResult(
        algorithm=f"A* ({heuristic})",
        source=source, goal=goal,
        path=[], cost=float("inf"),
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        frontier_log=frontier_log,
        found=False,
    )


# 4b. Time-Dependent A*  (TD-A*)
def td_astar(graph: CampusGraph, source: str, goal: str,
             departure_hhmm: str = "08:48",
             walking_speed_mps: float = 1.4) -> SearchResult:
    """
    Time-Dependent A* — extends A* to use w_eff(e, t) for edge costs.

    Parameters
    ----------
    departure_hhmm    : departure time as 'HH:MM' string
    walking_speed_mps : walking speed in m/s (default 1.4 m/s ≈ 5 km/h)

    State space: V x T = 34 x 96 = 3,264 (node, slot) pairs.
    """
    start_slot = graph.hhmm_to_slot(departure_hhmm)

    # heap entry: (f, tie_breaker, g, node, t_slot, path)
    counter = 0
    h0      = graph.h_haversine(source, goal)
    heap    = [(h0, counter, 0.0, source, start_slot, [source])]
    visited = {}   # {(node, slot): best_g}

    nodes_expanded  = 0
    nodes_generated = 1
    frontier_log    = []

    while heap:
        f, _, g, node, t_slot, path = heapq.heappop(heap)

        state = (node, t_slot)
        if state in visited and visited[state] <= g:
            continue

        visited[state] = g
        nodes_expanded += 1

        h_val = graph.h_haversine(node, goal)
        frontier_log.append({
            "step":   nodes_expanded,
            "node":   node,
            "t_slot": t_slot,
            "g":      round(g, 2),
            "h":      round(h_val, 2),
            "f":      round(f, 2),
        })

        if node == goal:
            return SearchResult(
                algorithm=f"TD-A* (depart {departure_hhmm})",
                source=source, goal=goal,
                path=path, cost=g,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                frontier_log=frontier_log,
            )

        for neighbour, _ in graph.neighbours(node):
            w   = graph.w_eff(node, neighbour, t_slot)
            g_new = g + w

            # Advance time slot by the time taken to walk this edge
            travel_slots = max(1, int(w / (walking_speed_mps * 15 * 60)))
            t_new = min(t_slot + travel_slots, 95)   # cap at last slot

            new_state = (neighbour, t_new)
            if new_state not in visited or visited[new_state] > g_new:
                h_nb  = graph.h_haversine(neighbour, goal)
                f_new = g_new + h_nb
                counter += 1
                heapq.heappush(heap, (f_new, counter, g_new,
                                      neighbour, t_new,
                                      path + [neighbour]))
                nodes_generated += 1

    return SearchResult(
        algorithm=f"TD-A* (depart {departure_hhmm})",
        source=source, goal=goal,
        path=[], cost=float("inf"),
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        frontier_log=frontier_log,
        found=False,
    )


# Internal helpers
def _get_heuristic(graph: CampusGraph,
                   name: str) -> Callable[[str, str], float]:
    """Return the named heuristic function bound to the graph."""
    if name == "haversine":
        return graph.h_haversine
    else:
        raise ValueError(f"Unknown heuristic '{name}'. ")


def _path_cost(graph: CampusGraph, path: list[str]) -> float:
    """Compute the total base-weight cost of a node path."""
    return sum(
        next(w for nb, w in graph.neighbours(path[i]) if nb == path[i + 1])
        for i in range(len(path) - 1)
    )
