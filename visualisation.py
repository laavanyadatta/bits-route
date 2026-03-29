"""visualisation.py

draw_graph(graph, result, title)
    Draw the campus graph with the found path highlighted.
plot_comparison(results)
    Bar chart comparing all four algorithms on nodes expanded and cost.
plot_empirical_vs_theory(empirical_data, theory_data)
    Scatter/bar comparison between predicted and actual node counts.
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
 
from graph_data import EDGES
from algorithms import SearchResult
from campus_graph import CampusGraph

ALGO_COLORS = {
    "BFS": "#7BC3FF", 
    "UCS": "#7BDE7E", 
    "Greedy (haversine)": "#FFEE6B", 
    "Greedy (euclidean)":"#FFA75F",
    "A* (haversine)": "#FF5E4F",
    "A* (euclidean)":"#E060B3", 
    "TD-A*": "#9E6BA5", }

NODE_DEFAULT = "#BBDEFB"
NODE_PATH = "#FFEE59"
NODE_WAYPOINT = "#FF9800"
NODE_SOURCE = "#66BB6A"
NODE_GOAL = "#3590FF"
EDGE_DEFAULT = "#6D8693"
EDGE_PATH = "#9E6BA5"


def _get_positions(coords: dict) -> dict[str, tuple[float, float]]:
    lat0, lon0 = coords["Clock Tower"]
    mpdlat = 111_320
    mpdlon = 111_320 * math.cos(math.radians(lat0))
    return {
        name: ((lon - lon0) * mpdlon, (lat - lat0) * mpdlat)
        for name, (lat, lon) in coords.items()}

def advance_time(hhmm, metres, speed_mpm=80.0):
    hh, mm = int(hhmm[:2]), int(hhmm[3:])
    total_min = int(hh * 60 + mm + metres / speed_mpm)
    return f"{(total_min // 60) % 24:02d}:{total_min % 60:02d}"

def draw_graph(graph: CampusGraph, result: SearchResult, title: str = "", save_path: str = None) -> None:
    pos = _get_positions(graph.coords)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#F5F5F5")
    fig.patch.set_facecolor("#FAFAFA")

    path_set = set(zip(result.path, result.path[1:]))  # set of path edges

    for u, v, _ in EDGES:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_path = (u, v) in path_set or (v, u) in path_set
        ax.plot([x0, x1], [y0, y1],
                color=EDGE_PATH if is_path else EDGE_DEFAULT,
                linewidth=3.0 if is_path else 0.8,
                zorder=1 if is_path else 0,
                alpha=1.0 if is_path else 0.5)

    for name, (x, y) in pos.items():
        if name == result.source:
            color, size, zorder = NODE_SOURCE, 180, 4
        elif name == result.goal:
            color, size, zorder = NODE_GOAL, 180, 4
        elif name in result.path:
            color, size, zorder = NODE_PATH, 140, 3
        else:
            color, size, zorder = NODE_DEFAULT, 80, 2

        ax.scatter(x, y, s=size, color=color, edgecolors="#37474F", linewidths=0.8, zorder=zorder)

        fontsize = 7.5 if name in result.path else 5.5
        weight = "bold" if name in result.path else "normal"

        label_offsets = { "Library": (6, 8), "New Workshop": (-12, -10),"Main Gate": (-12, -10),}
        dx, dy = label_offsets.get(name, (4, 4))

        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=fontsize, fontweight=weight, color="#212121", zorder=5)

    if len(result.path) >= 2:
        for i in range(len(result.path) - 1):
            u, v = result.path[i], result.path[i + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=EDGE_PATH, lw=1.5), zorder=5)

    legend_elements = [
        mpatches.Patch(color=NODE_SOURCE, label="Source"),
        mpatches.Patch(color=NODE_GOAL, label="Goal"),
        mpatches.Patch(color=NODE_PATH, label="Path nodes"),]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    info = (f"Algorithm : {result.algorithm}\n"
            f"Path cost : {result.cost:.1f} m\n"
            f"Hops : {len(result.path)-1}\n"
            f"Expanded : {result.nodes_expanded} nodes\n"
            f"Generated : {result.nodes_generated} nodes")

    ax.text(0.98, 0.98, info, transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.set_title(title or f"{result.algorithm}: "
                 f"{result.source} → {result.goal}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("East–West (metres from centroid)")
    ax.set_ylabel("North–South (metres from centroid)")
    ax.set_aspect("equal")

    plt.tight_layout()
    fname = save_path or f"path_{result.algorithm.split()[0].lower()}.png"
    plt.subplots_adjust(top=0.88)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close()

def draw_multistop(graph: CampusGraph, route_data: dict,title: str = "", save_path: str = None) -> None:
    pos   = _get_positions(graph.coords)
    stops = route_data["stops"]
    legs  = route_data["legs"]
 
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#F5F5F5")
    fig.patch.set_facecolor("#FAFAFA")
 
    edge_to_leg = {}
    for leg_idx, leg in enumerate(legs):
        for i in range(len(leg.path) - 1):
            u, v = leg.path[i], leg.path[i + 1]
            edge_to_leg[(u, v)] = leg_idx
            edge_to_leg[(v, u)] = leg_idx
 
    all_path_nodes = set()
    for leg in legs:
        all_path_nodes.update(leg.path)
 
    for u, v, _ in EDGES:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]; x1, y1 = pos[v]
        leg_idx = edge_to_leg.get((u, v), edge_to_leg.get((v, u), None))
        if leg_idx is not None:
            color = EDGE_PATH
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=3.0, zorder=2, alpha=0.9)
        else:
            ax.plot([x0, x1], [y0, y1], color=EDGE_DEFAULT, linewidth=0.8, zorder=0, alpha=0.4)
 
    for leg_idx, leg in enumerate(legs):
        color = color = EDGE_PATH
        for i in range(len(leg.path) - 1):
            u, v = leg.path[i], leg.path[i + 1]
            if u in pos and v in pos:
                ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle="->", color=color, lw=2.0), zorder=5)
 
    for name, (x, y) in pos.items():
        if name == stops[0]:       
            c, s, z = NODE_SOURCE, 200, 5
        elif name == stops[-1]:   
            c, s, z = NODE_GOAL,200, 5
        elif name in stops[1:-1]:    
            c, s, z = NODE_WAYPOINT, 170, 4
        elif name in all_path_nodes: 
            c, s, z = NODE_PATH, 130, 3
        else:
            c, s, z = NODE_DEFAULT, 70, 2
 
        ax.scatter(x, y, s=s, color=c, edgecolors="#37474F", linewidths=0.8, zorder=z)
 
        label_offsets = {"Library": (6, 8), "New Workshop": (-12, -10), "Main Gate": (-12, -10)}
        dx, dy = label_offsets.get(name, (4, 4))
        bold = name in all_path_nodes
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5 if bold else 5.5, fontweight="bold" if bold else "normal", color="#212121", zorder=6)
 
        if name in stops:
            idx = stops.index(name)
            ax.text(x, y, str(idx + 1), ha="center", va="center",fontsize=7, fontweight="bold", color="white", zorder=7)
 
    legend_els = [
        mpatches.Patch(color=NODE_SOURCE, label=f"1. {stops[0]} (start)"),
        mpatches.Patch(color=NODE_GOAL, label=f"{len(stops)}. {stops[-1]} (end)"),
    ]
    for i, wp in enumerate(stops[1:-1], 2):
        legend_els.append(mpatches.Patch(color=NODE_WAYPOINT, label=f"{i}. {wp} (waypoint)"))
    
    ax.legend(handles=legend_els, fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0, 1.02, 1, 0.1),loc="lower left", ncol=2,mode="expand",borderaxespad=0)
 
    total_cost = route_data["total_cost"]
    total_hops = route_data["total_hops"]
    total_exp  = sum(l.nodes_expanded for l in legs)

    ax.text(0.98, 0.98,
        f"Total dist: {total_cost:.1f} m\n"
        f"Total hops: {total_hops}\n"
        f"Nodes exp : {total_exp}\n"
        f"Departure : {route_data['times'][0]}\n"
        f"Arrival   : {advance_time(route_data['times'][0], total_cost)}",
        transform=ax.transAxes, va="top", ha="right",
        fontsize=6,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
 
    ax.set_title("")
    fig.text(0.5, 0.01, title or f"TD-A* Multi-Stop Route  ({len(stops)} stops)", ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("East-West (metres from Clock Tower)")
    ax.set_ylabel("North-South (metres from Clock Tower)")
    ax.set_aspect("equal")
    plt.tight_layout()
 
    fname = save_path or "multistop_route.png"
    plt.subplots_adjust(top=0.88, bottom=0.08)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close()


def plot_comparison(results: list[SearchResult], save_path: str = "comparison.png") -> None:
    names = [r.algorithm for r in results]
    expand = [r.nodes_expanded for r in results]
    generate = [r.nodes_generated for r in results]
    costs = [r.cost for r in results]
    colors = [ALGO_COLORS.get(r.algorithm, "#78909C") for r in results]

    x = np.arange(len(names))
    width = 0.28
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Algorithm Comparison — "
                 f"{results[0].source} → {results[0].goal}", fontsize=13, fontweight="bold")

    for ax, vals, ylabel, title in zip(axes, [expand, generate, costs],
            ["Count", "Count", "Metres"],
            ["Nodes Expanded", "Nodes Generated", "Path Cost (m)"]):

        bars = ax.bar(x, vals, color=colors, edgecolor="#37474F",linewidth=0.7, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01, f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f" Saved comparison: {save_path}")
    plt.close()


def plot_empirical_vs_theory(empirical, theory, save_path="empirical_vs_theory.png"):
    algos = list(empirical.keys())
    emp_v = [empirical[a] for a in algos]
    the_v = [theory[a] for a in algos]

    x = np.arange(len(algos))
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - 0.2, emp_v, 0.38, label="Empirical (actual)", color="#42A5F5", edgecolor="#1565C0", linewidth=0.8)
    bars2 = ax.bar(x + 0.2, the_v, 0.38, label="Theoretical bound", color="#EF9A9A", edgecolor="#F15D5D", linewidth=0.8, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=9)
    ax.set_ylabel("Nodes Expanded", fontsize=10)
    ax.set_title( f"Empirical vs Theoretical Node Expansion", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(list(bars1) + list(bars2), emp_v + the_v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(emp_v + the_v) * 0.01,
                f"{val}", ha="center", va="bottom", fontsize=8)

    ax.axhline(40, color="#37474F", linestyle=":", linewidth=1.2, label="N = 40 (finite graph ceiling)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved empirical vs theory: {save_path}")
    plt.close()
