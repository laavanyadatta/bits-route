"""visualisation.py

draw_graph(graph, result, title)
    Draw the campus graph with the found path highlighted.

plot_comparison(results)
    Bar chart comparing all four algorithms on nodes expanded and cost.

plot_empirical_vs_theory(empirical_data, theory_data)
    Scatter/bar comparison between predicted and actual node counts.

All plots are saved as PNG files & displayed .
"""

from graph_data import EDGES, COORDINATES
from algorithms import SearchResult
from campus_graph import CampusGraph
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import math
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend for file saving


ALGO_COLORS = {
    "BFS":                "#2196F3",   # blue
    "UCS":                "#4CAF50",   # green
    "Greedy (haversine)": "#FFE62C",   # yellow
    "Greedy (euclidean)": "#FF8F32",   # orange
    "A* (haversine)":     "#A51FBD",   # purple
}

NODE_DEFAULT = "#BBDEFB"
NODE_EXPANDED = "#FFEE59"    # yellow while expanding
NODE_PATH = "#EF5350"    # red when on final path
NODE_SOURCE = "#66BB6A"    # green
NODE_GOAL = "#3590FF"    # blue
EDGE_DEFAULT = "#90A4AE"
EDGE_PATH = "#FF56B6"


# Coordinate projection → plot positions
def _get_positions(coords: dict) -> dict[str, tuple[float, float]]:
    """
    Convert (lat, lon) → (x, y) in metres.
    Origin is set to Clock Tower.
    """

    lat0, lon0 = coords["Clock Tower"]

    mpdlat = 111_320
    mpdlon = 111_320 * math.cos(math.radians(lat0))

    return {
        name: ((lon - lon0) * mpdlon, (lat - lat0) * mpdlat)
        for name, (lat, lon) in coords.items()
    }


# 1. Static path map
def draw_graph(graph: CampusGraph, result: SearchResult, title: str = "", save_path: str = None) -> None:

   # Nodes on the path are shown in red, source in green, goal in blue.
    pos = _get_positions(graph.coords)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#F5F5F5")
    fig.patch.set_facecolor("#FAFAFA")

    path_set = set(zip(result.path, result.path[1:]))  # set of path edges

    # Draw all edges first (behind nodes)
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

    # Draw nodes
    for name, (x, y) in pos.items():
        if name == result.source:
            color, size, zorder = NODE_SOURCE, 180, 4
        elif name == result.goal:
            color, size, zorder = NODE_GOAL, 180, 4
        elif name in result.path:
            color, size, zorder = NODE_PATH, 140, 3
        else:
            color, size, zorder = NODE_DEFAULT, 80, 2

        ax.scatter(x, y, s=size, color=color,
                   edgecolors="#37474F", linewidths=0.8, zorder=zorder)

        # Label: smaller font for non-path nodes
        fontsize = 7.5 if name in result.path else 5.5
        weight = "bold" if name in result.path else "normal"
        # Custom offsets to avoid overlap
        label_offsets = {
            "Library": (6, 8),
            "New Workshop": (-12, -10),
            "Main Gate": (-12, -10),   # push right towards border
        }

        dx, dy = label_offsets.get(name, (4, 4))

        ax.annotate(name, (x, y),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=fontsize, fontweight=weight,
                    color="#212121", zorder=5)

    # Draw path arrows
    if len(result.path) >= 2:
        for i in range(len(result.path) - 1):
            u, v = result.path[i], result.path[i + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.annotate("",
                        xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->",
                                        color=EDGE_PATH, lw=2.0),
                        zorder=5)

    # Legend
    legend_elements = [
        mpatches.Patch(color=NODE_SOURCE, label="Source"),
        mpatches.Patch(color=NODE_GOAL, label="Goal"),
        mpatches.Patch(color=NODE_PATH, label="Path nodes"),
        mpatches.Patch(color=NODE_DEFAULT, label="Other nodes"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    # Annotations
    info = (f"Algorithm : {result.algorithm}\n"
            f"Path cost : {result.cost:.1f} m\n"
            f"Hops : {len(result.path)-1}\n"
            f"Expanded : {result.nodes_expanded} nodes\n"
            f"Generated : {result.nodes_generated} nodes")
    ax.text(0.98, 0.98, info, transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.85))

    ax.set_title(title or f"{result.algorithm}: "
                 f"{result.source} → {result.goal}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("East–West (metres from centroid)")
    ax.set_ylabel("North–South (metres from centroid)")
    ax.set_aspect("equal")

    plt.tight_layout()
    fname = save_path or f"path_{result.algorithm.split()[0].lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close()

# 2. Algorithm comparison bar chart


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
                 f"{results[0].source} → {results[0].goal}",
                 fontsize=13, fontweight="bold")

    for ax, vals, ylabel, title in zip(
            axes,
            [expand, generate, costs],
            ["Count", "Count", "Metres"],
            ["Nodes Expanded", "Nodes Generated", "Path Cost (m)"]):

        bars = ax.bar(x, vals, color=colors, edgecolor="#37474F",
                      linewidth=0.7, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved comparison: {save_path}")
    plt.close()


# 3. Empirical vs Theoretical comparison
def plot_empirical_vs_theory(
        empirical: dict[str, int],
        theory:    dict[str, int | str],
        query_label: str = "",
        save_path: str = "empirical_vs_theory.png") -> None:
    """
    Bar chart comparing actual (empirical) nodes expanded
    against the theoretical O(·) predictions from the report.

    Parameters
    ----------
    empirical : {algorithm_name: actual_nodes_expanded}
    theory    : {algorithm_name: theoretical_upper_bound}
    """
    algos = list(empirical.keys())
    emp_v = [empirical[a] for a in algos]
    the_v = [theory[a] for a in algos]

    x = np.arange(len(algos))
    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - 0.2, emp_v, 0.38, label="Empirical (actual)",
                   color="#42A5F5", edgecolor="#1565C0", linewidth=0.8)
    bars2 = ax.bar(x + 0.2, the_v, 0.38, label="Theoretical bound",
                   color="#EF9A9A", edgecolor="#B71C1C", linewidth=0.8,
                   alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=9)
    ax.set_ylabel("Nodes Expanded", fontsize=10)
    ax.set_title(
        f"Empirical vs Theoretical Node Expansion"
        + (f"\n{query_label}" if query_label else ""),
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(list(bars1) + list(bars2),
                        emp_v + the_v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(emp_v + the_v) * 0.01,
                f"{val}", ha="center", va="bottom", fontsize=8)

    # Horizontal line at N=34 (finite graph ceiling)
    ax.axhline(34, color="#37474F", linestyle=":", linewidth=1.2,
               label="N = 34 (finite graph ceiling)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved empirical vs theory: {save_path}")
    plt.close()
