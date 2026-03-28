"""empirical_validation.py
Compares empirical node expansion counts from the user's actual query against theoretical bounds from the D2 complexity analysis.
"""
from algorithms import SearchResult

N = 40  # total nodes in the campus graph
THEORY = {
    "BFS": N,
    "UCS": N,
    "Greedy (haversine)": 25,
    "A* (haversine)": 22,}

def run_valid(results: list[SearchResult]) -> dict[str, int]:
    empirical = {}
    for res in results:
        if res.algorithm in THEORY:
            empirical[res.algorithm] = res.nodes_expanded

    print("\n" + "=" * 60)
    print(f"  EMPIRICAL vs THEORY — {results[0].source} → {results[0].goal}")
    print("=" * 60)
    print(f"  {'Algorithm':<25} {'Empirical':>9} {'Theory':>8} {'Ratio':>7}")
    print("  " + "─" * 53)
    for algo, emp_val in empirical.items():
        theory_val = THEORY[algo]
        ratio = emp_val / theory_val
        print(f"  {algo:<25} {emp_val:>9}  {theory_val:>7}  {ratio:>6.2f}x")
    print()

    return empirical
