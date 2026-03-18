# graph_data.py
# Graph type : Sparse Weighted Undirected Graph
# Nodes : 36 verified campus locations (GPS coordinates)
# Edges : 76 walkable path segments with Haversine-verified distances

# All edge weights are in METRES.
# All coordinates are (latitude, longitude) in decimal degrees.

# NODE COORDINATES
# Each entry: "Node Name": (latitude, longitude)
NODE_COORDINATES = {
    "Clock Tower": (28.36412,  75.58704),
    "Meera Bhawan": (28.357125, 75.585593),
    "Ram Bhawan": (28.36230,  75.58636),
    "Budh Bhawan":(28.360722, 75.586769),
    "Krishna Bhawan":(28.36273,  75.58838),
    "Gandhi Bhawan":(28.36112,  75.58841),
    "Shankar Bhawan": (28.359770, 75.588494),
    "Vyas Bhawan":(28.358299, 75.588800),
    "Saraswati Temple":(28.358129, 75.587979),
    "Bhagirath Bhawan":(28.36146,  75.58960),
    "Vishwakarma Bhawan": (28.36294,  75.58924),
    "CVR Bhawan":(28.36156,  75.59107),
    "Malviya Bhawan": (28.361601, 75.585337),
    "Srinivasa Bhawan": (28.365601, 75.587347),
    "PIEDS":(28.36616,  75.58771),
    "Library":(28.365336, 75.588434),
    "Rana Pratap Bhawan":(28.362947, 75.590724),
    "Ashok Bhawan":(28.361347, 75.590787),
    "Lecture Theatre Complex":(28.365110, 75.589873),
    "FD1":(28.364287, 75.588781),
    "FD2": (28.363674, 75.588143),
    "FD3":(28.363420, 75.585924),
    "Food Ministry":(28.36489,  75.58826),
    "New Workshop": (28.365325, 75.587931),
    "All Night Canteen":(28.360141, 75.589384),
    "SAC":(28.360670, 75.585770),
    "Akshay Supermarket": (28.357148, 75.589956),
    "Cnot":(28.356218, 75.591457),
    "Looters":(28.361335, 75.585653),
    "BET-TACT": (28.362791, 75.585486),
    "Birla Balika Vidyapeeth":(28.359555, 75.586984),
    "Birla Shishu Vihar": (28.35738,  75.58727),
    "NAB": (28.36225,  75.58728),
    "Main Gate": (28.361022, 75.592976),
    "Vfast":(28.360911, 75.592074),
}

# EDGES
# Each tuple: ("Node A", "Node B", distance_in_metres)
# Distances are Haversine-verified straight-line values between GPS coords
EDGES = [
    ("Clock Tower",         "FD2",                      118.77),
    ("Clock Tower",         "FD3",                      134.10),
    ("Clock Tower",         "Food Ministry",             146.90),

    ("Meera Bhawan",        "Birla Shishu Vihar",        166.53),

    ("Ram Bhawan",          "Budh Bhawan",               179.97),
    ("Ram Bhawan",          "Krishna Bhawan",            203.35),
    ("Ram Bhawan",          "Malviya Bhawan",            126.73),
    ("Ram Bhawan",          "FD3",                       131.64),
    ("Ram Bhawan",          "NAB",                        90.19),

    ("Budh Bhawan",         "Gandhi Bhawan",             166.56),
    ("Budh Bhawan",         "SAC",                        97.92),
    ("Budh Bhawan",         "Looters",                   128.73),
    ("Budh Bhawan",         "Birla Balika Vidyapeeth",   131.46),
    ("Budh Bhawan",         "NAB",                       177.11),

    ("Krishna Bhawan",      "Gandhi Bhawan",             179.05),
    ("Krishna Bhawan",      "Vishwakarma Bhawan",         87.33),
    ("Krishna Bhawan",      "FD1",                       177.52),
    ("Krishna Bhawan",      "FD2",                       107.50),
    ("Krishna Bhawan",      "NAB",                       120.14),

    ("Gandhi Bhawan",       "Shankar Bhawan",            150.34),
    ("Gandhi Bhawan",       "Bhagirath Bhawan",          122.42),
    ("Gandhi Bhawan",       "Vishwakarma Bhawan",        218.06),
    ("Gandhi Bhawan",       "All Night Canteen",         144.68),
    ("Gandhi Bhawan",       "Birla Balika Vidyapeeth",   223.05),
    ("Gandhi Bhawan",       "NAB",                       167.37),

    ("Shankar Bhawan",      "Vyas Bhawan",               166.29),
    ("Shankar Bhawan",      "All Night Canteen",          96.36),

    ("Vyas Bhawan",         "Saraswati Temple",           82.53),
    ("Vyas Bhawan",         "Akshay Supermarket",        170.81),

    ("Saraswati Temple",    "Birla Balika Vidyapeeth",   186.07),
    ("Saraswati Temple",    "Birla Shishu Vihar",        108.40),

    ("Bhagirath Bhawan",    "Vishwakarma Bhawan",        168.30),
    ("Bhagirath Bhawan",    "CVR Bhawan",                144.27),
    ("Bhagirath Bhawan",    "Rana Pratap Bhawan",        198.58),
    ("Bhagirath Bhawan",    "Ashok Bhawan",              116.82),
    ("Bhagirath Bhawan",    "All Night Canteen",         148.18),

    ("Vishwakarma Bhawan",  "CVR Bhawan",                235.82),
    ("Vishwakarma Bhawan",  "Rana Pratap Bhawan",        145.21),
    ("Vishwakarma Bhawan",  "Ashok Bhawan",              233.00),
    ("Vishwakarma Bhawan",  "FD1",                       156.37),
    ("Vishwakarma Bhawan",  "FD2",                       134.84),

    ("CVR Bhawan",          "Rana Pratap Bhawan",        157.90),
    ("CVR Bhawan",          "Ashok Bhawan",               36.44),  # shortest edge

    ("Library",             "Lecture Theatre Complex",   143.02),
    ("Library",             "FD1",                       121.48),
    ("Library",             "Food Ministry",              52.43),  # 2nd shortest edge

    ("Rana Pratap Bhawan",  "Lecture Theatre Complex",   254.52),
    ("Rana Pratap Bhawan",  "FD1",                       241.55),
    ("Rana Pratap Bhawan",  "FD2",                       265.16),

    ("Malviya Bhawan",      "FD3",                       210.26),
    ("Malviya Bhawan",      "SAC",                       111.86),
    ("Malviya Bhawan",      "Looters",                    42.79),

    ("Srinivasa Bhawan",    "PIEDS",                      71.59),
    ("Srinivasa Bhawan",    "Library",                   110.36),
    ("Srinivasa Bhawan",    "Lecture Theatre Complex",   253.11),
    ("Srinivasa Bhawan",    "Food Ministry",             119.29),
    ("Srinivasa Bhawan",    "New Workshop",               64.86),

    ("PIEDS",               "Library",                   115.82),
    ("PIEDS",               "Lecture Theatre Complex",   241.71),
    ("PIEDS",               "New Workshop",               95.33),

    ("FD3",                 "BET-TACT",                   82.03),

    ("Food Ministry",       "New Workshop",               58.10),

    ("Ashok Bhawan",        "All Night Canteen",         191.91),

    ("Lecture Theatre Complex", "Food Ministry",         159.71),
    ("Lecture Theatre Complex", "New Workshop",          191.51),

    ("FD1",                 "FD2",                        92.43),
    ("FD1",                 "Food Ministry",              84.23),
    ("FD1",                 "New Workshop",              142.26),

    ("FD2",                 "Food Ministry",             135.70),
    ("FD2",                 "NAB",                       179.45),

    ("All Night Canteen",   "SAC",                       358.49),
    ("All Night Canteen",   "Birla Balika Vidyapeeth",   243.71),
    ("All Night Canteen",   "Vfast",                     276.79),

    ("Akshay Supermarket",  "Cnot",                      179.63),

    ("Main Gate",           "Vfast",                      89.12), 
    ("Vfast",               "Ashok Bhawan",              134.94),  
]

# Congestion schedule for L3 
# Format: { time_slot_index: {node_name: k_factor} }
# Time slot index = floor(minutes_since_midnight / 15)
# Slot 0  = 00:00–00:14,  Slot 31 = 07:45–07:59, Slot 35 = 08:45–08:59, etc.
# Only non-1.0 slots are listed.  All others default to k = 1.0.
# Helper: convert "HH:MM" to slot index

def _slot(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m) // 15

# Zone member lists (used to expand slot entries)
_Z1 = ["LTC", "FD1", "Library", "Food Ministry", "Srinivasa Bhawan","New Workshop", "Vishwakarma Bhawan", "PIEDS", "Lecture Theatre Complex"]
_Z1_ACADEMIC = ["Lecture Theatre Complex", "FD1", "Srinivasa Bhawan", "Vishwakarma Bhawan"]
_Z1_FULL = ["Lecture Theatre Complex", "FD1", "Food Ministry","Srinivasa Bhawan", "Vishwakarma Bhawan","New Workshop", "Library"]

_Z2 = ["NAB", "Clock Tower", "FD2", "FD3", "Krishna Bhawan", "Gandhi Bhawan", "Ram Bhawan", "Budh Bhawan"]

_Z3 = ["Meera Bhawan", "Birla Shishu Vihar","Birla Balika Vidyapeeth", "Saraswati Temple"]

_Z4_EVE = ["Shankar Bhawan", "All Night Canteen", "Akshay Supermarket"]
_Z4_NIGHT = ["All Night Canteen"]

_EXT = ["Bhagirath Bhawan", "Rana Pratap Bhawan","Ashok Bhawan", "CVR Bhawan", "SAC", "Malviya Bhawan", "Looters Truck", "Vfast"]

def _make_slot_entry(nodes, k):
    #Return {node: k} dict for a list of nodes.
    return {n: k for n in nodes}

# Build congestion schedule as {slot_index: {node: k}}
CONGESTION_SCHEDULE: dict[int, dict[str, float]] = {}

def _add_slots(start_hhmm, end_hhmm, nodes, k):
    # Mark every 15-min slot in [start_hhmm, end_hhmm) with congestion k for the given node list.
    s = _slot(start_hhmm)
    e = _slot(end_hhmm)
    entry = _make_slot_entry(nodes, k)
    for idx in range(s, e):
        if idx not in CONGESTION_SCHEDULE:
            CONGESTION_SCHEDULE[idx] = {}
        CONGESTION_SCHEDULE[idx].update(entry)

# 07:45–08:00  morning rush  (Z1 food + Z2 + Z3)
_add_slots("07:45", "08:00", ["FD1", "Food Ministry", "New Workshop","Srinivasa Bhawan", "Vishwakarma Bhawan","Malviya Bhawan"], 2.0)
_add_slots("07:45", "08:00", _Z2 + _Z3, 2.0)

# Classchange rushes (Z1 academic + Z2 + Z3, k=2.5)
for start, end in [("08:45", "09:00"), ("09:45", "10:00"), ("11:45", "12:00"), ("12:45", "13:00"),("14:45", "15:00"), ("15:45", "16:00"), ("16:45", "17:00"), ("17:45", "18:00")]:
    _add_slots(start, end, _Z1_ACADEMIC + ["FD1"], 2.5)
    _add_slots(start, end, _Z2, 2.5)
    _add_slots(start, end, _Z3, 2.0)

_add_slots("10:45", "11:00", ["Lecture Theatre Complex", "FD1","New Workshop", "Vishwakarma Bhawan"], 2.5)
_add_slots("10:45", "11:00", _Z2, 2.5)

# Full Z1 (including FM) for 11:45–, 12:45–, 16:45–, 17:45–
for start, end in [("11:45", "12:00"), ("12:45", "13:00"), ("16:45", "17:00"), ("17:45", "18:00")]:
    _add_slots(start, end, ["Food Ministry"], 2.5)

# 13:45–14:00  postlunch hostel return
_add_slots("13:45", "14:00", ["FD2", "FD3", "NAB"], 1.4)
_add_slots("13:45", "14:00", _Z3, 2.0)
_add_slots("13:45", "14:00", ["Shankar Bhawan"], 1.3)

# Extended Bhawans — same class-change + morning windows
for start, end in [("07:45", "08:00"), ("08:45", "09:00"),("09:45", "10:00"), ("11:45", "12:00"),("12:45", "13:00"), ("13:45", "14:00"),
                   ("14:45", "15:00"), ("15:45", "16:00"),("16:45", "17:00"), ("17:45", "18:00")]:
    _add_slots(start, end, ["Bhagirath Bhawan", "Rana Pratap Bhawan","Ashok Bhawan", "CVR Bhawan"], 2.0)

# 16:45–17:00 and 17:45–18:00  evening rush Z4 + extras
for start, end in [("16:45", "17:00"), ("17:45", "18:00")]:
    _add_slots(start, end, ["Shankar Bhawan", "All Night Canteen","Akshay Supermarket", "SAC","Vfast", "Looters"], 2.0)

# 18:00–19:00  ANC + SAC + Vfast
_add_slots("18:00", "19:00", ["All Night Canteen", "SAC", "Vfast","Looters"], 2.0)

# 20:00–02:00  night canteen light rush
_add_slots("20:00", "02:00", ["All Night Canteen", "Food Ministry", "Looters"], 1.4)

# 21:00–23:00  Cnot light rush  (node not in main edge list — stored anyway)
_add_slots("21:00", "23:00", ["Cnot"], 1.3)


# GRAPH STATISTICS 
def _compute_stats():
    from collections import defaultdict
    deg = defaultdict(int)
    for u, v, _ in EDGES:
        deg[u] += 1
        deg[v] += 1
    n = len(NODE_COORDINATES)
    e = len(EDGES)
    weights = [w for _, _, w in EDGES]
    isolated = [node for node in NODE_COORDINATES if deg[node] == 0]
    return {
        "nodes":            n,
        "edges":            e,
        "density_pct":      round(e / (n * (n - 1) / 2) * 100, 2),
        "min_weight_m":     min(weights),
        "max_weight_m":     max(weights),
        "avg_weight_m":     round(sum(weights) / len(weights), 2),
        "avg_degree":       round(2 * e / n, 2),
        "max_degree":       max(deg.values()),
        "max_degree_nodes": [k for k, v in deg.items() if v == max(deg.values())],
        "min_degree":       min(deg.values()),
        "min_degree_nodes": [k for k, v in deg.items() if v == min(deg.values())],
        "isolated_nodes":   isolated,
    }

GRAPH_STATS = _compute_stats()


# QUICK SELF-TEST 
if __name__ == "__main__":
    import math

    def haversine(c1, c2):
        R = 6_371_000
        la1, lo1 = math.radians(c1[0]), math.radians(c1[1])
        la2, lo2 = math.radians(c2[0]), math.radians(c2[1])
        a = (math.sin((la2 - la1) / 2) ** 2
             + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
        return round(2 * R * math.asin(math.sqrt(a)), 2)

    print("BITS Pilani Campus Graph — Data Verification")
    print("=" * 60)
    s = GRAPH_STATS
    print(f"  Nodes            : {s['nodes']}")
    print(f"  Edges            : {s['edges']}")
    print(f"  Density          : {s['density_pct']}%")
    print(f"  Avg degree       : {s['avg_degree']}")
    print(f"  Max degree       : {s['max_degree']} → {s['max_degree_nodes']}")
    print(f"  Min degree       : {s['min_degree']} → {s['min_degree_nodes']}")
    print(f"  Min edge weight  : {s['min_weight_m']} m")
    print(f"  Max edge weight  : {s['max_weight_m']} m")
    print(f"  Avg edge weight  : {s['avg_weight_m']} m")
    print(f"  Isolated nodes   : {s['isolated_nodes']}")
    print()

    print("Edge distance verification (Haversine vs stored):")
    print(f"  {'Edge':<55} {'Stored':>8}  {'Haversine':>10}  {'Diff':>7}")
    print("  " + "-" * 85)
    mismatches = []
    for u, v, d in EDGES:
        if u in NODE_COORDINATES and v in NODE_COORDINATES:
            hav = haversine(NODE_COORDINATES[u], NODE_COORDINATES[v])
            diff = abs(hav - d)
            flag = "  <-- CHECK" if diff > 1.0 else ""
            print(f"  {u+' — '+v:<55} {d:>8.2f}  {hav:>10.2f}  {diff:>7.2f}{flag}")
            if diff > 1.0:
                mismatches.append((u, v, d, hav, diff))
        else:
            print(f"  !! Missing coordinate for edge: {u} — {v}")

    print()
    if mismatches:
        print(f"  {len(mismatches)} edge(s) differ from Haversine by >1 m")
    else:
        print("  All edges within 1 m of Haversine")

    print()
