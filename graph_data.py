# graph_data.py - Sparse Weighted Undirected Graph
# All edge weights are in METRES.
# All coordinates are (latitude, longitude) in decimal degrees.
COORDINATES = {
    "Clock Tower":(28.36412,75.58704),
    "Meera Bhawan":(28.357125,75.585593),
    "Ram Bhawan":(28.36230,75.58636),
    "Budh Bhawan":(28.360722,75.586769),
    "Krishna Bhawan":(28.362567,75.588079),
    "Gandhi Bhawan":(28.361026,75.588324),
    "Shankar Bhawan":(28.359770,75.588494),
    "Vyas Bhawan":(28.358299,75.588800),
    "Saraswati Temple":(28.358129,75.587979),
    "Bhagirath Bhawan":(28.361138,75.589277),
    "Vishwakarma Bhawan":(28.362687,75.589040),
    "CVR Bhawan":(28.361886,75.590986),
    "Malviya Bhawan":(28.361601,75.585337),
    "Srinivasa Bhawan":(28.365601,75.587347),
    "PIEDS":(28.36616,75.58771),
    "Library":(28.365336,75.588434),
    "Rana Pratap Bhawan":(28.362947,75.590724),
    "Ashok Bhawan":(28.361389,75.590938),
    "Lecture Theatre Complex":(28.365110,75.589873),
    "FD1":(28.364287,75.588781),
    "FD2":(28.363570,75.587770),
    "FD3":(28.363390,75.586383),
    "Food Ministry":(28.36489,75.58826),
    "New Workshop":(28.365325,75.587931),
    "All Night Canteen":(28.360141,75.589384),
    "SAC":(28.360670,75.585770),
    "Akshay Supermarket":(28.357148,75.589956),
    "Cnot":(28.356218,75.591457),
    "Looters":(28.361335,75.585653),
    "BET-TACT":(28.362791,75.585486),
    "Birla Balika Vidyapeeth":(28.359555,75.586984),
    "Birla Shishu Vihar":(28.35738,75.58727),
    "NAB":(28.36225,75.58728),
    "Main Gate":(28.361022,75.592976),
    "Vfast":(28.360911,75.592074),
    "Gandhi Statue":(28.360181,75.586938),
    "Patel Statue":(28.360383,75.588402),
    "ANC Circle":(28.360499,75.589268),
    "Ashok Circle":(28.360752,75.591100),
    "Mandir Meera Intersection":(28.358046,75.587324), }

EDGES = [
    ("Akshay Supermarket", "Cnot", 179.63),
    ("Akshay Supermarket", "Vyas Bhawan", 170.81),

    ("All Night Canteen", "Akshay Supermarket", 337.48),
    ("All Night Canteen", "Vyas Bhawan", 212.64),

    ("ANC Circle", "All Night Canteen", 41.39),
    ("ANC Circle", "Ashok Circle", 181.45),
    ("ANC Circle", "Bhagirath Bhawan", 71.06),
    ("ANC Circle", "Patel Statue", 85.71),

  
    ("Ashok Bhawan", "Bhagirath Bhawan", 164.90),
    ("Ashok Bhawan", "CVR Bhawan", 55.46),
    ("Ashok Circle", "Ashok Bhawan", 72.58),
    ("Ashok Circle", "Cnot", 505.37),
    ("Ashok Circle", "Vfast", 96.93),

    ("BET-TACT", "FD3", 110.18),
    ("BET-TACT", "Malviya Bhawan", 133.12),
    ("BET-TACT", "Ram Bhawan", 101.46),

    ("Bhagirath Bhawan", "CVR Bhawan", 186.76),
    ("Bhagirath Bhawan", "Gandhi Bhawan", 94.08),
    ("Bhagirath Bhawan", "Rana Pratap Bhawan", 245.98),
    ("Bhagirath Bhawan", "Vishwakarma Bhawan", 173.80),

    ("Birla Shishu Vihar", "Meera Bhawan", 166.53),

    ("Budh Bhawan", "Looters", 128.73),
    ("Budh Bhawan", "NAB", 177.11),
    ("Budh Bhawan", "Ram Bhawan", 179.97),
    ("Budh Bhawan", "SAC", 97.92),

    ("Clock Tower", "FD2", 94.03),
    ("Clock Tower", "FD3", 103.54),

    ("CVR Bhawan", "Rana Pratap Bhawan", 120.73),
    ("CVR Bhawan", "Vishwakarma Bhawan", 210.21),

    ("FD1", "FD2", 127.05),
    ("FD1", "Food Ministry", 84.23),
    ("FD1", "Krishna Bhawan", 203.22),
    ("FD1", "Lecture Theatre Complex", 140.68),
    ("FD1", "Library", 121.48),
    ("FD1", "New Workshop", 142.26),
    ("FD1", "Rana Pratap Bhawan", 241.55),
    ("FD1", "Vishwakarma Bhawan", 179.71),

    ("FD2", "Food Ministry", 154.41),
    ("FD2", "Krishna Bhawan", 115.55),
    ("FD2", "NAB", 154.41),
    ("FD2", "Rana Pratap Bhawan", 297.22),
    ("FD2", "Vishwakarma Bhawan", 158.37),

    ("FD3", "Malviya Bhawan", 223.71),
    ("FD3", "Ram Bhawan", 121.22),

    ("Food Ministry", "Lecture Theatre Complex", 159.71),
    ("Food Ministry", "New Workshop", 58.10),

    ("Gandhi Bhawan", "Krishna Bhawan", 173.02),
    ("Gandhi Bhawan", "NAB", 170.17),
    ("Gandhi Bhawan", "Vishwakarma Bhawan", 197.54),
    ("Gandhi Statue", "Birla Balika Vidyapeeth", 69.75),
    ("Gandhi Statue", "Budh Bhawan", 62.39),
    ("Gandhi Statue", "SAC", 126.56),

    ("Krishna Bhawan", "NAB", 85.76),
    ("Krishna Bhawan", "Ram Bhawan", 170.80),
    ("Krishna Bhawan", "Vishwakarma Bhawan", 94.97),

    ("Lecture Theatre Complex", "New Workshop", 191.51),
    ("Lecture Theatre Complex", "Rana Pratap Bhawan", 254.52),

    ("Library", "Food Ministry", 52.43),  # 2nd shortest edge
    ("Library", "Lecture Theatre Complex", 143.02),
    ("Library", "PIEDS", 115.82),
    ("Library", "Srinivasa Bhawan", 110.36),

    ("Looters", "Malviya Bhawan", 42.79),  # shortest edge
    ("Looters", "SAC", 74.83),

    ("Main Gate", "Vfast", 89.12),
    ("Malviya Bhawan", "Ram Bhawan", 126.73),
    ("Mandir Meera Intersection", "Birla Balika Vidyapeeth", 171.06),
    ("Mandir Meera Intersection", "Birla Shishu Vihar", 74.24),
    ("Mandir Meera Intersection", "Saraswati Temple", 64.75),
    
    ("NAB", "Ram Bhawan", 90.19),

    ("Patel Statue", "Gandhi Bhawan", 71.90),
    ("Patel Statue", "Gandhi Statue", 145.00),
    ("Patel Statue", "Shankar Bhawan", 68.75),
    ("Shankar Bhawan", "Vyas Bhawan", 165.92),
    ("Vyas Bhawan", "Saraswati Temple", 82.33),

    ("PIEDS", "New Workshop", 95.33),
    ("PIEDS", "Srinivasa Bhawan", 71.59),
    ("Vishwakarma Bhawan", "Ashok Bhawan", 235.20),
    ("Vishwakarma Bhawan", "Rana Pratap Bhawan", 167.29),

]

# Only non-1.0 (congested) slots are listed; all others default to k = 1.0.

def _slot(hhmm: str) -> int:
    """Convert 'HH:MM' string to 15-min slot index (0–95)."""
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m) // 15

_Z1_ACADEMIC = [ "Lecture Theatre Complex", "FD1", "Srinivasa Bhawan", "Vishwakarma Bhawan",]
_Z1_FULL = [  "Lecture Theatre Complex", "FD1", "Food Ministry", "Srinivasa Bhawan", "Vishwakarma Bhawan", "New Workshop", "Library",]
_Z2 = [ "NAB", "Clock Tower", "FD2", "FD3","Krishna Bhawan", "Gandhi Bhawan", "Ram Bhawan", "Budh Bhawan",]
_Z3 = [  "Meera Bhawan", "Mandir Meera Intersection", "Birla Shishu Vihar", "Birla Balika Vidyapeeth", "Saraswati Temple",]
_Z4_EVE = [  "Shankar Bhawan", "All Night Canteen", "Akshay Supermarket",]
_Z4_NIGHT = ["All Night Canteen"] 
_EXT = [ "Bhagirath Bhawan", "Rana Pratap Bhawan", "Ashok Bhawan", "CVR Bhawan", "SAC", "Malviya Bhawan", "Looters", "Vfast",]

def _make_slot_entry(nodes: list, k: float) -> dict:
    return {n: k for n in nodes}

CONGESTION_SCHEDULE: dict[int, dict[str, float]] = {}

def _add_slots(start_hhmm: str, end_hhmm: str, nodes: list, k: float) -> None:
    s = _slot(start_hhmm)
    e = _slot(end_hhmm)
    entry = _make_slot_entry(nodes, k)

    slots = range(s, e) if s < e else list(range(s, 96)) + list(range(0, e))
    for idx in slots:
        CONGESTION_SCHEDULE.setdefault(idx, {}).update(entry)

#Congestion windows
_add_slots("07:45", "08:00", ["FD1", "Food Ministry", "New Workshop",  "Srinivasa Bhawan", "Vishwakarma Bhawan",  "Malviya Bhawan"], 2.0)
_add_slots("07:45", "08:00", _Z2 + _Z3, 2.0)

for start, end in [ ("08:45", "09:00"), ("09:45", "10:00"),  ("11:45", "12:00"), ("12:45", "13:00"),
    ("14:45", "15:00"), ("15:45", "16:00"), ("16:45", "17:00"), ("17:45", "18:00"),
]:
    _add_slots(start, end, _Z1_ACADEMIC + ["FD1"], 2.5)
    _add_slots(start, end, _Z2, 2.5)
    _add_slots(start, end, _Z3, 2.0)

_add_slots("10:45", "11:00", ["Lecture Theatre Complex", "FD1",  "New Workshop", "Vishwakarma Bhawan"], 2.5)
_add_slots("10:45", "11:00", _Z2, 2.5)

for start, end in [("11:45", "12:00"), ("12:45", "13:00"), ("16:45", "17:00"), ("17:45", "18:00")]:
    _add_slots(start, end, ["Food Ministry"], 2.5)

_add_slots("13:45", "14:00", ["FD2", "FD3", "NAB"], 1.4)
_add_slots("13:45", "14:00", _Z3, 2.0)
_add_slots("13:45", "14:00", ["Shankar Bhawan"], 1.3)

for start, end in [ ("07:45", "08:00"), ("08:45", "09:00"), ("09:45", "10:00"),
    ("11:45", "12:00"), ("12:45", "13:00"), ("13:45", "14:00"), ("14:45", "15:00"), ("15:45", "16:00"), ("16:45", "17:00"), ("17:45", "18:00"),
]:
    _add_slots(start, end, ["Bhagirath Bhawan", "Rana Pratap Bhawan", "Ashok Bhawan", "CVR Bhawan"], 2.0)

# 16:45–18:00 - evening rush
for start, end in [("16:45", "17:00"), ("17:45", "18:00")]:
    _add_slots(start, end, ["Shankar Bhawan", "All Night Canteen","Akshay Supermarket", "SAC", "Vfast", "Looters"], 2.0)

_add_slots("18:00", "19:00", ["All Night Canteen", "SAC", "Vfast", "Looters"], 2.0)
_add_slots("20:00", "02:00", ["All Night Canteen", "Food Ministry", "Looters"], 1.4)
_add_slots("21:00", "23:00", ["Cnot"], 1.3)

# GRAPH STATISTICS
def _compute_stats() -> dict:
    from collections import defaultdict
    deg = defaultdict(int)
    for u, v, _ in EDGES:
        deg[u] += 1
        deg[v] += 1

    n = len(COORDINATES)
    e = len(EDGES)
    weights = [w for _, _, w in EDGES]

    return {
        "nodes": n,
        "edges": e,
        "density_pct": round(e / (n * (n - 1) / 2) * 100, 2),
        "min_weight_m": min(weights),
        "max_weight_m": max(weights),
        "avg_weight_m": round(sum(weights) / len(weights), 2),
        "avg_degree": round(2 * e / n, 2),
        "max_degree": max(deg.values()),
        "max_degree_nodes": [k for k, v in deg.items() if v == max(deg.values())],
        "min_degree": min(deg.values()),
        "min_degree_nodes": [k for k, v in deg.items() if v == min(deg.values())],
        "isolated_nodes": [node for node in COORDINATES if deg[node] == 0],
    }
GRAPH_STATS = _compute_stats()
