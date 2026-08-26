"""Analyze the which-warmup sweep: first-session backlog and boredom rate vs warmup.

Reads session traces under data/{10K,100K}/ws{...}/ and prints the table used
in the README. Usage: python3 analyze.py
"""
import json, os, statistics

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def analyze(size, w):
    f = f"{ROOT}/{size}/ws{w}/0-session_trace.jsonl"
    if not os.path.exists(f):
        return None
    first = {}   # user_id -> (time, backlog) of first session
    endtype = {} # user_id -> end type of first session
    with open(f) as fh:
        for line in fh:
            e = json.loads(line)
            u = e["user_id"]
            if e["type"] == "start":
                if u not in first:
                    first[u] = (e["time"], e["backlog"])
            else:
                if u not in endtype:
                    endtype[u] = e["type"]
    completed = [(t, b) for u, (t, b) in first.items() if u in endtype]
    immediate = [u for u, (t, b) in first.items() if t <= w + 500 and u in endtype]
    all_bor = sum(1 for u in first if endtype.get(u) == "end_boredom")
    imm_bor = sum(1 for u in immediate if endtype[u] == "end_boredom")
    bl = [b for _, b in completed]
    return {
        "w": w, "n_first": len(first),
        "backlog_median": statistics.median(bl) if bl else 0,
        "first_boredom_pct": 100 * all_bor / len(first) if first else 0,
        "immediate_boredom_pct": 100 * imm_bor / len(immediate) if immediate else 0,
        "n_immediate": len(immediate),
    }

if __name__ == "__main__":
    print(f"{'size':>5} {'w':>6} {'median backlog':>14} {'first-sess boredom%':>19} {'imm(<=500t) boredom%':>21} {'n_imm':>7}")
    for size in ["10K", "100K"]:
        for w in [0, 100, 500, 1000, 2000, 5000, 10000]:
            r = analyze(size, w)
            if r:
                print(f"{size:>5} {r['w']:>6} {r['backlog_median']:>14} {r['first_boredom_pct']:>18.1f} "
                      f"{r['immediate_boredom_pct']:>20.1f} {r['n_immediate']:>7}")
