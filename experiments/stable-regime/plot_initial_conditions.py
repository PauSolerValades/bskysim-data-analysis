"""Side-by-side initial-conditions experiment: online % over time for
offline_startup_ratio in {0.0, 0.5, 1.0} on the same axes.

Usage: uv run python plot_initial_conditions.py <trace_root> <size> --w30K-h75K dirs
e.g.:  uv run python plot_initial_conditions.py /path/to/traces/10K
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import stable_regime, find_duration_and_warmup_time, topology_size

sns.set_theme(style="whitegrid")
plt.rcParams.update({"text.usetex": False, "axes.labelsize": 11, "font.size": 11,
                     "legend.fontsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10})

RUNS = [("r0", "0% online at start", "#0072B2"),
        ("r50", "50% online at start", "#E69F00"),
        ("r100", "100% offline at start", "#D55E00")]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace_root", help="e.g. .../traces/10K")
    p.add_argument("--bin_len", type=int, default=60)
    p.add_argument("--session", default="0-session_trace.jsonl")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))
    for tag, label, color in RUNS:
        trace = os.path.join(args.trace_root, f"w30K-h75K-{tag}")
        warmup, duration = find_duration_and_warmup_time(trace)
        n_users = topology_size(trace)
        counts = stable_regime(os.path.join(trace, args.session), warmup, duration, args.bin_len)
        t = [i * args.bin_len for i in range(len(counts))]
        ax.plot(t, [c / n_users * 100 for c in counts], label=label, color=color, alpha=0.9)

    ax.set_xlabel("Time since warmup end (s)")
    ax.set_ylabel("% online users")
    ax.set_title(os.path.basename(args.trace_root.rstrip("/")))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = args.output or os.path.join(args.trace_root, "initial_conditions.png")
    fig.savefig(out)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
