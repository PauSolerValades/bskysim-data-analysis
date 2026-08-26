"""Stability experiment analysis: initial-conditions convergence + equilibrium
online% + anchored stable-at, across 10K/100K/500K/1M.

Reads the binary session traces (TraceSession = 40B) with numpy — far faster
than JSONL. Layout: time f8, event_id u8, gen_id u8, user_id u4, pad u4,
type u4 (0=start,1=end_boredom,2=end), backlog u4.

Usage: uv run --with numpy python stability_analysis.py
"""
import os, sys, statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import find_duration_and_warmup_time, topology_size

sns.set_theme(style="whitegrid")
plt.rcParams.update({"text.usetex": False, "axes.labelsize": 11, "font.size": 11,
                     "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10})

ROOT = "/home/psoler/des-ctic-dev/steps/stability/traces"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "w2K-d60K")
os.makedirs(OUT, exist_ok=True)

DT = np.dtype([('time', '<f8'), ('event_id', '<u8'), ('gen_id', '<u8'),
               ('user_id', '<u4'), ('_pad', '<u4'), ('typ', '<u4'), ('backlog', '<u4')])

SIZES = ["10K", "100K", "500K", "1M"]
RATIOS = [("r0", "0% offline", "#0072B2"), ("r50", "50% offline", "#E69F00"), ("r100", "100% offline", "#D55E00")]
BIN = 60
WINDOW_BINS = 5
REL = 0.10

def online_curve(path, warmup, duration, N):
    a = np.fromfile(path, dtype=DT)
    t = a['time']; typ = a['typ']
    m = (t >= warmup) & (t <= warmup + duration)
    t, typ = t[m], typ[m]
    order = np.argsort(t, kind='stable')
    t, typ = t[order], typ[order]
    delta = np.where(typ == 0, 1, -1)
    cum = np.cumsum(delta)
    grid = np.arange(warmup, warmup + duration, BIN)
    pos = np.searchsorted(t, grid, side='right') - 1
    online = np.where(pos >= 0, cum[np.clip(pos, 0, len(cum) - 1)], 0)
    return online / N * 100

def anchored(x, rel):
    means = np.convolve(x, np.ones(WINDOW_BINS) / WINDOW_BINS, mode='valid')
    final = means[-1]
    lo, hi = final * (1 - rel), final * (1 + rel)
    bad = np.where((means < lo) | (means > hi))[0]
    last_bad = int(bad[-1]) if len(bad) else 0
    return (last_bad + 1 + WINDOW_BINS // 2) * BIN, final

print(f"{'size':>5} {'ratio':>6} {'equil %':>9} {'stable_at(±10%)':>16}   (median over 3 runs)")
if __name__ != "__main__":
    raise SystemExit
for size in SIZES:
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag, label, color in RATIOS:
        d = f"{ROOT}/{size}/w2K-d60K-{tag}"
        w, dur = find_duration_and_warmup_time(d)
        N = topology_size(d)
        curves, eqs, stabs = [], [], []
        for run in range(3):
            c = online_curve(f"{d}/{run}-session_trace.bin", w, dur, N)
            curves.append(c)
            t0, final = anchored(c, REL)
            eqs.append(final); stabs.append(t0)
        n = min(len(c) for c in curves)
        mean_curve = np.mean(np.array([c[:n] for c in curves]), axis=0)
        t_axis = np.arange(warmup_offset := 0, n) * BIN
        ax.plot(np.arange(n) * BIN, mean_curve, label=label, color=color, alpha=0.9)
        eq, st = statistics.median(eqs), statistics.median(stabs)
        print(f"{size:>5} {tag:>6} {eq:>9.2f} {st:>15.0f}s")
    ax.set_xlabel("Time since warm-up end (s)")
    ax.set_ylabel("% online users")
    ax.set_title(f"{size} users — initial-condition convergence")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"initial_conditions_{size}.png"))
    plt.close(fig)
    print(f"  saved {OUT}/initial_conditions_{size}.png")
