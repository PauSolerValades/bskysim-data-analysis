"""Plot the which-warmup results: first-session backlog and boredom vs warm-up.

Reads session traces under data/{10K,100K}/ws{...}/ and produces two combined
figures (one per metric, both network sizes) with the NEW warm-up grid.

Usage: uv run python plot_results.py
"""
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import analyze

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "text.usetex": False,
    "axes.labelsize": 11, "font.size": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

WARMUPS = [0, 100, 500, 1000, 2000, 5000, 10000]
SIZES = [("10K", "#0072B2"), ("100K", "#D55E00")]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

data = {size: [r for r in (analyze(size, w) for w in WARMUPS) if r] for size, _ in SIZES}

# --- Figure 1: median backlog at first session ---
fig, ax = plt.subplots(figsize=(8, 5))
for size, color in SIZES:
    xs = [r["w"] for r in data[size]]
    ys = [r["backlog_median"] for r in data[size]]
    ax.plot(xs, ys, marker="o", color=color, label=size)
ax.set_xlabel("Warm-up (ticks)")
ax.set_ylabel("Median backlog at first session (posts)")
ax.set_xticks(WARMUPS)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "first_session_backlog.png"))
plt.close(fig)
print("saved", os.path.join(OUT, "first_session_backlog.png"))

# --- Figure 2: first-session boredom rate ---
fig, ax = plt.subplots(figsize=(8, 5))
for size, color in SIZES:
    xs = [r["w"] for r in data[size]]
    ys = [r["first_boredom_pct"] for r in data[size]]
    ax.plot(xs, ys, marker="o", color=color, label=size)
ax.set_xlabel("Warm-up (ticks)")
ax.set_ylabel("First-session boredom (%)")
ax.set_xticks(WARMUPS)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "first_session_boredom.png"))
plt.close(fig)
print("saved", os.path.join(OUT, "first_session_boredom.png"))
