#!/usr/bin/env python3
"""Synthetic comparison of the Bluesky vs simulated repost power-law tails.

Both tails share the Bluesky lower cutoff xmin = 12 so the only difference is
the exponent. Left: CCDF on log-log axes. Right: density on linear axes.

The exponents are the fitted values reported in the thesis:
  Bluesky  alpha = 2.05  (firehose-analysis/cascade-creation/reposts_powerlaw.py)
  Sim      alpha = 2.9   (representative simulated run)

Output: plots/powerlaw_alpha_comparison.svg
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

XMIN = 12.0
ALPHA_BLUESKY = 2.05
ALPHA_SIM = 2.9

x = np.linspace(XMIN, 500, 500)
ccdf = lambda a: (x / XMIN) ** (1 - a)          # P(X >= x) for a Pareto tail
dens = lambda a: (a - 1) / XMIN * (x / XMIN) ** (-a)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(x, ccdf(ALPHA_BLUESKY), color="#1d9bf0", lw=2.2,
        label=f"Bluesky  $\\alpha={ALPHA_BLUESKY}$")
ax.plot(x, ccdf(ALPHA_SIM), color="#e53935", lw=2.2,
        label=f"Simulation  $\\alpha={ALPHA_SIM}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Reposts")
ax.set_ylabel("$P(X \\geq x)$")
ax.set_title("CCDF")
ax.legend()

ax = axes[1]
ax.plot(x, dens(ALPHA_BLUESKY), color="#1d9bf0", lw=2.2,
        label=f"Bluesky  $\\alpha={ALPHA_BLUESKY}$")
ax.plot(x, dens(ALPHA_SIM), color="#e53935", lw=2.2,
        label=f"Simulation  $\\alpha={ALPHA_SIM}$")
ax.set_xlabel("Reposts")
ax.set_ylabel("Density")
ax.set_title("Density")
ax.legend()

fig.tight_layout()
path = OUT / "powerlaw_alpha_comparison.svg"
fig.savefig(path, bbox_inches="tight")
print(f"Saved -> {path}")
