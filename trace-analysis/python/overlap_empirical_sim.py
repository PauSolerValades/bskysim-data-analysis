#!/usr/bin/env python3
"""Overlap the empirical (Bluesky) cascade-size tail with the 4 simulated datasets.

CCDF of cascade size (nodes, root included) on log-log axes: the empirical
firehose cascades vs the LIFO runs at 10K/100K/500K/1M. Visualises the
truncated tail discussed in @tbl-res-vs-data.

Empirical: firehose-analysis/cascade-creation/results/cascades_rows.csv
           (7 cols, no header; col 4 = cascade_size).
Simulated: des-ctic-dev/steps/final/datasets/<SIZE>/cascades.parquet.

Run with the sessions env (duckdb + matplotlib + seaborn):
    uv run --project /home/psoler/firehose-analysis/sessions python overlap_empirical_sim.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

EMP_CSV = Path("/home/psoler/firehose-analysis/cascade-creation/results/cascades_rows.csv")
SIM_ROOT = Path("/home/psoler/des-ctic-dev/steps/final/datasets")
OUT = Path("/home/psoler/tfm/report/images/results/overlap_empirical_sim.svg")
SIZES = ["10K", "100K", "500K", "1M"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "text.usetex": shutil.which("latex") is not None,
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
PALETTE = sns.color_palette("colorblind")


def ccdf(values: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CCDF from a histogram (values ascending, counts)."""
    order = np.argsort(values)
    v = values[order]
    c = counts[order]
    total = c.sum()
    tail = np.cumsum(c[::-1])[::-1]
    return v, tail / total


def main() -> None:
    db = duckdb.connect()

    t0 = time.time()
    print("== empirical ==")
    emp = db.execute(f"""
        SELECT column3 AS size, count(*) AS n
        FROM read_csv('{EMP_CSV}', header=false)
        GROUP BY column3 ORDER BY size
    """).fetchnumpy()
    ex, ey = ccdf(emp["size"].astype(float), emp["n"].astype(float))
    print(f"  empirical: {int(emp['n'].sum()):,} posts, max size {int(emp['size'].max())} "
          f"({time.time()-t0:.0f}s)")

    sims = {}
    for size in SIZES:
        t0 = time.time()
        p = SIM_ROOT / size / "cascades.parquet"
        d = db.execute(f"""
            SELECT CascadeSize AS size, count(*) AS n
            FROM read_parquet('{p}')
            GROUP BY CascadeSize ORDER BY CascadeSize
        """).fetchnumpy()
        sims[size] = ccdf(d["size"].astype(float), d["n"].astype(float))
        print(f"  {size}: {int(d['n'].sum()):,} posts, max size {int(d['size'].max())} "
              f"({time.time()-t0:.0f}s)")

    # ── plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(ex, ey, color="black", lw=2.2, label="Bluesky data")
    for i, size in enumerate(SIZES):
        sx, sy = sims[size]
        ax.plot(sx, sy, color=PALETTE[i], lw=1.6, label=size)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cascade size")
    ax.set_ylabel(r"$P(\mathrm{size} \geq x)$")
    ax.set_title("Cascade size tail: Bluesky vs. simulation")
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
