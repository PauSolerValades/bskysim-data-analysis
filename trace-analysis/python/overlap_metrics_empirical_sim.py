#!/usr/bin/env python3
"""Overlap the empirical and simulated cascade depth / width tails.

CCDF on log-log axes for ``CascadeDepth`` and ``MaxOutDegree``, Bluesky vs the
four LIFO datasets — the depth/width companions of ``overlap_empirical_sim.py``
(cascade size).

Only non-trivial cascades (``CascadeSize >= 2``) enter: a singleton has depth and
width 0, which a log axis cannot show. Size can include singletons, depth/width
cannot.

Run with the sessions env:
    uv run --project /home/psoler/firehose-analysis/sessions \
        python overlap_metrics_empirical_sim.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

EMP_CSV = Path("/home/psoler/firehose-analysis/cascade-creation/results/cascades_rows.csv")
SIM_ROOT = Path("/home/psoler/des-ctic-dev/steps/final/datasets")
OUT_DIR = Path("/home/psoler/tfm/report/images/results")
SIZES = ["10K", "100K", "500K", "1M"]

# (parquet column, empirical 0-indexed column, x label, file name, title)
METRICS = [
    ("CascadeDepth", 4, "Cascade depth", "overlap_depth_empirical_sim.svg",
     "Cascade depth tail: Bluesky vs. simulation"),
    ("MaxOutDegree", 5, "Max out-degree (width)", "overlap_width_empirical_sim.svg",
     "Cascade width tail: Bluesky vs. simulation"),
]

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
    con = duckdb.connect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for col, emp_col, xlabel, fname, title in METRICS:
        e = con.execute(f"""
            SELECT column{emp_col} AS v, count(*) AS n
            FROM read_csv('{EMP_CSV}', header=false)
            WHERE column3 >= 2
            GROUP BY column{emp_col} ORDER BY v
        """).fetchnumpy()
        ex, ey = ccdf(e["v"].astype(float), e["n"].astype(float))
        print(f"empirical {xlabel}: {int(e['n'].sum()):,} cascades")

        sims = {}
        for size in SIZES:
            p = SIM_ROOT / size / "cascades.parquet"
            d = con.execute(f"""
                SELECT {col} AS v, count(*) AS n
                FROM read_parquet('{p}')
                WHERE CascadeSize >= 2
                GROUP BY {col} ORDER BY v
            """).fetchnumpy()
            sims[size] = ccdf(d["v"].astype(float), d["n"].astype(float))
            print(f"  {size}: {int(d['n'].sum()):,} cascades")

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(ex, ey, color="black", lw=2.2, label="Bluesky data")
        for i, size in enumerate(SIZES):
            sx, sy = sims[size]
            ax.plot(sx, sy, color=PALETTE[i], lw=1.6, label=size)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$P(X \geq x)$")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        out = OUT_DIR / fname
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
