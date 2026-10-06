#!/usr/bin/env python3
"""Overlay the viral structural-virality density of Bluesky and the four LIFO datasets.

Single log-x density plot: the empirical firehose viral cascades (``depth >= 2``)
against the four simulated datasets. Companion to ``overlap_empirical_sim.py``
(same overlay for cascade size) and to ``viral_nu_density.py`` (per-dataset
panels).

The empirical and simulated viral sets run into 10^5-10^8 rows, so each KDE is
computed on a reservoir sample.

Run with the sessions env:
    uv run --project /home/psoler/firehose-analysis/sessions \
        python viral_nu_overlap.py
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
SIZES = ["10K", "100K", "500K", "1M"]
OUT = Path("/home/psoler/tfm/report/images/results/viral_nu_overlap.svg")
SAMPLE = 200_000

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


def main() -> None:
    con = duckdb.connect()

    # empirical cascades_rows.csv has no header; 0-indexed 4=depth, 6=nu
    emp = con.execute(f"""
        SELECT nu FROM (
            SELECT column6 AS nu
            FROM read_csv('{EMP_CSV}', header=false)
            WHERE column4 >= 2
        ) USING SAMPLE {SAMPLE} ROWS
    """).fetchnumpy()["nu"].astype(float)
    print(f"Bluesky: {len(emp):,} sampled, mean {emp.mean():.4f}, median {np.median(emp):.4f}")

    sims = {}
    for size in SIZES:
        p = SIM_ROOT / size / "cascades.parquet"
        nu = con.execute(f"""
            SELECT nu FROM (
                SELECT StructuralVirality AS nu FROM read_parquet('{p}')
                WHERE CascadeDepth >= 2
            ) USING SAMPLE {SAMPLE} ROWS
        """).fetchnumpy()["nu"].astype(float)
        sims[size] = nu
        print(f"{size}: {len(nu):,} sampled, mean {nu.mean():.4f}, median {np.median(nu):.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.kdeplot(x=emp, log_scale=(True, False), color="black", linewidth=2.2,
                ax=ax, label="Bluesky data")
    for (size, nu), color in zip(sims.items(), PALETTE):
        sns.kdeplot(x=nu, log_scale=(True, False), color=color, linewidth=1.6,
                    ax=ax, label=size)
    ax.axvline(2.0, color="grey", linestyle="--", linewidth=1.2,
               label=r"broadcast floor $\nu=2$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Structural virality $\nu(T)$")
    ax.set_ylabel("Density")
    ax.set_title("Structural virality of viral cascades: Bluesky vs. simulation",
                 fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
