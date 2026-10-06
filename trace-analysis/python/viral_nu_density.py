#!/usr/bin/env python3
"""Structural-virality density of the viral cascades, per dataset.

Mimics the empirical figure of the data chapter (firehose-analysis/
cascade-creation/viral_nu_stats.py): a log-x KDE of ``StructuralVirality`` for
the viral cascades (``CascadeDepth >= 2``), with the broadcast floor ``nu = 2``
(dashed) and the median (dotted) marked.

The viral sets run into 10^7-10^8 rows, so the KDE is computed on a reservoir
sample per dataset; mean/median are printed for reference.

Run with the sessions env:
    uv run --project /home/psoler/firehose-analysis/sessions \
        python viral_nu_density.py
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

SIM_ROOT = Path("/home/psoler/des-ctic-dev/steps/final/datasets")
SIZES = ["10K", "100K", "500K", "1M"]
OUT = Path("/home/psoler/tfm/report/images/results")
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
    OUT.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        path = SIM_ROOT / size / "cascades.parquet"
        if not path.exists():
            print(f"skip {size} (missing)")
            continue
        # ponytail: reservoir sample, the KDE is O(n^2); the shape is unchanged,
        # and the printed median is checked against the full-data table value.
        d = con.execute(f"""
            SELECT StructuralVirality AS nu
            FROM (SELECT StructuralVirality FROM read_parquet('{path}')
                  WHERE CascadeDepth >= 2)
            USING SAMPLE {SAMPLE} ROWS
        """).fetchnumpy()
        nu = d["nu"].astype(float)
        median = float(np.median(nu))
        print(f"{size}: {len(nu):,} sampled, mean {nu.mean():.4f}, median {median:.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.kdeplot(x=nu, log_scale=(True, False), color=PALETTE[0], linewidth=2,
                    ax=ax, label="viral cascades")
        ax.axvline(2.0, color=PALETTE[1], linestyle="--", linewidth=1.4,
                   label=r"broadcast floor $\nu=2$")
        ax.axvline(median, color=PALETTE[2], linestyle=":", linewidth=1.4,
                   label=rf"median $\nu={median:.2f}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Structural virality $\nu(T)$")
        ax.set_ylabel("Density")
        ax.set_title(f"Structural virality of viral cascades ({size})",
                     fontsize=12, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        out = OUT / f"viral_nu_density_{size}.svg"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
