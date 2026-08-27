#!/usr/bin/env python3
"""
Structural Virality — per-run and global, with 95% bootstrap CI.

For every dataset (cascades.parquet):
  - global: histogram + mean/median with bootstrap CI
  - per run: mean/median SV with bootstrap CI
  - fraction of star (SV=1.0) vs deeper (SV>1.0) cascades

Non-zero SV only (posts with >=1 repost; SV==0 are single-node "cascades").
Outputs (into --out-dir):
  sv_hist_global.png / sv_hist_run_<id>.png
  sv_global.csv            mean/median + CI (global)
  sv_per_run.csv           mean/median + CI per run
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def bootstrap_ci(values, stat, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    est = stat(arr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = stat(rng.choice(arr, size=arr.size, replace=True))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, lo, hi


def summarize(values, label, out_dir, n_boot=2000):
    arr = np.asarray(values, dtype=float)
    mean_est, mean_lo, mean_hi = bootstrap_ci(arr, np.mean, n_boot)
    med_est, med_lo, med_hi = bootstrap_ci(arr, np.median, n_boot)
    n_star = int((arr == 1.0).sum())
    return {
        "label": label,
        "n": int(arr.size),
        "mean": mean_est,
        "mean_ci_lo": mean_lo,
        "mean_ci_hi": mean_hi,
        "median": med_est,
        "median_ci_lo": med_lo,
        "median_ci_hi": med_hi,
        "std": float(arr.std()),
        "pct_star": 100.0 * n_star / arr.size,
    }


def plot_hist(values, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=60, color=sns.color_palette("colorblind")[0],
            edgecolor="white", alpha=0.85, density=True)
    ax.set_xlabel("Structural Virality")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(str(data_dir / "cascades.parquet"))
    print(f"loaded {df.height:,} cascades")

    nz = df.filter(pl.col("StructuralVirality") > 0)
    n_zero = df.height - nz.height
    print(f"SV==0 (no reposts): {n_zero:,}; SV>0: {nz.height:,}")

    # global
    sv = nz["StructuralVirality"].to_list()
    plot_hist(sv, out_dir / "sv_hist_global.png",
              "Structural virality (global)")
    rows = [summarize(sv, "global", out_dir, args.n_boot)]

    # per run
    for run_id, vals in (
        nz.group_by("RunID").agg(pl.col("StructuralVirality")).sort("RunID").iter_rows()
    ):
        v = list(vals)
        plot_hist(v, out_dir / f"sv_hist_run_{run_id}.png",
                  f"Structural virality (run {run_id})")
        rows.append(summarize(v, f"run_{run_id}", out_dir, args.n_boot))

    pl.DataFrame(rows).write_csv(out_dir / "sv_summary.csv")
    print(f"wrote {out_dir / 'sv_summary.csv'}")


if __name__ == "__main__":
    main()
