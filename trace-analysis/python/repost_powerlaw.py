#!/usr/bin/env python3
"""
Repost count power-law analysis.

For each run and globally:
  - histogram (linear + log-log) of total_reposts (>= 1)
  - power-law fit via the `powerlaw` package: alpha (gamma), xmin, sigma
  - likelihood-ratio tests power_law vs lognormal / exponential

Outputs (into --out-dir):
  repost_hist_<run>.png      per-run log-log histogram (run=global for pooled)
  powerlaw_fits.csv          per-run + global alpha/xmin/sigma/compare results
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import powerlaw

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def fit_one(values, label):
    """Fit power law to integer repost counts. Returns dict of results."""
    n = len(values)
    res = {"label": label, "n": n}
    if n < 50:
        res.update({"alpha": None, "xmin": None, "sigma": None,
                    "pl_vs_lognormal_R": None, "pl_vs_lognormal_p": None,
                    "pl_vs_exponential_R": None, "pl_vs_exponential_p": None,
                    "note": "n<50 skipped"})
        return res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = powerlaw.Fit(np.asarray(values, dtype=float),
                               discrete=True, verbose=False)
            res["alpha"] = fit.alpha
            res["xmin"] = fit.xmin
            res["sigma"] = fit.sigma
            for dist, key in [("lognormal", "lognormal"),
                              ("exponential", "exponential")]:
                R, p = fit.distribution_compare("power_law", dist)
                res[f"pl_vs_{key}_R"] = R
                res[f"pl_vs_{key}_p"] = p
        except Exception as e:  # noqa: BLE001
            res["note"] = f"fit failed: {e}"
    return res


def plot_loglog(values, out_path, title):
    if len(values) == 0:
        return
    counts, edges = np.histogram(values, bins=np.arange(0.5, max(values) + 1.5, 1.0))
    centers = np.arange(1, max(values) + 1)
    mask = counts > 0
    x, y = centers[mask], counts[mask]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(x, y, "o", markersize=3, alpha=0.7,
              color=sns.color_palette("colorblind")[0])
    if len(x) >= 2:
        lx, ly = np.log10(x), np.log10(y)
        slope, intercept = np.polyfit(lx, ly, 1)
        ax.loglog(x, 10 ** intercept * x ** slope, "--", color="crimson",
                  linewidth=2, label=f"$\\alpha \\approx {-slope:.2f}$")
    ax.set_xlabel("Total reposts")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(str(data_dir / "post_metrics.parquet"))
    nonzero = df.filter(pl.col("total_reposts") >= 1)
    print(f"loaded {df.height:,} posts, {nonzero.height:,} with >=1 repost")

    # global
    global_vals = nonzero["total_reposts"].to_list()
    plot_loglog(global_vals, out_dir / "repost_hist_global.png",
                "Repost count distribution (global)")
    rows = [fit_one(global_vals, "global")]

    # per run
    for run_id, reposts in (
        nonzero.group_by("run_id").agg(pl.col("total_reposts")).sort("run_id").iter_rows()
    ):
        vals = list(reposts)
        rows.append(fit_one(vals, f"run_{run_id}"))
        plot_loglog(vals, out_dir / f"repost_hist_run_{run_id}.png",
                    f"Repost count distribution (run {run_id})")

    pl.DataFrame(rows).write_csv(out_dir / "powerlaw_fits.csv")
    print(f"wrote {out_dir / 'powerlaw_fits.csv'}")
    # console summary of global + per-run alpha spread
    alphas = [r["alpha"] for r in rows[1:] if r.get("alpha") is not None]
    if alphas:
        print(f"per-run alpha: mean={np.mean(alphas):.3f} "
              f"std={np.std(alphas):.3f} min={np.min(alphas):.3f} max={np.max(alphas):.3f}")
    g = rows[0]
    print(f"global alpha={g.get('alpha')} xmin={g.get('xmin')} "
          f"pl_vs_lognormal_R={g.get('pl_vs_lognormal_R')} p={g.get('pl_vs_lognormal_p')} "
          f"pl_vs_exponential_R={g.get('pl_vs_exponential_R')} p={g.get('pl_vs_exponential_p')}")


if __name__ == "__main__":
    main()
