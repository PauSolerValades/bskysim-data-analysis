#!/usr/bin/env python3
"""Random-timeline experiment: results mirroring report sections 8.3/8.4/8.5.

Computes, for the random-drain datasets (steps/random-timeline/datasets/<SIZE>):
  8.3  Repost power-law (per-run discrete power-law fit, alpha/xmin/Vuong)
  8.4  Structural virality (cascade stats, broadcast vs viral, viral nu)
  8.5  Comparison with Bluesky data (table values)

Everything is pushed into DuckDB aggregates (the 1M cascades.parquet is ~123 GB,
so only tiny results are pulled into Python). Outputs JSON + 5 SVG plots.

Run with the sessions env (has powerlaw/polars/duckdb/matplotlib/seaborn):
    uv run --project /home/psoler/firehose-analysis/sessions \
        python results_83_85.py --sizes 10K 100K 500K 1M
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import duckdb
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import seaborn as sns

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATASETS = Path("/data/nfs/psoler/steps/random-timeline/datasets")
OUT = HERE / "output" / "results_83_85"
IMG_DIR = Path("/home/psoler/tfm/report/images/annex/random-timeline")

SIZES = ["10K", "100K", "500K", "1M"]

# Bluesky reference values (from report @tbl-res-vs-data / @sec-data-reposts)
BLUESKY = {
    "cascade_rate_pct": 16.32,
    "size_mean": 9.18, "size_median": 3, "size_max": 12720,
    "depth_mean": 1.50, "depth_median": 1, "depth_max": 131,
    "out_mean": 5.82, "out_median": 2, "out_max": 7768,
    "sv_mean": 1.454, "sv_median": 1.333, "sv_max": 50.27,
    "viral_sv_mean": 2.142, "viral_sv_median": 2.000, "viral_sv_max": 50.269,
    "broadcast_pct": 71.05,
    "alpha_mean": 2.05,
    "xmin_mean": 12,
}

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


def q(sql: str, db: duckdb.DuckDBPyConnection):
    return db.execute(sql).fetchall()


def cascades_path(size: str) -> Path:
    return DATASETS / size / "cascades.parquet"


# ── 8.4 structural virality (one scalar pass per size) ─────────────────
def structural_stats(db: duckdb.DuckDBPyConnection, size: str) -> tuple[dict, dict]:
    """Returns (structural, viral) dicts from a single scan."""
    p = cascades_path(size)
    row = q(f"""
        SELECT count(*) AS total_posts,
               sum((CascadeSize>=2)::INT) AS nontrivial,
               avg(CascadeSize) FILTER (WHERE CascadeSize>=2) AS size_mean,
               approx_quantile(CascadeSize, 0.5) FILTER (WHERE CascadeSize>=2) AS size_med,
               max(CascadeSize) AS size_max,
               avg(CascadeDepth) FILTER (WHERE CascadeSize>=2) AS depth_mean,
               approx_quantile(CascadeDepth, 0.5) FILTER (WHERE CascadeSize>=2) AS depth_med,
               max(CascadeDepth) AS depth_max,
               avg(MaxOutDegree) FILTER (WHERE CascadeSize>=2) AS out_mean,
               approx_quantile(MaxOutDegree, 0.5) FILTER (WHERE CascadeSize>=2) AS out_med,
               max(MaxOutDegree) AS out_max,
               avg(StructuralVirality) FILTER (WHERE CascadeSize>=2) AS sv_mean,
               approx_quantile(StructuralVirality, 0.5) FILTER (WHERE CascadeSize>=2) AS sv_med,
               max(StructuralVirality) AS sv_max,
               sum((CascadeSize>=2 AND CascadeDepth=1)::INT) AS n_broadcast,
               sum((CascadeSize>=2 AND CascadeDepth>=2)::INT) AS n_viral,
               sum(StructuralVirality) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2) AS v_s,
               sum(StructuralVirality*StructuralVirality) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2) AS v_s2,
               min(StructuralVirality) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2) AS v_mn,
               max(StructuralVirality) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2) AS v_mx,
               approx_quantile(StructuralVirality, 0.5) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2) AS v_med
        FROM read_parquet('{p}')
    """, db)[0]
    (total_posts, nontrivial, size_mean, size_med, size_max,
     depth_mean, depth_med, depth_max, out_mean, out_med, out_max,
     sv_mean, sv_med, sv_max, n_broadcast, n_viral,
     v_s, v_s2, v_mn, v_mx, v_med) = row
    structural = {
        "size": size,
        "total_posts": int(total_posts),
        "nontrivial": int(nontrivial),
        "cascade_rate_pct": round(100.0 * nontrivial / total_posts, 3),
        "size_mean": round(size_mean, 3), "size_median": int(size_med), "size_max": int(size_max),
        "depth_mean": round(depth_mean, 3), "depth_median": int(depth_med), "depth_max": int(depth_max),
        "out_mean": round(out_mean, 3), "out_median": int(out_med), "out_max": int(out_max),
        "sv_mean": round(sv_mean, 3), "sv_median": round(sv_med, 3), "sv_max": round(sv_max, 3),
        "n_broadcast": int(n_broadcast), "n_viral": int(n_viral),
        "broadcast_pct": round(100.0 * n_broadcast / (n_broadcast + n_viral), 2),
        "viral_pct": round(100.0 * n_viral / (n_broadcast + n_viral), 2),
    }
    n = int(n_viral)
    mean = v_s / n
    var = (v_s2 - v_s * v_s / n) / (n - 1)
    ci = 1.96 * (var / n) ** 0.5  # normal CI == bootstrap at these n
    viral = {
        "size": size,
        "n_viral": n,
        "viral_sv_mean": round(mean, 3),
        "viral_sv_ci": round(ci, 4),
        "viral_sv_median": round(v_med, 3),
        "viral_sv_min": round(v_mn, 3),
        "viral_sv_max": round(v_mx, 3),
    }
    return structural, viral


def viral_nu_values(db: duckdb.DuckDBPyConnection, size: str, max_n: int = 300_000) -> np.ndarray:
    """Reservoir sample of viral nu(T) (depth>=2), for a KDE density.

    ponytail: sampled in SQL so only max_n rows leave DuckDB (the 1M file is
    123 GB); the full scan is the same, but materializing 30M floats was slow.
    """
    p = cascades_path(size)
    sql = f"""
        SELECT StructuralVirality
        FROM (SELECT StructuralVirality FROM read_parquet('{p}')
              WHERE CascadeSize>=2 AND CascadeDepth>=2) t
        USING SAMPLE reservoir({max_n} ROWS)
    """
    return db.execute(sql).fetchnumpy()["StructuralVirality"]


# ── 8.3 repost power-law ────────────────────────────────────────────────
def repost_histograms(db: duckdb.DuckDBPyConnection, size: str) -> list[tuple[int, int, int]]:
    """Per-run histogram of repost counts (= CascadeSize-1, CascadeSize>=2)."""
    p = cascades_path(size)
    return q(f"""
        SELECT RunID, CascadeSize, count(*) AS c
        FROM read_parquet('{p}')
        WHERE CascadeSize>=2
        GROUP BY RunID, CascadeSize ORDER BY RunID, CascadeSize
    """, db)


def fit_powerlaw(values: np.ndarray) -> dict | None:
    """Discrete power-law fit + Vuong power_law vs lognormal."""
    values = values.astype(float)
    if len(values) < 50:
        return None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            fit = powerlaw.Fit(values, discrete=True, verbose=False)
            R, p = fit.distribution_compare("power_law", "lognormal")
        return {"alpha": float(fit.alpha), "xmin": float(fit.xmin),
                "sigma": float(fit.sigma), "R": float(R), "p": float(p)}
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--max-fit-n", type=int, default=400_000,
                    help="subsample cap per run for the power-law fit (speed)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()

    structural, viral, powerlaw_rows, nu_vals = [], [], [], {}
    for size in args.sizes:
        t0 = time.time()
        print(f"== {size} ==")
        s, v = structural_stats(db, size)
        structural.append(s)
        viral.append(v)
        print(f"  cascade rate {s['cascade_rate_pct']}%, size mean/max "
              f"{s['size_mean']}/{s['size_max']}, broadcast {s['broadcast_pct']}%")
        print(f"  viral nu: n={v['n_viral']:,} mean={v['viral_sv_mean']} "
              f"median={v['viral_sv_median']} max={v['viral_sv_max']}")

        nu_vals[size] = viral_nu_values(db, size)
        print(f"  nu values: {len(nu_vals[size]):,}")

        hist = repost_histograms(db, size)
        # group per run and fit
        runs = {}
        for run_id, cascade_size, c in hist:
            runs.setdefault(run_id, []).append((cascade_size, c))
        print(f"  {len(runs)} runs, fitting power-law per run...")
        fits = []
        for run_id in sorted(runs):
            pairs = runs[run_id]
            sizes_arr = np.array([p[0] for p in pairs])
            counts = np.array([p[1] for p in pairs], dtype=np.int64)
            vals = np.repeat(sizes_arr - 1, counts)
            # ponytail: subsample cap for the discrete power-law fit; alpha/xmin
            # are stable well below 400k points, full-fit if a run ever needs it
            if len(vals) > args.max_fit_n:
                idx = np.linspace(0, len(vals) - 1, args.max_fit_n).astype(np.int64)
                vals = vals[idx]
            r = fit_powerlaw(vals)
            if r:
                r["run"] = int(run_id)
                r["size"] = size
                fits.append(r)
        pl_rows = summarize_powerlaw(fits, size)
        powerlaw_rows.append(pl_rows)
        print(f"  alpha mean={pl_rows['alpha_mean']} xmin mean={pl_rows['xmin_mean']} "
              f"powerlaw runs={pl_rows['pl_runs']}")
        print(f"  {size} done in {time.time()-t0:.0f}s")

    # ── save JSON summary ───────────────────────────────────────────────
    summary = {"structural": structural, "viral": viral,
               "powerlaw": powerlaw_rows, "bluesky": BLUESKY}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'summary.json'}")

    # ── plots ───────────────────────────────────────────────────────────
    plot_nu_density(nu_vals, viral)
    alpha_sim = powerlaw_rows[-1]["alpha_mean"] if powerlaw_rows else 2.9
    plot_powerlaw_comparison(round(alpha_sim, 1))
    print("done")


def summarize_powerlaw(fits: list[dict], size: str) -> dict:
    alphas = np.array([f["alpha"] for f in fits])
    xmins = np.array([f["xmin"] for f in fits])
    n_pl = sum(1 for f in fits if f["R"] > 0 and f["p"] < 0.05)

    def ci(a):
        return round(1.96 * a.std(ddof=1) / np.sqrt(len(a)), 4)

    return {
        "size": size,
        "n_runs": len(fits),
        "alpha_mean": round(float(alphas.mean()), 3),
        "alpha_median": round(float(np.median(alphas)), 3),
        "alpha_ci": ci(alphas),
        "alpha_min": round(float(alphas.min()), 3),
        "alpha_max": round(float(alphas.max()), 3),
        "xmin_mean": round(float(xmins.mean()), 3),
        "xmin_median": round(float(np.median(xmins)), 1),
        "xmin_ci": ci(xmins),
        "xmin_min": int(xmins.min()),
        "xmin_max": int(xmins.max()),
        "pl_runs": n_pl,
    }


def plot_powerlaw_comparison(alpha_sim: float) -> None:
    """Synthetic CCDF+density: Bluesky alpha vs representative random alpha."""
    XMIN = 12.0
    ALPHA_BLUESKY = BLUESKY["alpha_mean"]
    x = np.linspace(XMIN, 500, 500)
    ccdf = lambda a: (x / XMIN) ** (1 - a)
    dens = lambda a: (a - 1) / XMIN * (x / XMIN) ** (-a)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, f, ylab, title in [(axes[0], ccdf, "$P(X \\geq x)$", "CCDF"),
                               (axes[1], dens, "Density", "Density")]:
        ax.plot(x, f(ALPHA_BLUESKY), color="#1d9bf0", lw=2.2,
                label=f"Bluesky  $\\alpha={ALPHA_BLUESKY}$")
        ax.plot(x, f(alpha_sim), color="#e53935", lw=2.2,
                label=f"Random timeline  $\\alpha={alpha_sim}$")
        ax.set_xlabel("Reposts")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend()
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    fig.tight_layout()
    out = IMG_DIR / "powerlaw_alpha_comparison.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_nu_density(nu_vals: dict, viral: list[dict], max_n: int = 300_000, seed: int = 42) -> None:
    """Log-x KDE density of viral nu(T) per size (matches viral_nu_stats.py).

    ponytail: KDE subsampled to max_n for speed; the density shape is stable
    well below this (heavily quantized nu), raise the cap if a tail wobble appears.
    """
    medians = {v["size"]: v["viral_sv_median"] for v in viral}
    rng = np.random.default_rng(seed)
    for size, nu in nu_vals.items():
        if len(nu) > max_n:
            nu = rng.choice(nu, size=max_n, replace=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.kdeplot(x=nu, log_scale=(True, False), color=PALETTE[0],
                    linewidth=2, ax=ax, label="viral cascades")
        ax.axvline(2.0, color=PALETTE[1], linestyle="--", linewidth=1.4,
                   label="broadcast floor $\\nu=2$")
        ax.axvline(medians[size], color=PALETTE[2], linestyle=":", linewidth=1.4,
                   label=f"median $\\nu={medians[size]:.2f}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Structural virality $\nu(T)$")
        ax.set_ylabel("Density")
        ax.set_title(f"Structural virality of viral cascades ({size})",
                     fontsize=12, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        out = IMG_DIR / f"viral_nu_density_{size}.svg"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
