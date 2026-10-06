#!/usr/bin/env python3
"""Cascade metrics (size, depth, width) for the final LIFO datasets.

For every dataset (``<dataset>/cascades.parquet``) and every tree metric
(``CascadeSize``, ``CascadeDepth``, ``MaxOutDegree``) over the non-trivial
cascades (``CascadeSize >= 2``):

- pooled mean / median / max;
- 95% CI of the mean under the two conventions used across the thesis:
    * ``within_run``     — metric per run, then CI over the ~100 run values
                           (unit of observation = the run; captures replication
                           noise; the more honest of the two);
    * ``within_cascade`` — pooled over all cascades, normal SEM
                           (unit = the cascade; with 10^7-10^8 cascades the CI
                           is near zero).

Also writes one rank-size figure per dataset (the three metrics sorted largest
to smallest, log-log), matching @fig-data-cascade-shape.

Run with the sessions env:
    uv run --project /home/psoler/firehose-analysis/sessions \
        python cascade_metrics.py
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SIM_ROOT = Path("/home/psoler/des-ctic-dev/steps/final/datasets")
SIZES = ["10K", "100K", "500K", "1M"]
OUT_IMG = Path("/home/psoler/tfm/report/images/results")
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_JSON = OUT_DIR / "cascade_metrics.json"
# (parquet column, table label, is the size column used to filter non-trivial)
METRICS = [
    ("CascadeSize", "Size", "CascadeSize"),
    ("CascadeDepth", "Depth", "CascadeSize"),
    ("MaxOutDegree", "Max out-degree", "CascadeSize"),
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


def per_run(con, path: Path) -> dict:
    """Per-run sums for every metric -> CI under both conventions."""
    q = f"""
        SELECT RunID,
               sum((CascadeSize>=2)::INT)                          AS n_nt,
               sum(CascadeSize)      FILTER (WHERE CascadeSize>=2) AS s_size,
               sum(CascadeSize*CascadeSize) FILTER (WHERE CascadeSize>=2) AS s2_size,
               sum(CascadeDepth)     FILTER (WHERE CascadeSize>=2) AS s_depth,
               sum(CascadeDepth*CascadeDepth) FILTER (WHERE CascadeSize>=2) AS s2_depth,
               sum(MaxOutDegree)     FILTER (WHERE CascadeSize>=2) AS s_out,
               sum(MaxOutDegree*MaxOutDegree) FILTER (WHERE CascadeSize>=2) AS s2_out
        FROM read_parquet('{path}')
        GROUP BY RunID
    """
    d = con.execute(q).fetchnumpy()
    n = d["n_nt"].astype(float)
    runs = len(n)
    out = {"n_runs": runs}
    for col, label, _ in METRICS:
        key = {"CascadeSize": "size", "CascadeDepth": "depth", "MaxOutDegree": "out"}[col]
        s = d[f"s_{key}"].astype(float)
        s2 = d[f"s2_{key}"].astype(float)
        run_mean = s / n
        ci_run = 1.96 * run_mean.std(ddof=1) / np.sqrt(runs)
        N = n.sum()
        var = (s2.sum() - s.sum() ** 2 / N) / (N - 1)
        ci_casc = 1.96 * np.sqrt(var / N)
        out[key] = {"ci_run": float(ci_run), "ci_cascade": float(ci_casc)}
    return out


def distribution(con, path: Path, col: str, filt: str | None) -> tuple[np.ndarray, np.ndarray]:
    where = f"WHERE {filt} >= 2" if filt else ""
    q = f"""
        SELECT {col} AS v, count(*) AS n
        FROM read_parquet('{path}') {where}
        GROUP BY {col} ORDER BY {col}
    """
    d = con.execute(q).fetchnumpy()
    return d["v"].astype(float), d["n"].astype(float)


def pooled(v: np.ndarray, n: np.ndarray) -> tuple[float, float, float]:
    N = n.sum()
    mean = float((v * n).sum() / N)
    median = float(v[np.searchsorted(np.cumsum(n), N / 2)])
    return mean, median, float(v.max())


def ranksize_xy(v: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ascending (value, count) -> descending rank-size step (x=rank, y=value)."""
    v, n = v[::-1], n[::-1]
    ends = np.cumsum(n)
    starts = np.concatenate([[1.0], ends[:-1] + 1])
    xs, ys = [], []
    for x0, x1, y in zip(starts, ends, v):
        xs += [x0, x1]
        ys += [y, y]
    return np.array(xs), np.array(ys)


def plot_dataset(size: str, curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (label, (x, y)), color in zip(curves.items(), PALETTE):
        ax.plot(x, y, color=color, linewidth=1.6, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cascade rank (largest first)")
    ax.set_ylabel("Cascade metric")
    ax.set_title(f"Cascade shape, {size}", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    out = OUT_IMG / f"cascade_shape_ranksize_{size}.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    report = {}
    for size in SIZES:
        path = SIM_ROOT / size / "cascades.parquet"
        if not path.exists():
            print(f"skip {size} (missing)")
            continue
        t0 = time.time()
        res = per_run(con, path)
        curves = {}
        for col, label, filt in METRICS:
            v, n = distribution(con, path, col, filt)
            mean, median, mx = pooled(v, n)
            key = {"CascadeSize": "size", "CascadeDepth": "depth", "MaxOutDegree": "out"}[col]
            res[key].update({"mean": mean, "median": median, "max": mx})
            curves[label] = ranksize_xy(v, n)
            print(f"  {size} {label:16s} mean={mean:.4g} median={median:g} max={mx:g} "
                  f"±run={res[key]['ci_run']:.4g} ±casc={res[key]['ci_cascade']:.4g}")
        report[size] = res
        plot_dataset(size, curves)
        print(f"{size}: {res['n_runs']} runs ({time.time() - t0:.0f}s)")
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
