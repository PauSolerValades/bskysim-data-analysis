#!/usr/bin/env python3
"""Queue vs attention: what the random-timeline experiment says about cascade width.

Reproduces the numbers behind the "Queue-Based Attention Bottleneck" section:

  1. overview   aggregate cascade stats, LIFO (steps/final) vs random
                (steps/random-timeline), for 10K/50K/100K/500K/1M.
  2. buckets    cascade size and first-hop broadcast width bucketed by the
                author's true follower count (from the .bin, see degree_map.py),
                LIFO vs random. This is where the "missing width" lives: mean
                size rises with followers but heavily compressed, and the heavy
                tail is hub-dominated.
  3. top        the largest cascades with their author's follower count.
  4. figure     mean size and P(size >= 50) vs author followers, LIFO vs random.

Requires: duckdb CLI on PATH (parquet queries), matplotlib, numpy.
Usage:
    python queue_experiment.py --repo ../../des-ctic-dev
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

import degree_map  # noqa: E402

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "text.usetex": shutil.which("latex") is not None,
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

DUCKDB = os.environ.get("DUCKDB", "duckdb")
HERE = Path(__file__).resolve().parent
SIZES = ["10K", "50K", "100K", "500K", "1M"]
TIMELINES = {"lifo": "final", "random": "random-timeline"}
# random-timeline data lives on the NFS scratch, not under the repo
RANDOM_ROOT = Path("/data/nfs/psoler/steps/random-timeline/datasets")

# log-spaced follower buckets (upper edge, label)
BUCKETS = [
    (10, "<=10"),
    (100, "11-100"),
    (1000, "101-1k"),
    (10000, "1k-10k"),
    (100000, "10k-100k"),
    (1 << 62, ">100k"),
]


def duck(sql: str) -> list[dict]:
    r = subprocess.run([DUCKDB, "-csv", "-c", sql], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"duckdb failed:\n{r.stderr}")
    return list(csv.DictReader(io.StringIO(r.stdout.replace("\r\n", "\n"))))


def cascades_path(repo: Path, timeline: str, size: str) -> Path:
    if timeline == "random":
        return RANDOM_ROOT / size / "cascades.parquet"
    return repo / "steps" / "final" / "datasets" / size / "cascades.parquet"


def bucket_case(col: str) -> str:
    parts = ["CASE"]
    for upper, label in BUCKETS:
        if upper == BUCKETS[-1][0]:
            parts.append(f"ELSE '{label}'")
        else:
            parts.append(f"WHEN {col}<={upper} THEN '{label}'")
    return " ".join(parts) + " END"


def bucket_index(col: str) -> str:
    parts = ["CASE"]
    for i, (upper, _) in enumerate(BUCKETS[:-1], start=1):
        parts.append(f"WHEN {col}<={upper} THEN {i}")
    return " ".join(parts) + f" ELSE {len(BUCKETS)} END"


def overview(repo: Path, sizes: list[str]) -> list[dict]:
    rows = []
    for timeline in TIMELINES:
        for size in sizes:
            p = cascades_path(repo, timeline, size)
            if not p.exists():
                continue
            sql = f"""
            SELECT count(*) AS total_posts,
                   round(100.0*sum((CascadeSize>=2)::INT)/count(*),3) AS pct_nontrivial,
                   round(avg(CascadeSize) FILTER (WHERE CascadeSize>=2),3) AS size_mean,
                   approx_quantile(CascadeSize, 0.5) FILTER (WHERE CascadeSize>=2) AS size_med,
                   max(CascadeSize) AS size_max,
                   round(avg(CascadeDepth) FILTER (WHERE CascadeSize>=2),3) AS depth_mean,
                   max(CascadeDepth) AS depth_max,
                   round(avg(MaxOutDegree) FILTER (WHERE CascadeSize>=2),3) AS out_mean,
                   max(MaxOutDegree) AS out_max,
                   round(avg(StructuralVirality) FILTER (WHERE StructuralVirality>0),3) AS sv_mean,
                   max(StructuralVirality) AS sv_max,
                   round(100.0*sum((CascadeDepth=1)::INT) FILTER (WHERE CascadeSize>=2)
                         / sum((CascadeSize>=2)::INT),2) AS broadcast_pct,
                   sum((CascadeSize>=2 AND CascadeDepth>=2)::INT) AS n_viral,
                   round(avg(StructuralVirality) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2),3) AS viral_sv_mean,
                   approx_quantile(StructuralVirality, 0.5) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2) AS viral_sv_med,
                   max(StructuralVirality) AS viral_sv_max
            FROM read_parquet('{p}')
            """
            row = duck(sql)[0]
            row.update(timeline=timeline, size=size)
            rows.append(row)
            print(f"  overview {timeline:6s} {size:>4s}: "
                  f"nontrivial={row['pct_nontrivial']}% "
                  f"size mean/max={row['size_mean']}/{row['size_max']} "
                  f"out mean/max={row['out_mean']}/{row['out_max']} "
                  f"broadcast={row['broadcast_pct']}%")
    return rows


def width_buckets(repo: Path, sizes: list[str], deg_dir: Path) -> list[dict]:
    rows = []
    for timeline in TIMELINES:
        for size in sizes:
            p = cascades_path(repo, timeline, size)
            if not p.exists():
                continue
            deg_csv = degree_map.ensure(repo, size, deg_dir)
            sql = f"""
            WITH deg AS (SELECT sim_id, indeg FROM read_csv('{deg_csv}')),
            c AS (
                SELECT c.CascadeSize, c.CascadeDepth, c.MaxOutDegree, d.indeg,
                       {bucket_case('d.indeg')} AS bucket,
                       {bucket_index('d.indeg')} AS bi
                FROM read_parquet('{p}') c JOIN deg d ON c.AuthorID = d.sim_id
            )
            SELECT bucket, bi, count(*) AS n,
                   round(avg(CascadeSize),4) AS mean_size,
                   sum((CascadeSize>=50)::INT) AS n_ge50,
                   round(sum((CascadeSize>=50)::INT)::DOUBLE/count(*),8) AS p_ge50,
                   round(avg(MaxOutDegree) FILTER (WHERE CascadeDepth=1 AND CascadeSize>=2),3) AS mean_first_hop,
                   max(CascadeSize) AS max_size
            FROM c GROUP BY bucket, bi ORDER BY bi
            """
            for row in duck(sql):
                row.update(timeline=timeline, size=size)
                rows.append(row)
            print(f"  buckets  {timeline:6s} {size:>4s}: done")
    return rows


def top_cascades(repo: Path, sizes: list[str], deg_dir: Path, n: int = 10) -> list[dict]:
    rows = []
    for timeline in TIMELINES:
        for size in sizes:
            p = cascades_path(repo, timeline, size)
            if not p.exists():
                continue
            deg_csv = degree_map.ensure(repo, size, deg_dir)
            sql = f"""
            WITH deg AS (SELECT sim_id, indeg FROM read_csv('{deg_csv}'))
            SELECT c.AuthorID, c.CascadeSize, c.CascadeDepth, c.MaxOutDegree,
                   round(c.StructuralVirality,3) AS sv, coalesce(d.indeg,0) AS author_followers
            FROM read_parquet('{p}') c JOIN deg d ON c.AuthorID = d.sim_id
            ORDER BY c.CascadeSize DESC LIMIT {n}
            """
            for row in duck(sql):
                row.update(timeline=timeline, size=size)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_width(rows: list[dict], sizes: list[str], out_path: Path) -> None:
    """2x2: rows = size, cols = mean size / P(size>=50), series = LIFO vs random."""
    labels = [b[1] for b in BUCKETS]
    xpos = np.arange(len(labels))
    colors = {"lifo": "#1d9bf0", "random": "#e53935"}

    fig, axes = plt.subplots(len(sizes), 2, figsize=(11, 4.2 * len(sizes)), squeeze=False)
    for r, size in enumerate(sizes):
        for c, metric in enumerate(["mean_size", "p_ge50"]):
            ax = axes[r][c]
            plotted = []
            for timeline in TIMELINES:
                sel = sorted((x for x in rows if x["size"] == size and x["timeline"] == timeline),
                             key=lambda x: int(x["bi"]))
                if not sel:
                    continue
                y = [float(x[metric]) for x in sel]
                plotted += y
                ax.plot(xpos[: len(y)], y, "o-", color=colors[timeline],
                        label="random timeline" if timeline == "random" else "LIFO")
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            if any(v > 0 for v in plotted):
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.set_xlabel("author followers" if r == len(sizes) - 1 else "")
            if metric == "mean_size":
                ax.set_ylabel(f"{size}\nmean cascade size" if c == 0 else "")
                ax.set_title("mean cascade size")
            else:
                ax.set_ylabel(f"{size}\nP(size $\\geq$ 50)" if c == 0 else "")
                ax.set_title("$P(\\mathrm{size} \\geq 50)$")
            if r == 0 and c == 0:
                ax.legend(frameon=True)
    fig.suptitle("Cascade width vs author followers (pooled over runs)", y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  figure -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=degree_map.default_repo())
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--out", type=Path, default=HERE / "output")
    args = ap.parse_args()

    repo, out = args.repo, args.out
    deg_dir = out / "degrees"

    print("== overview")
    ov = overview(repo, args.sizes)
    if ov:
        write_csv(out / "overview_lifo_vs_random.csv", ov)

    print("== width by author followers")
    wb = width_buckets(repo, args.sizes, deg_dir)
    if wb:
        write_csv(out / "width_by_author_followers.csv", wb)

    print("== top cascades")
    tc = top_cascades(repo, args.sizes, deg_dir)
    if tc:
        write_csv(out / "top_cascades.csv", tc)

    if wb:
        # only sizes with a giant-hub (>100k) bucket carry the width story
        hub_sizes = [s for s in args.sizes
                     if any(x["size"] == s and x["bucket"] == ">100k" for x in wb)]
        plot_width(wb, hub_sizes or [args.sizes[-1]], out / "width_vs_followers.svg")

    print("done")


if __name__ == "__main__":
    main()
