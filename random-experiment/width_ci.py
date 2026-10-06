#!/usr/bin/env python3
"""Per-bucket CIs for @tbl-queue-width (mean size, P(size>=50)).

Two conventions, per author-follower bucket:
  within-run      — metric computed per run, CI over the ~100 run values.
  within-cascade  — pooled over the bucket's cascades, normal SEM.

Run with the sessions env:
    uv run --project /home/psoler/firehose-analysis/sessions python width_ci.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np

import degree_map
from queue_experiment import bucket_case, bucket_index, cascades_path

SIM = {
    "lifo": Path("/home/psoler/des-ctic-dev/steps/final/datasets"),
    "random": Path("/data/nfs/psoler/steps/random-timeline/datasets"),
}
SIZES = ["500K", "1M"]
DEG_DIR = Path("/home/psoler/bskysim-data-analysis/random-experiment/output/degrees")
OUT = Path("/home/psoler/bskysim-data-analysis/random-experiment/output/width_ci.json")


def query(p: Path, deg_csv: Path) -> str:
    return f"""
        WITH deg AS (SELECT sim_id, indeg FROM read_csv('{deg_csv}')),
        c AS (
            SELECT c.RunID, d.indeg, c.CascadeSize,
                   {bucket_case('d.indeg')} AS bucket,
                   {bucket_index('d.indeg')} AS bi
            FROM read_parquet('{p}') c JOIN deg d ON c.AuthorID = d.sim_id
        )
        SELECT RunID, bucket, bi, count(*) AS n,
               sum(CascadeSize) AS s_size,
               sum(CascadeSize*CascadeSize) AS s2_size,
               sum((CascadeSize>=50)::INT) AS n_ge50
        FROM c GROUP BY RunID, bucket, bi ORDER BY bi
    """


def ci_run(vals: np.ndarray) -> float:
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))


def ci_cascade_mean(s: float, s2: float, n: int) -> float:
    if n < 2:
        return 0.0
    var = (s2 - s * s / n) / (n - 1)
    return float(1.96 * np.sqrt(var / n))


def ci_cascade_prop(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    return float(1.96 * np.sqrt(p * (1 - p) / n))


def main() -> None:
    db = duckdb.connect()
    out = {}
    for timeline, root in SIM.items():
        for size in SIZES:
            p = cascades_path(Path("/home/psoler/des-ctic-dev"), timeline, size)
            if not p.exists():
                continue
            deg_csv = degree_map.ensure(Path("/home/psoler/des-ctic-dev"), size, DEG_DIR)
            t0 = time.time()
            d = db.execute(query(p, deg_csv)).fetchnumpy()
            n_runs = len(np.unique(d["RunID"]))
            buckets = sorted(set(zip(d["bucket"].tolist(), d["bi"].tolist())),
                             key=lambda x: x[1])
            res = {"n_runs": n_runs, "buckets": {}}
            for bucket, bi in buckets:
                m = (d["bucket"] == bucket)
                n = d["n"][m].astype(float)
                s_size = d["s_size"][m].astype(float)
                s2_size = d["s2_size"][m].astype(float)
                n_ge50 = d["n_ge50"][m].astype(float)
                mean_size_run = s_size / n
                p_ge50_run = n_ge50 / n
                # pooled
                N = n.sum(); S = s_size.sum(); S2 = s2_size.sum(); G = n_ge50.sum()
                res["buckets"][bucket] = {
                    "mean_size": {
                        "within_run": round(ci_run(mean_size_run), 4),
                        "within_cascade": round(ci_cascade_mean(S, S2, int(N)), 4),
                    },
                    "p_ge50": {
                        "within_run": round(ci_run(p_ge50_run), 6),
                        "within_cascade": round(ci_cascade_prop(G / N, int(N)), 6),
                    },
                }
            out[f"{timeline}_{size}"] = res
            print(f"{timeline:6s} {size:>4s}: {n_runs} runs ({time.time()-t0:.0f}s)")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
