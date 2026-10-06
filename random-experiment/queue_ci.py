#!/usr/bin/env python3
"""Bootstrap CIs for the queue-experiment aggregate table (@tbl-queue-aggregate).

Computes, per size × timeline, two CI conventions for the pooled metrics:

  within-run      — the metric is computed per run, then a 95% CI is taken
                    over the ~100 run-level values (unit of observation = the
                    run; captures replication noise). Matches @tbl-res-reposts.
  within-cascade  — the metric is pooled over all cascades and the CI is the
                    normal SEM over the cascade population (unit = the cascade;
                    matches @tbl-res-viral-sv). With hundreds of millions of
                    cascades these CIs are essentially zero.

One DuckDB scan per (size, timeline): GROUP BY RunID with per-run sums, from
which both conventions follow.

Run with the sessions env:
    uv run --project /home/psoler/firehose-analysis/sessions python queue_ci.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np

SIM = {
    "lifo": Path("/home/psoler/des-ctic-dev/steps/final/datasets"),
    "random": Path("/data/nfs/psoler/steps/random-timeline/datasets"),
}
SIZES = ["10K", "100K", "500K", "1M"]
OUT = Path("/home/psoler/bskysim-data-analysis/random-experiment/output/queue_ci.json")

METRICS = ["pct_nt", "size_mean", "out_mean", "sv_mean", "viral_sv_mean", "broadcast_pct"]


def per_run_query(p: Path) -> str:
    return f"""
        SELECT RunID,
               count(*) AS n_posts,
               sum((CascadeSize>=2)::INT) AS n_nt,
               sum((CascadeDepth=1)::INT) FILTER (WHERE CascadeSize>=2) AS n_bcast,
               sum(CascadeSize) FILTER (WHERE CascadeSize>=2) AS s_size,
               sum(CascadeSize*CascadeSize) FILTER (WHERE CascadeSize>=2) AS s2_size,
               sum(MaxOutDegree) FILTER (WHERE CascadeSize>=2) AS s_out,
               sum(MaxOutDegree*MaxOutDegree) FILTER (WHERE CascadeSize>=2) AS s2_out,
               sum(StructuralVirality) FILTER (WHERE CascadeSize>=2) AS s_sv,
               sum(StructuralVirality*StructuralVirality) FILTER (WHERE CascadeSize>=2) AS s2_sv,
               sum(StructuralVirality) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2) AS s_vsv,
               sum(StructuralVirality*StructuralVirality) FILTER (WHERE CascadeSize>=2 AND CascadeDepth>=2) AS s2_vsv,
               sum((CascadeSize>=2 AND CascadeDepth>=2)::INT) AS n_viral
        FROM read_parquet('{p}')
        GROUP BY RunID
    """


def ci_across_runs(vals: np.ndarray) -> float:
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < 2:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(n))


def ci_across_cascades(mean: float, sumsq: float, ssum: float, n: int) -> float:
    """Normal SEM over the pooled cascade population."""
    if n < 2:
        return 0.0
    var = (sumsq - ssum * ssum / n) / (n - 1)
    return float(1.96 * np.sqrt(var / n))


def ci_proportion(p: float, n: int) -> float:
    """Binomial SE for a pooled proportion (in %)."""
    if n == 0:
        return 0.0
    return float(100 * 1.96 * np.sqrt(p / 100 * (1 - p / 100) / n))


def main() -> None:
    db = duckdb.connect()
    out = {}
    for timeline, root in SIM.items():
        for size in SIZES:
            p = root / size / "cascades.parquet"
            if not p.exists():
                continue
            t0 = time.time()
            d = db.execute(per_run_query(p)).fetchnumpy()
            n_runs = len(d["RunID"])

            # per-run metric values
            n_posts = d["n_posts"].astype(float)
            n_nt = d["n_nt"].astype(float)
            n_bcast = d["n_bcast"].astype(float)
            n_viral = d["n_viral"].astype(float)
            pr = {
                "pct_nt": 100 * n_nt / n_posts,
                "size_mean": d["s_size"] / n_nt,
                "out_mean": d["s_out"] / n_nt,
                "sv_mean": d["s_sv"] / n_nt,
                "viral_sv_mean": d["s_vsv"] / n_viral,
                "broadcast_pct": 100 * n_bcast / n_nt,
            }

            # pooled sums (across runs) for the within-cascade convention
            P = {k: float(v.sum()) for k, v in d.items()}

            res = {"n_runs": n_runs}
            for m in METRICS:
                res[m] = {
                    "within_run": round(ci_across_runs(pr[m]), 4),
                    "within_cascade": round(
                        _cascade_ci(m, P), 5
                    ),
                }
            out[f"{timeline}_{size}"] = res
            print(f"{timeline:6s} {size:>4s}: {n_runs} runs ({time.time()-t0:.0f}s)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


def _cascade_ci(m: str, P: dict) -> float:
    if m == "pct_nt":
        return ci_proportion(100 * P["n_nt"] / P["n_posts"], int(P["n_posts"]))
    if m == "broadcast_pct":
        return ci_proportion(100 * P["n_bcast"] / P["n_nt"], int(P["n_nt"]))
    spec = {
        "size_mean": ("s_size", "s2_size", "n_nt"),
        "out_mean": ("s_out", "s2_out", "n_nt"),
        "sv_mean": ("s_sv", "s2_sv", "n_nt"),
        "viral_sv_mean": ("s_vsv", "s2_vsv", "n_viral"),
    }[m]
    s, s2, n = spec
    return ci_across_cascades(P[s] / P[n], P[s2], P[s], int(P[n]))


if __name__ == "__main__":
    main()
