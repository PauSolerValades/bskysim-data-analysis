#!/usr/bin/env python3
"""Build follower-count (in-degree) maps straight from the simulation topology.

Why this exists (the trap it fixes):
    data/original/<n>_nodes.parquet does NOT have the same node ordering (or even
    node count) as the .bin the simulator actually reads: e.g. the 100K bin has
    99,835 nodes with the main hub at sim_id 97118, while the parquet's row order
    puts it at 97283 and reports 100,000 nodes. Joining cascades.AuthorID against
    parquet-derived degrees therefore silently produces wrong follower counts.
    Degrees MUST be taken from the same .bin the run consumed.

Binary layout (little-endian, bskysim/src/load-topology.zig):
    u32 n                     number of nodes
    u32 user_ids[n]           original ids (here already remapped to 0..n-1)
    u32 m                     number of edges
    u32 edges[2*m]            interleaved (actor_id, subject_id); actor follows subject

Follower count of sim user i = number of edges whose subject_id == i.

Usage:
    python degree_map.py --repo ../../des-ctic-dev --out output/degrees
"""

import argparse
import csv
from pathlib import Path

import numpy as np

SIZES = ["10K", "50K", "100K", "500K", "1M"]


def default_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "des-ctic-dev"


def build_degree(bin_path: Path) -> np.ndarray:
    """Return indeg[i] = number of followers of sim user i."""
    with open(bin_path, "rb") as f:
        n = int(np.fromfile(f, dtype="<u4", count=1)[0])
        np.fromfile(f, dtype="<u4", count=n)  # user_ids, unused
        m = int(np.fromfile(f, dtype="<u4", count=1)[0])
        edges = np.fromfile(f, dtype="<u4", count=2 * m)
    subjects = edges[1::2].astype(np.int64)
    assert subjects.max(initial=0) < n, f"{bin_path}: subject id out of range"
    deg = np.bincount(subjects, minlength=n)
    assert int(deg.sum()) == m, "every edge must contribute exactly one follower"
    return deg


def write_degree_csv(deg: np.ndarray, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_id", "indeg"])
        w.writerows(enumerate(int(d) for d in deg))


def ensure(repo: Path, size: str, out_dir: Path) -> Path:
    """Build (or reuse) the degree csv for `size`. Returns its path."""
    out_csv = out_dir / f"{size}.csv"
    if out_csv.exists():
        return out_csv
    bin_path = repo / "data" / "monotonic" / f"{size}_monotonic.bin"
    deg = build_degree(bin_path)
    write_degree_csv(deg, out_csv)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=default_repo())
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "output" / "degrees")
    args = ap.parse_args()

    for size in args.sizes:
        bin_path = args.repo / "data" / "monotonic" / f"{size}_monotonic.bin"
        deg = build_degree(bin_path)
        out_csv = args.out / f"{size}.csv"
        write_degree_csv(deg, out_csv)
        top = np.argsort(deg)[-3:][::-1]
        hubs = ", ".join(f"{int(i)}:{int(deg[i])}" for i in top)
        print(f"{size}: {len(deg):,} nodes, top hubs (sim_id:followers) {hubs} -> {out_csv}")


if __name__ == "__main__":
    main()
