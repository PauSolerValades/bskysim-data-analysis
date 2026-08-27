#!/usr/bin/env python3
"""Plot per-run RAM, normalized per worker, with min/max range and the estimate.

Reuses the parsing helpers from analyze.py. Produces one PNG per the min/max
range + estimate, so the worker count no longer distorts the comparison
(100K's ~1 TB peak was 12 workers × ~89 GB each).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import analyze


def fmt(v: float) -> str:
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot per-run RAM normalized per worker.")
    ap.add_argument("--ram-file", default="/tmp/ram-final.txt")
    ap.add_argument("--traces-dir", default="/home/psoler/des-ctic-dev/steps/final/traces")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    ram = analyze.parse_ram(Path(args.ram_file))
    traces_dir = Path(args.traces_dir)

    labels, mn, mx, est = [], [], [], []
    for size, start, end in analyze.SIZES:
        seg = [r for r in ram if start <= r[0] <= end + analyze.END_PAD]
        if not seg:
            continue
        peak_mb = max(r[2] for r in seg)
        times = analyze.parse_times(traces_dir / size / "execution_times.ssv")
        if not times:
            continue
        workers = len(times)
        slices = analyze.run_slices(start, times)
        peaks = [max((r[2] for r in ram if s <= r[0] <= e), default=None) for s, e, w, r in slices]
        vals = [p for p in peaks if p is not None]
        if not vals:
            continue
        labels.append(size)
        mn.append(min(vals) / workers)
        mx.append(max(vals) / workers)
        est.append(peak_mb / workers)

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))

    # min -> max range, normalized per worker
    for xi, lo, hi in zip(x, mn, mx):
        ax.vlines(xi, lo, hi, color="0.65", lw=2, zorder=2)
        ax.plot([xi - 0.18, xi + 0.18], [lo, lo], color="0.4", lw=1.5, zorder=2)
        ax.plot([xi - 0.18, xi + 0.18], [hi, hi], color="0.4", lw=1.5, zorder=2)

    # estimate marker (peak / workers), labeled
    ax.scatter(x, est, color="C0", s=45, zorder=3, label="estimate (peak / workers)")
    for xi, e in zip(x, est):
        ax.annotate(f"{fmt(e)} GB", (xi, e), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("size")
    ax.set_ylabel("RAM per run, per worker (GB, log)")
    ax.set_title("Per-run RAM (min → max bar, estimate marked), normalized per worker")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left")

    # annotate the min and max numbers (small, grey)
    for xi, lo, hi in zip(x, mn, mx):
        ax.annotate(fmt(lo), (xi, lo), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7, color="0.4")
        ax.annotate(fmt(hi), (xi, hi), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7, color="0.4")

    fig.tight_layout()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "ram_per_run.png"
    fig.savefig(png, dpi=150)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
