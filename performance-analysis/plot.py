#!/usr/bin/env python3
"""Plot per-run RAM, normalized per worker, with min/max range and the median.

Reuses the parsing helpers from analyze.py. All values are in GB and divided by
the worker count, so the worker count no longer distorts the comparison
(100K's ~1 TB peak was 12 workers × ~89 GB each).

The "estimate" marker is the MEDIAN of the per-run slice peaks (÷ workers) —
not the process peak ÷ workers — so it always lies inside the [min, max] range.
"""

from __future__ import annotations

import argparse
import statistics
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

    labels, lo, hi, med = [], [], [], []
    for size, start, end in analyze.SIZES:
        times = analyze.parse_times(traces_dir / size / "execution_times.ssv")
        if not times:
            continue
        workers = len(times)
        slices = analyze.run_slices(start, times)
        peaks = [max((r[2] for r in ram if s <= r[0] <= e), default=None) for s, e, w, r in slices]
        vals = [p for p in peaks if p is not None]  # MB
        if not vals:
            continue
        labels.append(size)
        lo.append(analyze.gb(min(vals)) / workers)               # GB per worker
        hi.append(analyze.gb(max(vals)) / workers)
        med.append(analyze.gb(statistics.median(vals)) / workers)

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))

    # min -> max range, normalized per worker
    for xi, l, h in zip(x, lo, hi):
        ax.vlines(xi, l, h, color="0.65", lw=2, zorder=2)
        ax.plot([xi - 0.18, xi + 0.18], [l, l], color="0.4", lw=1.5, zorder=2)
        ax.plot([xi - 0.18, xi + 0.18], [h, h], color="0.4", lw=1.5, zorder=2)

    # median (typical run) marker, labeled
    ax.scatter(x, med, color="C0", s=45, zorder=3, label="median (typical run)")
    for xi, m in zip(x, med):
        ax.annotate(f"{fmt(m)} GB", (xi, m), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("size")
    ax.set_ylabel("RAM per run, per worker (GB, log)")
    ax.set_title("Per-run RAM (min → max bar, median marked), normalized per worker")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left")

    # annotate the min and max numbers (small, grey)
    for xi, l, h in zip(x, lo, hi):
        ax.annotate(fmt(l), (xi, l), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7, color="0.4")
        ax.annotate(fmt(h), (xi, h), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7, color="0.4")

    fig.tight_layout()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "ram_per_run.png"
    fig.savefig(png, dpi=150)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
