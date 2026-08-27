#!/usr/bin/env python3
"""RAM scalability (big-O) of BskySim across dataset sizes.

Fits candidate complexity models to per-run RAM (normalized per worker, using
the MEDIAN per-run value per size) and plots the two power-law fit lines on a
log-log scatter -- mirroring python-utils/plot_time_complexity.py, but for
memory instead of time.

Usage:
    uv run python fit_ram_complexity.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze

TRACES = Path("/home/psoler/des-ctic-dev/steps/final/traces")
RAM_FILE = Path("/tmp/ram-final.txt")
OUTPUT = Path("output")

SIZES = ["10K", "50K", "100K", "500K", "1M"]
N = {"10K": 1e4, "50K": 5e4, "100K": 1e5, "500K": 5e5, "1M": 1e6}
BIG = ["100K", "500K", "1M"]


def per_size_median_gb(ram: list) -> dict[str, float]:
    """Median per-run peak RSS per size, in GB, normalized per worker."""
    out = {}
    for size, start, end in analyze.SIZES:
        times = analyze.parse_times(TRACES / size / "execution_times.ssv")
        if not times:
            continue
        workers = len(times)
        slices = analyze.run_slices(start, times)
        peaks = [max((r[2] for r in ram if s <= r[0] <= e), default=None) for s, e, w, r in slices]
        vals = [analyze.gb(p) / workers for p in peaks if p is not None]
        if vals:
            out[size] = statistics.median(vals)
    return out


def basis(name: str, x: np.ndarray) -> np.ndarray:
    return {"O(n)": x, "O(n log n)": x * np.log(x), "O(n^2)": x ** 2}[name]


def powerlaw(n: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = c * n^p in log-log space; return (p, c)."""
    p, logc = np.polyfit(np.log(n), np.log(y), 1)
    return float(p), float(np.exp(logc))


def fit_models(n: np.ndarray, y: np.ndarray, indent: str = "  ") -> None:
    ss_tot = np.sum((y - y.mean()) ** 2)
    for name in ("O(n)", "O(n log n)", "O(n^2)"):
        b = basis(name, n)
        a = np.sum(y * b) / np.sum(b ** 2)
        r2 = 1 - np.sum((y - a * b) ** 2) / ss_tot
        print(f"{indent}{name:<11} a={a:.4g}  R2={r2:.4f}")

    p, c = powerlaw(n, y)
    print(f"{indent}power law    RAM ~ n^{p:.3f}   (c={c:.4g})")


def main() -> None:
    ram = analyze.parse_ram(RAM_FILE)
    data = per_size_median_gb(ram)
    sizes = [s for s in SIZES if s in data]
    n = np.array([N[s] for s in sizes])
    y = np.array([data[s] for s in sizes])

    print("per-size median per-run RAM (GB, normalized per worker):")
    print(f"{'size':<6} {'n':>10} {'median GB':>11}")
    print("-" * 30)
    for s in sizes:
        print(f"{s:<6} {N[s]:>10.0f} {data[s]:>11.1f}")
    print()

    print(f"Fit on median, all sizes ({', '.join(sizes)}):")
    fit_models(n, y)

    n_big = np.array([N[s] for s in BIG])
    y_big = np.array([data[s] for s in BIG])
    print(f"\nFit on median, big only ({', '.join(BIG)}):")
    fit_models(n_big, y_big)

    # ---- log-log scatter + the two fit lines ----
    p_all, c_all = powerlaw(n, y)
    p_big, c_big = powerlaw(n_big, y_big)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(n, y, s=50, zorder=3)
    for s, xi, yi in zip(sizes, n, y):
        ax.annotate(s, (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    n_fit = np.geomspace(n.min(), n.max(), 100)
    ax.plot(n_fit, c_all * n_fit ** p_all, "--", color="0.3", lw=1.5,
            label=f"all sizes  RAM $\\propto n^{{{p_all:.2f}}}$")
    n_big_fit = np.geomspace(n_big.min(), n_big.max(), 100)
    ax.plot(n_big_fit, c_big * n_big_fit ** p_big, "--", color="crimson", lw=1.5,
            label=f"100K-1M  RAM $\\propto n^{{{p_big:.2f}}}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network size n")
    ax.set_ylabel("Median per-run RAM, per worker (GB)")
    ax.set_title("BskySim RAM scalability per dataset")
    ax.legend()

    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    out_path = OUTPUT / "ram_scalability.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
