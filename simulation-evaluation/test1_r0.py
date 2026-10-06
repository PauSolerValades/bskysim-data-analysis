"""Test 1: empirical reproduction number R0 from the cascade trees.

For every node that reposted, count its children in the cascade tree and take
the mean -> the empirical offspring mean of the branching process. A Galton-
Watson process with this mean is subcritical iff R0 < 1, in which case the
cascade size tail is exponentially bounded (no heavy tail).

Also reports the root "seed" term (mean direct reposts per root), which is
larger than R0: the root's fresh post gets more exposure than a later repost.

Usage:
    uv run python test1_r0.py 10K
    uv run python test1_r0.py 100K
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common import load_reposts, reposter_children

OUT = Path(__file__).resolve().parent / "output"


def summarize_offspring(ch: np.ndarray, size: str) -> dict:
    n = len(ch)
    r0 = float(ch.mean())
    return {
        "size": size,
        "n_reposter_nodes": n,
        "R0_mean_offspring": r0,
        "offspring_median": float(np.median(ch)),
        "offspring_max": int(ch.max()) if n else 0,
        "pct_zero_offspring": float((ch == 0).mean() * 100),
        "pct_one_or_more": float((ch >= 1).mean() * 100),
        "pct_two_or_more": float((ch >= 2).mean() * 100),
    }


def main() -> None:
    size = sys.argv[1] if len(sys.argv) > 1 else "10K"
    out_dir = OUT / size
    out_dir.mkdir(parents=True, exist_ok=True)

    reposts = load_reposts(size)
    n_reposts = reposts.height
    off = reposter_children(reposts)
    ch = off["children"].to_numpy()

    # Decompose reposts into first-level (parent = root) and deeper
    # (parent = a reposter). The reproduction number R0 is the mean
    # offspring of a *reposter* (secondary cases per infected node).
    sizes = reposts.group_by(["run_id", "post_id"]).len()
    n_cascades = sizes.height
    reposter_children_total = int(ch.sum())
    root_children_total = n_reposts - reposter_children_total

    row = summarize_offspring(ch, size)
    row["n_reposts"] = n_reposts
    row["n_cascades"] = n_cascades
    row["mean_cascade_size"] = float(sizes["len"].mean() + 1)
    row["root_children_total"] = root_children_total
    row["mean_seed_root_children"] = root_children_total / n_cascades
    row["reposter_children_total"] = reposter_children_total
    row["pct_deeper_reposts"] = 100.0 * reposter_children_total / n_reposts

    pl.DataFrame([row]).write_csv(out_dir / "r0_summary.csv")

    # Offspring histogram (log-y)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(0, ch.max() + 2)
    ax.hist(ch, bins=bins, color="tab:blue", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("children of a reposting node")
    ax.set_ylabel("count")
    ax.set_title(f"{size}: offspring distribution (mean R0 = {row['R0_mean_offspring']:.3f})")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="critical boundary R0=1")
    ax.axvline(row["R0_mean_offspring"], color="tab:red", lw=2, label="mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "offspring_distribution.svg", dpi=150)
    plt.close(fig)

    print(f"[{size}] R0 (repost reproduction)  = {row['R0_mean_offspring']:.4f}")
    print(f"[{size}] mean seed (root children)  = {row['mean_seed_root_children']:.3f}")
    print(f"[{size}] reposter nodes             = {n_reposts:,} reposts")
    print(f"[{size}] cascades                   = {n_cascades:,}")
    print(f"[{size}] mean cascade size          = {row['mean_cascade_size']:.3f}")
    print(f"[{size}] deeper reposts (depth>=2)   = {row['pct_deeper_reposts']:.1f}%")
    print(f"[{size}] zero-offspring nodes       = {row['pct_zero_offspring']:.1f}%")
    print(f"[{size}] wrote {out_dir}")


if __name__ == "__main__":
    main()
