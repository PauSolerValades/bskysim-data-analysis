#!/usr/bin/env python3
"""Temporal analysis: boredom, warmup attention decay, and new post traction.

Usage:
    python temporal_analysis.py --traces ../../traces/10K-warmup \\
                                --cascades ../../cascades/10K-warmup \\
                                --datasets ../../datasets/10K-warmup \\
                                -o output
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from utils import Config

# Thesis styling (mirrors firehose-analysis AGENTS.md; usetex off — no TeX on this box)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "text.usetex": False,
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(config.warmups)} warmup values: {config.warmups}")
    print(f"Runs per warmup: {config.num_runs}")
    print(f"Traces:   {config.traces_dir}")
    print(f"Cascades: {config.cascades_dir}")
    print(f"Datasets: {config.datasets_dir}")
    print(f"Output:   {config.output_dir}")
    print()

    all_data = load_all_runs(config)

    plot_boredom_timeline(all_data, config)
    plot_warmup_attention_decay(all_data, config)
    print("\nDone.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


CREATE_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Float64,
    "event_id": pl.Int64,
    "gen_id": pl.Int64,
    "user_id": pl.Int64,
    "post_id": pl.Int64,
}

ACTION_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Float64,
    "event_id": pl.Int64,
    "gen_id": pl.Int64,
    "user_id": pl.Int64,
    "post_id": pl.Int64,
    "parent_id": pl.Int64,
    "type": pl.String,
}

SES_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Float64,
    "event_id": pl.Int64,
    "gen_id": pl.Int64,
    "user_id": pl.Int64,
    "type": pl.String,
    "backlog": pl.Int64,
}


def _load_run(config: Config, warmup: float, run: int) -> dict[str, pl.DataFrame]:
    """Load all trace files for a single run — parallel I/O."""
    base = str(config.traces_dir / f"ws{warmup:g}" / str(run))

    def _read(kind: str, schema: dict) -> pl.DataFrame:
        return pl.read_ndjson(base + kind, schema_overrides=schema)

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_actions = ex.submit(_read, "-action_trace.jsonl", ACTION_SCHEMA)
        f_creates = ex.submit(_read, "-create_trace.jsonl", CREATE_SCHEMA)
        f_sessions = ex.submit(_read, "-session_trace.jsonl", SES_SCHEMA)
        return {
            "actions": f_actions.result(),
            "creates": f_creates.result(),
            "sessions": f_sessions.result(),
        }


def load_all_runs(config: Config) -> dict[int, dict[int, dict[str, pl.DataFrame]]]:
    """Load all trace data for every warmup × run combination.

    Returns ``{warmup: {run: {"actions": df, "creates": df, "sessions": df}}}``.
    """
    all_data: dict[int, dict[int, dict[str, pl.DataFrame]]] = {}
    for warmup in config.warmups:
        all_data[warmup] = {}
        for run in range(config.num_runs):
            all_data[warmup][run] = _load_run(config, warmup, run)
        print(f"  warmup={warmup} loaded")
    return all_data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _post_warmup_actions(
    actions: pl.DataFrame, creates: pl.DataFrame, warmup: float
) -> pl.DataFrame:
    """Return post-warmup actions with ``is_warmup`` column via native join.

    Much faster than the old map_elements / dict-lookup approach.
    """
    obs = actions.filter(pl.col("time") >= warmup)
    return obs.join(
        creates.select(
            pl.col("post_id"),
            (pl.col("time") < warmup).alias("is_warmup"),
        ),
        on="post_id",
        how="left",
    ).with_columns(pl.col("is_warmup").fill_null(False))


def _viridis_colors(n: int) -> list:
    """Return *n* evenly-spaced viridis colours."""
    return plt.cm.viridis(np.linspace(0.15, 0.95, n))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Curve computations (shared by per-warmup and overlay plots)
# ---------------------------------------------------------------------------


def _boredom_curve(all_data, config, warmup):
    """(xs, ys) cumulative % of users who ended a session in boredom, vs ticks after warmup."""
    parts = [
        all_data[warmup][run]["sessions"]
        .filter(pl.col("type") == "end_boredom")
        .select("time", "user_id")
        for run in range(config.num_runs)
    ]
    df = pl.concat(parts) if parts else pl.DataFrame(schema={"time": pl.Float64, "user_id": pl.Int64})
    if df.is_empty():
        return None, None
    df = df.filter(pl.col("time") >= warmup).with_columns(
        (((pl.col("time") - warmup) // config.bin_size) * config.bin_size).alias("bin")
    )
    n_users = (
        all_data[warmup][0]["sessions"]
        .filter(pl.col("type") == "start")
        .get_column("user_id")
        .n_unique()
    )
    if n_users == 0:
        return None, None
    first = df.group_by("user_id").agg(pl.col("bin").min().alias("first_bin"))
    max_bin = int(first["first_bin"].max())
    edges = np.arange(0, max_bin + 2 * config.bin_size, config.bin_size)
    counts, _ = np.histogram(first["first_bin"].to_numpy(), bins=edges)
    xs = edges[:-1]
    ys = np.cumsum(counts) / n_users * 100
    return xs, ys


def _attention_curve(all_data, config, warmup, bin_size, min_total):
    """(xs, ys) % actions on warmup posts, binned; (None, None) if empty."""
    parts = []
    for run in range(config.num_runs):
        d = all_data[warmup][run]
        obs = _post_warmup_actions(d["actions"], d["creates"], warmup)
        parts.append(
            obs.select(
                ((pl.col("time") - warmup) // bin_size).cast(pl.Int64).alias("bin"),
                pl.col("is_warmup"),
            )
        )
    if not parts:
        return None, None
    agg = (
        pl.concat(parts).group_by("bin")
        .agg(pl.len().alias("total"), pl.col("is_warmup").sum().alias("wp"))
        .sort("bin")
    )
    max_bin = agg["bin"].max()
    xs = np.arange(0, max_bin + 1) * bin_size
    full = pl.DataFrame({"bin": range(max_bin + 1)}).join(agg, on="bin", how="left").fill_null(0)
    ys = [
        wp / total * 100 if total > min_total else np.nan
        for wp, total in zip(full["wp"].to_list(), full["total"].to_list())
    ]
    return xs, ys


# ---------------------------------------------------------------------------
# Generic folder writer: per-warmup files + combined overlay
# ---------------------------------------------------------------------------


def _write_graph_folder(graph_name, combined_title, warmups, draw_one, draw_all, config):
    folder = config.output_dir / graph_name
    folder.mkdir(parents=True, exist_ok=True)
    colors = _viridis_colors(len(warmups))

    for i, w in enumerate(warmups):
        fig, ax = plt.subplots(figsize=(5, 4))
        draw_one(ax, w, colors[i])
        ax.set_title(f"warmup={w:g}")
        fig.tight_layout()
        fig.savefig(folder / f"{w:g}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_all(ax, warmups, colors)
    ax.set_title(combined_title)
    fig.tight_layout()
    fig.savefig(folder / "combined.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {graph_name}/ ({len(warmups)} warmups + combined.png)")


# ---------------------------------------------------------------------------
# Plot: boredom timeline
# ---------------------------------------------------------------------------


def plot_boredom_timeline(all_data, config):
    def one(ax, w, color):
        xs, ys = _boredom_curve(all_data, config, w)
        if xs is None:
            return
        ax.plot(xs, ys, "-", color=color, linewidth=1.5)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("% users bored")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2)

    def allw(ax, warmups, colors):
        for i, w in enumerate(warmups):
            xs, ys = _boredom_curve(all_data, config, w)
            if xs is None:
                continue
            ax.plot(xs, ys, "-", color=colors[i], label=f"w={w:g}", linewidth=1.5)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("% users bored")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.2)

    _write_graph_folder(
        "boredom_timeline",
        "Cumulative % of users ending a session in boredom over time",
        config.warmups, one, allw, config,
    )


# ---------------------------------------------------------------------------
# Plot: warmup attention decay
# ---------------------------------------------------------------------------


def plot_warmup_attention_decay(all_data, config):
    bin_size = config.bin_size

    def one(ax, w, color):
        if w == 0:
            ax.text(0.5, 0.5, "warmup=0\n(no warmup posts)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            return
        xs, ys = _attention_curve(all_data, config, w, bin_size, min_total=10)
        if xs is None:
            return
        ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.9)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("% actions on warmup posts")
        ax.set_ylim(-5, 105)

    def allw(ax, warmups, colors):
        for i, w in enumerate(warmups):
            if w == 0:
                continue
            xs, ys = _attention_curve(all_data, config, w, bin_size, min_total=50)
            if xs is None:
                continue
            ax.plot(xs, ys, "-", color=colors[i], label=f"w={w:g}", linewidth=1.5)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("% actions on warmup posts")
        ax.legend()
        ax.grid(True, alpha=0.2)

    _write_graph_folder(
        "warmup_attention_decay",
        f"Warmup post attention decay over time (binned every {bin_size} ticks, avg over runs)",
        config.warmups, one, allw, config,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
