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
    plot_new_post_traction(all_data, config)
    plot_combined_summary(all_data, config)
    plot_first_session_backlog(all_data, config)
    plot_sessions_actions(config)
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
    base = str(config.traces_dir / f"{warmup:g}-ticks" / str(run))

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
    """(centers, rate) of boredom-ended sessions after warmup; (None, None) if empty."""
    parts = [
        all_data[warmup][run]["sessions"]
        .filter(pl.col("type") == "end_boredom")
        .select("time")
        for run in range(config.num_runs)
    ]
    all_times_df = pl.concat(parts) if parts else pl.DataFrame(schema={"time": pl.Float64})
    if all_times_df.is_empty():
        return None, None
    arr = (
        all_times_df.filter(
            (pl.col("time") >= warmup) & (pl.col("time") <= warmup + 5000)
        )
        .select((pl.col("time") - warmup).alias("rel"))
        .to_series()
        .to_numpy()
    )
    if len(arr) == 0:
        return None, None
    bins = np.logspace(np.log10(1), np.log10(5100), 50)
    counts, _ = np.histogram(arr, bins=bins)
    rate = counts / np.diff(bins) / config.num_runs
    return (bins[:-1] + bins[1:]) / 2, rate


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


def _traction_curve(all_data, config, warmup, bin_size):
    """(xs, ys) impressions per new post, binned; (None, None) if empty."""
    imp_parts, np_parts = [], []
    for run in range(config.num_runs):
        d = all_data[warmup][run]
        actions, creates = d["actions"], d["creates"]
        obs = _post_warmup_actions(actions, creates, warmup)
        imp_parts.append(
            obs.filter(~pl.col("is_warmup"))
            .select(((pl.col("time") - warmup) // bin_size).cast(pl.Int64).alias("bin"))
        )
        np_parts.append(
            creates.filter(pl.col("time") >= warmup)
            .select(((pl.col("time") - warmup) // bin_size).cast(pl.Int64).alias("bin"))
        )
    if not imp_parts:
        return None, None
    imp_agg = pl.concat(imp_parts).group_by("bin").len(name="impressions").sort("bin")
    np_agg = pl.concat(np_parts).group_by("bin").len(name="new_posts").sort("bin")
    max_bin = max(
        imp_agg["bin"].max() if not imp_agg.is_empty() else 0,
        np_agg["bin"].max() if not np_agg.is_empty() else 0,
    )
    full_imp = pl.DataFrame({"bin": range(max_bin + 1)}).join(imp_agg, on="bin", how="left").fill_null(0)
    full_np = pl.DataFrame({"bin": range(max_bin + 1)}).join(np_agg, on="bin", how="left").fill_null(0)
    xs = np.arange(0, max_bin + 1) * bin_size
    ys = [
        (imp / config.num_runs) / max(np_count / config.num_runs, 1) if np_count > 0 else np.nan
        for imp, np_count in zip(full_imp["impressions"].to_list(), full_np["new_posts"].to_list())
    ]
    return xs, ys


def _first_session_backlogs(all_data, config, warmup):
    backlogs: list[int] = []
    for run in range(config.num_runs):
        sessions = all_data[warmup][run]["sessions"]
        ends = sessions.filter(pl.col("type").is_in(["end", "end_boredom"]))
        backlogs.extend(ends.unique(subset=["user_id"], keep="first")["backlog"].to_list())
    return backlogs


def _sessions_actions(config, warmup):
    path = config.datasets_dir / f"warmup-{warmup:g}" / "sessions.parquet"
    if not path.exists():
        print(f"  [skip] {path} not found — skipping sessions_actions for warmup={warmup}")
        return []
    return pl.read_parquet(str(path))["total_actions"].to_list()


# ---------------------------------------------------------------------------
# Generic folder writer: per-warmup files + 2x3 combined (warmups + overlay)
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

    n = len(warmups)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    flat = axes.flatten()
    for i, w in enumerate(warmups):
        draw_one(flat[i], w, colors[i])
        flat[i].set_title(f"warmup={w:g}")
    draw_all(flat[n], warmups, colors)
    flat[n].set_title("all warmups")
    for ax in flat[n + 1 :]:
        ax.set_visible(False)
    fig.suptitle(combined_title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(folder / "combined.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {graph_name}/ ({n} warmups + combined.png)")


# ---------------------------------------------------------------------------
# Plot: boredom timeline
# ---------------------------------------------------------------------------


def plot_boredom_timeline(all_data, config):
    def one(ax, w, color):
        centers, rate = _boredom_curve(all_data, config, w)
        if centers is None:
            return
        ax.loglog(centers, rate, ".-", color=color, linewidth=1, markersize=2)
        ax.axvline(x=1, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("Boredom ends / tick / run")
        ax.grid(True, alpha=0.2, which="both")

    def allw(ax, warmups, colors):
        for i, w in enumerate(warmups):
            centers, rate = _boredom_curve(all_data, config, w)
            if centers is None:
                continue
            ax.loglog(centers, rate, ".-", color=colors[i], label=f"w={w:g}", linewidth=1.5, markersize=2)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("Boredom ends / tick / run")
        ax.legend()
        ax.grid(True, alpha=0.2, which="both")

    _write_graph_folder(
        "boredom_timeline",
        "Boredom-ended sessions over time (log-log, per-tick rate, avg over runs)",
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
# Plot: new post traction
# ---------------------------------------------------------------------------


def plot_new_post_traction(all_data, config):
    bin_size = config.bin_size

    def one(ax, w, color):
        xs, ys = _traction_curve(all_data, config, w, bin_size)
        if xs is None:
            return
        ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.9)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("Impressions per new post")

    def allw(ax, warmups, colors):
        for i, w in enumerate(warmups):
            xs, ys = _traction_curve(all_data, config, w, bin_size)
            if xs is None:
                continue
            ax.plot(xs, ys, "-", color=colors[i], label=f"w={w:g}", linewidth=1.5)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("Impressions per new post")
        ax.legend()
        ax.grid(True, alpha=0.2)

    _write_graph_folder(
        "new_post_traction",
        f"New post traction over time (impressions/post, binned every {bin_size} ticks, avg over runs)",
        config.warmups, one, allw, config,
    )


# ---------------------------------------------------------------------------
# Plot: first-session backlog
# ---------------------------------------------------------------------------


def plot_first_session_backlog(all_data, config):
    def one(ax, w, color):
        bp = ax.boxplot(
            [_first_session_backlogs(all_data, config, w)],
            patch_artist=True, showfliers=False, widths=0.6,
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        ax.set_xticks([1], [f"{w:g}"])
        ax.set_xlabel("Warmup time")
        ax.set_ylabel("Backlog posts deleted on session end")
        ax.grid(True, alpha=0.2, axis="y")

    def allw(ax, warmups, colors):
        data = [_first_session_backlogs(all_data, config, w) for w in warmups]
        bp = ax.boxplot(
            data, tick_labels=[f"{w:g}" for w in warmups],
            patch_artist=True, showfliers=False, widths=0.6,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Warmup time")
        ax.set_ylabel("Backlog posts deleted on session end")
        ax.grid(True, alpha=0.2, axis="y")

    _write_graph_folder(
        "first_session_backlog",
        "Posts lost when first session ends (discarded from active timeline, all users)",
        config.warmups, one, allw, config,
    )


# ---------------------------------------------------------------------------
# Plot: sessions → actions (uses datasets, not traces)
# ---------------------------------------------------------------------------


def plot_sessions_actions(config):
    def one(ax, w, color):
        data = _sessions_actions(config, w)
        bp = ax.boxplot([data], patch_artist=True, showfliers=False, widths=0.6)
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        ax.set_xticks([1], [f"{w:g}"])
        ax.set_xlabel("Warmup time")
        ax.set_ylabel("Actions per session")
        ax.grid(True, alpha=0.2, axis="y")

    def allw(ax, warmups, colors):
        data, labels = [], []
        for w in warmups:
            data.append(_sessions_actions(config, w))
            labels.append(f"{w:g}")
        bp = ax.boxplot(
            data, tick_labels=labels,
            patch_artist=True, showfliers=False, widths=0.6,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Warmup time")
        ax.set_ylabel("Actions per session")
        ax.grid(True, alpha=0.2, axis="y")

    _write_graph_folder(
        "sessions_actions",
        "Total actions per session (all users, all runs)",
        config.warmups, one, allw, config,
    )


# ---------------------------------------------------------------------------
# Plot: combined summary — 3 separated overlay images + the 1x3 combined
# ---------------------------------------------------------------------------


def plot_combined_summary(all_data, config):
    folder = config.output_dir / "combined_summary"
    folder.mkdir(parents=True, exist_ok=True)
    warmups = config.warmups
    colors = _viridis_colors(len(warmups))

    def boredom_panel(ax, legend=True):
        for i, w in enumerate(warmups):
            centers, rate = _boredom_curve(all_data, config, w)
            if centers is None:
                continue
            ax.loglog(centers, rate, ".-", color=colors[i], label=f"w={w:g}", linewidth=1.5, markersize=2)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("Boredom ends / tick / run")
        ax.set_title("Boredom rate (log-log)")
        if legend:
            ax.legend()
        ax.grid(True, alpha=0.2, which="both")

    def attention_panel(ax, legend=True):
        for i, w in enumerate(warmups):
            if w == 0:
                continue
            xs, ys = _attention_curve(all_data, config, w, 100, min_total=50)
            if xs is None:
                continue
            ax.plot(xs, ys, "-", color=colors[i], label=f"w={w:g}", linewidth=1.5)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("% actions on warmup posts")
        ax.set_title("Warmup attention decay")
        if legend:
            ax.legend()
        ax.grid(True, alpha=0.2)

    def traction_panel(ax, legend=True):
        for i, w in enumerate(warmups):
            xs, ys = _traction_curve(all_data, config, w, 100)
            if xs is None:
                continue
            ax.plot(xs, ys, "-", color=colors[i], label=f"w={w:g}", linewidth=1.5)
        ax.set_xlabel("Ticks after warmup")
        ax.set_ylabel("Impressions per new post")
        ax.set_title("New post traction")
        if legend:
            ax.legend()
        ax.grid(True, alpha=0.2)

    # separated images (the three panels, each with all warmups overlaid)
    for name, panel in [("boredom", boredom_panel), ("attention_decay", attention_panel), ("new_post_traction", traction_panel)]:
        fig, ax = plt.subplots(figsize=(7, 5))
        panel(ax)
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=150)
        plt.close(fig)

    # same combined as before: 1x3
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    boredom_panel(axes[0])
    attention_panel(axes[1])
    traction_panel(axes[2])
    fig.suptitle(
        "Warmup dynamics: boredom → warmup drain → new content takeoff",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(folder / "combined_summary.png", dpi=150)
    plt.close(fig)
    print("  Saved combined_summary/ (boredom.png, attention_decay.png, new_post_traction.png, combined_summary.png)")


if __name__ == "__main__":
    main()
