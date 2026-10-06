"""Shared data-loading helpers for the simulation evaluation tests.

Data locations (hardcoded relative to this repo's parent):
- cascade TSVs: ../des-ctic-dev/steps/final/cascades/<SIZE>.tsv
- topology bins: ../des-ctic-dev/bskysim/data/<N>/size_monotonic.bin

Cascade TSV schema (tab-separated, header):
  run_id  post_id  user_id  parent_id  type  time
where type is one of creation | repost | propagation.
For a `repost` row, parent_id is the user it was seen from (the cascade
parent), so grouping repost rows by (run_id, post_id, parent_id) recovers
each node's number of children in the cascade tree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]  # /home/<user>
CASCADE_DIR = ROOT / "des-ctic-dev" / "steps" / "final" / "cascades"
TOPO_DIR = ROOT / "des-ctic-dev" / "bskysim" / "data"

# Cascade label -> topology directory name
SIZE_TO_TOPO_DIR = {
    "10K": "10000",
    "100K": "100000",
    "500K": "500000",
    "1M": "1000000",
}


def cascade_path(size: str) -> Path:
    return CASCADE_DIR / f"{size}.tsv"


def topo_path(size: str) -> Path:
    return TOPO_DIR / SIZE_TO_TOPO_DIR[size] / "size_monotonic.bin"


def load_reposts(size: str) -> pl.DataFrame:
    """Load repost rows only: run_id, post_id, user_id, parent_id."""
    return (
        pl.scan_csv(
            cascade_path(size),
            separator="\t",
            has_header=True,
            schema_overrides={
                "run_id": pl.UInt32,
                "post_id": pl.UInt32,
                "user_id": pl.UInt32,
                "parent_id": pl.Int64,
                "type": pl.String,
                "time": pl.Float64,
            },
        )
        .filter(pl.col("type") == "repost")
        .select(["run_id", "post_id", "user_id", "parent_id"])
        .collect()
    )


def reposter_children(reposts: pl.DataFrame) -> pl.DataFrame:
    """Per reposting node, its number of children in the cascade tree.

    Returns a DataFrame with columns run_id, post_id, user_id, children.
    Every node that reposted appears exactly once (leaves get children=0).
    """
    children = (
        reposts.group_by(["run_id", "post_id", "parent_id"])
        .len()
        .rename({"len": "children"})
    )
    reposters = reposts.select(["run_id", "post_id", "user_id"]).unique()
    return (
        reposters.join(
            children,
            left_on=["run_id", "post_id", "user_id"],
            right_on=["run_id", "post_id", "parent_id"],
            how="left",
        )
        .with_columns(pl.col("children").fill_null(0).cast(pl.UInt32))
        .select(["run_id", "post_id", "user_id", "children"])
    )


def parse_follower_counts(size: str) -> np.ndarray:
    """In-degree (number of followers) per node id, from the topology bin.

    Bin layout (all little-endian u32):
      num_nodes, user_ids[num_nodes], num_edges, edges[num_edges*2]
    where each edge is (actor_id, subject_id) = (follower, followed).
    """
    a = np.fromfile(topo_path(size), dtype="<u4")
    n = int(a[0])
    num_edges = int(a[1 + n])
    edges = a[2 + n : 2 + n + 2 * num_edges]
    subjects = edges[1::2]  # followed nodes
    return np.bincount(subjects, minlength=n).astype(np.int64)
