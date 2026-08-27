#!/usr/bin/env python3
"""Analyze bskysim RAM usage from /tmp/ram-final.txt for the five *final* runs.

Data source
-----------
`/tmp/ram-final.txt` is produced by `/tmp/ram-monitor.sh`, which loops every
~10 s and appends one line per sample:

    <epoch_seconds> cnt=<n> maxrss=<MB>

where `cnt` = number of live `bskysim` processes and `maxrss` = the largest
`VmRSS` (resident set) among them, KB -> MB. So each line is a 10 s snapshot
of the biggest `bskysim` process, *not* a per-run measurement.

Each size (10K / 50K / 100K / 500K / 1M) is run as ONE `bskysim` process
(`zig build` launches `bskysim -w<workers> -n100 ...`): all 100 runs happen
inside that process.  Therefore:

  * "peak RAM per process"  = max `maxrss` over that size's time window.
  * "RAM per run"           = peak `maxrss` during each individual run's time
                              slice, reconstructed from
                              `steps/final/traces/<size>/execution_times.ssv`
                              (which records `worker run_idx duration_ms`).

Caveats (read before trusting the numbers)
------------------------------------------
  * 10 s sampling of instantaneous RSS: the true peak between two samples is
    missed, so these are lower-bound approximations.
  * With `workers > 1` (10K/50K ran 16, 100K ran 12, 500K ran 2), several runs
    overlap and share one process, so a "per-run" slice still contains
    concurrent + accumulated memory -- it is an upper bound on an isolated run.
  * RSS accumulates across runs (state is reused, not fully released), so the
    per-run peak grows run-over-run for the large single-worker sizes.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

# --------------------------------------------------------------------------- #
# Final-run time windows.
#
# Provenance: `start` is the mtime of `steps/final/traces/<size>/used_config.json`
# (the first file bskysim writes), `end` is the mtime of the newest `*.bin` trace
# file (the last thing bskysim writes).  Timestamps are UTC epoch seconds.
# --------------------------------------------------------------------------- #
SIZES = [
    ("10K",  1787604396, 1787604442),
    ("50K",  1787788350, 1787788806),
    ("100K", 1787604621, 1787606233),
    ("500K", 1787608123, 1787649700),
    ("1M",   1787649994, 1787774759),
]

# Slack added only to the END of each window. `start` (used_config.json mtime) is
# written by the process itself, so the process is already running by then -- padding
# the start would capture the tail of the PREVIOUS process's RSS release (e.g. 10K
# follows a 668 GB run that is still freeing memory ~40 s before 10K starts). The
# peak RSS sample can land a few seconds AFTER the last trace file is written, hence
# a small end pad. Adjacent sizes are >= 2.5 min apart, so 30 s is safe.
END_PAD = 30


def parse_ram(path: Path) -> list[tuple[int, int, int]]:
    """Return [(epoch, cnt, maxrss_MB), ...]."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        ts = int(parts[0])
        cnt = int(parts[1].split("=")[1])
        rss = int(parts[2].split("=")[1][:-2])  # strip trailing "MB"
        rows.append((ts, cnt, rss))
    return rows


def parse_times(path: Path) -> dict[int, list[tuple[int, int]]] | None:
    """Parse execution_times.ssv -> {worker_id: [(run_idx, duration_ms), ...]}.

    Returns None if the file is missing/unparseable (then per-run is skipped).
    """
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith("batch"):
        return None
    per_worker: dict[int, list[tuple[int, int]]] = {}
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln:
            continue
        w, r, ms = ln.split()
        per_worker.setdefault(int(w), []).append((int(r), int(ms)))
    for w in per_worker:
        per_worker[w].sort()  # order each worker's runs by run_idx
    return per_worker


def run_slices(window_start: float, per_worker: dict[int, list[tuple[int, int]]]):
    """Reconstruct each run's [start, end] epoch seconds.

    All workers start at ~window_start concurrently; each worker runs its own
    runs back-to-back, so run i of worker w starts after the cumulative duration
    of its previous runs.
    """
    slices = []
    for w, runs in per_worker.items():
        cum = 0.0
        for run_idx, ms in runs:
            start = window_start + cum / 1000.0
            end = window_start + (cum + ms) / 1000.0
            slices.append((start, end, w, run_idx))
            cum += ms
    return slices


def gb(mb: float) -> float:
    return mb / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze final-run RAM usage.")
    ap.add_argument("--ram-file", default="/tmp/ram-final.txt")
    ap.add_argument("--traces-dir", default="/home/psoler/des-ctic-dev/steps/final/traces")
    ap.add_argument("--out-csv", default=None, help="optional dir to dump per-run CSV")
    args = ap.parse_args()

    ram = parse_ram(Path(args.ram_file))
    if not ram:
        raise SystemExit(f"no samples parsed from {args.ram_file}")
    traces_dir = Path(args.traces_dir)

    # samples grouped by size
    table1 = []  # (size, workers, runs, peak_GB)
    per_run = {}  # size -> [peak_GB per run, ...]

    for size, start, end in SIZES:
        seg = [r for r in ram if start <= r[0] <= end + END_PAD]
        peak_mb = max(r[2] for r in seg)

        times = parse_times(traces_dir / size / "execution_times.ssv")
        workers = len(times) if times else 0
        runs = sum(len(v) for v in times.values()) if times else 0
        table1.append((size, workers, runs, peak_mb))

        if times is None:
            per_run[size] = None
            continue

        slices = run_slices(start, times)
        peaks = []
        for s, e, w, r in slices:
            peak = max((rr[2] for rr in ram if s <= rr[0] <= e), default=None)
            peaks.append(peak)  # None -> run shorter than the sampling gap
        per_run[size] = peaks

    # ---- Table 1: peak RAM per process ----
    print("=" * 74)
    print("Table 1 - Peak RAM per process (one bskysim process per size)")
    print("=" * 74)
    print(f"{'size':>5} {'workers':>8} {'runs':>5} {'peak GB':>10}")
    print("-" * 74)
    for size, workers, runs, peak_mb in table1:
        print(f"{size:>5} {workers:>8} {runs:>5} {gb(peak_mb):>10.2f}")
    print()
    print("  workers/runs are read from execution_times.ssv (what actually ran),")
    print("  not from configs/build-configs/final/*.json (edited after the runs).")
    print()

    # ---- Table 2: RAM per run (normalized per worker) ----
    print("=" * 80)
    print("Table 2 - RAM per run (normalized per worker)")
    print("=" * 80)
    print(
        f"{'size':>5} {'workers':>8} {'runs':>5} {'min':>9} {'median':>9} {'max':>9}  (GB)"
    )
    print("-" * 80)
    for size, workers, runs, _ in table1:
        peaks = per_run.get(size)
        if not peaks or all(p is None for p in peaks):
            print(f"{size:>5} {workers:>8} {runs:>5}   (runs shorter than 10s sampling)")
            continue
        vals = [gb(p) / workers for p in peaks if p is not None]
        n_ok = len(vals)
        print(
            f"{size:>5} {workers:>8} {runs:>5} "
            f"{min(vals):>9.2f} {statistics.median(vals):>9.2f} {max(vals):>9.2f}"
            f"   ({n_ok}/{runs} runs resolved)"
        )
    print()
    print("  Each value = peak RSS during a run's time slice, divided by workers.")
    print("  min/median/max are the first / typical / last runs (RSS accumulates).")
    print("  For workers > 1 the slices overlap, so this is an upper bound on an")
    print("  isolated single run (shared topology isn't separated out).")

    if args.out_csv:
        out = Path(args.out_csv)
        out.mkdir(parents=True, exist_ok=True)
        for size, _, _, _ in table1:
            peaks = per_run.get(size)
            if not peaks:
                continue
            with (out / f"{size}_per_run.csv").open("w") as f:
                f.write("run_idx,peak_GB\n")
                for i, p in enumerate(peaks):
                    f.write(f"{i},{gb(p):.2f}\n" if p is not None else f"{i},\n")
        print(f"\nper-run CSVs written to {out}")


if __name__ == "__main__":
    main()
