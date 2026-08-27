# RAM Performance Analysis

Peak-RAM analysis of the five final `bskysim` simulation runs
(10K, 50K, 100K, 500K, 1M users), derived from the RAM monitor log
`/tmp/ram-final.txt`.

## Data source

`/tmp/ram-final.txt` is produced by `/tmp/ram-monitor.sh`, a loop that runs
every ~10 s and appends one line per sample:

```
<epoch_seconds> cnt=<n> maxrss=<MB>
```

- `cnt` — number of live `bskysim` processes at that instant (`pgrep -x bskysim`).
- `maxrss` — the largest `VmRSS` (resident set, KB→MB) across those processes.

Each line is therefore a **10 s snapshot of the biggest `bskysim` process**,
not a per-run measurement.

## How the results are derived

Each size is run as **one** `bskysim` process (`zig build` launches
`bskysim -w<workers> -n100 …`), so all runs happen inside a single process.

1. **Peak RAM per process** — the max `maxrss` over that size's time window.
2. **RAM per run** — the peak `maxrss` during each individual run's time slice,
   reconstructed from `steps/final/traces/<size>/execution_times.ssv`
   (records `worker run_idx duration_ms`).

The time window of each size is read from the trace files the process itself
writes: `used_config.json` mtime (first write, ≈ start) and the newest `*.bin`
trace mtime (last write, ≈ end).

## Results

### Table 1 — Peak RAM per process

| size | workers | runs | peak (GB) |
|------|---------|------|-----------|
| 10K  | 16      | 100  | 33.93     |
| 50K  | 16      | 100  | 424.85    |
| 100K | 12      | 100  | 1070.92   |
| 500K | 2       | 100  | 940.81    |
| 1M   | 1       | 98   | 638.03    |

`workers`/`runs` are read from `execution_times.ssv` (what actually ran), not
from `configs/build-configs/final/*.json` (edited after the runs).

### Table 2 — RAM per run

| size | est. single run (GB) | per-run min → max (GB) |
|------|----------------------|------------------------|
| 10K  | 2.1                  | 23.3 → 30.2            |
| 50K  | 26.6                 | 224.7 → 424.9          |
| 100K | 89.2                 | 651.0 → 1070.0         |
| 500K | 470.4                | 468.4 → 932.5          |
| 1M   | 638.0                | 311.8 → 633.3          |

- **est. single run** = peak-per-process ÷ workers, a rough estimate of one
  isolated run (assumes per-run memory is additive across workers).
- **per-run min → max** = peak RSS during each run's time slice. `min` is the
  first run, `max` the last run.

The process peaks look non-monotonic (100K > 500K > 1M) because they include
worker concurrency (12 vs 2 vs 1 workers). Normalized per worker, the
single-run footprint is monotonic: 2.1 → 638 GB.

## Caveats

- 10 s sampling of instantaneous RSS — the true peak between samples is
  missed, so all numbers are lower-bound approximations.
- With `workers > 1`, runs overlap and share one process, so a per-run slice
  still includes concurrent + accumulated memory (upper bound on an isolated
  run).
- RSS **accumulates** across runs (state is reused, not freed), so per-run
  peak grows run-over-run.

## Usage

```bash
uv run python analyze.py
# options:
#   --ram-file    path to the monitor log (default /tmp/ram-final.txt)
#   --traces-dir  path to the final trace dirs (default …/des-ctic-dev/steps/final/traces)
#   --out-csv     optional dir to dump per-run CSVs
```
