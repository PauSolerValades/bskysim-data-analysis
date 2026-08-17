#!/usr/bin/env bash
# Run the stable-regime analysis over the new stability traces (4 sizes × 6 durations).
# One plot set per (size, duration) — run_all_experiments.py can't do this layout
# (flat output/ names would collide across sizes).
set -euo pipefail
ROOT="/home/psoler/des-ctic-dev/steps/stability/traces"
cd "$(dirname "$0")"
for size in 10K 100K 500K 1M; do
    for d in d1000 d2000 d3000 d5000 d7000 d9000; do
        echo "[$size/$d] $(date)"
        uv run python main.py "$ROOT/$size/$d" --output "output/$size/$d"
    done
done
echo "=== ALL DONE ==="
