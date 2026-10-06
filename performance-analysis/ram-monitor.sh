#!/usr/bin/env bash
# Monitor bskysim RAM in real time, one output file per running config.
#
# Every INTERVAL seconds it scans all live `bskysim` processes, groups them by
# the build config that launched them (the `-Dconfig=...` argument of the
# ancestor `zig build`, found by walking the PPid chain), and appends to
#   <outdir>/<config>.txt
# one line per config per sample:
#   <epoch_seconds> cnt=<n> maxrss=<MB>
# where cnt = number of that config's bskysim processes and maxrss = the largest
# VmRSS (resident set, KB->MB) among them. That is the same format analyze.py
# already parses, so each per-config file can be passed straight to
# `--ram-file`. If bskysim was started by hand (no `zig build` ancestor), the
# label falls back to its --outputdir basename.
#
# Usage: ./ram-monitor.sh [-i seconds] [-o outdir]
#   -i  sampling interval in seconds (default 10)
#   -o  output directory (default: <this script's dir>/ram)
set -euo pipefail

INTERVAL=10
OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ram"

usage() { echo "usage: $0 [-i seconds] [-o outdir]" >&2; exit 2; }

while getopts ":i:o:h" opt; do
  case "$opt" in
    i) INTERVAL="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    h|*) usage ;;
  esac
done

mkdir -p "$OUTDIR"

# Config path (from `-Dconfig=...` in an ancestor's cmdline) -> file-safe label.
config_label() {
  local pid="$1" args ppid cfg
  while [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$pid" != "1" ]; do
    if [ -r "/proc/$pid/cmdline" ]; then
      args=$(tr '\0' ' ' < "/proc/$pid/cmdline")
      cfg=$(sed -nE 's/.*-Dconfig=([^ ]+).*/\1/p' <<<"$args")
      if [ -n "$cfg" ]; then
        cfg="${cfg%.json}"
        # configs/build-configs/random-timeline/100K.json -> random-timeline-100K
        local dir base
        dir=$(dirname -- "$cfg")
        base=$(basename -- "$cfg")
        if [ "$dir" = "." ] || [ "$dir" = "/" ]; then
          printf '%s' "$base"
        else
          printf '%s-%s' "$(basename -- "$dir")" "$base"
        fi
        return
      fi
    fi
    ppid=$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
    pid="$ppid"
  done
  # Fallback: bskysim run by hand -> use its --outputdir basename.
  args=$(tr '\0' ' ' < "/proc/$1/cmdline")
  outdir=$(sed -nE 's/.*--outputdir[= ]([^ ]+).*/\1/p' <<<"$args")
  if [ -n "$outdir" ]; then
    printf '%s' "$(basename "$outdir")"
  else
    printf 'bskysim'
  fi
}

echo "monitoring bskysim every ${INTERVAL}s -> $OUTDIR" >&2

while true; do
  ts=$(date +%s)
  declare -A cnt=() max=()
  for pid in $(pgrep -x bskysim || true); do
    rss_kb=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
    [ -n "${rss_kb:-}" ] || continue
    label=$(config_label "$pid")
    rss_mb=$(( rss_kb / 1024 ))
    cnt[$label]=$(( ${cnt[$label]:-0} + 1 ))
    if [ "${max[$label]:-0}" -lt "$rss_mb" ]; then max[$label]=$rss_mb; fi
  done
  for label in "${!cnt[@]}"; do
    printf '%s cnt=%s maxrss=%sMB\n' "$ts" "${cnt[$label]}" "${max[$label]}" >> "$OUTDIR/$label.txt"
  done
  sleep "$INTERVAL"
done
