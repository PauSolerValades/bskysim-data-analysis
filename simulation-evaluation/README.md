# Simulation evaluation

Empirical tests for the hypotheses behind the missing heavy tail:

1. **`test1_r0.py`** — the process is subcritical. Computes the empirical
   reproduction number R0 (mean offspring per reposting node) from the cascade
   trees, plus the root "seed" term. R0 < 1 => exponentially bounded tail, so
   no heavy tail is possible under the homogeneous IC model.

The "content would restore the tail" argument is analytic, not empirical; see
the *Branching-Process Derivation of the Missing Tail* appendix in the report
(`src/annex/branching-math.typ`). The "missing width" argument is likewise a
hypothesis section in the report (`src/8-results.typ`, `@sec-missing-width`),
not a measurement.

## Data (read-only)

- cascade TSVs: `../des-ctic-dev/steps/final/cascades/<SIZE>.tsv`
- topology bins: `../des-ctic-dev/bskysim/data/<N>/size_monotonic.bin`

## Usage

```bash
uv run python test1_r0.py 10K
```

Output goes to `output/<SIZE>/`.
