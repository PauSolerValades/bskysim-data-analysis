# Stable Regime

> **Question:** does the simulation's session layer reach a steady state, and how
> long does it take to get there?

> **Answer:** yes — the online-user fraction converges to the same equilibrium
> (~2.1–2.4%) regardless of the initial condition, in **~26–32K ticks** after
> warm-up for networks ≥ 100K users. The minimum horizon is therefore
> **warm-up 2,000 + transient 30,000 + measurement window**.

---

## 1. What this experiment measures

The simulation has two layers with very different timescales:

1. **Content / timelines** — calibrated by the `which-warmup` experiment
   (warm-up length = 2,000 ticks, see `../which-warmup/`).
2. **Sessions / online presence** — how many users are online at any instant,
   governed by the fitted session-duration and inter-session-gap distributions.

This experiment characterises layer 2. It asks: if the system starts from some
arbitrary online fraction, does the online population settle to a stable value,
and after how long?

The quantity measured is **% online users over time**, for three different
initial conditions set by `offline_startup_ratio`:

| tag | `offline_startup_ratio` | meaning |
|---|---|---|
| `r0` | 0.0 | everyone online at warm-up end |
| `r50` | 0.5 | half online |
| `r100` | 1.0 | everyone offline |

If all three collapse onto the same plateau, the equilibrium is a property of the
*system* (the session distributions), not of the initial state — which is exactly
what a steady-state claim needs.

## 2. Method

- **Networks:** 10K, 100K, 500K, 1M (monotonic topologies).
- **Warm-up:** 2,000 ticks (locked from the `which-warmup` experiment).
- **Duration:** 60,000 ticks (long enough to observe the transient *and* a plateau).
- **Replications:** 3 per (size, ratio); results are medians over the 3 runs.
- **Detection:** the online fraction is bin-averaged (60 s bins) and smoothed with
  a 300 s rolling mean. Stability is the earliest time after which the rolling mean
  stays within **±10% of its final value** for the rest of the run.

Reproduce with:

```bash
uv run --with numpy python stability_analysis.py
```

The analysis reads the binary session traces (`0-session_trace.bin`, 40-byte
records) with numpy, so the whole 4×3×3 matrix is processed in seconds.

## 3. Results

### Equilibrium online fraction (final, median over 3 runs)

| size | r0 (0% offline) | r50 (50%) | r100 (100%) |
|---|---|---|---|
| 10K  | 1.90 % | 1.85 % | 1.71 % |
| 100K | 2.20 % | 2.24 % | 2.21 % |
| 500K | 2.40 % | 2.37 % | 2.39 % |
| 1M   | 2.09 % | 2.08 % | 2.10 % |

### Stabilization time (anchored ±10% of final)

| size | stable at (post-warm-up) |
|---|---|
| 10K  | ~58K s — noise-limited, see §4.3 |
| 100K | ~30–32K s |
| 500K | ~26–29K s |
| 1M   | ~28–31K s |

Plots: `output/w2K-d60K/initial_conditions_{10K,100K,500K,1M}.png`.

## 4. Findings

### 4.1 Initial-condition independence holds at every size

Within each size the three initial conditions land on the same equilibrium
(within ~0.2 pp; tightest at 1M: 2.08–2.10 %). The equilibrium is a property of
the system, not of how it starts. This is the central steady-state claim, now
backed by a 4×3 matrix rather than a single curve.

### 4.2 The size effect is non-monotonic — and that is topology, not noise

Online % climbs 1.8 % (10K) → 2.2 % (100K) → 2.4 % (500K) and then *drops* to
2.1 % (1M). 1M is lower at every time point (2.27 % vs 2.62 % at t=30K), so this
is not a drift artifact: the `1M_monotonic` topology is sparser than
`500K_monotonic`, which produces sparser timelines → more boredom-shortened
sessions → a lower equilibrium online fraction. The monotonic datasets are not a
pure "same shape, scaled" family; equilibrium online % is topology-dependent.

### 4.3 Convergence by ~30K ticks, with two caveats

- **10K's ~58K is not a slower equilibrium.** At ~180 online users, Poisson noise
  is ~±7 %, i.e. comparable to the ±10 % detection band, so noise alone pushes the
  "last violation" to the end of the window. The honest 10K statement is "settles
  early, but cannot be pinned tighter than ±10 % at this N".
- **500K and 1M keep drifting ~0.1 pp per 15K ticks at the end** (e.g. 2.49→2.38
  for 500K over 45–60K). This is the heavy-tail users filtering out permanently:
  the approach to equilibrium is asymptotic, not exact. For practical purposes
  (cascade analysis, engagement metrics) the drift is negligible.

## 5. Minimum horizon — the recommendation

Combining `which-warmup` (warm-up = 2,000) with this experiment (transient ≈
30,000):

| phase | ticks |
|---|---|
| warm-up (fill timelines) | 2,000 |
| transient (online fraction settles) | ~26–32K |
| **minimum to *reach* steady state** | **~34,000** |
| recommended buffer before measuring | +8–10K |

**Recommendation: horizon = 42,000 ticks** (warm-up 2,000 + duration 40,000).
That is safely past the ~30K convergence for every size ≥ 100K and leaves ~10K of
clean plateau to collect measurements. Add your measurement window on top if the
experiment needs to observe long-lived cascades:

```
horizon = 2,000 (warm-up) + 30,000 (transient) + measurement_window
```

For the 10K network specifically, the equilibrium is reached early but is
noise-limited at the ±10 % level — if 10K must be reported with a tight stability
estimate, use more replications rather than a longer horizon.

## 6. Where this sits

- `../which-warmup/` — how long warm-up must be (locked: 2,000 ticks).
- this experiment — when the session layer stabilises (≈ 30K ticks) and at what
  online fraction (~2.1–2.4 %, topology-dependent).
- Together they fix the two free temporal parameters of the simulation, so the
  remaining experiments (cascades, datasets) can run in a well-defined steady
  state.
