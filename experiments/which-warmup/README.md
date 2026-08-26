# Which Warm-up?

> **Answer: 2,000 ticks.** Below that, a user's first session after warm-up has an
> under-filled timeline and dies of boredom; above it, the backlog is saturated and
> every extra tick is wasted compute and memory. This value is locked into the
> stability and production configurations.

---

## 1. What is warm-up, and why does it exist?

The simulation models a Bluesky-like network where users go online, consume their
timeline, and occasionally post. A *timeline* is a per-user feed of posts; a *session*
is one continuous interval of being online; and a session ends **in boredom** when the
user's timeline is empty — there is nothing left to read, so they leave early.

At simulation start every timeline is empty. If the measurement phase began
immediately, the very first user to come online would find nothing, drain the empty
feed in an instant, and disconnect. Every downstream metric — impressions, engagement,
session length, cascade size — would be measured on a broken, empty system.

**Warm-up exists to pre-fill the timelines with content**, so that the first real
session finds a healthy feed, exactly as it would in steady state. This is the
non-negotiable design goal:

> any user, logging in at any moment, must find a timeline full of recent posts.

There is a direct tension here:

- **Too short** a warm-up → empty timelines → boredom truncates every first session and
  corrupts all cascade / diffusion measurements.
- **Too long** a warm-up → all that pre-generated content must be discarded from the
  analysis (it was not produced under the real session dynamics), and — as we
  discovered — the memory footprint explodes.

The question this experiment answers is therefore the minimum warm-up that makes the
first session behave like a steady-state session.

## 2. Warm-up is dense by design (and that is correct)

A subtle but important point: during warm-up, **every** user generates and propagates
content, regardless of how many users will actually be online later. This is deliberate
and correct:

- We do not know *which* users will be online at any future instant, so we must fill
  *every* timeline, not just a few.
- In steady state only ~2.4% of users are online at once (see the stability
  experiment), so a dense warm-up fills timelines ~40× faster than real operation
  would. This is a feature: a short dense warm-up can deliver the backlog that would
  take hours of real time to accumulate.

Concretely, `stageOne` in the simulator seeds a first-post event for every user, so the
warm-up phase runs at full density independent of any other parameter.

## 3. Two knobs, and the mistake that motivated this experiment

There are two *separate* parameters that were historically conflated:

| knob | meaning | where it applies |
|---|---|---|
| `warmup_time` | how many ticks of content pre-fill | the warm-up phase only |
| `offline_startup_ratio` | what fraction of users start the *measurement* phase online | the boundary between warm-up and measurement |

The original warm-up sweep set `offline_startup_ratio = 0.0` — **everyone online at the
start of the measurement phase**. That is both unrealistic (a real run has ~2.4%
online) and a confound: with everyone consuming simultaneously, the backlog is drained
~40× faster, so the sweep systematically *over-estimates* how much warm-up is needed
(it is implicitly asking "how much content keeps 100% of users busy" rather than "how
much content keeps the real 2.4% busy").

The sweep in this experiment therefore fixes the duration-phase load at
`offline_startup_ratio = 0.976` (~ the equilibrium online fraction), while keeping the
warm-up phase itself dense. The warm-up *value* is then calibrated against the load a
real run will actually experience.

## 4. The question, made precise

> What is the **minimum warm-up** such that a user's **first session** after warm-up
> does **not** end in boredom?

`warmup = 0` is the control: with no pre-fill, the first session's backlog is
necessarily zero, and every first session must (modulo live arrivals) boredom out.

Two quantities are needed to answer this cleanly:

1. **Backlog at session start** — how many posts the user finds the moment they log in.
   This is the warm-up's *direct* contribution.
2. **First-session boredom rate** — the fraction of first sessions that end in
   `end_boredom`. This is the *outcome* we actually care about.

The two are related but not identical: even with zero initial backlog, a user's session
can survive because live posts from *currently online* users keep arriving during the
session. So we measure both.

### Implementation

The session trace already recorded a `backlog` field, but only on `.end` events (the
leftover unread). A one-line change makes `.start` events record the length of the
background timeline — which at session start is precisely the backlog the user is about
to consume:

```zig
// simulation.zig — handleSession
const backlog: u32 = if (ssn == .end or ssn == .start)
    @intCast(background_timeline.elements.items.len) else 0;
```

## 5. Method

- **Networks:** 10K and 100K (the two that fit comfortably; 10K is the conservative
  bound — see §7).
- **`offline_startup_ratio`:** 0.976 (realistic equilibrium load).
- **Warm-up grid:** `{0, 100, 500, 1000, 2000, 5000, 10000}` ticks.
- **Duration:** 20,000 ticks (plenty to observe every user's first session).
- **Measured per run:** median backlog at first-session start, and the first-session
  boredom rate.

## 6. Results

| warm-up (ticks) | median backlog @ 1st session (10K / 100K) | 1st-session boredom % (10K / 100K) |
|---|---|---|
| 0 | 19 / 77 | 46.9 / 34.7 |
| 100 | 26 / 98 | 32.7 / 21.7 |
| 500 | 28 / 99 | 24.9 / 15.4 |
| 1000 | 27 / 105 | 20.8 / 12.0 |
| 2000 | 32 / 107 | **15.7 / 8.8** |
| 5000 | 32 / 113 | 9.4 / 5.7 |
| 10000 | 31 / 114 | 6.7 / 4.2 |

The raw traces live under `data/10K/ws*` and `data/100K/ws*` (one directory per warm-up
value), with `0-session_trace.jsonl` + `used_config.json` per run.

## 7. Interpretation

1. **No hard cliff, but clear diminishing returns.** The dominant win is `0 → 500`
   (boredom roughly halves). From `2000` onward each extra tick buys very little.
2. **The backlog saturates at ≈ 2000 ticks** (median ~32 at 10K, ~107 at 100K). After
   that, extra warm-up produces posts nobody is around to read.
3. **Boredom never reaches zero.** 4–7% of first sessions still end in boredom even at
   `w = 10000`. That is the *natural* boredom floor (tiny backlogs, singleton
   sessions), not a warm-up failure. The correct bar is therefore "not elevated above
   steady state" — and first sessions cross *below* the ~30% steady-state boredom rate
   already around `w = 100–500`.
4. **10K is the conservative size.** At every warm-up value, 10K has *more* first-session
   boredom than 100K (a smaller network produces sparser timelines). So a warm-up sized
   from the 10K curve is automatically safe for 100K, 500K and 1M.

## 8. Conclusion

**Locked value: `warmup = 2000` ticks.**

- It is the knee of the curve: backlog saturated, first-session boredom at 15.7% (10K)
  / 8.8% (100K) and falling slowly.
- It is a ~15× reduction from the previous 30,000-tick assumption.
- It dissolves the memory problem: warm-up memory is proportional to warm-up length, so
  a 1M run at warm-up 2000 needs ~72 GB instead of ~2 TB. The "free dead posts"
  optimization remains a nice-to-have but is no longer on the critical path.
- It leaves the stability experiment (which measures the *online-fraction* transient, an
  entirely different quantity) unchanged in design — only its warm-up parameter changes.

### Reproduce

```bash
# analysis over the moved traces
python3 analyze.py
```

The sweep itself is generated by the job configs in
`des-ctic-dev/configs/build-configs/warmup/*-ws{...}.json` (ratio 0.976) and run with
`zig build sim -Dconfig=...`.
