# Random-timeline experiment (queue-based attention bottleneck)

Analysis of the `-Dtimelinerandom` experiment: the same CTIC simulation run with
the per-user timeline drained **uniformly at random** instead of LIFO, so that an
older post has the same chance of being read as a fresh one (a poor man's
recommender). Dataset: `steps/random-timeline` (10K/50K/100K/500K/1M, 100 runs

each, under `/data/nfs/psoler/steps/random-timeline/datasets`), compared against
the LIFO run `steps/final` (10K/100K/500K/1M under `des-ctic-dev`).

## What the experiment shows

- **Queue order is an allocation valve, not the capacity.** Randomising slightly
  raises the extreme tail (max cascade size 779 -> 892 at 500K; hub broadcast
  first-hop max 162 -> 155 with more large hub cascades at 100K) but makes the
  bulk *more* broadcast-shaped (broadcast share 80.3% -> 82.5%). Aggregate
  metrics barely move, so drain order is a second-order lever.
- **Reach rises with followers but heavily compressed.** With the author's true
  follower count, mean cascade size goes ~1.00 (<=10 followers) -> ~2.83
  (10k-100k) -> **~174-184 (>100k)** at 500K. The heavy tail is almost entirely
  the giant hub: at 500K, 73% (LIFO) / 70% (random) of posts by >100k-follower
  authors reach >=50 reposts, vs ~0.03% for 10k-100k authors and ~0% below 1k.
- **The random queue helps the giant hub but not mid-tier hubs.** At 500K the
  >100k bucket's first-hop broadcast width goes 16.6 -> 26.4 under random (+60%)
  and max cascade size 779 -> 892, while the 10k-100k bucket drops 2.33 -> 2.09
  and the aggregate broadcast share rises. So ordering decides *which* of a
  giant hub's posts snowball; it doesn't create capacity.
- **Why width is missing.** `out-degree = impressions x conversion`, and both
  factors are capped: (1) each session reads a bounded number of posts shared
  across everyone followed, so even a 211k-follower hub gets a small fraction of
  its audience; (2) conversion is a homogeneous 1.2% (no content), so no post can
  exceed baseline; (3) delivery is in-network only, so first-hop impressions are
  bounded by the follower count. Real max out-degree 7,768 needs ~647k impressions
  at 1.2%, more than the largest hub's 408k followers -- so out-of-network
  exposure (recommendation) or above-baseline conversion (content) is required.

## The trap this folder fixes

`data/original/<n>_nodes.parquet` has a **different node ordering** (and count)
than the `.bin` the simulator reads (100K: 99,835 nodes, hub at `sim_id` 97118 vs
the parquet's 97283). Joining `cascades.AuthorID` against parquet-derived degrees
silently gives wrong follower counts and makes the width line look flat.
`degree_map.py` derives degrees from the `.bin` itself.

## Usage

```bash
# needs: duckdb CLI on PATH, python with numpy + matplotlib + seaborn
python degree_map.py  --repo ../../des-ctic-dev        # build follower maps
python queue_experiment.py --repo ../../des-ctic-dev   # tables + figure
python results_83_85.py                                # appendix (8.3/8.4/8.5)
```

Outputs into `output/`:

| file | contents |
|------|----------|
| `overview_lifo_vs_random.csv` | aggregate cascade stats per size/timeline |
| `width_by_author_followers.csv` | n / mean size / P(size>=50) / first-hop by follower bucket |
| `top_cascades.csv` | largest cascades with author followers |
| `width_vs_followers.svg` | mean size and P(size>=50) vs author followers, LIFO vs random, hub-bearing sizes |
| `degrees/<size>.csv` | cached `sim_id,indeg` from the topology `.bin` |
| `results_83_85/summary.json` | random-timeline repost power-law + structural virality + vs-Bluesky tables |
