# Round 5 certified-capture dump-science batch

**Status: all six rows answered by the first analyst and independently
re-derived by the second analyst (section below) — promotion rule satisfied.**
All results use capture manifest
`2fc2ec9dd4d6b684a473847f6fbd4541d4326b49d7d6456b463ee5722399a75f`
and the publicly timestamped execution plan at `88dd002`.

| Row | Registered outcome |
|---|---|
| R5-C lifecycle | 4786 sustained onset at L26; 3290 has no registered positive-coverage onset; L39/40 handoff gradual/mixed (0/6 sharp) |
| R5-C geometry | carrier <1% failed (maximum cell 10.82%); flip-band discontinuity failed; globals-rotate-more passed (p=0.015625) |
| LF3 | failed into the surprise branch: random-text tail clock max abs(r)=0.9750 at L59, search-wide p=1/127; BOS transient also present |
| LF8 | all three failed: minimum cross-text cosine 0.3852 at L53; no global/local family split; 47 chirality candidates |
| LF9 | failed: capacity peaks at L11, though L65 is the unique minimum; bias decreases far share at all registered mid layers |
| R5-B | depth profile passed with unique peak L41 and L65 collapse; random-nearest/code-farthest failed (code nearest, needles farthest) |

Each row has a `RESULTS.md`, machine-readable JSON, compressed numerical dump,
input hashes, source hash, and an overwrite guard. Artifact hashes and array
finiteness were checked after execution. The surprising R5-C carrier maximum
was independently spot-checked from the full L20/needles state and communal
basis (`0.1081893562` direct versus `0.1081892867` reported).

## Execution-order deviation

The execution plan says the unchanged mechanical A6 re-certifications precede
these newly unblocked rows. The primary agent mistakenly ran these six rows
first. This is a procedural deviation, not an outcome-dependent method change:
all six estimators and verdict mappings were committed publicly before their
outcomes, none consumes a mechanical re-certification result, and no threshold,
band, input, control, or failure disposition changed. The mechanical
re-certifications remain next and this deviation is not repaired by rewriting
timestamps or rerunning outcomes.

## Independent second-analyst re-derivation (2026-07-18)

All six rows were re-derived from the raw certified dumps with independent
implementations — no producer imports; correlations via explicit centered dot
products, Gram-eigensystem participation ratios, hand-rolled Holm, and the
lifecycle coverage taken from the massive-census artifact family (a different
artifact family than the producer's states) with a direct-state cross-check.
Scripts: `scripts/round5_batch_verify_{rvec_meters,states_r5b,bos_pr,rotation}.py`;
artifacts: `verification_*.json` in this directory.

- **LF3** — tail statistic, argmax layer/regressor, and search-wide
  circular-shift p reproduce exactly (0.9750178, L59, log1p, p = 1/127);
  prose control identical (0.9393 at L11, p = 1/127). BOS displacement ranks
  identical under the producer's midrank convention (random 127.5/128,
  prose 123.5/128).
- **LF8** — minimum stability cosine exact (0.38521707, L53, prose–needles);
  family statistic exact (−0.00619929), own-seed null p 0.7428 vs 0.7481;
  chirality census reproduces exactly 47 candidates.
- **LF9** — the paired per-head aggregation reproduces every reported effect
  digit-for-digit; all seven signs are also invariant under the marginal
  (aggregate-then-difference) reading.
- **R5-B** — both clauses and all six centrality medians match to the
  reported precision.
- **R5-C lifecycle** — onset 26 / no-onset reproduced from the census
  artifacts; direct-state cross-check agrees bitwise; channel 3290 never
  crosses the threshold anywhere in the window.
- **R5-C geometry** — carrier maximum cell exact (0.1081894, L20/needles;
  "component 1" resolved against the basis artifact's own one-indexed naming,
  confirmed by the frozen `live_share` column-0 median 0.641155); rotation
  contrast exact (0.0234676, p = 0.015625, positive in all six texts —
  driven by templated +0.061 and needles +0.060); PR flip-band failure
  confirmed (unique max at destination L64, outside L13–28).

Two convention differences were reconciled, neither material: midrank versus
inclusive BOS percentiles, and paired versus marginal LF9 aggregation. No
discrepancy of substance was found. With the registered controls passing and
this re-derivation on record, all six rows satisfy the promotion rule.
