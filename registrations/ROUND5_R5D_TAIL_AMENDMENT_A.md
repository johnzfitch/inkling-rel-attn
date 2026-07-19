# R5-D tail Amendment A — use the already-frozen mode-0 decay envelope

This amendment is registered before any R5-D GPU preflight or propagated
causal outcome. It replaces only section 3 of
`ROUND5_R5D_EXECUTION_AMENDMENT.md`. All arms, readouts, arithmetic, causal
predictions, thresholds, and other gates are unchanged.

## Disclosed failed preflight

The first registered tail builder (`round5_r5d_tail.py`, commit `88a4099`)
fit a separate continuity-constrained signed two-exponential to each raw table
row. Its literal finite/continuity/overshoot gate passed for 176/176 rows, but
its required diagnostics showed that this coordinate system does not contain
individually identifiable decay curves:

- median fit R2 was negative at 5/11 layers and below 0.10 at 9/11;
- 126/176 slow rates hit the lower bound;
- an SVD-coordinate diagnostic did not repair the general problem (only the
  leading, high-energy modes were reliably decay-shaped).

The full first report is retained at
`analysis/round5/r5d/tail_fit.json`; its dump remains immutable and unused.
No NLL, model logit, attention redistribution, or other causal result was
computed. Proceeding with those poorly identified rowwise extrapolations would
turn "heal the wall" into an arbitrary intervention even though the narrow
preflight gate technically passed.

## Corrected tail definition

Use the two-exponential decay envelope that was already fit and frozen for LF6
before this R5-D execution work. Its private fit manifest is hash-bound by the
committed public LF6 result (`analysis/round5/lf6/lf6_mi_mimicry.json`). For
each global layer, take the registered positive mode-0 `exp2` parameters
`(a1,r1,a2,r2)` and define

`g_L(u) = [a1 exp(-r1(1023+u)) + a2 exp(-r2(1023+u))] /
          [a1 exp(-r1*1023) + a2 exp(-r2*1023)]`.

Then append, for every raw table row j,

`P_L[j,1023+u] = P_L[j,1023] * g_L(u)`, for `u=1..7168`.

This keeps all learned cells `d<1024` bit-identical, makes the continuous
extension exactly equal each learned endpoint at `u=0`, preserves the full
16-dimensional endpoint orientation, and uses one already-observed,
layer-specific decay envelope rather than pretending the arbitrary raw-row
coordinates have separately identifiable rates. It introduces no new fit and
therefore no new optimization or selection degrees of freedom.

## Replacement gate

Before GPU execution require:

1. the private LF6 fit-manifest hash equals the hash recorded in the committed
   LF6 public result;
2. every LF6 projection input hash equals the current frozen projection file;
3. all 22 amplitudes and rates are finite and strictly positive;
4. each normalized envelope is finite, positive, monotonically non-increasing,
   exactly 1 at `u=0` to `1e-12`, and strictly smaller at `u=7168`;
5. all 11 extended `16 x 7168` tables are finite and their maximum magnitude
   does not exceed the corresponding learned endpoint magnitude.

Failure stops the campaign and requires another public amendment. The first
rowwise-tail report remains part of the audit trail; it is not deleted or
reinterpreted as an R5-D result.
