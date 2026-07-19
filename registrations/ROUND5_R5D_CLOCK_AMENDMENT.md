# Round 5 R5-D clock-freeze amendment (arm-family addition)

This amendment adds one intervention family to the R5-D campaign
operationalized in `ROUND5_R5D_EXECUTION_AMENDMENT.md`. It changes no
existing arm, threshold, gate, tail fit, or readout. Its commit timestamp is
the registration event. No clock-freeze outcome exists or has been inspected:
the interventions below have never been run.

Precedence under ledger rule 4: the runnable-now queue is empty and R5-D is
the next registered science state change; this family attaches to that
campaign before its runner is built, so it displaces nothing.

## Disclosed peek (motivating exploration)

The LF3-certified clock was characterized exploratorily on 2026-07-18 from
the certified capture (artifacts committed under
`analysis/round5/r5d_clock/exploration/`; all quantities below are BASELINE
values and are peeked):

- the positional drift of tail block-mean r-vectors is rank-1 dominated at
  all 11 globals (PC1 share 0.33–0.82, |corr(PC1, log p)| 0.72–0.97), ~72%
  separable as one 16-dim mode broadcast to nearly all heads with same-sign
  gains; `log(p+128)` fits the strongest coordinate with R² 0.956;
- through the frozen projection bank the drift is multiplicative kernel
  annealing at L53/L59 (cos(drift, mean kernel) −0.79 to −0.91; median-head
  gain 1.70→0.88 and 1.61→0.95 across the tail on `06_random`; median signed
  gain–log-p correlation −0.739 and −0.633) and near-null at L65
  (0.97→1.01, +0.619);
- a needle-row near-mass check is dilution-confounded and is treated as no
  evidence either way.

Every registered clause below concerns quantities from the not-yet-run
ablated forwards.

## 1. Frozen inputs

`scripts/round5_clock_freeze_build.py` builds
`analysis/round5/r5d_clock/clock_freeze.npz` (+ manifest with input hashes)
BEFORE any clock arm executes; it refuses overwrite. Definitions:

- **Blocks and regressor.** Token positions 64..8191 in 127 consecutive
  64-token blocks; block midpoint `m_b = start_b + 31.5`; regressor
  `x_b = log1p(m_b)`.
- **Clock direction `G_L`** for `L ∈ {53, 59, 65}`: from the certified
  `replay/rvec_L{L}_06_random.npy`, flatten to `[8192, 1024]` float64, take
  the 127 block means, and set `G_L` to the per-coordinate OLS slope on
  `x_b`, L2-normalized to a unit 1024-vector.
- **Anchor mean `r̄_L`** for the same layers: the all-token mean flattened
  r-vector over the six certified v1 texts (49,152 tokens, float64).
- **Sham direction `S`:** seeded Gaussian in R^1024, orthogonalized against
  `G_59`, unit-normalized. Seed: first eight bytes (unsigned big-endian) of
  SHA-256 of `<sha256-of-this-amendment-file>:clock_sham_L59`.

## 2. New intervention arms (five)

`clock_freeze_L53`, `clock_freeze_L59`, `clock_freeze_L65`,
`clock_freeze_L53_L59` (joint), `clock_sham_L59`.

At each target layer, on the production r-vector of every token (flattened
1024, FP32): `r̃ = r − ⟨r − r̄_L, D⟩ D`, where `D = G_L` (sham arm: `D = S`,
anchor `r̄_59`); cast back to the production dtype. Only the relative-logit
path consumes `r̃`; Q/K/V, content logits, the residual stream, and all
other layers are untouched — the mirror image of the carrier-out arm, acting
on the r-projection output instead of its input. This pins the clock
coordinate at its frozen mean; it does not remove the mode's static
contribution.

Prefix reuse follows the parent amendment exactly: single-layer arms start
from the certified state entering the target layer; the joint arm starts
from `hidden_L52` and intervenes at both L53 and L59 during propagation.

## 3. Additional dump

Each clock arm saves the FP16 pre-intervention r-vectors at every intervened
layer for all six texts (`r̃` is then derivable exactly from the frozen
inputs). All parent §5 readouts (per-position NLL/logit deltas, needle
splits, locus meters) apply unchanged.

## 4. Frozen verdict rules

- **CK1 — mechanism (kernel flattening).** For an arm and its intervened
  layer, compute from the arm's `r̃` on `06_random`: block means
  `B_b ∈ R^{64×16}`, per-block realized kernels `K_b = B_b P_L` against the
  frozen bank, per-head gain `g_{b,h} = ⟨K_{b,h}, K̄_h⟩ / ⟨K̄_h, K̄_h⟩`
  (`K̄` = mean over blocks), and the median over 64 heads of
  `|corr(g_{·,h}, x)|`. CK1 passes only if this statistic is `≤ 0.20` in
  BOTH `clock_freeze_L53` and `clock_freeze_L59` AND `≥ 0.50` in
  `clock_sham_L59`. Ties fail.
- **CK2 — behavior (log-extremes cost, joint arm).** Pool per-block ΔNLL as
  the mean over the six texts of the block-mean `NLL_arm − NLL_baseline`
  (targets 64..8191, the same 127 blocks). CK2 passes only if the Spearman
  correlation of pooled per-block ΔNLL with `|x_b − mean_b(x)|` is strictly
  positive AND its 5,000-draw block-bootstrap 95% lower bound (resampling
  the 32 aligned 256-token superblocks; seed rule
  `<sha256-of-this-amendment-file>:ck2_bootstrap`) is strictly positive.
  Rationale on record: freezing a log-clock at its mean should hurt most at
  the log-extremes; if the clock is epiphenomenal to prediction this fails.
- **CK3 — wall exemption.** `|pooled ΔNLL(clock_freeze_L65)| < 0.005`, the
  wall-heal scale.
- Needle splits and locus meters for the clock arms are registered causal
  readouts with no numerical prediction; they receive no invented
  thresholds.

Verdicts are computed only after all five arms are sealed. Independent
raw-dump re-derivation is required before promotion to certified.

## 5. Additional startup gates

Alongside every parent §4 gate: a toy test proving the freeze operator
(`⟨r̃ − r̄_L, D⟩ = 0` to FP32 resolution; `⟨S, G_59⟩ ≤ 1e−12`; frozen-input
hashes match the sealed artifact), and a locus identity check that the
pre-intervention r-vectors of `clock_freeze_L53` on `06_random` are bitwise
equal to the certified capture's `rvec_L53_06_random.npy`.
