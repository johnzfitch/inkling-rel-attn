# R5-D clock-transfer preregistration (CK1b)

This registers the empirical clause that CK1's real-arm test turned out not
to carry. Its commit timestamp is the registration event.

**Why CK1 needs this.** `G_L` was frozen as the OLS slope of the certified
`06_random` block means on the CK1 regressor. For the single-layer clock
arms, the sealed pre-intervention r-vectors are bitwise that same data, and
projecting an OLS-slope direction out of its own fitting data annihilates
the regressor correlation in every coordinate identically. The real-arm CK1
statistic was therefore an algebraic identity (second-analyst demonstration:
float64 residual ≤3e-13; `analysis/round5/r5d/verification.json`). The
identity holds ONLY for the fitting text: whether projecting out the
random-fit direction flattens the drift on OTHER texts is an empirical
question about whether the clock is one shared direction.

**Precedence (ledger rule 4):** the runnable-now queue is empty; this test
reads only sealed, already-committed dumps and blocks the final
interpretation of an otherwise-certified campaign row.

## Disclosed peeks

All `06_random` gain statistics at every layer; signed gain-correlation
medians for `01_prose_en` at L41/L53/L59/L65; coordinate-level tail
correlations for all six texts at L59 (templated 0.9703, needles 0.9494 on
coordinate 650). The gain-level median-|corr| statistics for `03_templated`
and `05_needles` have never been computed, frozen or baseline, at any layer.

## Frozen test

Cells: text T ∈ {`03_templated`, `05_needles`} × layer L ∈ {53, 59}. Inputs:
the sealed `clock/rvec_pre_L{L}_{T}.npy` from arms `clock_freeze_L53` and
`clock_freeze_L59`, and the sealed `clock_freeze.npz`.

For each cell compute the CK1 statistic exactly as registered (127 tail
block means, frozen bank product, per-head gain against the block-mean
kernel, median over 64 heads of |Pearson corr| with `log1p(start+31.5)`),
twice: on the raw pre-intervention vectors (**baseline**) and on
`r̃ = r − ⟨r − r̄_L, G_L⟩ G_L` (**transferred freeze**, `G_L` and `r̄_L` from
the sealed artifact, float64).

**CK1b passes only if, in all four cells, baseline > 0.50 AND transferred
freeze < 0.20.** Ties fail. A baseline below 0.50 fails that cell (the
prediction stakes both the presence of the drift on unseen-by-G texts and
its one-directionness). `01_prose_en`, `02_code`, `04_multilingual` are
reported descriptively (prose cells are peeked; no verdict weight).

Outcome artifacts: `analysis/round5/r5d/ck1b_transfer.json` + ledger update
in the same commit as the verdict.
