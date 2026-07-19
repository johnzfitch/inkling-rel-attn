# Corrections to the exploration_q24 package (2026-07-20)

The first analyst audited this exploratory package; the second analyst
verified the audit with corrected code (`q_corrected_v2.py`, hash-stamped,
overwrite-protected). Original scripts and outputs are retained above for
provenance. None of this touches the certified 72-arm verdicts
(`../verification.json`).

## Accepted corrections (verified independently)

1. **Baseline-NLL semantics bug (material).** The arm token dumps' `nll`
   field is the INTERVENED readout; baseline is `nll − delta_nll`. The
   original q_token.py misused it. Consequences, now corrected:
   - The reported 0.51–0.80 "surprisal gates vulnerability" correlations
     (Q4/Q22) were artifacts of correlating ΔNLL with a field containing
     ΔNLL. True per-text Pearson correlations of ΔNLL(bias_off_L29) with
     baseline NLL: prose −0.070, code +0.023, templated +0.206,
     multilingual −0.036, needles −0.149, random +0.010. **There is no
     tokenwise surprisal gating.** The aperture null stands.
   - Q9's beneficiary-difficulty claim survives on natural texts with the
     true baseline (e.g. prose L65 beneficiaries 1.69 vs 0.11 overall) but
     is near-flat on random (12.96 vs 12.54).
   - Q14's clock-vs-difficulty correlations become mildly negative
     (−0.007..−0.104), still no coherent population.
2. **Q5 convention.** Shares are now reported against positive gross
   damage: top 0.1% / 1% / 10% of tokens carry 5.6% / 28.2% / 81.0% of
   gross. (The earlier >100% figure divided by net.)
3. **Q8 mediation contradiction (conceptual, important).** Near-off at L29
   PRESERVES or slightly increases 4–128 attention mass (prose
   0.263→0.277) while reproducing 84% of the damage. Aggregate mid-band
   mass loss is therefore a side effect of full bias-off, not the
   mechanism.
4. **Stencil hypothesis confirmed at the kernel level.** L29's mean
   realized kernel is negative at d=0 (−0.49..−1.73 across texts) and
   positive at d=1–3; the contrast mean(d1–3)−d0 correlates 0.774 with
   per-text causal cost (n=6, exploratory). The operative object is a
   local contrastive stencil — suppress self, boost immediate neighbors —
   whose detailed shape, not band totals, carries the effect.
5. **Q11/Q15/Q16 subgroup claims downgraded** to hypotheses: the
   below/above-seam and needle-improvement means have intervals/permutation
   p-values spanning zero.
6. **Q21 chirality distribution corrected:** candidates are spread across
   depth (23/47 at L0–28, 24/47 at L29–65), not concentrated early.
7. **Q23 semantics + emission:** tables now committed; `delta[p]` scores
   the prediction made FROM the class token. Matched analysis keeps
   sentence starts (+0.135, p≈0.011) and pronouns (+0.081, p≈0.030);
   function words and rare-BPE do not survive matching.

## One audit point rejected, with evidence

The audit held that the rising-head joint arm is a "selected-head" arm and
cannot be compared with all-head singles. Code and registration disagree:
`intervene_bias(kind='rising_heads_off')` returns `zeros_like(bias)` — all
64 heads — and the parent amendment states "all 64 heads at each of L0–L4
are rising." The joint arm is semantically identical to simultaneous
all-head bias-off at L0–L4, so the superadditivity comparison stands:
joint +0.01988 vs singles-sum +0.00459 (4.3×).
