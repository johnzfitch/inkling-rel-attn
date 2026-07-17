# Corpus v2 registered results

Executed 2026-07-17 from the private, gitignored corpus-v2 artifacts. The
original prediction commit is `7fb84ab`; the honest public-registration boundary
is `65b220c`, as recorded in `CORPUS_V2_AMENDMENT_A1.md`. The exact execution
plan was public at `4d39da1`, and all measurement/readout sources were public at
`c834fdf` before the model forward began.

No prediction, class definition, threshold, direction, or failure rule was
changed after outcome access.

## Capture and validation

The single GPU pass processed both 8,192-token arms through all 66 layers in
9.63 minutes. It sealed 132 FP16 r-vector arrays and two FP32 next-token-NLL
arrays (134 artifacts total). Normalized attention inputs were omitted as
registered optional inputs.

Independent validation hashed all 134 artifacts and checked 1,107,312,638
values. Every r-vector shape/dtype/finiteness check passed; both NLL files had
8,191 finite values, exact target positions, and target IDs matching the private
corpus. The validator reported no errors.

## P-v2-1 — novelty gate: pass

| Arm | Mean NLL (nats) | Registered threshold |
|---|---:|---:|
| Slack human | 2.558635 | >= 1.5 |
| Math LLM | 1.613229 | >= 1.5 |

Both arms exceed 1.5 nats and Slack is more surprising than math-LLM. Neither
arm is below the 1.0-nat replacement boundary. P-v2-1 passes and both arms are
eligible for the aperture readouts.

## P-v2-2 — aperture–surprisal law: pass

The registered statistic averages five layerwise, within-512-token-bin aperture
percentiles and correlates that score with own-token NLL separately in each of
16 bins.

| Arm | Median bin Spearman | Positive bins | Decision |
|---|---:|---:|---|
| Slack human | 0.152384 | 16/16 | pass |
| Math LLM | 0.114283 | 13/16 | pass |

Both arms satisfy median rho > 0 and at least 12/16 positive bins. The crossed
true-null pairings do not satisfy the rule:

- math aperture / Slack NLL: median `0.003503`, 9/16 positive;
- Slack aperture / math NLL: median `0.000166`, 8/16 positive.

P-v2-2 and its control pass.

## P-v2-3 — message-start widening: pass

The 311 frozen Slack message-start positions have a registered LF4 aperture
effect of `+0.0433594`, with one-sided Holm-corrected `p = 0.0029997`.
The same position mask on math-LLM is null (`effect = -0.00742188`,
`p_holm = 1.0`). P-v2-3 and its control pass.

## P-v2-4 — pronouns and function words narrow: fail jointly

| Class | N | Effect | Holm p | Arm decision |
|---|---:|---:|---:|---|
| Pronouns | 946 | -0.0460938 | 0.00029997 | pass |
| Function words | 2,016 | -0.00585938 | 0.659334 | fail |

Pronouns replicate the revised narrow direction strongly. They do not flip back
wide on novel casual human text, so the registered pronoun falsifier survives.
Function words are slightly negative but not significant. Because P-v2-4
requires both classes to pass, the joint prediction fails; the data do not
support a blanket closed-class-narrow claim.

Both cross-arm class masks are null:

- pronouns on math-LLM: `effect = +0.0144531`, `p_holm = 1.0`;
- function words on math-LLM: `effect = -0.00820313`,
  `p_holm = 0.429557`.

## Independent confirmation: pass

The non-importing verifier:

- rebuilt all private class positions exactly;
- reproduced both NLL means;
- recomputed all ten aperture arrays from raw r-vectors and projections with
  maximum absolute difference `0.0` from the blocked dump;
- reproduced every primary and crossed Spearman correlation;
- reproduced all six permutation p-values and Holm corrections; and
- reproduced all prediction fields.

All eight methodology gates, including every true-null control, passed.

## Registered scorecard

- P-v2-1: **pass**
- P-v2-2: **pass**
- P-v2-3: **pass**
- P-v2-4: **fail** (pronouns pass; function words fail)

Result: 3/4 registered predictions pass. The aperture–surprisal law and
message-start widening replicate on genuinely novel data. The important
pronoun-direction reversal also replicates, but it should be described as a
pronoun result rather than generalized to all closed-class tokens.

Private text, IDs, sidecars, class positions, NLL arrays, r-vectors, and
aperture dumps remain gitignored and are not part of the public result commit.
