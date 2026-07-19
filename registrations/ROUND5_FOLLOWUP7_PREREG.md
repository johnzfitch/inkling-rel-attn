# Round 5 seven-experiment causal follow-up preregistration

This document freezes the seven post-R5-D follow-ups requested after the
exploratory Q1--Q24 audit. Its commit timestamp is the registration event.
No intervention, patch, full-vocabulary, fresh-text bias-off, or follow-up
outcome described here has been run or inspected at registration time.

The motivating observations are explicitly peeked: the certified R5-D dumps,
the exploratory `exploration_q24` package, and its accepted corrections at
commit `9999752`. These experiments test the resulting story; they are not a
blind continuation of the original Round-5 predictions.

This work takes precedence under ledger rule 4 because the runnable-now queue
is empty, LF5-b remains gated on the separate D3 bracket-corpus decision, and
the user explicitly authorized all seven experiments with the full GPU and CPU.

## Common execution contract

- Model, tokenizer, six v1 texts, 8,192-token length, NVFP4 layer loader,
  BF16 content-plus-bias addition, FP32 softmax, q-chunk 512, lossless D4
  prefix states, and final readout arithmetic are inherited unchanged from
  `ROUND5_R5D_EXECUTION_AMENDMENT.md`.
- A no-intervention production-layer replay must be bitwise equal to the
  certified D4 state. A new exact-copy `bias_off_L29_fullvocab` arm must be
  bitwise equal, for target logit, log-normalizer, NLL, and probability, to
  the certified `bias_off_L29` token dump before its new metrics are used.
- Every arm starts from the certified state entering its first changed layer,
  propagates through L65, writes an initially incomplete immutable manifest,
  and is resumable only at complete-arm boundaries. Failed directories stay
  on record.
- Every arm saves per-token target readouts and full-vocabulary sufficient
  statistics: strict target rank (ties do not outrank), top-1 id/probability
  and correctness, predictive entropy, multiclass Brier score, target margin,
  and the parent target-logit/log-normalizer/NLL/probability fields. The full
  vocabulary matrix is consumed chunkwise and is not retained.
- Primary pooled effects average all 49,146 next-token observations from the
  six equal-length v1 texts. Intervals use 5,000 shared 256-token block
  bootstrap draws. Seeds are SHA-256 derived from this registration file's
  byte hash plus the named statistic. Holm adjustment is used for each family
  explicitly named below. Exact equality boundaries fail.
- Inputs are built before GPU execution into a sealed NPZ plus JSON manifest,
  with hashes for every source. Builders and runners refuse overwrite.
- A producer analyzer and an independent verifier must recompute results from
  raw dumps without importing one another. No result is certified until both
  agree and every artifact hash is checked.

## F7-1: signed L29 near-field stencil

At L29, run these eight all-head bias interventions:

`d0_off`, `d1_off`, `d2_off`, `d3_off`, `d1_3_off`, `restore_d0`,
`restore_d1_3`, and `stencil_only_d0_3`.

An `off` arm zeros exactly the named integer distance(s) and changes nothing
else. A `restore` arm starts from bias-off and restores the certified deployed
bias only at the named distance(s). `stencil_only_d0_3` restores d=0..3 on an
otherwise bias-off L29. Existing certified `bias_off_L29` and `near_off_L29`
are references, not rerun outcomes for this family.

Predictions, informed by the peeked negative-d0/positive-d1..3 stencil:

1. At least one of the four singleton-off arms has pooled delta-NLL > 0 with a
   Holm-adjusted 95% block-bootstrap lower bound > 0.
2. `stencil_only_d0_3` rescues at least 50% of certified full bias-off damage,
   where rescue is `1 - cost(stencil_only)/cost(bias_off_L29)`.
3. Singleton additivity, the d1--3 group interaction, each restore arm, all
   per-text effects, and attention-band redistribution are frozen descriptive
   readouts; no post-outcome threshold will be invented.

## F7-2: L23/L29/L35 shoulder interactions

Run semantically identical all-head full-bias-off joints:

`bias_off_L23_L29`, `bias_off_L29_L35`, `bias_off_L23_L35`, and
`bias_off_L23_L29_L35`.

The registered backup prediction is positive interaction for both adjacent
L29 pairs and the triple: joint cost minus the sum of the corresponding
certified single-layer costs is > 0. The family passes only if all three
contrasts have Holm-adjusted 95% block-bootstrap lower bounds > 0. The L23/L35
pair is a negative-architecture control and is reported without a directional
requirement.

## F7-3: mean, centered, carrier, and non-carrier r-space components

For each v1 text, let `mu_t` be its certified baseline mean 1,024-dimensional
L29 r-vector. Let `g` be the unit r-space image of communal hidden carrier 1:
the certified L29 `r_proj` weight multiplied by row 0 of the certified hidden
basis. Decompose `mu_t = c_t + n_t`, with `c_t=(mu_t dot g)g`, and each token
`r = mu_t + delta`.

Run six r-path-only arms at L29:

- `r_remove_mean`: `r' = r - mu_t`;
- `r_remove_centered`: `r' = mu_t`;
- `r_remove_carrier_all`: `r' = r - (r dot g)g`;
- `r_remove_noncarrier_all`: `r' = (r dot g)g`;
- `r_remove_carrier_mean`: `r' = r - c_t`;
- `r_remove_noncarrier_mean`: `r' = r - n_t`.

Q/K/V and the residual stream remain untouched. Predictions:

1. Removing the static mean and removing its non-carrier component each
   reproduce at least 50% of certified `bias_off_L29` cost.
2. Removing centered token variation costs at most 25% of full bias-off cost.
3. Removing only the carrier part of the mean has absolute pooled cost < .005.
4. The all-carrier/all-noncarrier complementary arms are reported as causal
   decomposition checks; finiteness and exact algebra are startup gates, not
   scientific outcomes.

## F7-4: causal head localization and rescue

Before GPU work, rank L29 heads without causal outcomes. For each head and
text, form the baseline mean realized kernel and its signed stencil score
`mean(K[d=1..3]) - K[d=0]`. Rank heads by the median score across six texts,
descending, breaking ties by lower head index. Freeze Q1..Q4 as consecutive
16-head quartiles and top-8/top-16/top-32 prefixes in the sealed input file.

Run four quartile-off arms, `head_q1_off` through `head_q4_off`; one
`head_top16_only` arm retaining full bias only for Q1; and three nested
`head_top08_stencil_only`, `head_top16_stencil_only`, and
`head_top32_stencil_only` arms retaining only d=0..3 on those heads. The
all-64 stencil reference is F7-1's `stencil_only_d0_3`.

Predictions: Q1-off is the largest quartile cost, with its paired difference
from every other quartile having Holm-adjusted bootstrap lower bound > 0; and
the top-16 stencil rescues at least 50% of full bias-off damage. Nested rescue
curves and top-16-full versus top-16-stencil are descriptive.

## F7-5: needle query-state patch

On `05_needles`, run L29 bias-off and, immediately after the L29 residual is
formed, replace only the 24 frozen query-position residuals with the certified
baseline `hidden_L29` values. A sham arm patches 24 frozen eligible positions
drawn from positions 256..8190, excluding every position within eight tokens
of a needle query. Its seed is derived from this registration hash. Both arms
then propagate L30..L65 normally.

The patch prediction passes only if query patching rescues at least 50% of the
certified mean recall-target damage and the 5,000-draw query bootstrap lower
bound on rescue is > 0, while sham rescue is < 10% and its interval contains
zero. A baseline-to-baseline query patch must be bitwise identical as a startup
gate.

## F7-6: text-specific and union clock subspaces

At L53 and L59, reproduce the registered 127 block means and log-position
regressor separately for all six v1 texts. The per-text clock direction is the
unit coordinatewise OLS slope. Freeze the six-direction SVD union basis at
each layer; require numerical rank six. Also freeze leave-one-text-out bases
from the other five directions and a seeded six-dimensional sham basis
orthogonal to the full union.

Run `clock_union_L53`, `clock_union_L59`, `clock_union_L53_L59`,
`clock_pertext_L53_L59`, `clock_loto_L53_L59`, and
`clock_sham6_L53_L59`. At each locus, pin the selected subspace to that text's
certified mean: `r' = r - U U^T(r-mu_t)`. Per-text uses its one direction;
LOTO uses the five-text basis that excludes the current text.

Own-text/per-text and full-union flattening are operator checks, not science
verdicts, because the fitted slope lies in the removed space by construction.
The non-circular transfer prediction is that LOTO reduces every held-out
text's median-head absolute kernel-gain/log-position correlation below .20 at
both layers, while sham remains at least .50. Behaviorally, every real joint
clock arm is predicted equivalent to baseline at 8k: absolute pooled
delta-NLL < .005. Failure of that equivalence is evidence that the omitted
text-specific clock subspace is functional.

## F7-7: full-vocabulary mechanism and fresh class replication

Produce full-vocabulary sufficient statistics from certified v1 baseline L65
states and the exact-copy `bias_off_L29_fullvocab` arm. The ranking prediction
passes if bias-off increases pooled mean `log1p(target_rank)` and decreases
top-1 accuracy, with both 95% block-bootstrap intervals excluding zero.
Calibration evidence is declared only if 20-bin equal-width top-1 ECE changes
by at least .01 and its bootstrap interval excludes zero. Entropy, Brier,
margin, per-text results, and alternate 10-bin ECE are frozen descriptive
readouts.

For fresh replication, run untouched baselines for `07b_slack_multi` and
`08_math_llm` from embeddings, require NLL equality to the independently
certified corrected corpus-v2 capture, save their L28 prefixes, then run exact
L29 bias-off. Primary classes are query-aligned: delta index p scores target
p+1. Slack first-content and pronoun positions come verbatim from frozen
`corpus_v2/depth_classes.json`. Math unit starts come verbatim from its frozen
sidecar; math pronouns use the exact committed corpus-v2 tokenizer-offset
selector and lexicon, with positions sealed before execution.

Class contrasts use controls matched within 512-token position block and
baseline-NLL decile, with 10,000 seeded matched resamples. The fresh-casual
replication passes only if both Slack first-content and pronoun contrasts are
positive with Holm-adjusted p < .05. Math classes test domain transfer and are
reported in the same frozen family but do not gate the Slack replication
verdict because only ten provider-unit starts exist. Target-aligned p-1
versions are secondary and explicitly labeled.

## Completion rule

All seven rows remain OPEN until every required arm/job is sealed. A failed
prediction is an answer, not a reason to refit a threshold, regroup heads,
change clock rank, or choose different classes. Any execution defect requires
a public amendment before replacement output is inspected.
