# Corpus v2 execution plan — capture, novelty gate, and registered readouts

This document operationalizes `CORPUS_V2_SPEC.md` after the pre-measurement
construction correction in `CORPUS_V2_AMENDMENT_A1.md`. It is written before
any corpus-v2 model forward, NLL observation, r-vector aperture computation, or
registered readout.

Public registration boundary: commit `65b220c`, which makes the original local
registration commit `7fb84ab` and Amendment A1 reachable from the remote
branch. P-v2-1 through P-v2-4 are unchanged.

## 1. Private inputs and class freeze

The following remain under the already-public `corpus_v2/` gitignore rule:

- `07_slack_human.ids.npy`, text, and sidecar;
- `08_math_llm.ids.npy`, text, and sidecar;
- the private corpus manifest; and
- a generated class-position manifest.

Before capture, a public selector script creates the private class manifest
from IDs and sidecars only. It freezes:

- all 311 Slack `unit_start_tokens` as `message_starts`;
- pronouns using the exact Round-5 single-token selector and word list; and
- function words using the exact Round-5 single-token selector and word list.

A token is a class member only when decoding that token ID alone, stripping
outer whitespace, and case-folding yields a complete ASCII word in the frozen
list. Demonstratives remain excluded from the pronoun list, matching the
Round-5 implementation. Positions, decoded text, and sidecar metadata are not
published.

## 2. GPU capture

Run one standard streaming forward over both 8,192-token arms and all 66
layers, using the validated checkpoint loader and true with-bias attention
path. Capture:

- `rvec[L,text]` as FP16 `[8192,64,16]` for every layer and arm; and
- next-token NLL as FP32 for target token positions 1 through 8,191.

The NLL record stores `target_position`, `target_id`, and `nll`, so own-token
alignment is explicit: aperture at token `t` pairs with the loss whose target
position is `t`. Normalized attention inputs and attention meters are omitted;
the registration made normalized inputs optional, and neither registered
readout consumes them. Omitting the measurement-only without-bias softmax does
not change the model's with-bias output.

The capture refuses to overwrite a nonempty output directory. Its manifest
records hashes for all 134 artifacts, all relevant public sources, the private
input manifest, checkpoint/config, environment, exact text/layer set, and the
public registration boundary. It records no NLL mean.

An independent validator, which does not import the capture implementation,
must hash and inspect all 132 r-vector files and both NLL files. Required:

- exact shapes and dtypes;
- all values finite;
- NLL target IDs equal private corpus IDs 1 through 8,191;
- artifact hashes equal the capture manifest; and
- no missing or extra registered artifacts.

No registered outcome is read before that validation passes.

## 3. P-v2-1 novelty gate

For each arm, compute the float64 mean over all 8,191 stored FP32 NLL values.

- Data eligibility: both means must be at least `1.0` nat. An arm below `1.0`
  is marked contaminated and must be replaced; aperture readouts remain closed.
- P-v2-1 passes only if both means are at least `1.5` nats and
  `mean(slack_human) > mean(math_llm)`.
- A mean in `[1.0,1.5)` is a valid but prediction-failing arm; the remaining
  readouts proceed without reinterpretation.

## 4. Aperture dump

Only the five registered mid-depth global layers are needed: L23, L29, L35,
L41, and L47. For every token and arm, compute in float64 from raw r-vectors and
the frozen projection table:

```text
A[L,t] = sum_h sum_{d=129}^{1023} |b[L,t,h,d]|
         ---------------------------------------
         sum_h sum_{d=0}^{1023} |b[L,t,h,d]|
```

This is the Round-5 full-support aperture, not the history-truncated diagnostic.
Ten result arrays are dumped and hashed before any class or surprisal statistic
is evaluated.

## 5. P-v2-2 aperture–surprisal law

For each arm and layer, convert aperture to mid-rank percentiles separately
inside each fixed 512-token position bin. Average those five layerwise
percentiles for each token. Within each of the 16 bins, compute Spearman
correlation between that averaged aperture score and own-token NLL; token zero
is omitted because it has no next-token loss targeting it.

For an arm, the registered decision passes when:

- the median of its 16 bin correlations is greater than zero; and
- at least 12 of 16 bin correlations are positive.

P-v2-2 passes only when both arms pass.

True-null control: preserve positions but cross the arms, correlating Slack
aperture with math-LLM NLL and math-LLM aperture with Slack NLL. The control
passes only if neither crossed pairing satisfies the registered P-v2-2
decision. This control affects confirmation, not the already-registered primary
decision fields.

## 6. P-v2-3 and P-v2-4 class tests

Use the exact Round-5 LF4 statistic:

1. convert aperture to mid-rank percentiles within fixed 256-token bins and per
   layer;
2. average each token's percentile across the five mid-depth globals; and
3. compute `median(class score) - 0.5`.

Run 10,000 deterministic within-bin label permutations, preserving the class
count in every bin. The three-test primary family is:

- message starts, positive direction (P-v2-3);
- pronouns, negative direction (P-v2-4); and
- function words, negative direction (P-v2-4).

Apply Holm correction across these three one-sided tests at family-wise alpha
0.05. P-v2-3 passes when its effect has the registered sign and corrected
`p < 0.05`. P-v2-4 passes only when both of its class arms do.

True-null control: apply each frozen Slack position mask to the unrelated
math-LLM aperture scores and repeat the same permutation tests. Holm-correct the
three control p-values as a separate family. A class claim is confirmable only
when its corresponding cross-arm mask is null (`p_holm >= 0.05`).

## 7. Independent confirmation

The independent verifier must not import the main aperture/readout code. It:

- recomputes all ten aperture arrays from raw r-vectors and projections using a
  separately structured token-block calculation;
- independently rebuilds the private class positions and requires exact
  equality with the pre-capture freeze;
- recomputes NLL means, all 32 primary/crossed Spearman correlations, all six
  class/control permutation tests, Holm corrections, and decision fields; and
- compares numerical effects and p-values to the main reports.

Raw aperture agreement tolerance is absolute error at most `1e-12`. No claim is
promoted unless capture validation, class equality, numerical rederivation, and
the applicable true-null control all pass. Prediction outcomes remain separate
from methodology/confirmation outcomes.

## 8. Execution order

1. Commit and push this plan plus all four public implementation scripts.
2. Generate and validate the private class-position freeze.
3. Run the two-arm GPU capture once.
4. Run independent full capture validation.
5. Evaluate P-v2-1; stop and replace only if an arm is below 1.0 nat.
6. Dump the ten aperture arrays.
7. Evaluate P-v2-2 through P-v2-4 and their frozen controls.
8. Run independent confirmation.
9. Commit only public reports; never add private corpus or dump artifacts.

Unrelated Round-3 worktree changes remain out of scope and unstaged.
