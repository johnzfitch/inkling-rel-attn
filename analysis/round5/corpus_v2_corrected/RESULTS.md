# Corpus v2 corrected-arithmetic and depth-resolved results

Executed 2026-07-17 PDT (2026-07-18 UTC) from the private, gitignored corpus
artifacts. The original corpus-v2 predictions were registered at `7fb84ab`.
Amendment A6 (`061bb04`) corrected the deployed-model dtype boundary, A7
(`205f975`) froze this execution, and A8 (`93665e2`) froze historical-source
validation after a concurrent byte-identical registration-file move.

No registered prediction, class, layer band, statistic, threshold, direction,
seed, permutation count, control, or failure rule was changed after outcome
access. The dtype-incorrect reports in `../corpus_v2/` remain on disk as the
provisional historical record; this report supersedes them for in-situ results.

## Corrected capture and validation

One GPU pass captured four 8,192-token texts through all 66 layers: the two
frozen v2.0 arms (`07_slack_human`, `08_math_llm`), the fresh v2.1 replication
arm (`07b_slack_multi`), and `01_prose_en` for P-d4. It sealed 264 FP16
r-vector arrays and four FP32 next-token-NLL arrays, 268 artifacts total. The
model-forward wall time was 782.241 seconds (13.04 minutes).

The startup gate reproduced installed stock eager attention bitwise for both a
global and a sliding case under BF16 add-before-upcast arithmetic; both maximum
output deltas were exactly zero. The gate uses compact toy shapes: it certifies
the dtype-boundary and mask semantics, not production-size kernel reduction
order.

The first independent validation attempt correctly rejected later checkout
drift after the registration documents were reorganized during the capture.
That failed report is preserved in
[`capture_validation.json`](capture_validation.json). A8 registered the
non-relaxing correction: authenticate capture sources against exact Git blobs
at the manifest's `capture_git_head`, while reporting current-checkout drift
separately. The second attempt, recorded in
[`capture_validation_v2.json`](capture_validation_v2.json), passed with no
errors after independently:

- rehashing all 33 production checkpoint shards;
- rehashing and validating all 268 artifacts, including shapes, dtypes, and
  finiteness;
- checking NLL target alignment; and
- authenticating the exact capture-time source tree at `7dc4460`.

The manifest SHA-256 is
`312a057a389fd2c0c47c2d2ef9dd6d5abfdb111a432aad8b19e68cd6a641d84e`.
The only current-checkout drift reported is `corpus_v2_capture.py`, whose
capture-time blob is authenticated; its later change only updates registration
paths.

## P-v2-1 — novelty gate: pass

| Arm | Mean NLL (nats) | Status |
|---|---:|---|
| `07_slack_human` | 2.557112 | registered gate arm |
| `08_math_llm` | 1.616911 | registered gate arm |
| `07b_slack_multi` | 2.571714 | nonregistered descriptive arm |
| `01_prose_en` | 0.113929 | nonregistered descriptive arm; memorization caveat |

Both registered arms exceed 1.5 nats, Slack is more surprising than math-LLM,
and neither arm reaches the below-1.0 replacement rule. P-v2-1 passes and the
two arms are eligible for the aperture readouts.

## Corrected corpus-v2 readouts

### P-v2-2 — aperture-surprisal law: pass

| Arm | Median within-bin Spearman | Positive bins | Decision |
|---|---:|---:|---|
| Slack human | 0.148028 | 16/16 | pass |
| Math LLM | 0.120362 | 13/16 | pass |

Both satisfy the registered median-rho-above-zero and at-least-12/16-positive
rule. The crossed pairings remain null: math aperture with Slack NLL has median
rho `-0.001033` and 8/16 positive bins; Slack aperture with math NLL has median
rho `+0.001092` and 8/16 positive bins. P-v2-2 and its control pass.

### P-v2-3 — message-start widening: pass

The 311 frozen Slack message starts have aperture effect `+0.051172` with
one-sided Holm-adjusted `p = 0.00159984`. Applying the same mask to math-LLM is
null (`effect = -0.010547`, `p_holm = 1.0`). P-v2-3 and its control pass.

### P-v2-4 — pronouns and function words narrow: fail jointly

| Class | N | Effect | Holm p | Class decision |
|---|---:|---:|---:|---|
| Pronouns | 946 | -0.039844 | 0.00059994 | pass |
| Function words | 2,016 | -0.004297 | 0.651335 | fail |

Pronouns remain significantly narrow on genuinely novel casual text, so the
registered pronoun memorization falsifier does not fire. Function words do not
support the broader closed-class claim. Because the registered prediction
requires both classes, P-v2-4 fails. Both crossed math-arm controls are null
(`p_holm = 1.0` for pronouns and `0.389961` for function words).

### Effect of the A6 arithmetic correction

| Quantity | Provisional | Corrected | Decision changed? |
|---|---:|---:|---|
| Slack mean NLL | 2.558635 | 2.557112 | no |
| Math-LLM mean NLL | 1.613229 | 1.616911 | no |
| Slack median aperture-surprisal rho | 0.152384 | 0.148028 | no |
| Math-LLM median aperture-surprisal rho | 0.114283 | 0.120362 | no |
| Message-start effect | +0.043359 | +0.051172 | no |
| Pronoun effect | -0.046094 | -0.039844 | no |
| Function-word effect | -0.005859 | -0.004297 | no |

A6's public prediction that the arithmetic correction would not reverse the
registered outcomes is borne out: the scorecard remains 3/4, with P-v2-4's
same pronoun-pass/function-word-fail decomposition.

## Depth-resolved registered readouts

The fixed shallow-global band is `{L17, L23, L29}` and the fixed deep-global
band is `{L35, L41, L47, L53, L59}`. Each effect below is the median of
per-layer position-matched effects. Holm correction is across P-d1 through
P-d4.

| Prediction | Registered test | Observed | Holm p | Decision |
|---|---|---:|---:|---|
| P-d1 | speaker labels: deep >= +0.10 and shallow <= 0 | deep +0.132812; shallow -0.082031 | 0.00039996 | pass |
| P-d2 | pronouns: shallow <= -0.05 and terminal abs(effect) < 0.03 | shallow -0.093750; terminal +0.040039 | 0.00039996 | fail |
| P-d3 | Slack first-content: deep <= -0.05 | deep +0.083984 | 1.0 | fail |
| P-d4 | prose sentence starts: deep >= +0.05 | deep +0.173828 | 0.00039996 | pass |

All source-arm random-position, math random-position, and crossed-math
true-null controls pass.

The frozen v2.1 sidecar contains 232 unit starts and 231 valid `+2`
first-content positions. A truncation-boundary artifact leaves one token in an
overlap between two registered classes; it was disclosed before execution and
retained rather than edited after registration.

P-d1 cleanly confirms the delimiter depth flip: speaker labels are narrow in
the shallow band and strongly wide in the deep band. P-d2 confirms strong
shallow pronoun narrowing, but fails its second conjunct because the effect is
`+0.040039` at `{L53, L59}`, outside the registered strict `< 0.03` collapse
threshold. This is a threshold failure despite a highly significant shallow
effect, not an absence of the shallow phenomenon.

P-d3 is falsified in the opposite direction. Slack first-content effects by
deep layer are `-0.052734, +0.076172, +0.083984, +0.083984, +0.099609` at
`L35, L41, L47, L53, L59`: the shallow narrowing does not persist, and the
deep band widens. P-d4 formally passes, with prose sentence starts strongly
wide deep. But its proposed prose-versus-chat contrast is not supported,
because Slack first-content also widens deep once P-d3 reverses. The formal
P-d4 result and the failed joint mechanistic rationale should therefore be
reported separately.

## Independent confirmation

[`verification.json`](verification.json) records a clean independent pass. The
verifier recomputed all v2 and depth effects, permutation p-values, Holm
adjustments, controls, thresholds, and decision booleans, and independently
recomputed nine selected aperture positions from raw r-vectors and projections
in every one of the 32 aperture files. It reproduced both scorecards exactly
and reported no errors. The compact primary records are
[`novelty.json`](novelty.json),
[`readouts.json`](readouts.json), and
[`depth_readouts.json`](depth_readouts.json).

## Scorecards

- Corpus v2: P-v2-1 pass, P-v2-2 pass, P-v2-3 pass, P-v2-4 fail (3/4).
- Depth resolved: P-d1 pass, P-d2 fail, P-d3 fail, P-d4 pass (2/4).
- Every registered true-null control passes.
- Independent capture validation and independent readout confirmation pass.

Private texts, token IDs, sidecars, frozen class positions, NLL arrays,
r-vectors, and aperture dumps remain gitignored and are not included in the
public result commit.
