# Round 5 P-e preregistration — boundary context-dose response

This document must be committed before construction, capture, or inspection of
any fresh P-e outcome data. The commit timestamp is the registration event.

## Priority and scope

P-e is gated on a fresh paired corpus and corrected capture, so it does not
displace any row currently listed as runnable-now in `QUESTIONS.md`. The
prediction is frozen now because the hypothesis was exposed by completed
Figure 2 synthesis; execution remains behind the runnable-now queue.

## Prior disclosure

The following observations are known before registration and are
hypothesis-generating only:

- corrected deep-band speaker-label widening is `+0.240234375` on the v2.0
  single-thread arm and `+0.1328125` on the v2.1 multi-conversation arm;
- exploratory deep-band aperture percentiles were approximately `0.58` at
  ordinary message boundaries and `0.66` at channel boundaries, with some
  channel-boundary values as high as `0.94`.

The v2.0 and v2.1 arms and all exploratory channel/message readouts are
therefore excluded from P-e hypothesis testing. They may appear only in a
clearly labeled historical comparison.

## Mechanistic hypothesis

Deep-global boundary widening is a context-dose response: it increases with
the amount of accumulated prior context that a boundary makes old, closes, or
demotes. A single long thread should therefore widen more strongly at message
boundaries than matched short conversations, and higher-scope
channel/conversation boundaries should widen more than ordinary message
boundaries when they retire more context.

For P-e, **retired-context dose** is fixed before capture as the token count
from the opening delimiter of the segment being closed to the tested boundary.
The dose is computed from frozen structural metadata only; no aperture value,
loss, or hidden-state statistic may affect boundary selection or dose bins.

## Fresh paired design

Freeze a new held-out message pool with no message or conversation overlap
with `07_slack_human` or `07b_slack_multi`. Source (feasibility measured
before registration): the group/public channels (`C*.json`) of the private
archive, never used by any prior arm — 19 channels with human text, 1,008,273
chars total, largest single channel 807,211 chars across 11,599 messages and
15 distinct users. The DM pool is exhausted (3,823 unused chars). Speaker
overlap with prior arms cannot be excluded — the archive owner participates
throughout — and is disclosed rather than prohibited; pseudonymization is
per-conversation as in v2.1, so no speaker identity is shared across renders
at the token level. Render the same ordered messages in two 8,192-token arms,
matching token budget and speaker-label syntax as closely as tokenizer
constraints permit:

1. **single-thread:** one continuous conversation (one pseudonym map for the
   whole arm);
2. **multi-conversation:** the same messages partitioned into multiple
   conversation units, with partition points fixed before capture.

**Partition render rule (frozen):** a partition point is rendered exactly as a
channel switch is rendered in v2.1 — the pseudonym map resets (speakers
re-letter from A in each unit) and the sidecar records the unit boundary.
No separator tokens are inserted; text is otherwise identical between renders.
The single-thread render differs only in keeping one pseudonym map.

Freeze ordinary message starts, higher-scope conversation/channel starts,
retired-context dose, and paired-message identifiers before any corrected
aperture capture. Record and report any token-position mismatch between paired
renders; do not repair pairs using outcomes.

The aperture instrument is unchanged: within each 256-token position bin,
convert corrected aperture values to midrank percentiles. The deep-global band
is fixed as `{L35, L41, L47, L53, L59}`. A boundary's token-level deep score is
the mean of its five per-layer percentiles. The class-level effect remains the
median of per-layer class medians minus `0.5`, matching the corrected corpus-v2
figures.

## Registered predictions

- **P-e1 — primary dose slope.** Across fresh boundaries, deep-band boundary
  score has positive Spearman association with `log2(1 + retired-context
  dose)`. Test one-sided with 10,000 deterministic structure-block
  permutations; the permutation seed is derived from the committed corpus
  freeze hash. Report the coefficient and interval even if the test fails.
- **P-e2 — paired arm consequence.** On paired ordinary message starts, the
  single-thread class-level speaker-label effect is greater than the
  multi-conversation effect. Test the paired difference one-sided by permuting
  arm labels within paired messages. The observed `+0.240` versus `+0.133`
  values set direction only and are not pooled with the fresh estimate.
- **P-e3 — boundary-scope consequence.** Before dose adjustment, fresh
  higher-scope channel/conversation boundaries have a greater median deep-band
  score than ordinary message boundaries. As a mechanistic discriminator,
  also report the scope contrast after exact dose-bin matching: attenuation of
  the raw scope contrast supports a dose account; persistence supports a
  boundary-type account. This adjusted contrast is diagnostic, not a rescue
  criterion for P-e1.

P-e1 is the sole primary verdict. P-e2 and P-e3 are preregistered directional
consequences and are Holm-adjusted as a two-test secondary family. No threshold,
band, dose definition, pairing rule, or boundary subset may be changed after
outcome access.

## Controls and failure handling

Cross every frozen boundary mask onto the existing corrected `08_math_llm`
capture at the same absolute positions (control-only reuse, as in the P-d
family) and include position-matched random-token masks. Report the
shallow-global `{L17, L23, L29}` profile as a depth control, but do not substitute
it for the registered deep band. If paired rendering cannot preserve at least
80% of message-start pairs, stop before capture and amend the design; do not
lower the pairing requirement after inspection. Null or reversed results are
reported as P-e failures, not as grounds for redefining context dose.
