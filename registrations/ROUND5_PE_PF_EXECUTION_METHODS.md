# Round 5 P-e/P-f execution-method amendment

This amendment freezes the operational details left implicit by
`ROUND5_APERTURE_CONTEXT_DOSE_PREREG.md` and
`ROUND5_APERTURE_ANCHOR_PREREG.md`. Its commit timestamp is the registration
event. At this point the paired corrected r-vector files exist, but no paired
aperture value has been computed or inspected. Pre-frozen class metadata and
the already disclosed v2.1 exploratory profile were used only to validate the
implementation.

All registered bands, thresholds, classes, controls, sample sizes, direction
rules, and multiplicity families remain unchanged.

## Common instrument and artifact gates

- Compute aperture with the unchanged frozen LF4 estimator from the certified
  A6-corrected paired r-vectors.
- Refuse any r-vector, projection, aperture file, private freeze, or capture
  manifest whose shape, dtype, hash, completeness flag, or bound provenance
  differs from the certified inputs.
- Convert aperture to midrank percentiles independently inside each 256-token
  position bin. The shallow band is L17/L23/L29 and the deep band is
  L35/L41/L47/L53/L59.
- Derive every pseudorandom seed from the SHA-256 of the committed public
  paired-corpus freeze plus a named test label.
- Write only aggregate results publicly. The private paired corpus, token
  classes, raw r-vectors, and token-level aperture arrays remain gitignored.

## P-e operational details

P-e1 is primary on the multi-conversation arm. A token's band score is the
mean of its per-layer percentiles. The observed statistic is Spearman rho
between the deep score and `log2(1 + retired_context_dose)`. A structure block
is the frozen `segment_open_token`; dose labels are independently permuted
within the eight blocks for 10,000 one-sided draws. The reported 95% interval
is a 5,000-draw cluster bootstrap that resamples the eight complete structure
blocks with replacement. The shallow-band calculation is a depth control.

The crossed-math control uses the identical boundary rows, doses, and blocks
at the same absolute positions. Because each frozen position-matched random
mask is a set rather than a row mapping, ordinary and higher-scope masks are
each sorted and paired in order with the corresponding sorted boundary rows;
this deterministic mapping is used only for the random control. A token may
occur in both independently frozen random masks. Permutations therefore act
on row identity, not unique token value.

P-e2 uses the registered class effect (median of per-layer class medians minus
0.5) in each arm. For each of the 563 frozen paired ordinary messages, a
10,000-draw one-sided permutation independently swaps its two arm labels; the
test statistic is single-thread effect minus multi-conversation effect.

P-e3 uses the median deep token score for higher-scope minus ordinary
boundaries. Its 10,000 one-sided draws permute scope labels among boundary-row
identities within 256-token position bins while preserving each bin's class
count. The diagnostic dose bins are exactly
`floor(log2(1 + retired_context_dose))`; each higher-scope score is contrasted
with the median ordinary score in its bin, and the median of those contrasts
is reported. The shallow, crossed-math, and frozen-random versions are
controls. P-e2 and P-e3 raw p-values form the registered two-test Holm family.

## P-f operational details

For P-f1, each anchor's main test uses the registered deep-band class effect
and 10,000 one-sided stratified position-mask permutations. Crossed-math and
random-mask controls use two-sided permutations. The three main p-values, the
three math-control p-values, and the three random-control p-values are each
Holm-adjusted within their own family. Each anchor passes only with main
effect at least +0.05, adjusted main p below .05, and neither control detected
at adjusted p below .05. P-f1 requires all three anchors. Its valid
intersection-union p for the top-level family is the maximum of the three raw
main p-values; the internal adjusted criteria remain additional gates.

P-f2 preserves the exact reducer behind the disclosed v2.1 value `+0.18165`:
for each token, subtract the mean shallow percentile from the mean deep
percentile, then take the mean over ordinary message starts. Starts must be
valid for the complete historical offset profile, `-4..+11`, meaning
`start >= 4` and `start + 11 < 8192`; no additional outcome-dependent or
message-length filter is introduced. The offset-0 interval is a 5,000-draw
paired-message bootstrap of that mean. The body statistic is the median of
the six mean differences at offsets 6 through 11. Its top-level p-value is a
10,000-draw one-sided paired-message sign-flip test of the offset-0 mean.

For P-f3, shallow and deep label-colon effects each receive a 10,000-draw
one-sided negative permutation test. The conjunction p-value is the maximum
of those two p-values (an intersection-union test), and both registered
effects must be at most -0.05.

The P-f1 intersection-union p, P-f2 sign-flip p, and P-f3
intersection-union p form the registered three-test top-level Holm family.
Substantive thresholds remain conjunctive gates after adjustment.

## Failure and confirmation discipline

The first sealed output under the source commit implementing this amendment is
the registered result. No failed prediction, control firing, small class, or
borderline p-value permits a rerun with changed methods. Results are marked
answered but pending confirmation until an independent analyst reproduces
them from the sealed aperture artifacts.
