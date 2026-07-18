# Round 5 certified dump-science execution plan

This document is an outcome-blind operational amendment to the Round 5
registration. Its commit timestamp is the execution-plan registration event.
It fixes estimators and verdict mappings that the original question-level
registration left qualitative. It does not change any scientific prediction,
corpus, layer band, channel, control, or failure disposition.

At the time of this amendment, the only widened-capture values inspected were
the independent integrity report and two meter schemas (array names, shapes,
dtypes, metadata, and no outcome summaries). The registered outcomes below
have not been computed. Previously disclosed peeks remain disclosed: the
channel-4786/3290 approximate lifecycle, the four-arm L53 mean-r cosine anomaly,
the old-arithmetic LF4/corpus-v2 results, and all peeks listed in the parent
registrations.

## Frozen input and execution order

The sole confirmatory input is the independently certified capture at
`dumps/round5/widened_corrected_capture/`, manifest SHA-256
`2fc2ec9dd4d6b684a473847f6fbd4541d4326b49d7d6456b463ee5722399a75f`.
Every outcome script must require the public validation report to say
`passed=true`, `D4_satisfied=true`, zero state non-finites, and the same
manifest hash. The six v1 texts are confirmatory; the two paired arms are used
only by P-e/P-f and labeled descriptive R5-C replications.

Execution order is: unchanged mechanical A6 re-certifications; targeted R5-C
channel lifecycle; broad R5-C; LF3; LF8 (with L53 printed first); LF9; R5-B;
then the already fully specified P-e/P-f analyses. No result may silently
overwrite an existing artifact. Each row writes a machine-readable dump,
source/input hashes, and a human-readable result before ledger promotion.

Unless a parent registration fixes a seed, deterministic resampling seeds are
the first eight bytes of SHA-256 of
`<capture-manifest-sha256>:<test-name>`, interpreted unsigned big-endian. Holm
means the standard step-down adjustment. All reported inequalities are
evaluated without rounding.

## R5-C targeted channel lifecycle

Channels 4786 and 3290 are zero-indexed. `hidden_Lxx_TEXT.npy` is the residual
output of layer xx. BF16 payloads are decoded losslessly to float32; no value is
clipped, imputed, or deleted. For every layer, text, and channel, compute over
all 8,192 tokens: signed mean, median, RMS, median absolute deviation about the
median, absolute q90/q95/q99/q99.9/max, variance divided by the sum of all
6,144 coordinate variances, and coverage `mean(abs(h) > 30000)`. Cross-text
dispersion is both the six-text range and MAD for every statistic.

The broadcast-onset statistic is coverage. For each channel separately, a
candidate layer l is sustained on a text when the three adjacent changes
l-1->l, l->l+1, and l+1->l+2 are all strictly positive. Its onset is the first
l in L23--28 sustained by at least four of six texts; absence is a failure for
that channel. Alignment is reported against the already registered flip band,
not used to move it.

The L39->40 handoff statistic is signed mean activation. A text is `sharp` in
a 256-token-block bootstrap replicate only when L39->40 is the unique largest
absolute adjacent change for each channel among transitions L35->36 through
L43->44 and the two L39->40 changes have opposite signs. Use 5,000 resamples
of the 32 aligned blocks. A text supports `sharp` when at least 95% of its
replicates are sharp. The registered aggregate verdict is sharp only with at
least five of six supporting texts; otherwise it is gradual/mixed. Ties fail
the unique-maximum condition. Full trajectories and percentile intervals are
reported whatever the verdict.

## R5-C broad hidden-state geometry

For layer L, its input is `hidden_embed` at L0 and `hidden_L(L-1)` otherwise;
its output is `hidden_LL`. Per text and layer, use all tokens for median and
mean L2 norm and for the median tokenwise residual rotation
`1 - cosine(input, output)`.

Intrinsic-dimension calculations use the fixed 128-token stratified sample per
text: offsets 31, 95, 159, and 223 within each non-overlapping 256-token bin.
After coordinate centering, participation ratio is
`(sum(lambda))^2 / sum(lambda^2)` from the exact sample Gram matrix. The k-NN
estimate uses Euclidean distance, k=10, local Levina--Bickel estimates
`[(1/(k-1))*sum(j=1..k-1, log(T_k/T_j))]^-1`, and their median. A zero neighbor
distance is an integrity failure, not a dropped sample.

The positional carrier is communal basis component 1 in
`analysis/subspace_anatomy/common_bases_top4.npz`, using the same layer index as
the residual output. Its hidden-variance share is the all-token variance of
the scalar projection divided by total centered hidden variance. The registered
`<1%` prediction passes strictly only if every layer's six-text median is below
0.01; the maximum cell is also reported. The advertised read-energy contrast
uses the already frozen `live_share` field from the same basis artifact. Angle
to the bulk is the carrier's principal angle to the top-32 state PCs on the
fixed sample, computed from the exact Gram eigensystem.

For the flip-band prediction, local/global scope-change boundaries are excluded
so they cannot manufacture the result. For participation ratio and median
rotation separately, take the absolute adjacent change in the six-text median
at every remaining same-scope boundary. A registered discontinuity is detected
when either metric's unique largest such change lands at a destination layer
in L13--28 and its signed direction agrees in at least four of six texts. Ties
fail. All ranked changes are retained.

For the global-rotation prediction, each text contributes median rotation over
the 11 global layers minus median rotation over the 55 local layers. Use the
mean of the six paired differences and the exact 64-way one-sided sign-flip
test. The prediction passes when the observed mean is positive and p <= 0.05.
Neighbor-local contrasts for every global layer are descriptive diagnostics.
BOS/sink geometry is reported as the first-64-token mean-state displacement
from the remaining-token mean, without an additional registered verdict.

## LF3 absolute-position counter

Flatten each r-vector to 1,024 coordinates. `06_random` is primary and
`01_prose_en` is the registered control. Report per-coordinate Pearson
correlation with token index on all tokens. For the tail test, average within
the 127 complete 64-token blocks after positions 0--63 and correlate each
coordinate with both block midpoint and `log1p(midpoint)`.

The global-counter statistic is the maximum absolute tail correlation over
both regressors, all coordinates, and all 66 layers. Its exact null circularly
shifts the 127 position-block labels through every 126 nonzero shift, applying
the same shift at every layer. The plus-one randomization p-value therefore
controls the entire search. `No global counter` passes when p > 0.05. The full
sorted squared-correlation spectrum is the registered position-explained
variance spectrum.

For localization, each layer ranks the Euclidean displacement of the first
64-token mean from the other-token mean among the analogous displacements of
all 128 blocks. A BOS transient is present when the median first-block
percentile across layers is at least 0.95. LF3's prediction passes only when
the transient is present and the random-arm global-counter test passes. Prose
is reported identically but cannot rescue the random-arm verdict.

## LF8 fiber orientation

For each layer and v1 text, flatten r to 1,024 coordinates and take its
all-token mean. Content stability is every one of the 15 six-text pairwise
cosines. The registered `>0.9 at every layer` statement passes only if the
minimum of all 990 values is strictly above 0.9. L53 is computed and printed
before any other layer because its old-arithmetic anomaly was disclosed.

For depth families, cosine-normalize the 66 six-text-mean vectors. The statistic
is `0.5*(mean GG + mean LL) - mean GL`. Its depth-matched null independently
chooses one pseudo-global from each consecutive six-layer block (L0--5 through
L60--65), 10,000 times; the registered split passes for a positive statistic
with one-sided p <= 0.05.

For chirality, compute Fisher skewness for each centered coordinate and text.
At each layer, run two-sided one-sample t-tests over the six text-level
skewnesses, Holm-adjust across 1,024 coordinates, and require both adjusted
p < 0.05 and absolute median skewness > 0.25 for a candidate. The `no chirality
beyond the mean` prediction passes only if there are zero candidates across
all layers. This census is explicitly low-confidence as registered.

## LF9 long-range bandwidth

For each layer, text, head, and condition, normalize the meter's distance-mass
vector to sum one. Far-field share is probability at d > 256. Distance entropy
is `-sum(p*log(p))` over nonzero cells and effective count is its exponential;
also report the same entropy conditional on d > 256. Aggregate first over
heads by median, then over the six texts by median.

The registered depth prediction passes when the global layer with maximum
with-bias far share lies in L23--47 and L65 is the unique minimum among global
layers. The registered bias-direction prediction passes when the with-minus-
without far-share effect is positive at every one of L23, L29, L35, L41, L47
and negative at both L5 and L65. Ties or zero effects fail. All layers, texts,
heads, conditions, entropies, and contrasts remain in the dump.

## R5-B prompt dependence

For each text, layer, and head, the live bias curve is the all-token mean
r-vector multiplied in float64 by that layer's frozen 16-by-extent projection.
For curves a and b, normalized pairwise distance is
`||a-b||_2 / (0.5*(||a||_2+||b||_2))`; zero denominators are integrity failures.
The layer dispersion is the median over 64 heads and 15 text pairs.

The depth prediction passes when the unique maximum layer dispersion lies in
L23--47, the median dispersion in L23--47 exceeds that in L0--5, and L65 is
below every global-layer dispersion in L23--47. For text centrality, distance
to the six-text arithmetic-mean curve is computed per head and summarized by
the median over all layers and heads. The registered ordering passes only when
random is the unique nearest text and code the unique farthest. Report both
component verdicts and their conjunction.

## Unchanged rows and confirmation discipline

LF4 and the other A6 re-certifications retain their already frozen statistics;
only their input paths and provenance gates change to the corrected capture.
P-e and P-f retain every estimator, band, seed rule, multiplicity family, and
control in their dedicated preregistrations. R5-D is a later GPU campaign and
is not amended here.

As required by the parent registration, LF3/LF8/LF9 are answered by the first
analyst's artifacts but are not promoted to `certified` until an independent
raw-dump re-derivation and controls agree. Failures and inverted results are
kept under the registered names; they do not trigger threshold or band edits.
