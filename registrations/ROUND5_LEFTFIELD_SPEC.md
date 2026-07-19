# Round 5 — Core Program + Left-Field Curiosity Register (pre-registered 2026-07-16)

## Round 5 core: four questions

### R5-A. The Atlas — map the actual structure, end to end (dumps only, no GPU)

One navigable object unifying everything measured so far: per (layer, head) —
kernel family + fitted rates, near-field class, aperture (far-field share), seam
sign, weight-vs-live energy, communal-carrier alignment; per layer — basis-family
membership, long-range capacity, content sensitivity (R5-B), flip-band membership.
Deliverable: `analysis/round5/atlas.json` + a single poster figure (66×64 tiles ×
~8 channels) telling the depth story end to end: rising band (L0–5) → flip band
(L13–28, humped kernels) → decay regime → deep-global hump cluster → L65 wall.
The Atlas is descriptive glue — no new claims, but every cell cites its dump.

### R5-B. Prompt-dependence — does the structure change with input?

The table is fixed; the *realized* transport b(d; v) is input-dependent. From the
per-text Tier-2 meters + captured r-vectors: per layer, cross-text dispersion of the
live bias curves (normalized pairwise distance over the 6 corpus texts), giving a
depth profile of content sensitivity. Peek disclosure: needles/templated texts were
already seen to damp deep-global seam bias (C1-act per-text tables). Per-token
resolution = LF4. **Prediction:** sensitivity is low at L0–5 (carrier fixed early),
peaks mid-depth (L23–47), and collapses at L65 (the wall is unconditional); random
text sits nearest the layer mean, code farthest. Confidence: medium.

### R5-C. Activation-space geometry — what does it look like in there?

From capture/ (66 layers × 6 texts × 8192 hidden states + r-vectors): per layer —
intrinsic dimension (participation ratio + k-NN estimator), norm growth, adjacent-
layer rotation; the positional carrier's share of hidden variance and its angle to
the bulk; BOS/sink transient geometry (ties LF3); whether hidden geometry shows a
discontinuity at the flip band (the C5 transition seen from the *state* side, not
the weight side) and at local↔global boundaries. **Predictions:** (i) the positional
carrier is a tiny fraction of hidden variance (<1%) despite carrying ~70% of
positional read energy — position rides a narrow, protected channel; (ii) a
detectable intrinsic-dim or rotation discontinuity at the flip band; (iii) global
layers rotate the state more than locals. Confidence: medium on all three.

### R5-D. Ablation campaign — what happens when pieces come out? (Tier-3, GPU)

The causal phase: streaming forward passes (~20 min/arm on the 4090) with
interventions, measuring per-token NLL on the corpus + needle-retrieval logits
(probability of the correct codeword continuation at the 24 recall positions) +
attention redistribution meters. Registered arms:

1. **bias-off(L)** — remove the positional bias at one layer, propagating (unlike
   the Tier-2 counterfactual, which never propagated): 11 globals + 5 sampled locals.
2. **carrier-out(L)** — project only communal component 1 out of the r-vectors;
   tests whether the scalar carrier IS the functional channel (vs the other 15 dims).
3. **near-vs-far** — zero b(d<4) vs zero b(d>128) at the same layers; which part of
   the kernel carries the loss?
4. **heal the wall** — extend the table past d=1024 with the fitted decay tail at
   global layers; is the hard zero load-bearing or incidental at 8k context?
5. **head-class arms** — rising heads off in early locals; negative-seam heads off
   at L11.

**Predictions:** single-layer bias-off barely moves NLL (redundancy) except at the
L0–5 band and L65 (ΔNLL > 0.05 nats there); carrier-out(L) ≈ bias-off(L) within 20%
(the scalar carrier is effectively the whole channel); near-field ablation dominates
far-field by ≥5× in ΔNLL; healing the wall changes far-field attention mass but
moves NLL < 0.005 nats at 8k (the wall is incidental at this context length — its
cost should appear only beyond ~10k). Confidence: medium-high on near≫far, medium
on the rest. Dump-first: every arm dumps its meters + logit diffs before any verdict.

---

# Left-field register (LF1–LF11)

Premise: Rounds 1–4 certified the learned transport as a **true null** for structured
positional mechanisms — no oscillation anywhere (fractional-cycle wins audited out),
no cross-head parameter ladder (rate spread 2.7× vs RetNet's 128×), decay-only,
low-rank. A certified null is an instrument: against a flat background, *any*
structure that survives a control is signal. This register hunts in left field on
purpose. The commit timestamp of this file is the pre-registration.

Ground rules (as Rounds 3–4): dump-first; predictions registered blind below, before
any targeted analysis; every test names its null/control; analyses read from existing
dumps where possible (weights/, dumps/round3/, dumps/tier2/ + capture/) — items
needing new compute say so.

## Prior peeks (disclosure)

During the Round-4 audits, the following adjacent readouts already happened and are
NOT blind: (a) head-mean bias **band steps** (16-token windows) at d ∈ {256, 320, 384,
448, 512, 576, 640, 704, 768} — only d=512 was sign-consistent (33/33); (b) FFT of
detrended mid-range bias curves (no crisp period); (c) two-rate fits of mode-0 curves
(rates recorded in analysis/round4/). Tests below are framed to ask *new* questions,
noted per item where they touch a peeked quantity.

---

## LF1 — Architectural numerology: single-distance pips at powers of two

**Q.** Does the table treat exact power-of-two distances specially — single-distance
discontinuities (pips) at d ∈ {32, 64, 128, 256, 2048…}, as opposed to the band steps
already examined? Training data and architecture are saturated with powers of two;
the 512 echo proves architecture can imprint. **Test.** Per global layer, z-score of
|b(d) − local median| at each power of two vs the distribution over all d; multiple-
comparison controlled. Peek note: band steps at these d were seen (null); *pips* were
not examined. **Prediction (registered):** no pips beyond d=512 survive control —
except 128 is the most likely surprise (the tau constant is 128,000 = 128·1000).
Confidence: medium. **Surprise looks like:** a pip at 128 or 2048 → architecture
constants leak into the learned table at more than one scale.

## LF2 — Linguistic scales: sentence and paragraph knees

**Q.** Token distance is a proxy for linguistic distance. Does the kernel have
curvature knees at text-statistic scales — sentence (~15–40 tokens) or paragraph
(~100–250)? **Test.** Segmented-regression knee detection on log|mode-0| per layer;
null = knee positions from refits of the smooth 2-exp surrogate + jitter. Corpus-side
scale measurement (sentence/paragraph token-length distributions from 01/04) done
FIRST and frozen, so "knee matches scale" is well-defined. **Prediction:** a sentence-
scale knee in ≥6/11 global layers; no paragraph knee (the window barely holds one
paragraph). Confidence: medium-low. **Surprise:** knees at *both* scales, aligned
across layers → the kernel is piecewise-linguistic, not smooth-decay.

## LF3 — A counter hiding in a relative mechanism

**Q.** The r-channel is nominally relative. Did the model sneak an *absolute* position
signal into it? **Test.** From capture/rvec on 06_random (content-free): per-channel
correlation with position; spectrum of position-explained variance; localization
(BOS transient vs global drift). Controls: shuffled-position null; repeat on prose to
separate content drift from position proper. **Prediction:** a position-correlated
component exists but is confined to the first ~64 positions (attention-sink
transient); no global counter. Confidence: medium. **Surprise:** a monotone
log-position channel across all 8k positions → an absolute clock inside the relative
fiber, and a mechanism for the tau regime worth chasing.

## LF4 — The zoom lens: per-token aperture control (flagship)

**Q.** Transport is b(d; v) — every token gets its own curve via its r-vector. Is the
16-dim v-channel a *per-token aperture dial*? Do tokens that need distant context
(pronouns, closing brackets, sentence-initial tokens, rare tokens) flatten their
decay to look farther? **Test.** Per token: aperture = far-field share of its own
bias curve (Σ|b(d)| for d>128 / total), from rvec ⊗ proj, layers pooled by depth
band. Token classes frozen ex ante (pronouns, closers `)]}`, sentence starts,
rare-BPE, function words) from ids alone. Null: class labels shuffled within
position-matched bins. **Prediction:** closing brackets and pronouns sit above the
median aperture in code/prose respectively at mid-depth globals; function words below.
Confidence: medium-high — this is the (ρ, δ, σ) framework's δ(v): content-modulated
decay, and it would upgrade the transport from "a table" to "a controlled instrument."
**Surprise in the other direction:** aperture is content-blind (pure carrier) → the
16-dim interface is even more vestigial than C2 suggested.

## LF5 — Instrument: offline pair-level attention (no GPU pass)

Capture holds every layer's input hidden states; weights hold every Q/K/r projection.
So full attention rows for ANY (layer, query) are recomputable offline on CPU —
pair-level analysis at will, no new streaming pass. **Deliverable:** a verified
`offline_row(layer, text, q)` (parity vs the 24 captured needle rows must be exact to
fp16) + a dump of rows at pre-named loci: bracket-closers (02), sentence starts (01),
heartbeat lines (03), 64 random controls per text. Everything downstream
(LF4-pairwise, bracket matching, LF-deck below) consumes these dumps. **Prediction:**
parity passes; bracket-closers show matched-open attention above distance-matched
baseline at ≥3 global layers (medium confidence).

## LF6 — Power-law mimicry: is the 2-exp kernel a quadrature of MI(d)?

**Q.** Mutual information between tokens in natural text famously decays as a *power
law*, but the model chose exponential-family kernels. Sums of exponentials mimic
power laws over bounded windows. Is F9-with-two-rates a two-point quadrature of the
corpus's MI curve — i.e., did SGD approximate the *right* power law with the *wrong*
family? **Test.** (i) Measure corpus MI(d) (subword MI, d up to 1024, prose+code,
frozen estimator); (ii) fit k-exp mixtures k=1,2,3 and pure power law to mode-0
far-fields; (iii) compare kernel-vs-MI shape (rank correlation on log-log). Peek
note: 2-exp fits exist (rates known); k=1,3 and the MI comparison are new.
**Prediction:** BIC order 2-exp > power > 1-exp on most globals, 3-exp ≉ better than
2-exp; kernel shape tracks MI(d) shape with rank-corr > 0.9 on prose. Confidence:
medium. **Surprise:** 3-exp keeps winning → the kernel is genuinely multi-scale and
"two rates" was our resolution limit, not the model's choice.

## LF7 — MTP parentage: did the drafter fork the trunk?

**Q.** MTP drafter layers have trunk-like transport. Are they *copies* of specific
trunk layers — a fork at some depth — or fresh solutions of the same shape? **Test.**
Procrustes/subspace distance from each MTP layer's composed operator set to every
trunk layer's; nearest-parent profile; null = distances among unrelated trunk pairs.
**Prediction:** MTP layers' nearest parents cluster in the deep trunk (L30+), and all
8 drafters share one parent neighborhood (they're a family). Confidence: medium.
**Surprise:** parents scattered or early → the drafter re-derived transport
independently; the "same shape" is convergent, not inherited — a mini universality
result inside one checkpoint.

## LF8 — Orientation of the fiber: is there an anchor?

**Q.** The live mean r-vector defines an orientation in the 16-dim fiber per layer.
Is that orientation content-stable, and does it form depth families? (The sign/
orientation question asked of a production model.) **Test.** cos(mean-r) across the
6 texts per layer (content stability); across layers (family structure); odd-moment
census of the centered r distribution (is anything *chiral* beyond the mean?).
**Prediction:** content stability cos > 0.9 at every layer; cross-layer structure
reproduces the global/local family split from subspace anatomy; centered odd moments
≈ 0 (no chirality beyond the carrier). Confidence: medium-high on the first two,
low information on the third — that's the point of asking.

## LF9 — The long-range bandwidth budget

**Q.** How many nats of attention does each layer actually spend beyond d=256, and
where does the model concentrate its long-range capacity? **Test.** From tier2
meters: far-field mass share and an entropy-style effective-count per (layer, text);
depth profile; with/without-bias comparison isolates how much capacity the *bias*
buys. **Prediction:** capacity peaks at mid-depth globals (L23–L47), collapses at
L65; the bias *increases* far capacity at mid layers (the contrast-enhancer effect)
while decreasing it at L5/L65. Confidence: medium-high (extrapolates the needle
result; the depth-peak location is the genuinely blind part).

## LF10 — Postcard to 1M context: the wall gets stronger

**Q.** tau = 1 + 0.1·log(pos/128000) multiplies query AND bias on global layers.
A falsifiable forecast anyone with hardware can test later. **Prediction (registered
with numbers):** at pos = 10⁶, tau = 1.205572502, so seam logit steps multiply by
~1.206: layer-5-style with-bias inside/outside mass ratios grow from the measured
1.48–1.65 (8k ctx) to ≈ 1.60–1.83, and the L65 in-extent bias floor rises ~+0.53
logits → the terminal wall's mass cliff steepens from ~8× toward ~13×. Direction:
long-context Inkling gets MORE near-sighted in its final integration, relying even
more on pure content matching for long range. Caveat frozen with the prediction:
assumes content-logit statistics stay comparable at 1M.

## LF11 — Null-instrument calibration (the inversion)

The certified null is itself an asset: Inkling's transport channels are now
**negative controls for structure detectors**. Any oscillation/ladder/holonomy
detector that fires on these channels is exposing its own artifact — we already
caught three published numbers this way (C4's constant, C5's gauge, P3's fractional
cycles). **Deliverable:** a frozen "null benchmark" bundle — the specific channels
+ the certification (what was excluded and at what resolution) — so future detectors
(ours or anyone's) are run here first. **Prediction:** at least one more of our own
standard tools, run naively against the bundle, reports spurious structure.
Confidence: high — that is the track record; the value is finding *which* tool.

---

## Priority and cost

| Item | Data | Compute | Priority |
|---|---|---|---|
| LF4 zoom lens | capture (have) | CPU hours | **1 — flagship** |
| LF5 offline-row instrument | capture+weights (have) | CPU hours | **1 — enabler** |
| LF3 hidden counter | capture (have) | minutes | 2 |
| LF9 bandwidth budget | tier2 meters (have) | minutes | 2 |
| LF6 MI mimicry | corpus+dumps (have) | CPU hours | 2 |
| LF8 fiber orientation | capture (have) | minutes | 3 |
| LF1 pips | tier2+weights (have) | minutes | 3 |
| LF7 MTP parentage | weights (have) | minutes | 3 |
| LF2 knees | dumps (have) | minutes | 3 |
| LF10 postcard | none (forecast) | none | registered |
| LF11 null bundle | curation | none | registered |

Confirmation discipline: an LF item promoted to a claim needs its control to pass AND
an independent re-derivation from raw dumps (Round-4 rule). Everything else stays an
exploratory artifact.
