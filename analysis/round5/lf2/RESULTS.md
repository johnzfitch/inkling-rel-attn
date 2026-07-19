# LF2 — linguistic knees: ANSWERED, independently certified inverted result

**Question** (ROUND5_LEFTFIELD_SPEC.md): does the kernel have curvature knees
at text-statistic scales — sentence (~15–40 tokens) or paragraph (~100–250)?

**Registered prediction:** sentence-scale knee in ≥6/11 global layers; no
paragraph knee. **FAILED — inverted.** Sentence knees: **0/11**. Paragraph
knees: **8/11** (Holm-significant breakpoints inside the frozen paragraph
range).

## What was found

- Frozen scale ranges (measured and committed BEFORE knee detection, from
  01_prose_en + 04_multilingual): sentence [7, 44] tokens, paragraph
  [21, 160]. Disclosure: the ranges overlap on [21, 44]; no observed
  breakpoint landed in the overlap, so no verdict depends on it.
- **Corrected claim structure (audit 2026-07-17), three tiers:**
  **eight** paragraph-range hinges under the registered IID null (L17, L23,
  L29, L35, L41, L47, L53, L59; median breakpoint 124); **seven** survive
  block-residual sensitivity (L47 loses Holm significance at block sizes 16
  and 32 — the registered IID null is optimistic, residual lag-1
  autocorrelation reaches 0.915; `lf2_block_sensitivity.json`, independently
  matching the auditor's recomputation); **six** are fitted rise-to-decay
  crests — L47 rises on both sides of its hinge (+0.00074 → +0.00027) and
  L59 decays on both sides (−0.00001 → −0.00039), so "all eight are crests"
  (this file's first version) was an overgeneralization.
- L11 (246) and L5 (260) break outside both frozen ranges — the two
  farthest-looking early globals live at a longer scale (L5 also drops just
  below Holm significance under block resampling). L65, the wall layer,
  never reaches significance.

## Consistency with LF6 (answered the same day)

LF6's family race found the far field (d ≥ 32) single-exponential in 9/11
globals — no second decay rate. No contradiction: sums of positive decaying
exponentials can only FLATTEN in log space, never rise or steepen, so the
crest is invisible to that family while the hinge detector sees it. Together:
rise to a paragraph-scale crest, then one clean exponential tail.

## Caveats

- d≈128 adjacency: the crest cluster (56–144, median ~116) brackets 128.
  LF1 rules out a single-distance pip at 128, but a smooth architectural
  scale near 128 cannot be dissociated from a corpus-statistic scale by this
  test alone. A discriminator (e.g., knee position vs corpus-specific
  paragraph statistics across corpora) would need its own registration.
- Resolution fix, disclosed: the first run used 200 surrogates, whose Holm
  floor (11 × 1/201 = 0.0547) cannot reach 0.05 by construction; raised to
  2,000 before any verdict was recorded. Observed statistics identical in
  both runs; only the null resolution changed.

## Provenance

Frozen scales: [`corpus_scales.json`](corpus_scales.json) (written before
detection; refuses overwrite). Dump-first:
`dumps/round5/lf2/lf2_knees.npz` + manifest with input hashes. Full table:
[`lf2_knees.json`](lf2_knees.json). Script: `scripts/round5_lf2_knees.py`.

**Independent verification.** A no-producer-import verifier rebuilt every
log curve from the original weight banks using a Gram eigensystem, implemented
the hinge scan through residualized regressors, regenerated all 22,000 IID
surrogates and all 44,000 circular-block surrogates from their fixed seeds,
and reproduced every breakpoint and decision. Maximum IID-statistic difference
was 9.1e-12; all 22,000 null breakpoint identities matched exactly. The
8 IID / 7 block-robust / 6 crest hierarchy is certified with zero errors:
[`verification.json`](verification.json); script:
`scripts/round5_lf2_verify.py`.
