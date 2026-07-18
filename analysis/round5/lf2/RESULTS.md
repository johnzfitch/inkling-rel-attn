# LF2 — linguistic knees: ANSWERED, prediction failed inverted (paragraph crest, no sentence scale)

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
- 10/11 global layers have Holm-significant breakpoints (2-exp-surrogate
  null, 2,000 replicates); L65 (the wall layer) does not. Breakpoints:
  L47 74, L53 56, L35 92, L65 114*, L29 116, L17 132, L23 136, L59 140,
  L41 144 (all in the paragraph range), L11 246, L5 260 (outside both
  ranges — the two farthest-looking early globals live at a longer scale).
- **The "knees" are crests, not decay kinks.** In all 8 paragraph-range
  layers the fitted slope BEFORE the breakpoint is slightly positive
  (log₁₀|mode-0| rising ~+0.0005–0.0014/token) and turns negative after:
  the kernel rises gently to a crest at ~56–144 tokens, then decays. This
  is the weight-level face of the deep-global hump family, now localized:
  **the crest sits at paragraph scale, consistently across mid/deep globals.**

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
