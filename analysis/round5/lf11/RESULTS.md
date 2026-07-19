# LF11 — null-instrument calibration: ANSWERED, prediction passed (fresh specimen caught)

**Question** (ROUND5_LEFTFIELD_SPEC.md): the certified null is an asset —
Inkling's transport channels are negative controls for structure detectors.
Deliverable: a frozen null-benchmark bundle. **Registered prediction:** at
least one more of our own standard tools, run naively against the bundle,
reports spurious structure. Confidence was high; the value is *which* tool.

**Prediction PASSED, with numbers.** The damped-sinusoid-vs-exponential BIC
comparison — our own Round-4 family-race component, run exactly as a careless
analyst would (no cycle-count check) — "detects oscillation" in **8/11 global
layers**, with ΔBIC up to 584. Every winning fit is a fractional-cycle
phantom: fitted periods 727 to 9.2×10⁷ tokens inside a 392-token window
(0.004–0.54 cycles). The one-line audit rule — *require at least one full
fitted cycle inside the window* — kills all eight. **0/11 survive.**
[`naive_demo.json`](naive_demo.json).

This is the fourth tool caught by the null (after C4's free constant, C5's
gauge inflation, and P3's fractional-cycle wins), and the first caught
prospectively by the frozen bundle rather than during an audit.

## The bundle

[`bundle.json`](bundle.json): 148 weight files (66 trunk + 8 MTP layers ×
{proj bank, wr_du}) with SHA-256 hashes, plus:

- **Certified absent** (what a detector must NOT find): oscillation,
  cross-head rate ladders, power-of-two pips, corpus-MI shape tracking —
  each with its resolution statement and evidence pointer.
- **Certified present** (positive controls a detector SHOULD find): the
  d=512 echo, the L65 wall, the paragraph crest, the two near-field motifs.
  A detector that misses these is blind; one that finds more than the
  absent-list allows is hallucinating.
- **Calibration rules** distilled from every failure this campaign caught:
  cycle-count gates, window-resolvable extrapolation, uniform-relative
  thresholds, independent re-derivation before promotion.
