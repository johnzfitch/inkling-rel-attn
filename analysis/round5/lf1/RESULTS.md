# LF1 — power-of-two pips: ANSWERED, clean null (prediction passed)

**Question** (ROUND5_LEFTFIELD_SPEC.md): does the learned table treat exact
power-of-two distances specially — single-distance discontinuities (pips) —
given that training data and architecture are saturated with powers of two and
the d=512 echo proves architecture can imprint?

**Registered prediction:** no pips survive multiple-comparison control; d=128
flagged in advance as the most likely surprise (tau constant 128,000).

**Verdict: no pips anywhere. Prediction PASSED.**

- 66 global-family tests (11 global layers × d ∈ {16, 32, 64, 128, 256, 512})
  and 275 local-family tests: **zero Holm-significant survivors** in either
  family.
- d=128, the pre-flagged suspect: strongest layer L17 at raw p = 0.040,
  Holm 1.0. Nothing.
- Strongest candidate anywhere: L17 at d=256, raw p = 0.013, Holm 0.86 —
  exactly what 341 tests of a null should produce.
- The known d=512 echo does NOT appear as a pip (all raw p ≥ 0.15): the echo
  is a band-level step, not a single-distance discontinuity. The two
  statistics dissociate cleanly, which is the right behavior for both
  findings.

**Method** (frozen in the script header before outcomes): pip(d) =
|c(d) − median(±8 window excl. d)| / MAD(window) on each layer's mode-0 proj
curve; empirical p against all eligible d; Holm within family. Dump-first:
`dumps/round5/lf1/lf1_pips.npz` (+ per-row bank series for the secondary
read), manifest with input hashes. Full test table:
[`lf1_pips.json`](lf1_pips.json). Script:
`scripts/round5_lf1_pips.py`.

**Interpretation.** The table's smoothness is real, not an artifact of coarse
statistics: even at single-distance resolution, SGD left no architectural
numerology in the learned kernel. Combined with the Round-4 battery this
tightens the "shape without structure" picture — the only distance the table
singles out is 512, and only as a scope handoff, not a pip.
