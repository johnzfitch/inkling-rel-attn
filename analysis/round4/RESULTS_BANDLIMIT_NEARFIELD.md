# R4-W (Whittaker band-limit) + R4-N (near-field battery) — ANSWERED

The two ROUND4_SPEC.md sections that were never run. Implemented as
registered; method details frozen in `scripts/round4_bandlimit_nearfield.py`
before outcomes. Inputs: the existing round-3 dumps (mode_curves,
perhead_svd) — weight-level, unaffected by A5/A6.

## R4-W — the table is heavily oversampled but not purely band-limited

- Mode-0 effective bandwidth f90: **median 0.0195 cycles/token** (90% of
  spectral energy below one cycle per ~51 tokens) — against a Nyquist of
  0.5, the table carries its dominant structure at ~4% of available
  bandwidth. Combined with effective rank ~2 of 16: the model uses a thin
  slice of the table's capacity in BOTH rank and bandwidth.
- But sinc reconstruction after decimation is not free: rel. RMSE median
  **7.3% at 2× and 9.1% at 4×** — a genuine small broadband component rides
  on the smooth kernel. The table could be stored at quarter resolution at
  the cost of ~9% RMS distortion, not zero.
- Outliers worth naming: L05 (f90 = 0.352) and L17 (0.305) have broadband
  mode-0 — the far-looking early-global (L5, the seam-sensitive layer)
  spends its curve capacity very differently from the smooth mid-depth
  kernels.

Full table: [`bandlimit.json`](bandlimit.json).

## R4-N — near field: a real handoff break, and exactly two depth-split motifs

- **Handoff discontinuity** (near-field vector vs far-field fit extrapolated
  back, resolvable-component bounds per the C4 lesson): median **0.37**
  far-field-RMS units, q90 1.22, max 3.10 (extreme heads at L48/L64). The
  near field is NOT the smooth continuation of the far family — spec
  prediction P5 ("d<8 is not any smooth family") is **supported**.
- **Motif structure is discrete and binary.** Silhouette selects k=2
  (0.514, well above k=3–8): cluster 0 (n=2094, median depth 17) is
  **"self-suppress"** — d=0 pinned near zero, then a flat plateau; cluster 1
  (n=2130, median depth 49) is **"self-inclusive"** — d=0 present with a
  mild d=1 peak, gently decaying. Shallow layers mostly refuse to attend to
  the current position through the positional channel; deep layers include
  it. Per-layer motif counts are in the JSON for the Atlas.

Full table: [`nearfield.json`](nearfield.json).

## Fitting notes (disclosed)

Two pre-verdict repairs, both recorded: (1) unbounded extrapolation rates let
the fitter hide huge fast components invisible on the fit window that
exploded at d<8 (the C4 free-constant failure mode in miniature) — bounded to
window-resolvable components (r ≤ 0.35, |a|,|c| ≤ 5× window peak);
(2) a NaN-key sort corrupted the top-discontinuity list — filtered.
