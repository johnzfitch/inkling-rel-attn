# Round 5 — decision D1 resolved: widened corrected capture scope

Decision D1 (`QUESTIONS.md` decision queue) is resolved in favor of widening:
the next GPU capture pass — the same pass that captures the P-e paired arms —
additionally captures, for all six v1 corpus texts
(01_prose_en … 06_random), under the A6-corrected arithmetic with full A8
provenance (git-blob authentication, shard records, package records,
stock-parity gate):

1. per-layer r-vectors (all 66 layers, [8192, 64, 16] fp16), and
2. the Tier-2 distance-marginal meters (with-bias and without-bias softmax
   mass sums, bias and content sums, per layer × text).

## What this closes

- Every artifact currently stamped PROVISIONAL re-certifies: the LF4 aperture
  numbers, the LF5-adjacent in-situ readings, and the meter-based findings
  (d=1024 seam, d=512 echo, L65 wall, heartbeat induction) get corrected-
  arithmetic replacements evaluated with unchanged definitions. Expectation
  (registered in A6, already confirmed on the corpus-v2 side): no verdict
  flips at Δp ≤ 0.025.
- LF3 (r-channel counter, needs 06_random rvec), LF8 (fiber orientation,
  needs all-text rvec), LF9 (bandwidth budget, needs corrected meters), and
  R5-B (content-sensitivity depth profile, needs corrected meters) move from
  gated to runnable the moment the pass lands.

## Rules

Old dumps remain immutable; corrected outputs write new directories. No
threshold, definition, or verdict rule changes anywhere. Evaluation order
after the pass: re-certifications first (mechanical re-runs), then the
newly-unblocked rows under their existing registrations, then P-e/P-f.
