# LF7 — MTP parentage: ANSWERED (rerun complete), prediction failed → no fork; awaiting independent verification

**Question** (ROUND5_LEFTFIELD_SPEC.md): are the 8 MTP drafter layers *copies*
of specific trunk layers — a fork at some depth — or fresh solutions of the
same shape?

**Registered prediction:** nearest parents cluster in the deep trunk (L30+)
and all 8 drafters share one parent neighborhood. **FAILED.** The registered
surprise branch is what the data shows: **the drafter re-derived transport
independently — convergent shape, not inheritance.**

## Rerun history (all disclosed)

Three dump passes: K_STORE=64 (bug: silently degraded the primary metric),
640 (bug, caught by external audit: trunk energy-90% ranks reach 776, so
406/2,145 trunk-trunk null pairs used truncated bases with an unmatched √k
normalization), and the final 1024 (full row-space bound; `chordal()` now
raises on any k beyond the stored basis; the dump stores the full bases so it
is self-contained for independent re-derivation). As the audit predicted:
MTP-to-trunk distances are identical across the 640/1024 runs (MTP ranks top
out at 616); only the null cleaned up (median 0.948 → 0.945, p05 0.901).

## Evidence (final numbers, K_STORE=1024)

**Hidden-side read subspaces: no inheritance, by two independent metrics.**
- Primary (chordal distance between energy-90% right-singular subspaces of
  `wr_du`, k = min(k90ᵢ, k90ⱼ)): every MTP layer's *nearest* trunk layer sits
  at the **55th–73rd percentile** of the 2,145-pair trunk-trunk null
  (distances 0.948–0.958 vs null median 0.945). No MTP distance approaches
  the fork threshold (5th percentile, 0.901).
- Independent sketch metric (Frobenius cosine of fixed-seed sketched Grams):
  agrees the landscape is flat and picks entirely different argmins (0/8
  agreement) — "parent identities" are noise over a featureless landscape.
  The frozen promotion gate for any parent-identity claim correctly fails.

**Curve-side bank shapes: a strong secondary observation, NOT promoted.**
All 8 MTP proj banks are closer (16-dim row-space angle, shared d<512 window)
to one deep-trunk bank — L51 (deep local) ×7, L47 (global) ×1 — than ANY two
trunk layers are to each other. The mode-0-cosine re-derivation does not
confirm the neighborhood (1/8), but its null is saturated (p95 = 0.996 — the
decay-only result again), so it discriminates poorly. Stays an observation;
a sharper discriminator (decay-mode-excluded angles) would need registering.

## Status

Verdict on record; the LEFTFIELD promotion rule still requires an independent
re-derivation from the (now self-contained) dump by a second analyst before
"certified". The dump carries full bases, spectra, all four distance
matrices, and input hashes: `dumps/round5/lf7/lf7_dump.npz` + manifest.
Tables: [`lf7_parentage.json`](lf7_parentage.json),
[`lf7_rederivation.json`](lf7_rederivation.json).

## Interpretation

The drafters did not fork the trunk: they rebuilt qualitatively the same
transport (decay-only, same family — Round 2 E4) on hidden-space read
directions that share nothing measurable with any trunk layer's. A mini
universality result inside one checkpoint: the kernel shape is the attractor;
the coordinates implementing it are incidental.
