# LF7 — MTP parentage: REOPENED (audit 2026-07-17) — rerun in progress

**Audit findings that reopened this row:** (1) K_STORE=640 still truncated
the primary null — trunk energy-90% ranks reach 776, so 406/2,145 trunk-trunk
null pairs were computed on truncated bases with an unmatched √k
normalization (MTP ranks top out at 616, so the MTP-trunk distances
themselves were intact); (2) the dump did not store the bases it claimed;
(3) the dump-manifest source hash predates the committed script. Rerun at
K_STORE=1024 with a self-contained dump is in progress; the fixed-k and
sketch metrics — both unaffected — already support "no close parent", so the
conclusion is expected to survive, but nothing below is citable until the
rerun replaces these numbers and an independent verifier confirms.

---

# (superseded pending rerun) LF7 — MTP parentage: prediction failed → no fork

**Question** (ROUND5_LEFTFIELD_SPEC.md): are the 8 MTP drafter layers *copies*
of specific trunk layers — a fork at some depth — or fresh solutions of the
same shape?

**Registered prediction:** nearest parents cluster in the deep trunk (L30+)
and all 8 drafters share one parent neighborhood. **FAILED.** The registered
surprise branch is what the data shows: **the drafter re-derived transport
independently — convergent shape, not inheritance.**

## Evidence

**Hidden-side read subspaces: no inheritance, by two independent metrics.**
- Primary (chordal distance between energy-90% right-singular subspaces of
  `wr_du`): every MTP layer's *nearest* trunk layer sits at the 49th–73rd
  percentile of the 2,145-pair trunk-trunk null — i.e. no trunk layer is any
  closer to a drafter than a median pair of unrelated trunk layers. No MTP
  distance approaches the fork threshold (5th percentile, 0.901; MTP minima
  0.948–0.958 vs null median 0.948).
- Independent sketch metric (Frobenius cosine of fixed-seed sketched Grams):
  agrees the landscape is flat (best similarities at the 53rd–56th null
  percentile) and picks entirely different argmins (0/8 agreement) — the
  "parent identities" are noise over a featureless landscape. The frozen
  promotion gate for any parent-identity claim correctly fails.

**Curve-side bank shapes: a strong secondary observation, NOT promoted.**
All 8 MTP proj banks are closer (16-dim row-space angle, shared d<512 window)
to one deep-trunk bank — L51 (deep local) ×7, L47 (global) ×1 — than ANY two
trunk layers are to each other (0.667–0.682 vs trunk-trunk minimum 0.698;
0.0th percentile). The registered independent re-derivation (mode-0 cosine,
`lf7_rederivation.json`) does **not** confirm the neighborhood (1/8) — but its
null is saturated (p95 = 0.996; mode-0 curves are near-collinear trunk-wide,
which is just the decay-only result again), so it discriminates poorly. Per
the promotion rule this stays an observation; a discriminating re-derivation
(e.g., decay-mode-excluded subspace angles) would need registering first.

## Interpretation

The drafters did not fork the trunk. They rebuilt qualitatively the same
transport (decay-only, same family — Round 2 E4) on hidden-space read
directions that share nothing measurable with any trunk layer's. That is a
mini universality result inside one checkpoint: the *kernel shape* is the
attractor, the *coordinates implementing it* are incidental. Whether the
banks' fine structure specifically converges on the deep-trunk shape (the
L47/L51 observation) is left open pending a sharper registered discriminator.

## Provenance

Dump-first: `dumps/round5/lf7/lf7_dump.npz` (spectra, 640-dim bases, all four
74×74 matrices; input hashes in the manifest). Method constants frozen in
`scripts/round5_lf7_mtp_parentage.py` header before outcomes; the first dump's
k-truncation bug (stored 64 basis directions, silently degrading the primary
metric) was found and fixed BEFORE any verdict was recorded — both runs'
numbers appear in the session record. Full tables:
[`lf7_parentage.json`](lf7_parentage.json),
[`lf7_rederivation.json`](lf7_rederivation.json).
