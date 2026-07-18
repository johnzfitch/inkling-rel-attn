# LF6 — MI mimicry: ANSWERED, prediction failed — the kernel is not a quadrature of corpus MI

**Question** (ROUND5_LEFTFIELD_SPEC.md): mutual information between tokens in
natural text decays as a power law; the model chose exponential-family
kernels. Is the two-rate kernel a two-point quadrature of the corpus MI
curve — the right power law approximated with the wrong family?

**Registered prediction:** BIC order 2-exp > power > 1-exp on most globals;
3-exp not decisively better; kernel tracks MI(d) with rank correlation > 0.9
on prose. **FAILED on both substantive clauses.**

## What was found

- **Shape tracking: nowhere near.** Spearman(log MI, log |mode-0|) over
  d ∈ [16, 1023] on prose (305,465 tokens of the long source text,
  Miller-Madow-corrected, top-64 alphabet): 0.33–0.66 across the 11 globals
  (median ≈ 0.61), against the registered 0.9. **0/11.**
- **Family race: the far field is SIMPLER than predicted.** On d ∈ [32,
  extent): 1-exp wins outright in 9/11 layers — the staged 2-exp fit finds
  zero SSE improvement (BIC gap exactly the parameter penalty). Only L65
  (the wall layer, decisively two-scale) and marginally L5 support 2-exp.
  The power law never wins. 3-exp is never decisive (11/11, vacuously where
  fits collapse to fewer components).
- Two fitting passes were run: pure random restarts, then staged
  initialization (seeded from the (k−1)-exp solution); they agree everywhere
  except L5. Provenance correction (audit): only the staged pass's manifest
  exists on disk — the first pass's BIC table was overwritten and survives
  only in the session record. The report now binds the fits-manifest hash.

## Synthesis with LF2 (interpretation, not a registered LF6 result)

LF2 (same day) suggests why 2-exp finds nothing: several mid/deep global
kernels rise toward a paragraph-range hinge before decaying, and
positive-amplitude exponential sums cannot express a rise. The
"rise → crest → single exponential tail" geometry is a post-hoc synthesis of
LF2 + LF6 — the registered d≥32 race does not directly test it, and it needs
its own artifact before promotion (the audit's post-hinge diagnostic supports
1-exp after the hinge in 9/11). Likewise, "long range is delegated to content
matching" is an interpretation connecting LF6 to the needle/heartbeat
results, not an LF6 finding.

## Caveats

- Prose MI beyond d ≈ 500 approaches the corrected noise floor; the rank
  correlation window is as frozen, and no threshold was adjusted after
  outcome access. Code MI (40,802 tokens) is descriptive only, as frozen.
- The registered "2-exp" intuition came from Round-4 fits whose window
  included the near field (d < 32), where a genuine fast component lives;
  this test's far-field-only window is the reason the answer differs, not a
  contradiction.

## Provenance

Dump-first: `dumps/round5/lf6/lf6_mi.npz` (+ `mi_manifest.json` with text
hashes and shuffle floors), `fits_manifest.json` (both passes' BIC tables and
parameters). Full report: [`lf6_mi_mimicry.json`](lf6_mi_mimicry.json).
Script: `scripts/round5_lf6_mi_mimicry.py`.
