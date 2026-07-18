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
- The two LF6 fitting passes are both recorded: the first (pure random
  restarts) and the staged rerun (seeded from the (k−1)-exp solution) agree
  everywhere except L5; the collapse is a data property, not an optimizer
  artifact — confirmed by the crest geometry below.

## Synthesis with LF2

LF2 (same day) shows why 2-exp finds nothing: the mid/deep global kernels
RISE to a crest at ~56–144 tokens before decaying, and positive-amplitude
exponential sums cannot express a rise. The true far-field geometry is:
gentle rise → paragraph-scale crest → single exponential tail. Corpus MI has
the opposite far-field character (slow power-law flattening). **SGD did not
approximate the corpus statistics' shape — it built a shorter-sighted,
simpler kernel and delegates genuine long range to content matching** (the
needle/heartbeat results), which is where the power-law work actually
happens.

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
