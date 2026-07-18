# Round 5 — decision D4 resolved: lossless full-depth residual states

This amendment is registered by its public commit timestamp, before the D4
runner extension, production capture, or inspection of any new hidden-state
outcome.

## Decision

D4 is resolved in favor of adding a lossless full-depth residual-state capture
to the already registered D1 widened pass. A later standalone D4 pass would
repeat the same 66-layer forward computation; capturing the states while that
pass is already resident adds storage and host transfer, not another model
evaluation.

The scope is deliberately not restricted to global layers. R5-C's registered
flip-band discontinuity, adjacent-layer rotation, and local-versus-global
predictions require all layer transitions.

## Frozen artifact contract

For each of the eight widened-pass arms (the six v1 texts plus the two fresh
P-e/P-f paired arms), capture:

1. the normalized embedding residual entering layer 0 (`hidden_embed`), and
2. the residual output of every layer L0 through L65 (`hidden_L00` …
   `hidden_L65`).

Each state is the complete `[8192, 6144]` deployed BF16 tensor, persisted
bit-for-bit as an uncompressed NumPy `uint16` payload. No FP16 conversion,
clipping, imputation, finite-only deletion, dimensional projection, token
subsampling, or channel subsampling is permitted. The layer output is copied
only after the unmodified layer forward returns and is not fed back from host
storage.

This adds 67 × 8 = 536 artifacts and 53,955,526,656 raw payload bytes
(approximately 50.25 GiB, before small `.npy` headers). Together with D1, the
production manifest must contain exactly 2,324 artifacts.

## Confirmatory versus secondary use

The six frozen v1 texts remain the confirmatory population for every R5-C
prediction and for the channel-4786/3290 lifecycle preregistration. Capturing
the two fresh paired arms prevents another GPU pass and permits descriptive
replication, but they may not be substituted into, pooled into, or used to
rescue a failed six-text confirmatory verdict.

The state convention for layer L is:

- input to L: `hidden_embed` for L0, otherwise `hidden_L{L-1}`;
- output of L: `hidden_L{L}`;
- the rotation attributable to L compares those two states.

The registered R5-C predictions, bands, channel identities, lifecycle rules,
thresholds, and controls are unchanged. D4 does not alter any P-e/P-f class,
boundary, dose, pairing, or outcome rule.

## Validation and admissibility

The independent validator must authenticate the D4 amendment as a historical
Git blob at the capture's recorded HEAD; require the exact 536-path inventory,
shape, `uint16` storage dtype, artifact hash, and finite BF16 exponent pattern;
and mark D4 satisfied only when the complete capture passes. The validator must
not decode through FP16.

The overflowed historical FP16 residual files remain inadmissible. This
amendment does not rehabilitate or compare against them.

## Interrupted preflight disclosure

Immediately before this decision, a no-D4 D1 production preflight began at
commit `62eb071` and was stopped when D4 was requested. It produced no
preflight report, capture directory, model activation, loss, attention row, or
registered outcome. At most, it read public inputs/checkpoint bytes for startup
authentication. No GPU capture began.
