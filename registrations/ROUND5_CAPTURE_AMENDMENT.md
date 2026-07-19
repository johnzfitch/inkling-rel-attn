# Round 5 LF5 capture amendment — lossless attention inputs

Status: frozen before recapture and before any registered LF5/LF4 outcome.

Parent registrations:

- Round 5 preregistration: commit `34278b4`, 2026-07-16 21:48.
- LF5/LF4 implementation plan and directing-session addendum: commit
  `d4e2579`, 2026-07-16.

This is a technical data-integrity amendment. It changes no registered
prediction, token contrast, control, depth band, or decision direction.

## Discovery record

The Tier-2 activation follow-up wrote the residual stream after each layer as
FP16. A pre-build audit for LF5 found that L8's saved output contains non-finite
values on all six corpus texts, making the saved input to L9 and later layers
lossy. The directing-session review independently verified the boundary and
recorded a sparse overflow density consistent with massive/sink activations
crossing FP16's maximum finite value (65,504), rather than broad state drift.

The live model pass was not non-finite. All 396 captured relative vectors and
all 66 x 24 captured needle attention rows are finite. The corruption occurred
only in the diagnostic serialization:

```text
BF16 residual stream -> FP16 NPY -> overflow to infinity
```

Consequently, clipping, imputing, or substituting the overflowed values would
be an unregistered approximation and cannot support an all-layer LF5 claim.

## Remediation

Repeat the same 66-layer x 6-text forward pass with the same checkpoint,
verbatim token IDs, layer order, attention implementation, query chunk size,
dtypes, and model code. Add a non-mutating forward hook to each decoder layer's
`input_layernorm` and save the actual normalized BF16 tensor entering attention.

BF16 is stored losslessly by viewing its payload as `uint16` in NPY. Each file
has logical dtype BF16, physical dtype `uint16`, and shape `[8192, 6144]`. The
new pass writes only below `dumps/round5/`; it never overwrites Tier-2 data.

At near-zero marginal model cost, the pass also captures:

1. replay r-vectors and the 24 needle rows at every layer, for an exact
   non-interference check;
2. every layer-output coordinate with `abs(hidden) > 30000`, stored as
   `(position, channel, value)` before any FP16 conversion; and
3. per-token next-token NLL for all six texts, using the checkpoint final norm,
   `/24` logits multiplier, unembedding matrix, and unpadded 200,058-token
   vocabulary.

The NLL is an integrity baseline for later R5-D, not an ablation result. Its
pre-result sanity gate is: every NLL is finite, mean English-prose NLL is
positive and below the uniform-vocabulary value `ln(200058)`, and mean random
text NLL is greater than mean English-prose NLL.

## Capture acceptance gate

The recapture is accepted only if all of the following hold:

- 396 normalized-input files exist, decode to finite BF16, and have shape
  `[8192, 6144]`;
- replay r-vectors and replay needle rows are bit-identical to the corresponding
  original FP16 payloads;
- massive-activation files cover all 66 x 6 layer/text pairs, including empty
  files where no coordinate crosses the threshold;
- six NLL arrays contain exactly 8,191 finite next-token losses and pass the
  frozen sanity gate above;
- checkpoint-index, config, tokenizer, corpus-ID, source, and output hashes are
  present; and
- an atomic manifest records every file's logical/physical dtype, shape, byte
  count, SHA-256, provenance, completion flag, and wall time.

Failure of any condition stops LF5 production work. A failure is preserved in
the manifest; no threshold or source is changed after inspection.

## LF5 parity amendment

The original plan's single bitwise CPU parity requirement conflated model
wiring with cross-backend numerical reduction. The directing-session addendum
froze this two-tier gate before implementation:

### Tier 1 — REPLAY wiring parity

Run `offline_row` on the same GPU with the original BF16 dtypes and Tier-2 query
chunk boundaries. Compare all 66 x 24 needle rows with the original capture.
Every FP16 bit, including signed zero, must match. This certifies tensor wiring,
head grouping, convolution, normalization, bias, mask, scaling, and softmax.

### Tier 2 — PRODUCTION CPU characterization

Run the CPU/FP32 production path on the same 66 x 24 rows. Every row must meet:

- maximum elementwise `abs(delta_probability) <= 1e-3`;
- absolute row-sum error `<= 1e-3` per head;
- per-head argmax agreement `== 100%`; and
- `KL(original || offline) <= 1e-6` per row/head, evaluated with a numerically
  safe zero convention and reported before aggregation.

Passing both tiers is the registered verified LF5 deliverable. Failure of any
production row stops semantic row generation and triggers diagnosis. Diagnostic
statistics cannot turn a failed threshold into a pass.

## Frozen class clarifications

The implementation-plan addendum also froze two classification details before
any aperture computation:

- `{this, that, these, those}` are excluded from the LF4 pronoun class and are
  not moved into the function-word class; and
- token positions occupied by either planted mention of every codeword in
  `05_needles` are excluded from the primary rare-BPE class. An including-needles
  variant is secondary only.

The committed `analysis/round5/loci.json` is the machine-readable freeze of all
loci, classes, seeds, hashes, and selector definitions.

## Housekeeping

All raw amendment outputs live under `dumps/round5/`, which is already covered
by the repository's `dumps/` ignore rule. No ignore rule is changed.
