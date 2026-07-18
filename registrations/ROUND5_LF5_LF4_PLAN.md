# Round 5 implementation plan — LF5 offline attention, then LF4 zoom lens

Status: planning only; no Round 5 analysis or instrument implementation has begun.

Prepared after the public preregistration at commit `34278b4` (2026-07-16
21:48). This document operationalizes LF5 and LF4. It does not replace the
registered questions, predictions, controls, or confirmation rule in
`ROUND5_LEFTFIELD_SPEC.md`.

## 1. Scope and order

The build order is fixed:

1. Freeze provenance, query loci, class definitions, and pass/fail criteria.
2. Build and certify LF5's `offline_row(layer, text, q)` primitive.
3. Produce the registered LF5 row dump and bracket-matching readout.
4. Build LF4's per-token aperture metric on the already captured r-vectors.
5. Run the registered position-matched controls and an independent
   re-derivation before promoting either result to a claim.

R5-A/B/C/D and LF1-3/LF6-11 are out of scope for this first build. The LF5
interfaces and dump schema should nevertheless be reusable by R5-D.

## 2. Read-only preflight findings

The repository is clean at `34278b4`. The local inputs are present:

- 66 layers, 64 query heads, 8 global or 16 local KV heads, head dimension
  128, relative dimension 16;
- six fixed 8,192-token ID arrays with hashes in `corpus/manifest.json`;
- 396 r-vector captures, one per layer and text, all finite;
- 66 captured needle-row files, each holding 24 rows of shape
  `[24, 64, 8192]` in FP16;
- the complete NVFP4 checkpoint and the existing random-access shard reader;
- extracted relative projections in `weights/`.

One preflight finding changes the implementation path but not a prediction:
the residual-stream captures were written as FP16. The first non-finite input
to attention occurs at L9 for every text (the saved output of L8 has overflowed
values). Of the global layers, only L5 has a finite saved input; L11-L65 do not.
The original forward pass and its captured r-vectors/attention rows are finite:
the loss happened only while serializing the large residual stream to FP16.

Consequences:

- LF4 is unaffected because all 396 r-vector files are finite.
- LF5 can be developed against L0-L8 from the existing hidden captures.
- Exact Q/K reconstruction for L9-L65 is impossible from the overflowed files:
  replacing infinities, clipping, or inferring 6,144 hidden coordinates from a
  16-dimensional relative channel would be an unregistered approximation.
- A full all-layer LF5 certification therefore needs the small protocol
  amendment in section 3. Without it, LF5 must be reported incomplete rather
  than silently weakened.

## 3. Pre-build protocol amendment for LF5

Before inspecting any registered LF5 or LF4 outcome, add and publicly timestamp
a short capture amendment that records the FP16 overflow, its discovery check,
and this remediation. Predictions and statistical tests remain unchanged.

Run one replacement capture pass using the same checkpoint, corpus IDs, layer
order, model forward, and attention implementation as Tier 2. Add a forward hook
on each decoder layer's `input_layernorm` and save the actual normalized tensor
entering attention, not the unnormalized residual stream. Preserve its BF16 bits
losslessly as a `uint16` NPY plus dtype/shape metadata; do not cast it to FP16.
Write into a new directory and never overwrite the Tier-2 capture.

Expected cost is approximately the prior capture pass (17.6 minutes on the
4090) and about 38 GiB. The drive currently has more than 400 GiB free.

The amendment pass is accepted only if:

- all 396 normalized-input files have the registered shape and are finite after
  decoding BF16;
- corpus, tokenizer, config, checkpoint-index, and source hashes are recorded;
- newly observed r-vectors and needle rows are bit-identical to the original
  capture, proving the hook did not change the forward stream; and
- a complete atomic manifest lists every artifact, byte count, dtype, shape,
  and SHA-256.

If a new GPU pass is not approved, continue only with an explicitly partial LF5
implementation/certification on L0-L8, do not evaluate the registered
three-global-layer bracket prediction, and proceed independently to LF4.

## 4. Frozen LF5 design

### 4.1 Public interface

Implement a library module plus a thin CLI. The core interface is:

```python
offline_row(layer: int, text: str, q: int, *, compact: bool = False) -> OfflineRow
```

`OfflineRow` contains key positions, per-head content logits, positional bias,
total logits, with-bias attention, and without-bias attention. The default form
is `[64, 8192]` so it can be compared directly with the captured rows. Compact
form stores only the causal support and records its start/stop offsets.

The implementation will reuse the repository's `ShardReader` but load only the
attention tensors needed for one layer: input norm, Q/K projections, K short
convolution, Q/K head norms, and the relative projection. It will not load or
execute V/O projections or the MLP.

For each `(layer, text)`:

1. Decode the captured BF16 normalized attention input.
2. Project all keys, apply the exact four-tap residual short convolution in
   FP32, reshape into KV heads, and apply the checkpoint K RMS norm.
3. Project only requested query positions, reshape into 64 heads, and apply the
   Q RMS norm.
4. Repeat KV heads by the model's mapping: 4 query heads per KV head locally,
   8 globally.
5. Compute content logits with scaling `1/128`.
6. Use the captured r-vector for the query and the checkpoint relative table to
   compute `b(q-k; v)`. It is zero outside the table extent.
7. Apply the exact causal mask; local support is
   `[max(0, q-511), q]`, while global support is `[0, q]`.
8. Apply float32 softmax and cast the stored row to FP16. Tau is exactly 1 for
   all positions in this 8k corpus.

Key states are cached only for the current `(layer, text)` and released before
advancing, keeping memory bounded. No dense `[Q,K]` attention matrix is formed.

### 4.2 Parity gate

"Exact to FP16" is frozen as bitwise equality of the FP16 payload, not an
`allclose` tolerance. Signed-zero differences also count as mismatches.

Development starts with L0, L5, L11, L23, L41, and L65, then expands to every
one of the 24 needle queries at all 66 layers. Certification requires every
stored element, including masked zeros, to match the existing capture.

The parity report also records mismatch count, maximum absolute error, error
quantiles, row-sum error, argmax agreement, and top-k overlap. Those diagnostics
do not turn a failed bitwise gate into a pass. If CPU/GPU reduction order prevents
bitwise parity, the instrument may be reported as numerically characterized but
not as the registered verified LF5 deliverable until the discrepancy is resolved.

### 4.3 Query loci frozen before row computation

Selectors consume the committed `*.ids.npy` arrays verbatim and decode them with
the hashed Inkling tokenizer. They never re-tokenize the `.txt` files. The first
selector run writes `analysis/round5/loci.json`; that file is reviewed and
committed before any LF5 row or LF4 aperture is read.

- Bracket closers, `02_code`: use Python lexical `OP` tokens outside strings and
  comments, retain `)`, `]`, and `}`, and use a typed stack to record the matched
  opener. Abort on a lexical/matching error rather than changing parsers after
  seeing attention.
- Sentence starts, `01_prose_en`: the first non-whitespace lexical token at the
  beginning of the decoded stream or after `.`, `?`, or `!`, allowing closing
  quote/bracket characters between the terminator and whitespace.
- Heartbeat lines, `03_templated`: the final non-newline token on every decoded
  line containing the exact ASCII field `HEARTBEAT`.
- Random controls: 64 positions per text, eight sampled without replacement
  from each 1,024-token position block, excluding named loci. Seeds are derived
  from `SHA256("34278b4|LF5|<text>")` and stored in the loci manifest.

All eligible semantic loci are retained; they are not selected after looking at
attention. The manifest stores token IDs, decoded fragments, character offsets,
query positions, matched opener positions where applicable, selector version,
and input hashes.

### 4.4 LF5 production dump

Generate rows for the frozen loci at all 66 layers. Store them under the ignored,
local-only `dumps/round5/lf5/` tree. Use a lossless ragged representation:
`qpos`, support start/stop, `indptr`, and contiguous `[key, head]` FP16 arrays.
At minimum preserve with-bias attention; content logits, positional bias, and
without-bias attention use the same support-index schema so pair-level
counterfactuals do not require recomputation.

Writes are atomic and resumable at `(layer, text)` granularity. A manifest
contains source/output hashes, code version, numeric backend, elapsed time, and
completion state. Only compact JSON summaries, figures, and provenance are
committed under `analysis/round5/lf5/`.

### 4.5 Registered bracket readout

For each syntactically matched closer/opener pair, compute head-mean attention
to the matched opener. Distance controls are other earlier openers of the same
bracket type within `max(8, round(0.1*d))` tokens of the matched distance. Require
at least eight controls; otherwise expand once to the same integer log2-distance
bin, and exclude the pair if eight still do not exist. This rule is applied
without inspecting weights.

The layer statistic is the median log ratio of matched-opener attention to the
median control attention. A within-query random-designation test over eligible
control openers uses 10,000 deterministic permutations. Across the 11 global
layers, use Holm correction at family-wise alpha 0.05. The registered prediction
passes only if at least three global layers have a positive ratio and corrected
one-sided `p < 0.05`.

## 5. Frozen LF4 design

### 5.1 Primary aperture metric

For layer `L`, token `t`, head `h`, relative channel `c`, and distance `d`:

```text
b[L,t,h,d] = sum_c rvec[L,t,h,c] * rel_proj[L,c,d]

A[L,t] = sum_h sum_{d=129}^{E_L-1} |b[L,t,h,d]|
         ------------------------------------------------
         sum_h sum_{d=0}^{E_L-1}   |b[L,t,h,d]|
```

`E_L` is 512 locally and 1,024 globally. The primary token scalar is the ratio
of sums across heads, not the mean of 64 unstable per-head ratios. A zero
denominator is recorded as missing. Headwise ratios are diagnostics only.

The primary metric uses the whole learned support even for tokens near the start
of a text: it measures the token's aperture setting, not how much history happens
to be available at that position. A history-truncated version is reported only
as a secondary diagnostic. Computation uses float64 accumulation from the raw
r-vector and projection dumps, in token blocks; no full
`[8192,64,extent]` tensor is retained.

The primary depth bands are fixed to the global layers:

- early: L5, L11, L17;
- mid-depth: L23, L29, L35, L41, L47;
- late: L53, L59;
- wall: L65 alone.

The registered directional tests use the five mid-depth globals. Every layer is
still dumped separately so no band average can hide the L65 wall or a sign flip.

### 5.2 Token classes

All classes are functions of token IDs plus deterministic decoding and are
stored in the frozen loci/class manifest before aperture computation.

- Pronouns: single-token, case-folded complete words in
  `{i, me, my, mine, myself, you, your, yours, yourself, yourselves, he, him,
  his, himself, she, her, hers, herself, it, its, itself, we, us, our, ours,
  ourselves, they, them, their, theirs, themselves, this, that, these, those}`.
- Closing brackets: the syntactic `)]}` positions selected for LF5.
- Sentence starts: the prose positions selected for LF5.
- Rare BPE: non-whitespace token IDs occurring at most twice across the five
  non-random corpus texts, with special/control tokens excluded.
- Function words: single-token, case-folded complete words in
  `{a, an, the, and, or, but, if, then, than, as, at, by, for, from, in, into,
  of, on, onto, to, with, without, is, am, are, was, were, be, been, being, do,
  does, did, have, has, had, can, could, may, might, must, shall, should, will,
  would, not}`.

The three registered directional contrasts are closers in code, pronouns in
English prose, and function words in English prose. Sentence starts and rare BPE
are secondary, multiplicity-controlled observations.

### 5.3 Position-matched null and decision rule

Within each text, convert aperture to mid-rank percentiles separately inside
fixed 256-token position bins and per layer. Average a token's percentile over
the five mid-depth global layers, then use `median(class percentile) - 0.5` as
the contrast statistic. This directly operationalizes "above/below the median"
while removing coarse position imbalance.

Shuffle class labels within the same 256-token bins, preserving class counts,
for 10,000 deterministic permutations. Use a one-sided test in the registered
direction and Holm correction over the three primary contrasts. A directional
prediction passes only when its sign agrees and corrected `p < 0.05`. Report
the raw effect, permutation interval, class count, per-layer effects, and a
cluster bootstrap over 256-token blocks; never report only a p-value.

Controls and diagnostics:

- apply each semantic text's class-position mask to `06_random` as a true-null
  negative control;
- repeat with 128- and 512-token position bins as frozen sensitivity checks;
- report the available-history aperture separately to expose any early-position
  artifact; and
- retain per-head effects to show whether a pooled result is broad or driven by
  a few heads.

### 5.4 Independent confirmation

The main LF4 implementation uses blocked matrix multiplication. A second audit
script must not import that computational core: it directly loops over the raw
16-channel r-vectors and projection table for all primary-class tokens plus a
frozen matched control sample. It recomputes curves, apertures, class effects,
and permutation signs from source files and their hashes.

Likewise, LF5 parity and the bracket statistic receive a scalar reference path
on frozen sentinel rows. A claim is promoted only when the registered control
passes and the independent path agrees in direction and numerical effect within
the precision declared in the artifact manifest.

## 6. Planned repository artifacts

Implementation names may change only before result-bearing execution, and any
change is recorded in the run manifest.

```text
ROUND5_CAPTURE_AMENDMENT.md              # protocol note, committed before recapture
scripts/round5_capture_attention_inputs.py
scripts/round5_offline_attention.py      # LF5 library + CLI
scripts/round5_lf5_analyze.py
scripts/round5_lf4_zoom_lens.py
scripts/round5_lf4_verify.py             # independent raw re-derivation
analysis/round5/loci.json                # frozen before outcomes
analysis/round5/lf5/                     # compact results, parity, figure/report
analysis/round5/lf4/                     # compact results, controls, figure/report
dumps/round5/attention_inputs/            # ignored raw BF16-bit capture
dumps/round5/lf5/                         # ignored pair rows
dumps/round5/lf4/                         # ignored per-token aperture dumps
```

Every JSON result includes schema version, source paths and hashes, registration
commit, code commit/dirty state, package versions, numeric dtype/backend, seeds,
and the exact pass/fail rule. Figures are generated only from dump artifacts.

## 7. Execution gates and handoff

1. **Approval gate:** review this plan and authorize either the capture amendment
   or the explicitly partial L0-L8 LF5 path.
2. **Freeze gate:** commit the amendment (if approved) and the loci/class manifest
   before computing registered outcomes.
3. **Capture gate:** normalized inputs and replayed r-vectors/needle rows pass the
   integrity checks.
4. **LF5 parity gate:** all 66 x 24 captured needle rows are bit-identical in
   FP16. No production semantic-row dump before this passes.
5. **LF5 confirmation gate:** bracket control and independent reference agree.
6. **LF4 input gate:** all r-vector/projection shapes, dtypes, hashes, and finite
   checks pass.
7. **LF4 confirmation gate:** position-matched null, negative control, and
   independent re-derivation agree.
8. **Handoff:** publish dump-first summaries, then expose the certified LF5 API
   to LF4 pairwise follow-ups and later R5-D attention-redistribution readouts.

No causal ablation, wall healing, or new Round 5 claim is part of this execution
until LF5 and LF4 have crossed their respective gates.

## 8. Review addendum (directing session, 2026-07-16) — four amendments, frozen pre-build

Preflight claim independently verified: first non-finite capture is L8's OUTPUT
(input to L9) on every text sampled, including 06_random; r-vector spot-checks
finite. Overflow density ~1 element per 512 sampled rows on random tokens →
consistent with a massive-activation / attention-sink coordinate exceeding the
FP16 ceiling (65,504) from L8 on, not broad distribution drift.

**A1 — Split the parity gate BEFORE building (Section 4.2 risk).** Bitwise FP16
equality across different matmul reduction orders (CPU BLAS vs the original CUDA
bf16 kernels, and even CUDA-to-CUDA with different chunk shapes) is unlikely to
pass and was not what the capture certified. Freeze a two-tier gate now, not
post-hoc: (i) REPLAY parity — offline_row run on the same GPU, same dtypes, same
qchunk boundaries as tier2_stream: bitwise FP16 equality required, certifies the
wiring; (ii) PRODUCTION characterization — the CPU/f32 path on all 66×24 needle
rows: max |Δp| ≤ 1e-3 per element, row-sum error ≤ 1e-3, per-head argmax
agreement 100%, KL(orig‖offline) ≤ 1e-6 per row. Passing (i)+(ii) = the
registered verified LF5 deliverable; failing (ii) at any row = stop and diagnose.

**A2 — The amendment pass captures two more things (near-zero marginal cost).**
(a) Per-token NLL of the true next token for all 6 texts (final norm + LM head on
the existing forward): a few MB, gives R5-D its baseline arm for free, and a sane
prose NLL is the only end-to-end integrity check the plan currently lacks.
(b) The massive-activation census: for every layer, dump (position, channel,
value) of all |h| > 30,000 before FP16 serialization — identifies the sink
coordinate(s) that caused the overflow; feeds LF3's BOS-transient question.

**A3 — Class-definition tweaks (Section 5.2), frozen now.** (a) Drop
{this, that, these, those} from the pronoun class — demonstratives are
determiner-like in most occurrences and dilute the registered contrast; they
join neither class. (b) Exclude text 05's planted codeword subtoken positions
from the rare-BPE class (they are engineered retrieval targets, not natural rare
tokens); a with-needles variant may be reported as a secondary diagnostic.

**A4 — Housekeeping.** `dumps/round5/` is already covered by the repository's
`dumps/` gitignore rule; no new ignore entry needed. The capture amendment note
should cite this addendum's overflow verification as its discovery record.

With A1–A4 applied: plan APPROVED, including the ~18-minute recapture (option 1
at gate 1). The L0–L8 partial path is not needed.
