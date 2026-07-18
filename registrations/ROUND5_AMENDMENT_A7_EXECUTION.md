# Round 5 Amendment A7 — corrected-run execution freeze

This amendment is registered before any A6-corrected model capture or outcome
read. It resolves implementation details exposed while wiring the runner; it
does not change a prediction, threshold, depth band, class lexicon, or the
10,000-permutation count in A6 / `ROUND5_DEPTH_RESOLVED_PREREG.md`.

## 1. Four-arm corrected capture

The fresh corrected capture contains all 66 r-vectors plus next-token NLL for
`07_slack_human`, `08_math_llm`, `07b_slack_multi`, and `01_prose_en`. The
fourth arm is required for P-d4; including its NLL is symmetric bookkeeping,
not a new gate. Output paths are new and provisional v2.0 artifacts are never
overwritten. Startup must pass a bitwise BF16 global + sliding comparison of
the compact measuring path against stock Transformers eager attention.

## 2. v2.1 sidecar boundary disposition (pre-outcome)

The built `07b_slack_multi` IDs and text are intact. Its sidecar labels all
8,192 tokens with unit indices 0..231 (`n_units_used = 232`), but the stored
`unit_start_tokens` list has 231 entries. The sole omitted transition is token
1024, the first channel join: a BPE token straddles the raw character boundary,
so the character-span start detector misses it. This was found by the class
freeze before any model result was read.

For P-d1/P-d3, message starts are therefore frozen as token 0 plus every change
in the already-built `token_unit_index` array. This yields the intended 232
speaker-label positions. First-content remains exactly start + 2; a start whose
+2 is outside the 8,192-token capture is excluded and counted. IDs and text are
unchanged. The builder is corrected to emit transition-derived starts on any
future rebuild.

## 3. Depth statistic and controls

For every layer, aperture is converted to midrank percentile within fixed
256-token position bins. A class's per-layer effect is median percentile minus
0.5; a band's effect is the median of its registered per-layer effects.

The directed permutation test samples the registered class count without
replacement inside each 256-token bin, uses one sampled position mask across
all layers in the band, recomputes the band-median effect, and repeats 10,000
times with a deterministic A6-derived seed. The four primary p-values are Holm
adjusted together. A P-d prediction passes only when every numerical threshold
in its preregistered conjunction passes and its directed Holm p is below 0.05.

Each primary mask is crossed onto `08_math_llm` using the same band and
direction. Deterministic random-position masks match the class count in every
position bin separately on the source arm and math arm. Crossed-math, source-
random, and math-random p-values form three separate four-test Holm families;
each is a true-null pass when Holm p is at least 0.05. All effects and verdicts
are reported even if a control fires.

## 4. Unchanged corpus-v2 replication

P-v2-1..4 are recomputed only on the two original v2.0 arms with their original
class freeze, seeds, layers, thresholds, directions, and decision rules. NLLs
for v2.1 Slack and prose are descriptive and do not enter P-v2-1.

## 5. Provenance

The corrected capture manifest records hashes for all 33 checkpoint shards,
the stock Inkling implementation, relevant source/spec files, every input ID
array, and versions plus resolved module paths for NumPy, tokenizers, torch, and
Transformers. Raw captures and private classes remain gitignored.
