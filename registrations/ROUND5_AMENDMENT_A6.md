# Round 5 Amendment A6 — attention dtype-boundary erratum, fixes, corpus v2.1

## 1. Material erratum: dtype boundary in the measurement fork

Our measuring attention path upcast content and positional bias to FP32
separately and added them in FP32 (`scripts/tier2_stream.py`). Stock Inkling
eager attention adds the BF16 tensors first — taking the BF16 rounding at the
addition — and upcasts only inside the softmax
(`transformers/models/inkling/modeling_inkling.py`, `attn_weights =
attn_weights + position_bias` then `softmax(..., dtype=float32)`). Verified
against both sources. On six real LF5 sentinel layers the discrepancy changes
14.4–70.2% of stored FP16 attention words (worst Δp 0.02498, worst KL 0.00145,
headwise argmax agreement 96.81–99.87%).

The earlier "bitwise replay" gates compared rows captured and replayed by the
SAME implementation — they certified self-consistency, not stock parity. The
registered "true with-bias attention path" claim was therefore wrong as stated.

**Scope.** Affected (deviations bounded by the sentinel numbers above): all
in-situ artifacts — Tier-2 meters, both capture passes (residual stream,
r-vectors, needle rows, NLL), LF5 row dumps, LF4 apertures, corpus-v2 readouts.
NOT affected: all weight-level results (Rounds 1–4 battery, fingerprints,
oscillation audit, subspace anatomy geometry — none touch the attention add).

**Status change.** All four corpus-v2 outcomes (P-v2-1…4) and the in-situ LF4/
LF5 numbers are PROVISIONAL pending a corrected recapture. Given effect sizes
(0.04–0.35) versus worst-case Δp 0.025, we expect no verdict to flip; that
expectation is a prediction, not a result. No threshold, direction, or decision
rule changes. Old artifacts remain immutable; corrected runs write new dirs.

**Fix applied** (this commit): BF16 add before upcast in
`tier2_stream.py` (measuring fork) and both GPU/CPU paths of
`round5_offline_attention.py`. The instrument and any recapture now share the
corrected arithmetic; the next replay-parity gate certifies against freshly
captured stock-arithmetic rows.

## 2. Verifier independence fix

`corpus_v2_verify.py` recomputed effects and p-values independently but
compared decision BOOLEANS against the main report's own fields (tautological
for P-v2-3/P-v2-4). Fixed: the verifier now derives its own booleans from its
own recomputed effects and Holm values. (Independent hand-check of the current
fields found them correct: SciPy reproduced all 64 Spearman correlations within
1.1e-16; an alternate permutation implementation preserved every decision.)

## 3. Lower-severity dispositions

- `corpus_v2_freeze_classes.py` now refuses to overwrite an existing freeze
  without `--force` (which itself requires a public amendment).
- The math-LLM builder's ≥40-char assistant-turn filter was in the public
  builder before measurement but omitted from the spec prose — documented here.
- 6/2,016 function-word selections are subword fragments; removing them leaves
  the effect unchanged (unadjusted p 0.6682). Definition unchanged; noted.
- Next capture manifests must record checkpoint shard hashes and package
  versions/module paths (NumPy, tokenizers, torch, transformers).

## 4. Corpus v2.1 slack arm (recipe registered here)

v2.0's frozen recipe consumed a single two-person DM thread (all 311 messages
from one channel) — novelty intact, generality poor. New arm `07b_slack_multi`:
top 8 DM channels by human chars, up to 1,024 tokens each in rank order,
continuing down the ranking on underfill (+64-token surplus for BPE joins,
trimmed to 8,192). Built: 8,192 tokens, 232 messages, 9 conversations. The
v2.0 arm and its results remain immutable for comparison.

## 5. One corrected capture pass covers everything

The next GPU pass (corrected arithmetic, new output dirs, provenance additions
from §3): recapture 07_slack_human + 08_math_llm, capture 07b_slack_multi, all
with r-vectors + NLL. Downstream: re-evaluate P-v2-1…4 unchanged; then the
depth-resolved registration (`ROUND5_DEPTH_RESOLVED_PREREG.md`) evaluates on
these corrected captures only.
