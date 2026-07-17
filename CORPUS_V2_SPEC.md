# Corpus v2 — novel-text arms (pre-registered before build)

Motivation: the A2a NLL capture showed corpus v1's natural texts are substantially
memorized (prose 0.11 / code 0.71 / multilingual 0.10 nats), and the exploratory
aperture–surprisal correlation implies memorization compresses every content-
conditional readout. Corpus v2 adds two arms of **private, unmemorizable** text.
The raw text and token ids are PRIVATE and live only in `corpus_v2/` (gitignored
before creation — commit d4e2579's successor; ids reconstruct the text verbatim).
This spec and the builder are public; the data never is.

## Arms (two, deliberately minimal)

**07_slack_human** — private workplace Slack, 0% LLM.
Frozen recipe: DM channels only (`D*.json`), ranked by total human chars
descending; within channel, messages ascending by `ts`; message filter = has
`user`, no `bot_id`, no `subtype`, nonempty `text`. Speakers pseudonymized per
channel in first-appearance order (`A:`, `B:`, …). Cleaning: U+FFFD → `'`,
`<url|label>` → label, `<url>` → url, `<@UID>` → pseudonym or `@user`, HTML
entities unescaped. Concatenate `{speaker}: {text}\n`; take the first 8,192
tokens. Sidecar (measured post-tokenization): per-token channel/speaker labels +
message-start token positions.

**08_math_llm** — assistant-authored technical prose from the user's math
conversations; provider-diverse by construction.
Frozen recipe: providers claude → chatgpt → gemini in equal thirds (2,731 /
2,731 / 2,730 tokens); files lexicographic per provider, skipping duplicate
exports (filenames containing `(`); parse `messages` (or `clean.messages`),
keep `role == "assistant"` turns in order; U+FFFD → `'`; concatenate turns with
blank-line separators. Sidecar: per-token provider + turn index.

Both arms run through the standard capture protocol (r-vectors + per-token NLL;
normalized attention inputs optional) whenever the next GPU pass is authorized.

## Registered predictions (blind — written before the build)

- **P-v2-1 (novelty gate):** mean next-token NLL ≥ 1.5 nats on both arms
  (memorized v1 texts: 0.10–0.71), and slack_human > math_llm (LLM prose is
  more predictable than casual human chat). Confidence: high. If an arm comes
  in < 1.0, that arm is contaminated (memorized or degenerate) and is replaced,
  not reinterpreted.
- **P-v2-2 (zoom-lens law, promoted from peeked-exploratory):** aperture
  correlates positively with own-token surprisal within 512-token position bins
  at the five mid-depth globals, median Spearman > 0 with ≥ 12/16 bins positive,
  on BOTH arms. Confidence: medium-high. This is the registered version of the
  finding first seen (and disclosed) on v1.
- **P-v2-3 (segment-start widening replication):** message-start tokens in
  07_slack_human sit above the position-matched aperture median (same statistic
  as LF4; sentence starts were +0.117 on memorized prose). Confidence: medium.
- **P-v2-4 (revised class direction — a real test of the updated story):**
  pronouns AND function words in 07_slack_human both sit BELOW the
  position-matched aperture median. Note this prediction follows the Round-5
  *result* (closed-class → narrow), i.e. we now predict the direction we
  previously got wrong. Confidence: medium. If pronouns flip back above median
  on novel casual text, the closed-class story is wrong and memorization was
  doing the work.

## Confirmation discipline

As ROUND5: dump-first; the builder writes a manifest with SHA-256 of every
artifact; class/sidecar definitions frozen here; claims promoted only after the
registered control passes and an independent re-derivation agrees.
