# Inkling relative-attention analysis

Weight-level and in-situ analysis of the learned relative-position mechanism in
[thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling)
(952B MoE, 66 layers, 64 heads, d_rel=16, rel_extent 1024 global / 512 local),
run on a single RTX 4090 via HTTP range requests + a streaming NVFP4 prefill pass.

## Plain-language summary

When a language model reads text, every word needs a sense of *how far back* each
earlier word is — otherwise "the dog bit the man" and "the man bit the dog" look the
same. Most models are given a hand-designed distance formula (RoPE, ALiBi, etc.).
Inkling is unusual: it **learned its sense of distance from scratch** during training,
as a table of numbers nobody designed. This project reads that table — first directly
from the weights, then by running the model and watching the mechanism operate — all
on one consumer GPU.

What we found, in plain terms:

- **The learned rule is simple: closer = more attention, fading smoothly with
  distance.** No rhythm, no waves. The model was free to learn wave-like patterns
  (the way older hand-designed schemes work); everywhere we looked, fits that
  seemed wavy turned out to be fractions of one slow hump, not real oscillation.
- **It resembles known designs in silhouette but copied none of them.** The fade-out
  is exponential-ish like RetNet's, but RetNet spreads its heads' fade speeds over a
  ~128× range in a fixed ladder; Inkling's heads all fade within a narrow ~3× band.
  SGD converged on the same *shape* engineers picked, with a different organization.
- **Its sense of distance ends at 1024 tokens.** Beyond that there is literally no
  positional signal — a hard wall in the table. Yet the model still finds things past
  the wall: when we planted rare facts ~1100 tokens back, most layers retrieved them
  as well as facts inside the wall, using pure content matching ("these words look
  like those words"). Distance gets it oriented; content does the long-range work.
- **The final layer is a strict gatekeeper.** Layer 65 adds a large bonus to
  everything within 1024 tokens, effectively ignoring anything older — the model's
  last step is deliberately near-sighted.
- **The layers divide the labor.** Five of six layers can only see 512 tokens back;
  the remaining "global" layers learned to boost exactly the region *past* 512 —
  they pick up precisely where the near-sighted layers stop.
- **Early layers look far, deep layers focus near**, and the handoff around
  layers 13–28 is a fairly abrupt regime switch, passing through hump-shaped
  (rise-then-fall) curves rather than morphing smoothly.

## Highlights

- The transport b(d) is **decay-dominated and low-rank** (~1.5–3 of 16 modes); no
  genuine oscillation anywhere — damped-sinusoid/RoPE family wins are fractional-cycle
  artifacts (`analysis/round4/oscillation_audit.json`).
- Family battery (F1–F10, pre-registered): **exp-decay shape without any named
  scheme's ladder** — cross-head decay rates span only ~2.7× (RetNet's ladder ~128×).
- The d=1024 extent seam is real and bias-caused, but **needle retrieval is
  seam-robust at mid/deep global layers**; the penalty is confined to L5 (and the
  L65 terminal wall). Bias acts as a contrast enhancer, not a gate
  (`analysis/needles/`).
- **d=512 echo**: global layers' bias steps up exactly past the local window size
  (33/33 layer×texts) — division of labor between scopes written into the table.
- L65 is a hard 1024-token attention wall (+2.6 bias floor in-extent, ~8× mass cliff).

## Layout

- `QUESTIONS.md` — **the question ledger**: every registered question with its
  current status (open / answered / provisional / forecast) and the gate on it.
  Start here to see what is actually answered vs pending.
- `registrations/` — all pre-registered specs, preregs, execution plans, and
  amendments; the git timestamp of each file is its registration, and contents
  are immutable after registration (corrections arrive as new amendments)
- `scripts/` — extraction (range requests), Round 3 dumps, Tier-2 streaming runner,
  Round 4 battery/fingerprints/curiosity, audits
- `analysis/` — results (JSON + figures) per round
- `corpus/` — 6×8192-token pre-tokenized measurement corpus (the runner consumes
  `*.ids.npy` verbatim; never re-tokenize)
- `docs/` — model reference material

Not in the repo (local-only, re-derivable): `nvfp4/` (592GB checkpoint),
`dumps/` (raw meter/capture dumps, 52GB), `weights/` (extracted tensors, 1.9GB).

Methodology: dump-first — every experiment dumps raw data at full fidelity before
any analysis; analyses read only from dumps.
