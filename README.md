# Inkling relative-attention analysis

Weight-level and in-situ analysis of the learned relative-position mechanism in
[thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling)
(952B MoE, 66 layers, 64 heads, d_rel=16, rel_extent 1024 global / 512 local),
run on a single RTX 4090 via HTTP range requests + a streaming NVFP4 prefill pass.

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

- `ROUND{2,3,4}_SPEC.md`, `TIER2_SPEC.md` — pre-registered experiment specs
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
