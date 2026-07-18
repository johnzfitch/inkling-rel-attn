# Round 2 experiment specifications — Inkling relative-attention transport

Author: research-assistant session, 2026-07-15.
Builder: implement each experiment as a standalone script in `R:\inkling\scripts\`,
outputs under `R:\inkling\analysis\round2\`. Python 3.14, numpy 2.5, scipy 1.18,
torch 2.11+cu128 (RTX 4090, 24GB), matplotlib if needed (`pip install` allowed).
Reuse the range-fetch helpers in `scripts/extract_rel_attn.py` verbatim for any
new Hub reads. All Round 1 tensors are already on disk in `R:\inkling\weights\`
(float32 .npy): `layerNN_rel_logits_proj.npy` [16, extent], `layerNN_wr_du.npy`
[1024, 6144], plus `_meta.json` (extent + is_local per layer).

Background: `R:\inkling\analysis\VERIFICATION.md` documents three defects in the
Round 1 fit that E1 exists to fix. Read it before implementing.

---

## E1 — `fit_transport_models.py`: model-selection refit of the mode curves

Replaces Round 1's single damped-sinusoid fit, which had a sigma/phi degeneracy
and was over-parameterized (see VERIFICATION.md issues 1–3).

For each layer, SVD `proj = U S Vt` (as in Round 1). For each mode k with
`S[k] >= 0.05 * S[0]` (skip noise modes), take the curve `y = S[k] * Vt[k]`,
sign-canonicalized so `y[:8].mean() >= 0`. Split it:

- **near field**: d in [0, 8) — do not fit; store the 8 raw values verbatim.
- **far field**: d in [8, extent) — fit ALL of the following model ladder with
  `scipy.optimize.curve_fit` (maxfev 20000, seeds as in Round 1 where
  applicable; wrap in try/except and record failures):
  1. `const`: a                                        (1 param)
  2. `exp`: a*exp(-delta*d) + c                        (3 params)
  3. `exp2`: a1*exp(-d1*d) + a2*exp(-d2*d) + c         (5 params, Prony-style;
     seed d1, d2 at 1x and 5x the exp-fit delta)
  4. `log`: a + b*log1p(d)                             (2 params, T5-bucket-like)
  5. `dsin`: a*exp(-delta*d)*cos(rho*d) + c            (4 params — NOTE: no free
     phi; phase is absorbed by sign canonicalization + the constant. This kills
     the Round 1 degeneracy. Seed rho from dominant non-DC FFT bin.)
- Select the winner by **BIC** (n = extent-8). Also record AICc.
- Report per mode: winning family, its params, BIC of every family, R² of
  winner, plus two derived scale-free quantities that ARE comparable across
  layers: `amp0` = fitted value at d=8, and `d_half` = distance at which the
  fitted curve (minus its constant) falls to half of amp0 (numeric root-find;
  None for const/log).

Claim to test: dominant modes prefer `exp`/`exp2` everywhere, and `dsin` wins
only where |rho| is genuinely nonzero. Output
`analysis/round2/transport_fits.json` keyed by layer -> mode. Print a per-layer
one-line summary (winner family + d_half of mode 0).

## E2 — `per_head_transport.py`: per-head composed transport + reshape test

Weight-level per-head object: for head h with row-block `Wr_h` (16 rows of
wr_du), the hidden-state-to-distance-logits map is `C_h = proj.T @ Wr_h`
([extent, 6144]). Everything data-independent about head h's positional
behavior lives in C_h.

1. **Reshape hypothesis test** (must run first): the 1024 rows of wr_du could be
   head-major (`reshape(64, 16, 6144)`) or d_rel-major (`reshape(16, 64, 6144)`
   then transpose). Discriminator: for each hypothesis, for each head, compute
   the top-1 SVD energy fraction of C_h (`S0² / ΣS²`). The correct grouping
   should yield more concentrated (coherent) per-head operators. Report mean
   concentration per hypothesis per layer for layers {0, 23, 40, 65}; pick the
   winner globally; abort with a loud warning if the two are within 2% (then
   run the rest under both and emit both outputs).
2. For every layer and every head (66 x 64): SVD of C_h (use
   `torch.linalg.svd` on GPU, float32 — 4224 SVDs of [extent, 6144]; batch as
   [64, extent, 6144] per layer to keep it fast; ~10GB VRAM headroom is plenty
   at these sizes, but chunk to 16 heads per batch if allocation fails).
3. Per head store: top 8 singular values, and the top-2 left singular vectors
   (distance profiles, length extent) downsampled to 128 points (mean-pool) to
   keep the JSON manageable; also the full profile for d in [0, 32) (near
   field, no pooling).
4. **Head taxonomy**: classify each head's top distance profile by simple rules:
   `prev_token` (|profile| has >50% of its L2 mass in d in [0,4)),
   `decay` (monotone-ish decreasing envelope: Spearman rho of |profile| vs d
   < -0.5), `flat` (max-min < 0.1 * global max), `other`. Report the count per
   class per layer.

Outputs: `analysis/round2/per_head_summary.json` +
`analysis/round2/head_profiles.npz` (arrays: profiles [66,64,2,128], near
[66,64,32], svals [66,64,8]). Print per-layer class counts.

## E3 — `depth_and_extent.py`: spectrum-vs-depth + local/global comparison

All inputs already on disk; no network, no GPU needed.

1. Singular spectrum heatmap data: matrix [66, 16] of singular values,
   plus participation ratio per layer -> `analysis/round2/spectrum_depth.json`.
2. Local-vs-global: for each global layer g in {5,11,...,65}, correlate each of
   its 16 mode curves truncated to d<512 against all 16 mode curves of the
   adjacent local layers g-1 and g+1 (Pearson, after sign canonicalization).
   Record the best-match correlation matrix. Question: do global layers just
   re-learn the local transport plus a long tail, or something different?
3. Near-field census: for every layer, the layer-mean curve
   `mean_k |S_k * Vt_k|` over d in [0, 32). Question: is the previous-token
   spike (VERIFICATION.md issue 3) universal or layer-band-specific?
4. Plots (matplotlib, PNG, 150 dpi, into `analysis/round2/figs/`):
   spectrum-vs-depth heatmap; participation ratio vs depth with global layers
   marked; near-field mean curves as small multiples (11 x 6 grid).

## E4 — `extract_mtp.py`: extend dump to the MTP drafter layers

`mtp.safetensors` holds 8 MTP layers with the same attn structure
(`model.mtp.layers.N.transformer_block.attn.{wr_du.weight, rel_logits_proj.proj}`,
N in 0..7; weight map already lists them — confirm names against the index
before fetching, and note mtp_config.local_layer_ids = [0,2,4,5,6,7] for the
local/global split). Fetch via range requests into
`R:\inkling\weights\mtp\`, then run the E1 fit and E3 items 1+3 on them.
Question: does the drafter learn the same transport as the trunk, or a cheaper
one? Output `analysis/round2/mtp_spectrum.json`.

## Ground rules for the builder

- Pure weight-level analysis only; nothing here needs activations or the full
  checkpoint. Total new downloads: only E4 (~200MB).
- Every fitted quantity must be identifiable — no free phase next to a free
  amplitude (that was Round 1's bug).
- Every script: runnable standalone via `python <script>.py`, idempotent,
  prints progress per layer, writes JSON with `float()`-cast values only.
- Do not modify Round 1 scripts or outputs; Round 2 outputs live under
  `analysis/round2/`.
- If any measured quantity contradicts a claim in this spec or in
  VERIFICATION.md, print a loud `[CONTRADICTION]` line rather than massaging it.
