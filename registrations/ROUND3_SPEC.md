# Round 3 specifications — dump-first restructure

Author: research-assistant session, 2026-07-15 (late evening). Supersedes the
earlier same-day version of this file.
Builder: the user's own agent. Same conventions as ROUND2_SPEC.md (Python 3.14,
invoke as `python`, JSON plain floats, `[CONTRADICTION]` protocol, Round 1/2
outputs immutable).

## Methodology (binding for this and all future rounds)

**Dump first, analyze second.** Every experiment is split into a Dump script
(D*) and an Analysis script (A*):
- D* scripts produce full-fidelity artifacts (npy/npz): ALL modes/components,
  full resolution, no top-k truncation, no thresholds, no pooling, no
  classification. Anything expensive to recompute (network fetch, GPU compute)
  MUST be dumped. Where an object has bounded rank (all our composed operators
  have rank <= 16), dump the complete object.
- A* scripts read ONLY from dumps (never network, never GPU), so every result
  is independently re-runnable from the dumped raw data, and the dumps remain
  available for non-pre-registered pattern hunting.
- Dumps -> `R:\inkling\dumps\round3\`; analysis outputs ->
  `R:\inkling\analysis\round3\`; figures -> `analysis\round3\figs\`.

Round 2 hygiene debt this round repairs: E2 kept only top-8/16 singular
values, top-2/16 left vectors pooled 4x, and discarded ALL right singular
vectors (the hidden-space read-directions). D0 backfills this.

## Context — read first
- `R:\inkling\analysis\VERIFICATION.md` (updated 2026-07-15 evening): official
  forward pass known from transformers main (`models/inkling/modular_inkling.py`,
  v5.14.0). Head-major wr_du layout CONFIRMED from source; bias zero outside
  0 <= d < rel_extent; q/k RMS-normed, scaling 1/head_dim; global layers apply
  tau = 1 + 0.1*log(clamp((pos+1)/128000, min=1)) to query AND bias.
- Existing raw dumps: `weights\layerNN_{wr_du,rel_logits_proj}.npy` (66 layers,
  verified byte-identical to Hub), `weights\mtp\` (8 MTP layers).

---

## D0 — `dump_per_head_svd.py` (GPU, ~5 min, ~2.2 GB)

For every trunk layer (66) and MTP layer (8), every head h (64):
C_h = proj.T @ Wr_h  with the CONFIRMED head-major blocks
(wr.reshape(64, 16, 6144)). rank(C_h) <= 16, so dump the COMPLETE economy SVD:
- S: all 16 singular values
- U: [extent, 16] left singular vectors (distance profiles, full resolution)
- V: [16, 6144] right singular vectors (hidden-space read directions)
Sign convention: flip each (U[:,k], V[k,:]) pair so U[:8,k].sum() >= 0.
One npz per layer: `dumps/round3/perhead_svd/layer{NN}.npz` and
`mtp{N}.npz` with arrays S [64,16], U [64,extent,16], V [64,16,6144]
(float32). Include a `checksums.json` (sha256 per npz). Validate: for 3 random
(layer, head) pairs, reconstruct U @ diag(S) @ V and compare to a freshly
computed C_h (max abs diff < 1e-3, bf16-scale tolerance).

## D1 — `dump_mode_curves.py` (CPU, seconds, ~30 MB)

Layer-level SVD at full fidelity (Round 2 stored only fits of it): per trunk +
MTP layer, SVD of proj [16, extent]; dump S [16], U [16,16], Vt [16, extent]
(float32, same sign convention) to `dumps/round3/mode_curves/layer{NN}.npz` /
`mtp{N}.npz`. Trivially recomputable, but dumped anyway so A* scripts have a
single canonical source and analyses are byte-reproducible.

## D2 — `dump_sconv.py` (network, small)

Fetch ALL SConv tensors for all 66 trunk + 8 MTP layers via range requests
(reuse extract_rel_attn.py helpers): `attn.k_sconv.weight`,
`attn.v_sconv.weight`, `attn_sconv.weight`, `mlp_sconv.weight` (and the MTP
`transformer_block.*` equivalents). Look up exact names AND shapes from the
safetensors index first; save raw arrays verbatim as
`dumps/round3/sconv/layer{NN}_{name}.npy` plus a `_meta.json` of
name -> shape -> dtype. NO derived quantities in this script.

## D3 — `dump_norms_and_scales.py` (network, small)

While we're at the well: fetch per layer `attn.q_norm.weight`,
`attn.k_norm.weight`, `attn_norm.weight`, `mlp_norm.weight` (trunk + MTP) into
`dumps/round3/norms/`. Not needed by any pre-registered A* below — dumped
precisely so unregistered patterns (e.g., per-head-dim norm structure
interacting with the positional pathway) can be looked for later without a new
fetch round.

---

## A1 — `validate_mechanism.py`

Re-implement `InklingRelativeLogits.forward` locally (matmul + distance gather
+ mask, semantics from modular_inkling.py; do NOT pip-upgrade transformers).
For layers {0, 5, 33, 65}: 5 random unit r-vectors per head (seed 0), compute
bias via (a) the gather path over a [64, extent+64] position grid spanning the
extent boundary, (b) our analysis path b(d) = (r @ proj)[d] zeroed outside
[0, extent). Assert max |a-b| < 1e-5. Verify tau(10^6) ≈ 1.2058. Output
`analysis/round3/mechanism_validation.json`. Reads `weights/` only.

## A2 — `positional_horizon.py`

Reads D1 (trunk + MTP). Per layer, per mode k (ALL 16 modes now — report
singular values alongside so the reader can weight them; no 5% threshold at
dump level): with y = S[k]*Vt[k]:
- edge_amp = mean |y[extent-16 : extent]|
- interior_amp = mean |y[extent//4 : extent//2]|
- cliff_ratio = edge_amp / (interior_amp + 1e-12)
Per head (reads D0, full-resolution U now, not pooled): same three numbers on
S[h,0] * U[h,:,0].
Global layers: also tau-amplified edge_amp (x1.2058).
Output `analysis/round3/positional_horizon.json`, fig
`figs/cliff_ratio_depth.png` (mode-0 cliff_ratio vs depth, global layers
marked), verdict field: "smooth" (median mode-0 cliff_ratio < 0.05) /
"cliff" (> 0.25) / "mixed". Print 5 worst layers.

## A3 — `head_taxonomy_v2.py`

Reads D0 (full-resolution profiles — Round 2's pooled npz is superseded).
Per head, on p = |S[h,0] * U[h,:,0]|:
- near_mass_ratio = (sum p[:4]^2 / sum p^2) / (4/extent)
- classes: `near_focused` (near_mass_ratio > 20 AND argmax p < 4),
  `decay` (Spearman rho(p, d) < -0.5), `rising` (rho > +0.5 — Round 2's
  early-layer anomaly gets its own class), `flat` ((max-min) < 0.1*max),
  else `other`. Apply rules in that order.
- also d_peak = argmax p, effective_range = min d* with cum L2 mass >= 0.9.
Output `analysis/round3/head_taxonomy_v2.json` (per-layer counts, per-head
class/d_peak/effective_range), figs `taxonomy_depth_stack.png`,
`d_peak_hist.png`. Flag any layer with rising > 32/64. Cross-check: recompute
Round 2's E2 class counts from D0 under the OLD rules and confirm they match
per_head_summary.json (validates D0 against the Round 2 GPU run); print
`[CONTRADICTION]` if not.

## A4 — `sconv_analysis.py`

Reads D2 only. Infer tap/channel axis order from dumped shapes (state the
inference in a comment + JSON field). Per layer/kernel: tap-energy profile,
effective delay (center of mass of |taps|), fraction of near-identity channels
(tap_t0 > 0.9 of L1 mass). Question: smoothers, differentiators, or
passthroughs — and depth / local-vs-global / k-vs-v-vs-residual differences?
Output `analysis/round3/sconv_summary.json` + `figs/sconv_taps_depth.png`.

---

## Run order & effort
D0 (GPU) -> D1 -> D2+D3 (network) -> A1 -> A2/A3/A4 in any order.
Everything after D0 is CPU + disk. Estimated total new disk: ~2.5 GB.
A* scripts must not import torch.cuda or urllib — that's the enforcement of
dump-first.
