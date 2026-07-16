# Verification of Round 1 (extraction + spectrum dump) — 2026-07-15

## Data integrity: PASS
- Re-fetched `layers.33.attn.rel_logits_proj.proj` and `layers.7.attn.wr_du.weight`
  from the Hub via independent range requests; byte-identical to the saved `.npy` files.
- 66/66 layers present, shapes match config (local: [16,512], global: [16,1024];
  wr_du always [1024,6144]). Local/global pattern matches `config.json` exactly.
- Files are float32 on disk (bf16 upcast at save time) — lossless, 2x size.

## Decomposition audit: 3 issues found

### Issue 1 (CONFIRMED BUG, interpretation-level): sigma/phi degeneracy
When the fitted rho ~ 0 (nearly every dominant mode), the damped-sinusoid model
`sigma*exp(-delta*d)*cos(rho*d+phi)` degenerates: only `sigma*cos(phi)` is
identified. Measured examples:

| layer | reported sigma | phi    | sigma*cos(phi) (identified qty) |
|-------|---------------|--------|---------------------------------|
| 0     | 747.21        | 1.5694 | 1.02                            |
| 3     | 1229.24       | 4.7115 | -1.06                           |
| 28    | 0.59          | 3.7999 | -0.47                           |

=> The raw `sigma` column in `rel_attn_spectrum.json` is NOT comparable across
layers and must not be interpreted. The singular values are unaffected (they
come from SVD, not the fit). Round 1's depth-trend claim was based on singular
values, so it stands.

### Issue 2 (model misspecification): rho adds nothing; decay-only fits as well or better
Refit of dominant modes with `a*exp(-delta*d) + c` (3 params vs 4):

| layer | R² damped-sin | R² decay+const |
|-------|--------------|----------------|
| 0     | 0.939        | 0.947          |
| 28    | 0.973        | 0.970          |
| 42    | 0.645        | 0.782          |
| 64    | 0.975        | 0.949          |

=> Dominant modes are decay-shaped, not oscillatory. The sinusoid family is
over-parameterized for them. Proper model selection (AIC/BIC across a family
ladder) is needed before claiming any rho is real.

### Issue 3 (unmodeled structure): near-field mismatch at d=0
Even the identified amplitude `sigma*cos(phi)` disagrees with the actual curve
value at d=0 (layer 0: model 1.02 vs actual 0.28). The curves have structure in
the first few distances (previous-token-attention-like spikes) that a single
smooth decay cannot capture. Near-field (d < ~8) should be modeled/reported
separately from the far-field tail.

## New finding from audit: the transport is strongly low-rank
Participation-ratio effective rank of the 16-mode spectrum:
layer 0: 2.6, layer 5 (global): 3.1, layer 23 (global): 3.2, layer 40: 1.7,
layer 65 (global): 1.4. The 16-dim relative feature bank effectively uses
~1.5–3 modes. This makes per-mode interpretation tractable.

## Standing assumptions — RESOLVED 2026-07-15 (evening) against official source

`huggingface/transformers` main branch contains the official day-0
implementation (`src/transformers/models/inkling/modular_inkling.py`, shipped
in v5.14.0; local install is 5.12.1 which is why it wasn't found initially).
Verified directly from that source:

- CONFIRMED: `proj [16, extent]` is shared across heads
  (`InklingRelativeLogits` has one `proj` per attention module).
- CONFIRMED: head-major layout. `r_proj = Linear(hidden, num_heads*d_rel)` then
  `view(*, num_heads, -1)` => wr_du rows [16h:16h+16] belong to head h
  (hypothesis A). Round 2 E2's winner was correct — though note the
  concentration discriminator it used was later shown unreliable out-of-sample
  (layer 3 prefers A strongly, layer 33 prefers B strongly), so the source is
  the load-bearing evidence, not the statistic.
- CONFIRMED: bias construction matches our reconstruction exactly:
  `rel_logits = relative_states @ proj`, gathered at d = q_pos - k_pos,
  masked to ZERO outside 0 <= d < rel_extent.
- NEW (not previously known): q/k are per-head RMS-normed and attention scaling
  is 1/head_dim (not 1/sqrt); on non-sliding layers a log-length factor
  tau = 1 + 0.1*log(clamp((pos+1)/128000, min=1)) multiplies both the query and
  the position bias.
- Checkpoint-name mapping confirmed in `conversion_mapping.py`:
  wq_du->q_proj, wk_dv->k_proj, wv_dv->v_proj, wr_du->r_proj, wo_ud->o_proj.
- CORRECTION of the Round 1 note below: the research subagent's report cited
  this transformers file accurately; the coordinating session wrongly dismissed
  it as fabricated. Its other citations (vLLM PR, tml-fa4) remain unverified,
  and the docstring in modular_inkling.py does reference "sglang RelLogitsProj
  + the FA4 score_mod", lending them plausibility.

## Architectural consequence worth flagging
Position bias is identically zero for d >= rel_extent. Global layers therefore
carry learned positional signal only over the first 1024 distances; past that
the model runs effectively position-blind (NoPE-like), aside from causal
masking and the tau query scaling, out to the advertised 1M context.
