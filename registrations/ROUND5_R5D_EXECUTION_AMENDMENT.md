# Round 5 R5-D causal-ablation execution amendment

This amendment operationalizes R5-D in `ROUND5_LEFTFIELD_SPEC.md` without
changing its interventions, predictions, or thresholds. Its commit timestamp
is the registration event. No propagated-ablation NLL, needle logit, or
attention-redistribution outcome exists or has been inspected at this point.

The certified A6-corrected D1+D4 capture is the baseline and restart source.
All six frozen v1 texts are confirmatory. The paired P-e/P-f arms are excluded:
R5-D was registered on the original corpus and needle readout.

The ledger's runnable-now queue is empty, and every dump-only row plus P-e/P-f
has been executed. R5-D is therefore the next registered science state change.
LF5-b remains gated on the separate D3 engineered-corpus decision and is not
displaced by this GPU campaign.

## 1. Intervention inventory

The five sampled locals are fixed as L0--L4. Together with all eleven globals,
the single-layer set is:

`{0,1,2,3,4,5,11,17,23,29,35,41,47,53,59,65}`.

Run these 67 propagated intervention arms:

- `bias_off_Lxx` at each of the 16 layers;
- `carrier_out_Lxx` at each of the same 16 layers;
- `near_off_Lxx` and `far_off_Lxx` at each of the same 16 layers;
- one `wall_heal_global` arm;
- one `rising_heads_off_L00_L04` arm;
- one `negative_seam_heads_off_L11` arm.

A baseline-final calibration is also produced from the certified saved L65
state, but it is not counted as an intervention arm.

For a single-layer arm at L, load the lossless BF16 residual entering L from
D4 (`hidden_embed` for L0, otherwise `hidden_L{L-1}`), intervene at L, and
propagate through L65. This is exact prefix reuse, not a truncated causal arm:
all upstream layers are identical to baseline bit-for-bit. The wall arm starts
from baseline `hidden_L04`; the L0--L4 head-class arm starts from the embedding;
the L11 head-class arm starts from `hidden_L10`.

## 2. Exact interventions

- **Bias off:** replace the gathered positional bias by exact BF16 zero for all
  64 heads at the target layer. Content logits, Q/K/V, and every other layer
  are unchanged.
- **Carrier out:** communal component 1 is row 0 of the certified per-layer
  hidden-space basis in
  `analysis/subspace_anatomy/common_bases_top4.npz`. On the target layer's
  normalized attention input `x`, compute `x_perp = x - (x dot c)c` in FP32,
  cast `x_perp` to BF16, and feed it only to `self_attn.r_proj`. Q/K/V and the
  residual stream still receive the original `x`. Thus this removes precisely
  the scalar hidden carrier from the r-vector path; it is not a projection of
  the whole hidden state and not an r-space proxy.
- **Near off:** set `b(d)=0` for integer backward distances `d<4`; retain all
  other in-extent bias values.
- **Far off:** set `b(d)=0` for integer backward distances `d>128`; retain
  distances 0 through 128 inclusive.
- **Wall heal:** keep every learned table entry at `d<1024` bit-identical and
  append a fitted decaying continuation to each of the 16 raw table rows at
  every global layer. No local layer is changed.
- **Head classes:** "off" means positional bias off for the selected heads,
  not removal of their content-attention outputs. The rising arm uses the
  committed Round-3 taxonomy at the registration commit: all 64 heads at each
  of L0--L4 are `rising`. The L11 negative-seam class is frozen from the
  certified corrected mean live bias over `d=1008..1023`, averaged over the six
  texts. Its 45 zero-indexed heads are:

  `{0,1,2,4,5,6,7,8,9,10,12,13,14,15,17,18,21,23,25,27,29,31,32,34,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,60,62}`.

## 3. Wall-tail fit

Fit each raw row of each global `16 x 1024` projection table independently on
the fixed window `d=512..1023`. The continuation is a continuity-constrained
signed two-exponential:

`f(1023+u) = a_s exp(-r_s u) + (y_1023-a_s) exp(-r_f u)`.

This makes the continuous fit equal the learned endpoint at `u=0`; the first
new table cell is `u=1` (`d=1024`). Rates obey
`r_s >= 1/(2*1024)`, `r_f >= r_s`, and both are at most 0.1. The signed slow
amplitude is bounded to ten times the maximum absolute value of that observed
row. Use deterministic multi-start nonlinear least squares in the original
signed value space and keep the minimum-SSE solution. Evaluate through
`d=8191`, store the tail dump before any causal verdict, and record every fit,
source hash, endpoint step, R2, and bound contact.

The tail build must stop before GPU execution unless all 176 row fits are
finite, the `u=0` reconstruction is exact to `1e-6`, every tail is finite, and
no extrapolated magnitude exceeds ten times its row's observed maximum. A
failed gate requires a public amendment; no post-outcome refit is allowed.

## 4. Deployed arithmetic and startup gates

The runner copies the A6 production arithmetic: content and intervened bias
are added in BF16, then softmaxed in FP32. The existing compact relative-logit
path, causal/sliding masks, q-chunk 512, NVFP4 layer loader, tokenizer, corpus,
checkpoint, and package provenance remain unchanged.

Before the first intervention, require:

1. Git-blob authentication of this amendment, runner, tail builder, stock-parity
   dependencies, and all frozen class/basis inputs;
2. the independently certified D1+D4 capture manifest and every reused state
   hash, shape, BF16 dtype, and finiteness check;
3. baseline final-state NLL bitwise equality to the six certified NLL arrays;
4. a production-shape L65 no-intervention replay on `06_random` whose BF16
   layer output is bitwise equal to certified `hidden_L65_06_random`;
5. toy tests proving near/far boundaries, selected-head isolation, carrier
   orthogonalization, wall preservation below 1024, finite tail behavior, and
   baseline attention parity.

No intervention arm starts if a gate fails. Each arm writes to a fresh
directory, preserves a failed manifest, and is resumable only at arm
boundaries; a complete arm is immutable and skipped by the batch driver.

## 5. Dump-first readouts

For every arm and text, save all 8,191 target positions with target id, target
logit, log-normalizer, NLL, probability, and their differences from baseline.
For text 05, separately index the 24 frozen recall query positions `q` and the
correct continuation `ids[q+1]`; report target-logit difference, probability,
log-probability difference, and below/above-seam split.

Save full per-head distance meters at every intervention locus: one layer for
each single-layer arm, all eleven globals for wall healing, L0--L4 for the
rising-head arm, and L11 for the negative-seam arm. Downstream meter changes
are not silently inferred; the causal downstream readouts are final logits and
NLL. Meter summaries report mass in `d<4`, `4<=d<=128`, `d>128`, and (globals)
`d>=1024`, plus attention effective count.

The primary arm cost is the pooled mean `delta_NLL = NLL_arm-NLL_baseline`
over all six equal-length texts. Also report each text, the five non-random
texts, a 5,000-draw 256-token block-bootstrap interval, and all token-level
arrays. Random is never dropped from the registered primary. Seeds derive from
this amendment's committed hash.

## 6. Frozen verdict rules

- **Bias-off depth prediction:** pass only if pooled delta NLL is `>0.05` at
  every L0--L5 arm and L65, and `abs(delta_NLL)<=0.05` at each other sampled
  global.
- **Carrier equivalence:** at every one of the 16 layers,
  `abs(delta_carrier-delta_bias) <= 0.20*abs(delta_bias)`. All 16 must pass;
  zero baseline effect receives no denominator floor or rescue rule.
- **Near dominates far:** the mean pooled delta NLL across the 16 near-off
  arms divided by the corresponding mean across the 16 far-off arms is at
  least 5. The denominator must be strictly positive; otherwise the prediction
  fails. Layerwise ratios are descriptive.
- **Wall incidental at 8k:** pass when the absolute pooled delta NLL of
  `wall_heal_global` is `<0.005`. Far-field attention-mass change is reported
  with no retrofitted minimum-effect gate.
- The two head-class arms and all needle splits are registered causal
  readouts but had no numerical prediction in the parent spec; they receive no
  invented pass/fail threshold.

Verdicts are computed only after every required arm is sealed. Partial batches
remain `RUNNING`, never preliminary results. Independent raw-dump
re-derivation is required before promotion to certified.
