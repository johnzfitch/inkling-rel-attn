# Tier 2 — streaming forward pass on the 4090

Author: research-assistant session, 2026-07-15/16. Complements ROUND2/ROUND3.
Goal: one streaming prefill pass over Inkling-NVFP4 that yields the two
measurements the black-box tiers cannot produce:

- **(a) the seam**: attention mass vs backward distance `d` at global layers,
  straddling d=1024 — is there an attention-pattern discontinuity in situ.
- **(b) push-outward vs content-decay**: the net early-layer attention profile,
  adjudicating the "rising bias pushes attention outward" story against the
  objection that content decay beats it.

Same pass, both answers, zero generation.

## Established facts (verified this session, not assumed)

**Model.** 975B total / 41B active, 66 layers, hidden 6144, 64 heads,
head_dim 128. Global layers `{5,11,17,23,29,35,41,47,53,59,65}` (11 of 66):
`num_kv_heads=8`, `rel_extent=1024`, **unbounded** causal attention. The other
55 are local: `num_kv_heads=16`, `sliding_window=512`, and — from source —
`rel_extent = sliding_window_size if is_sliding else rel_extent`, i.e. **512**.
MoE: 256 experts, 6/token, 2 shared, `intermediate_size=3072`. Layers 0–1 dense
(`dense_mlp_idx=2`), layer 2 MoE.

**The seam exists only at global layers.** At local layers `rel_extent ==
sliding_window == 512`: the bias cutoff and the attention mask coincide exactly,
so no discontinuity is possible. At global layers the bias is zero for d>=1024
(`masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)`) while
attention still reaches arbitrarily far back. Past d=1024 tokens are attended
with *exactly zero* positional bias. That is the seam, and it is well-posed.

**Every rising head is in a LOCAL layer.** From `analysis/round3/head_taxonomy_v2.json`:
layers 0–4, 6–10, 12–13, 15–16, 18–19, 21–22, 24 (all extent 512). Layer 5 —
the first global layer — is 59/64 `decay`, **zero** rising. So dispute (b) lives
entirely inside a 512-token window: the outward push is bounded at d<512 by
construction, whoever is right about the mechanism. (b) does not need long
context; only (a) does.

**tau is dormant below 128k.** `tau = 1 + 0.1*log(clamp((pos+1)/128000, min=1))`
=> tau ≡ 1 exactly for pos < 128000. An 8k pass says **nothing** about log
scaling, and no claim about the log-scaling regime may be sourced to it. The
pos=1e6 value is **1.20557** (ROUND3_SPEC and earlier notes carry x1.2058 /
~1.2058 — a rounding error; 1.20557 is exact).

**Quantization scope (`hf_quant_config.json` + confirmed).** The ONLY NVFP4
tensors in the checkpoint are `mlp.experts.{w13,w2}_weight` for layers **3–65**.
`exclude_modules` covers every layer's `attn`, `attn_norm`, `attn_sconv`,
`mlp_norm`, `mlp_sconv`, `mlp.gate`, `mlp.shared_experts`, plus embed/unembed
and layer 2's `mlp.experts`. **The entire measurement pathway — attention,
wr_du, rel_logits_proj, norms, sconv — is BF16 at full precision.** Quantization
reaches the result only indirectly, through routed-expert MLP outputs in the
residual stream.

## Environment facts

- **R: is exFAT.** `hf download` livelocks on `.cache/huggingface/.gitignore.lock`
  because filelock cannot take byte-range locks on exFAT. Hence
  `scripts/tier2_download.py` (parallel HTTP Range, no lock files, resumable).
- R: = Crucial T700 (Gen5 NVMe, ~12GB/s), 1100GB free; NVFP4 is 592GB / 33 shards.
- Measured HF throughput ~300-450 MB/s => ~25-30 min for the checkpoint.
- 4090 is Ada (sm_89): **no native FP4**. NVFP4 kernels in vLLM/SGLang/TRT are
  Blackwell-only, and `modular_inkling.py` is BF16-only (no fp4/scale2/dequant
  path). So we dequantize ourselves — `scripts/tier2_nvfp4.py`.
- System RAM 61.6GB — enough to double-buffer two 8.7GB layers pinned.
- transformers 5.14.0 lives in `.venv-tier2` (`--system-site-packages`, reusing
  torch 2.11+cu128). The system 5.12.1 is untouched, so ROUND3 A1's
  independent-reimplementation guarantee still holds.

## Corrections to the original Tier-2 sketch

1. **"~15B/layer @ 4-bit = 7.5GB, load / apply / discard"** — size is right
   (~8.7GB/layer packed), but a layer **cannot** be dequantized: 14.6B params at
   bf16 = 29GB > 24GB VRAM. Keep the layer **packed** in VRAM and dequantize
   **expert-by-expert** JIT (113MB each, 256/layer). Same idea, one level finer.
2. **No airllm / accelerate disk-offload path.** They cannot drive a bespoke
   NVFP4 MoE. The per-layer loader is custom.
3. **`w13_weight` is INTERLEAVED, not concatenated.** `conversion_mapping.py`:
   `WeightConverter("mlp.experts.w13_weight" -> "mlp.experts.gate_up_proj",
   operations=[Interleave(dim=1)])`. Assuming `[gate; up]` concat produces
   garbage that still runs. Mitigation: our dequant output is layout-identical
   to the BF16 checkpoint tensor (validated), so we dequantize into a virtual
   BF16 state dict and let transformers' own conversion machinery apply
   Interleave rather than reimplementing it.

## Dequant validation (`scripts/tier2_validate_dequant.py`) — PASS

Fetches ONE expert (layer 3, expert 0, w13) from BOTH repos via Range requests
(~95MB) and compares — settles the format without the 1.9TB BF16 download.

| hypothesis | rel_err | corr |
|---|---|---|
| lo-nibble-first + scale + scale2 | 0.0870 | 0.99974 |
| hi-nibble-first (wrong) | 1.4142 | -0.00001 |
| no scale2 (wrong) | 3696.87 | 0.99868 |

`rel_err=0.087` is **not** a bug and an absolute threshold is the wrong test: an
**ideal nearest-representable-value quantizer** using the same stored block
scales scores **0.0869** on this data — identical to ours. We sit exactly at
FP4's intrinsic floor. Optimal global rescale = 0.996 (no systematic scale
error). 90.3% of elements are the exact nearest value; the ~10% one-ULP
differences cost nothing, since our rel_err equals the ideal quantizer's.
Pass criteria are therefore: corr>0.99, rel_err within 2% of the ideal
quantizer, and clearly beating the wrong hypotheses.

Format: `value = E2M1_LUT[nibble] * scale_e4m3[block_of_16] * scale2[expert]`,
low nibble = even element, blocks along the last (input/contraction) axis.

## Measurement design

Replace `eager_attention_forward` with a measuring variant. It already receives
`position_bias` separately from the content logits, which is what makes the
decisive test nearly free:

**Run the softmax twice on the same content logits — with and without the bias
term.** That isolates the bias's *causal* contribution to attention mass vs d,
settling (b) directly instead of inferring it from a net profile. Return the
with-bias output as the true forward path.

**Scope of the "without" condition — state this in any writeup.** The
counterfactual holds each layer's *inputs fixed*: downstream layers still
receive the true with-bias residual stream. This is exactly the right causal
quantity for both questions **as posed** — "what does the bias do to *this
layer's* attention" — but it is **not** "what would the model do if the bias did
not exist," which would compound across all 66 layers. One sentence, so nobody
oversells it later.

Per layer accumulate, chunked over query blocks to bound memory:
- `mass_with[h,d]`, `mass_without[h,d]`, `count[d]`
- `bias_mean[h,d]`, `content_mean[h,d]` (the decomposition)

**Per-head all the way to the dump.** The `[h,d]` accumulators are never
aggregated over heads at dump time, and analysis must join against
`analysis/round3/head_taxonomy_v2.json` (`head_class`) **before** any pooling.
The dispute is about *rising* heads specifically; averaging all 64 heads mixes
the effect with its opposite and would wash out the very thing being tested.

d ranges over [0,512) at local layers (band only — cheap) and [0,seq) at global.
At seq=8192, a 512-query chunk is [1,64,512,8192] f32 ≈ 1.07GB, alongside the
8.7GB packed layer. Attention FLOPs are ~1.1 TFLOP/global layer — negligible;
the pass stays I/O-bound, which is the whole point.

## Code-review fixes applied (Codex review, 2026-07-16)

The first seq-8192 run thrashed; a review found real issues, since fixed:
- **#2 (the big one): full position_bias materialization.** Inkling builds the
  dense [1,64,Q,Q] bias (8.6GB bf16, ~17GB through the tau float-cast) BEFORE the
  attention interface runs, so content-chunking never bounded the largest tensor.
  Fixed by patching `InklingRelativeLogits.forward` to return the COMPACT
  rel_logits [1,H,Q,extent] (~1-2GB) and doing the per-chunk distance gather
  inside `measuring_attention`. `tier2_parity_bias.py` proves the reconstruction
  is BIT-EXACT vs the original gather for extent 512 and 1024.
- **#1 layer free**: `del layer` now happens in the loop before the next
  `build_layer` allocates (was leaking one full layer).
- **#3 output isolation**: smoke writes to `dumps/tier2/_smoke/`; filenames carry
  `_s{seq}`; writes are atomic (tmp+rename). No smoke/partial run can clobber
  production data.
- **#4 / #10 guards**: runner asserts layers start at 0 and are contiguous (no
  resume support), and that `--seq <= corpus length` (no silent truncate).
- **#8 dequant validation**: now a 4-case battery (w13+w2, layers 3 & 40,
  different experts); the no-scale2 hypothesis and a two-sided floor bound both
  gate the pass. (Caught a dim-order bug in the test itself.)
- **#9 provenance**: `manifest.json` records checkpoint/config/tokenizer/script
  SHAs, library+CUDA versions, seq, qchunk, file list, and a `complete` flag.
- **#11 RMSNorm parity**: embedding norm now uses the library `InklingRMSNorm`
  (exact cast order), not a hand-rolled version.
- **#14 needles caveat**: the corpus needle gaps are ~896-1000 vs ~1048-1152,
  NOT a matched 1023/1024 pair, so the needle arm cannot isolate the one-token
  step. The all-pairs `mass_with(d)` accumulator captures exact distances
  regardless, so question (a) is unaffected; the "designed needle instrument"
  language is downgraded to "needle arm biases samples to straddle the seam".

## Performance notes (learned during build)

- The per-(q,k) `scatter_add` in f64 was the sink (~800B ops over the run).
  Replaced by a DIAGONAL reduction: gather each tensor along keys so axis d is
  distance, sum over queries to [H, dmax], in f32. Mass conservation unchanged
  (sum_d mass == n_queries, exact).
- `dmax` is adaptive: local (sliding) layers only populate d<512, so dmax=window
  there and dmax=seq only at the 11 global layers. Cuts the meter cost ~16x on
  55 of 66 layers.
- GPU ran at 100% util (compute-bound, not disk/thrash) — confirms the pass is
  bound by the attention+measurement, not I/O, at seq 8192.

## FINDINGS (full run, 66 layers x 6 texts, seq 8192, 2026-07-16)

Analysis: `scripts/tier2_analyze.py` -> `analysis/tier2/tier2_findings.json`.
Integrity: 396/396 dumps conserve mass and are finite.

**(a) Seam — CONFIRMED across depth.** Bias-attributable step in mass_with across
d=1024 is positive at 10/11 global layers; `without_step` (content) is an order
of magnitude smaller everywhere (content is seam-blind). Strengthens through
mid/late depth. Exceptions to note, not bury: **layer 11** is near-null and the
only one leaning negative (46% heads positive, in-window bias -0.089); **layer
65** is dramatic (in-window bias +2.66, step +3.8e-4, 100% heads). Matches Round
4 C1's blind `predicted_seam_direction = positive (drop)`.

**(b) Push-outward — REJECTED in situ.** On rising heads (Round 3 taxonomy; all
in local layers 0-28), the bias BOOSTS near tokens (b(1)~+2.0) and SUPPRESSES far
tokens (b(400)~-1.5), zero-crossing ~d=32. Its contribution to attention mass
(mass_with - mass_without) has NEGATIVE slope over the far-field band at every
rising-head layer and every text, unanimously. So the bias concentrates attention
INWARD and is on the SAME side as content decay, not opposing it. Fable's "rising
bias pushes attention outward" is wrong; GPT's content-decay objection holds and
is strengthened. The static weights were ambiguous because Round 3's "rising"
labels rising MAGNITUDE of the operator profile, and magnitude != sign — a rising
|b| with negative sign is rising SUPPRESSION. The forward pass measures the real
signed logit (no sign-canonicalization), so this is a cleaner sign read than the
weight-space C1 test. Empirically vindicates code-review point #13.

## First result (seam, question (a)) — CONFIRMED at seq 2048, layer 5

The twice-softmax decomposition cleanly separates bias from content at the seam:
`mass_without` is CONTINUOUS across d=1024 (step ~-1e-5, content is blind to the
seam) while `mass_with` steps DOWN by +1.2..1.9e-4, essentially all of it
bias-attributable, consistent across random/prose/code/needles. Bias snaps to
exactly 0 outside 1024 and is POSITIVE (+0.3..0.45) just inside. So there IS an
in-situ attention discontinuity at the seam, it is caused by the positional
bias, and its direction matches Round 4 C1's blind `predicted_seam_direction =
positive (drop)`.

## Run plan

seq=8192, full 66 layers, **3-5 diverse texts** — not one.

Content decay is a property of the *data*: code, prose, and repetitive text have
very different content-match profiles, so a single 8k sequence would make the
(b) verdict a fact about that document rather than about the model. Each pass is
minutes, so this is cheap insurance. Corpora: English prose, code, repetitive/
templated text, multilingual, and a natural long-range-dependency document.
Dump per-text accumulators separately (never pre-averaged across texts) so
between-text variance is measurable, and report (b) per text — if the sign of
the effect flips across texts, that IS the finding.

Download (~25-30 min) -> stream 592GB per pass (~1-3 min of pure I/O at Gen5
speeds; weights are re-read per text, which is fine).

Dump-first, per ROUND3 methodology: the pass writes full-fidelity accumulators
to `dumps/tier2/` (no thresholds, no pooling); analysis reads only dumps.
