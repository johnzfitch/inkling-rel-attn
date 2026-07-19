"""
D-TIER2 primitives -- streaming prefill over Inkling-NVFP4 on a single 4090.

This module holds the pieces that were validated against BF16 ground truth
before committing to the 592GB pass:
  - ShardReader / layer name mapping (mmap random access to the 33 shards)
  - deinterleave (the w13 -> gate_up layout op, matches transformers' Interleave)
  - PackedExperts (keeps a layer's experts PACKED in VRAM, dequantizes one
    expert at a time inside the library's own expert loop)
  - Meter + measuring_attention (twice-softmax: with and without the bias term)

The full streaming forward lives in tier2_run.py. See TIER2_SPEC.md.

Why a layer cannot simply be dequantized: 14.6B params at bf16 = 29GB > 24GB
VRAM. So the packed experts (~7.3GB) stay resident and each expert (~113MB) is
dequantized just-in-time. Only mlp.experts.{w13,w2} for layers 3-65 are NVFP4;
everything else is BF16.

The "without bias" softmax holds each layer's INPUTS fixed -- downstream layers
still receive the true with-bias stream. It answers "what does the bias do to
THIS layer's attention", not "what would the model do with no bias at all".
"""
import json
import os
import struct

import torch

from tier2_nvfp4 import dequant_nvfp4

# safetensors dtype -> (torch dtype, bytes/elem)
_ST_DT = {"BF16": (torch.bfloat16, 2), "F8_E4M3": (torch.float8_e4m3fn, 1),
          "F16": (torch.float16, 2), "F32": (torch.float32, 4), "F64": (torch.float64, 8),
          "U8": (torch.uint8, 1), "I8": (torch.int8, 1), "I16": (torch.int16, 2),
          "I32": (torch.int32, 4), "I64": (torch.int64, 8), "BOOL": (torch.bool, 1)}

NVFP4_DIR = r"R:\inkling\nvfp4"
GLOBAL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}

# source(checkpoint) -> target(module) renames for one layer, prefix stripped.
# Pure renames (values unchanged); experts are handled separately (dequant+interleave).
RENAME = {
    "attn.wq_du.weight": "self_attn.q_proj.weight",
    "attn.wk_dv.weight": "self_attn.k_proj.weight",
    "attn.wv_dv.weight": "self_attn.v_proj.weight",
    "attn.wr_du.weight": "self_attn.r_proj.weight",
    "attn.wo_ud.weight": "self_attn.o_proj.weight",
    "attn.q_norm.weight": "self_attn.q_norm.weight",
    "attn.k_norm.weight": "self_attn.k_norm.weight",
    "attn.k_sconv.weight": "self_attn.k_sconv.conv1d.weight",
    "attn.v_sconv.weight": "self_attn.v_sconv.conv1d.weight",
    "attn.rel_logits_proj.proj": "self_attn.rel_logits_proj.proj",
    "attn_sconv.weight": "attn_sconv.conv1d.weight",
    "mlp_sconv.weight": "mlp_sconv.conv1d.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    # MoE non-expert
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.bias": "mlp.gate.e_score_correction_bias",
    "mlp.gate.global_scale": "mlp.gate.global_scale",
    # dense layers 0-1
    "mlp.global_scale": "mlp.global_scale",
}


class ShardReader:
    """Random access to safetensors shards via direct byte-range reads.

    Deliberately NOT safe_open/mmap: Windows charges an mmap of a 17GB shard
    against the commit limit, which blows a small pagefile (os error 1455). We
    parse the header once per shard and seek/read only the bytes we need, so
    resident memory is exactly one tensor at a time.
    """

    def __init__(self, d=NVFP4_DIR):
        self.dir = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self._hdr = {}   # shard -> (header dict, data_base_offset)

    def _header(self, shard):
        if shard not in self._hdr:
            with open(os.path.join(self.dir, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            self._hdr[shard] = (hdr, 8 + n)
        return self._hdr[shard]

    def has(self, name):
        return name in self.map

    def _read(self, shard, lo, hi):
        with open(os.path.join(self.dir, shard), "rb") as f:
            f.seek(lo)
            return f.read(hi - lo + 1)

    def get(self, name, device="cpu"):
        shard = self.map[name]
        hdr, base = self._header(shard)
        info = hdr[name]
        dt, isz = _ST_DT[info["dtype"]]
        start, end = info["data_offsets"]
        raw = self._read(shard, base + start, base + end - 1)
        t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(info["shape"])
        return t.to(device) if device != "cpu" else t

    def get_expert(self, name, e, device="cpu"):
        """One expert slice from a [E, ...] tensor, reading only that slice."""
        shard = self.map[name]
        hdr, base = self._header(shard)
        info = hdr[name]
        dt, isz = _ST_DT[info["dtype"]]
        shape = info["shape"]
        per = 1
        for s in shape[1:]:
            per *= s
        start = info["data_offsets"][0]
        lo = base + start + e * per * isz
        raw = self._read(shard, lo, lo + per * isz - 1)
        t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape[1:])
        return t.to(device) if device != "cpu" else t

    def layer_keys(self, i):
        p = f"model.llm.layers.{i}."
        return [k for k in self.map if k.startswith(p)]


def deinterleave(t, dim):
    """transformers' Interleave(dim): (g0,u0,g1,u1,...) -> [gate_block; up_block].
    Safe on the PACKED uint8 experts because FP4 packing is on the LAST dim, and
    for w13 the interleave axis (intermediate) is dim 1, distinct from packing."""
    shape = list(t.shape)
    shape[dim:dim + 1] = [shape[dim] // 2, 2]
    return t.reshape(shape).transpose(dim, dim + 1).reshape(t.shape).contiguous()


class PackedExperts:
    """Holds one layer's routed experts PACKED on GPU; dequantizes per-expert.

    Monkeypatched onto the library InklingExperts instance so the library's exact
    routing/accumulation loop is reused; only the two `self.gate_up_proj[e]` /
    `self.down_proj[e]` reads are replaced by JIT dequant. For layer 2 (BF16,
    not quantized) `bf16=True` stores plain bf16 weights instead.
    """

    def __init__(self, gate_up, gu_scale, gu_scale2, down, d_scale, d_scale2, bf16_gu=None, bf16_down=None):
        self.gate_up, self.gu_scale, self.gu_scale2 = gate_up, gu_scale, gu_scale2
        self.down, self.d_scale, self.d_scale2 = down, d_scale, d_scale2
        self.bf16_gu, self.bf16_down = bf16_gu, bf16_down

    def gate_up_e(self, e):
        if self.bf16_gu is not None:                        # layer 2: bf16 experts live on CPU
            return self.bf16_gu[e].to("cuda", non_blocking=True)
        return dequant_nvfp4(self.gate_up[e], self.gu_scale[e], self.gu_scale2[e], torch.bfloat16)

    def down_e(self, e):
        if self.bf16_down is not None:
            return self.bf16_down[e].to("cuda", non_blocking=True)
        return dequant_nvfp4(self.down[e], self.d_scale[e], self.d_scale2[e], torch.bfloat16)


def patched_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Byte-for-byte the library InklingExperts.forward, except gate_up_proj[e]
    / down_proj[e] come from JIT dequant via self._packed."""
    import torch.nn.functional as F
    pk = self._packed
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for ei in hit:
        ei = ei[0]
        if ei == self.num_experts:
            continue
        pos, tok = torch.where(mask[ei])
        cur = hidden_states[tok]
        gate, up = F.linear(cur, pk.gate_up_e(ei)).chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        h = F.linear(h, pk.down_e(ei))
        h = h * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, h.to(final.dtype))
    return final


# ------------------------------------------------------------------ measurement

class Meter:
    """Per-(layer,text) accumulators over backward distance d. Per-head, ALWAYS.
    Never aggregated over heads here -- analysis joins Round 3's taxonomy first.

    Accumulation is by DIAGONAL, not scatter: for a query chunk we gather each
    tensor along keys so that axis d indexes distance q-k, then sum over queries
    to [H, dmax]. That reduces ~q*K per-element scatters (the original f64 sink)
    to one gather + one reduction per chunk, in f32.
    """

    def __init__(self, n_heads, dmax, device="cuda"):
        self.n_heads, self.dmax = n_heads, dmax
        z = lambda: torch.zeros(n_heads, dmax, dtype=torch.float64, device=device)
        self.mass_with, self.mass_without = z(), z()
        self.bias_sum, self.content_sum = z(), z()
        self.count = torch.zeros(dmax, dtype=torch.float64, device=device)

    def add_chunk(self, w_with, w_without, bias, content, q_start, sliding, window):
        """w_*/bias/content: [H, qc, K]. Sums each tensor along diagonals of
        constant distance d=q-k into [H, dmax]. Causality is the key-index bound
        (k>=0); sliding layers additionally drop d>=window."""
        H, qc, K = w_with.shape
        dev = w_with.device
        qpos = q_start + torch.arange(qc, device=dev)
        d = torch.arange(self.dmax, device=dev)
        kidx = qpos[:, None] - d[None, :]                 # [qc, dmax] key index for each (row, d)
        valid = (kidx >= 0) & (kidx < K)
        if sliding:
            valid = valid & (d[None, :] < window)
        gidx = kidx.clamp(0, K - 1).unsqueeze(0).expand(H, -1, -1)   # [H, qc, dmax]
        vf = valid.unsqueeze(0)                            # [1, qc, dmax]
        for acc, src in ((self.mass_with, w_with), (self.mass_without, w_without),
                         (self.bias_sum, bias), (self.content_sum, content)):
            g = torch.gather(src, 2, gidx)                 # [H, qc, dmax]
            acc += (g * vf).sum(1).double()                # -> [H, dmax]
            del g
        self.count += (valid.sum(0)).double()

    def to_npz(self):
        c = self.count.clamp(min=1).cpu().numpy()
        d = dict(mass_with=self.mass_with.cpu().numpy(), mass_without=self.mass_without.cpu().numpy(),
                 bias_sum=self.bias_sum.cpu().numpy(), content_sum=self.content_sum.cpu().numpy(),
                 count=self.count.cpu().numpy())
        d.update(mean_mass_with=d["mass_with"] / c, mean_mass_without=d["mass_without"] / c,
                 mean_bias=d["bias_sum"] / c, mean_content=d["content_sum"] / c)
        return d


# module-level handle the measuring interface reads (set per layer by the runner)
_ACTIVE = {"meter": None, "sliding": False, "window": 512, "qchunk": 512,
           "needle_qpos": None, "needle_rows": None}
# capture state for the follow-up activation pass (r-vectors + needle attention rows)
_CAPTURE = {"enabled": False, "rvec": None}


def compact_relative_logits_forward(self, relative_states, query_positions, key_positions):
    """Replacement for InklingRelativeLogits.forward that returns the COMPACT
    rel_logits [batch, H, q_len, rel_extent] instead of gathering it up to the
    full [batch, H, q_len, kv_len] position bias.

    The original materializes an 8.6GB [1,64,8192,8192] tensor (and the tau
    float-cast in InklingAttention doubles it) before our interface ever runs,
    which pinned VRAM and forced WDDM spill. rel_logits is only
    [1,64,8192,<=1024] (~2GB), and our measuring_attention does the per-chunk
    distance gather itself with a bounded [H, qchunk, K] scratch. Semantically
    identical: bias(q,k) = rel_logits[q, q-k] for 0<=q-k<rel_extent else 0, which
    is exactly what the original gather+masked_fill produced. We stash rel_extent
    so the caller can rebuild the bias; the query_positions/key_positions are
    contiguous-from-offset in prefill. rel_extent is recovered downstream as the
    tensor's last-dim size (the tau float-cast in InklingAttention would drop any
    stashed attribute, so we deliberately rely on shape, not an attribute).

    Capture hook: relative_states [1, q, H, d_rel] are the per-token r-vectors
    that drive the positional table -- the live-activation counterpart to the
    weight-space read directions (D0 V). Stashed here for C2/C3 enrichment.
    """
    if _CAPTURE["enabled"]:
        _CAPTURE["rvec"] = relative_states.detach()[0].to("cpu", torch.float16)  # [q,H,d_rel]
    return (relative_states @ self.proj).transpose(1, 2)   # [b, H, q, rel_extent]


def measuring_attention(module, query, key, value, attention_mask, scaling,
                        dropout=0.0, position_bias=None, **kwargs):
    """Attention interface with the twice-softmax measurement baked in.

    Signature matches transformers' eager interface; registered so
    InklingAttention.forward calls it. `position_bias` here is the COMPACT
    rel_logits [1, H, Q, rel_extent] from compact_relative_logits_forward (tau
    already applied upstream; tau==1 for all pos<128000, so identity here).
    Per query-chunk we gather the chunk's bias [H, qc, K] from rel_logits by
    distance, so the largest live tensor scales with qchunk, not Q*K. Returns the
    TRUE with-bias output; the without-bias softmax is measurement only and never
    feeds the residual stream.
    """
    meter = _ACTIVE["meter"]
    sliding, window, qchunk = _ACTIVE["sliding"], _ACTIVE["window"], _ACTIVE["qchunk"]
    groups = module.num_key_value_groups
    kx = key.repeat_interleave(groups, dim=1)
    vx = value.repeat_interleave(groups, dim=1)
    H, Q, D = query.shape[1], query.shape[2], query.shape[3]
    K = kx.shape[2]
    extent = position_bias.shape[-1] if position_bias is not None else 0
    out = torch.empty(1, H, Q, D, dtype=query.dtype, device=query.device)
    kpos = torch.arange(K, device=query.device)
    neg = torch.finfo(torch.float32).min

    for s in range(0, Q, qchunk):
        e = min(s + qchunk, Q)
        qpos = torch.arange(s, e, device=query.device)
        d = qpos[:, None] - kpos[None, :]                          # [qc, K] backward distance
        causal = d >= 0
        if sliding:
            causal = causal & (d < window)
        content = (torch.matmul(query[:, :, s:e], kx.transpose(2, 3)) * scaling)[0]   # [H,qc,K]
        if position_bias is not None:
            rel_chunk = position_bias[0, :, s:e, :]                # [H, qc, extent]
            in_ext = (d >= 0) & (d < extent)                       # [qc, K]
            gi = d.clamp(0, extent - 1).unsqueeze(0).expand(H, -1, -1)   # [H, qc, K]
            b = torch.gather(rel_chunk, 2, gi).masked_fill(~in_ext.unsqueeze(0), 0.0)
        else:
            b = torch.zeros_like(content)
        m = ~causal
        cf = content.float()
        # A6 dtype boundary: stock eager adds content+bias in BF16 (rounding included),
        # then softmaxes in FP32 -- replicate exactly (was: separate upcasts, FP32 add).
        w_with = torch.softmax((content + b).float().masked_fill(m, neg), -1)
        if meter is not None:
            w_without = torch.softmax(cf.masked_fill(m, neg), -1)
            meter.add_chunk(w_with, w_without, b.float(), cf, s, sliding, window)
            del w_without
        # needle capture: full per-head attention row for recall query positions
        nq = _ACTIVE["needle_qpos"]
        if nq is not None:
            for qp in nq:
                if s <= qp < e:
                    _ACTIVE["needle_rows"][int(qp)] = w_with[:, qp - s, :].detach().to("cpu", torch.float16)
        out[:, :, s:e] = torch.matmul(w_with.to(query.dtype).unsqueeze(0), vx)
        del content, b, cf, w_with
    # library expects (attn_output[b,q,h,d], attn_weights)
    return out.transpose(1, 2).contiguous(), None
