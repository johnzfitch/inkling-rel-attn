"""
Validate the NVFP4 dequant against BF16 ground truth.

Fetches ONE expert (layer 3, expert 0, w13) from both the NVFP4 repo and the
BF16 repo via HTTP Range requests (~95MB total) and compares. This settles the
format -- nibble order, scale application, axis layout -- without downloading
the 1.9TB BF16 checkpoint.

Pass criteria: NVFP4 is lossy, so we do NOT expect exact equality -- and an
absolute error threshold is the wrong test. Measured rel_err is ~0.087, which
is NOT a bug: an IDEAL nearest-representable-value quantizer on this same
weight distribution scores 0.0869 too. FP4's intrinsic floor is ~8.7% here.
So the principled criteria are:
  - correlation > 0.99
  - rel_err within 2% of the ideal quantizer's rel_err (i.e. we sit at the
    quantization floor, with no systematic scale/layout error on top)
  - the WRONG hypotheses (swapped nibble order, missing scale2) score clearly
    worse, which is what makes this a real test rather than a rubber stamp.
"""
import json
import struct
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, r"R:\inkling\scripts")
from tier2_nvfp4 import E2M1_LUT, dequant_nvfp4, unpack_fp4

NV = "https://huggingface.co/thinkingmachines/Inkling-NVFP4/resolve/main/"
BF = "https://huggingface.co/thinkingmachines/Inkling/resolve/main/"
LAYER, EXPERT = 3, 0

_hdr_cache = {}


def header(base, shard):
    if (base, shard) in _hdr_cache:
        return _hdr_cache[(base, shard)]
    url = base + shard
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"}), timeout=60)
    n = struct.unpack("<Q", r.read(8))[0]
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=8-{8+n-1}"}), timeout=120)
    hdr = json.loads(r.read(n))
    _hdr_cache[(base, shard)] = (hdr, 8 + n)
    return _hdr_cache[(base, shard)]


DT = {"BF16": (torch.bfloat16, 2), "F8_E4M3": (torch.float8_e4m3fn, 1), "F32": (torch.float32, 4),
      "U8": (torch.uint8, 1), "I64": (torch.int64, 8)}


def fetch_slice(base, index, name, expert=None):
    """Fetch a tensor, or one expert's contiguous slice along dim 0."""
    shard = index[name]
    hdr, off = header(base, shard)
    info = hdr[name]
    dt, isz = DT[info["dtype"]]
    shape = info["shape"]
    start, end = info["data_offsets"]
    if expert is not None:
        per = int(np.prod(shape[1:]))
        lo = off + start + expert * per * isz
        hi = lo + per * isz - 1
        out_shape = shape[1:]
    else:
        lo, hi = off + start, off + end - 1
        out_shape = shape
    req = urllib.request.Request(base + shard, headers={"Range": f"bytes={lo}-{hi}"})
    raw = urllib.request.urlopen(req, timeout=600).read()
    t = torch.frombuffer(bytearray(raw), dtype=dt)
    return t.reshape(out_shape)


def check_case(nv_idx, bf_idx, layer, expert, which):
    """Validate one (layer, expert, w13|w2) against BF16 ground truth.
    w13 is interleaved (dim 1 on the expert slice); w2 is a plain rename."""
    from tier2_stream import deinterleave
    from transformers.core_model_loading import Interleave
    p = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
    packed = fetch_slice(NV, nv_idx, p, expert)
    scale = fetch_slice(NV, nv_idx, p + ".scale", expert)
    scale2 = fetch_slice(NV, nv_idx, p + ".scale2")[expert]
    ref_raw = fetch_slice(BF, bf_idx, p, expert).float()

    if which == "w13":
        # On the FULL tensor [E, 2*inter, hidden] the interleave axis is dim 1
        # (what build_layer uses); on this per-expert slice [2*inter, hidden] it
        # is dim 0. Interleave(dim=0) on the BF16 ref matches.
        packed_d = deinterleave(packed, 0)
        scale_d = deinterleave(scale, 0)
        ref = Interleave(dim=0).convert({"x": ref_raw}, None, ["y"])["y"]
    else:
        packed_d, scale_d, ref = packed, scale, ref_raw

    got = dequant_nvfp4(packed_d, scale_d, scale2, torch.float32)
    if got.shape != ref.shape:
        print(f"  L{layer} e{expert} {which}: SHAPE {tuple(got.shape)} vs {tuple(ref.shape)}  -> FAIL")
        return False
    corr = float(torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1])
    rel = float((got - ref).norm() / ref.norm())

    # ideal nearest-representable quantizer using the SAME stored block scales.
    sub = slice(0, min(512, ref.shape[0]))
    r_, g_ = ref[sub], got[sub]
    bs = scale_d[sub].float().repeat_interleave(16, -1) * scale2.float()
    cand = E2M1_LUT.view(1, 1, 16) * bs.unsqueeze(-1)
    nearest = cand.gather(-1, (cand - r_.unsqueeze(-1)).abs().argmin(-1, keepdim=True)).squeeze(-1)
    ideal = float((nearest - r_).norm() / r_.norm())
    ours = float((g_ - r_).norm() / r_.norm())

    # wrong hypotheses that MUST score worse (both used in the decision now).
    lo = E2M1_LUT[(packed_d & 0x0F).long()]; hi = E2M1_LUT[((packed_d >> 4) & 0x0F).long()]
    swap = torch.stack([hi, lo], -1).reshape(*packed_d.shape[:-1], packed_d.shape[-1] * 2)
    swap = swap * scale_d.float().repeat_interleave(16, -1) * scale2.float()
    c_swap = float(torch.corrcoef(torch.stack([swap.flatten(), ref.flatten()]))[0, 1])
    nos2 = dequant_nvfp4(packed_d, scale_d, None, torch.float32)
    rel_nos2 = float((nos2 - ref).norm() / ref.norm())

    # two-sided floor: within 2% ABOVE ideal (not optimal) and not implausibly
    # BELOW it (would mean we somehow beat the optimal quantizer -> a bug/leak).
    at_floor = (ours <= ideal * 1.02) and (ours >= ideal * 0.98)
    beats_swap = corr > c_swap + 0.05                       # nibble order discriminated
    beats_nos2 = rel_nos2 > rel * 5                          # scale2 genuinely required
    ok = corr > 0.99 and at_floor and beats_swap and beats_nos2
    print(f"  L{layer:2d} e{expert} {which:3}: corr={corr:.5f} rel={rel:.4f} "
          f"floor(ideal={ideal:.4f}/ours={ours:.4f})={at_floor} "
          f"swap={c_swap:+.3f} nos2_rel={rel_nos2:.1f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("fetching indices...")
    nv_idx = json.load(urllib.request.urlopen(NV + "model.safetensors.index.json", timeout=120))["weight_map"]
    bf_idx = json.load(urllib.request.urlopen(BF + "model.safetensors.index.json", timeout=120))["weight_map"]
    # battery: both tensors, a shallow and a deep MoE layer, different experts.
    cases = [(3, 0, "w13"), (3, 0, "w2"), (40, 7, "w13"), (40, 7, "w2")]
    print(f"validating {len(cases)} (layer, expert, tensor) cases vs BF16 ground truth:")
    results = [check_case(nv_idx, bf_idx, L, e, w) for (L, e, w) in cases]
    ok = all(results)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  ({sum(results)}/{len(results)} cases)")
    print("  rel_err ~0.087 is NVFP4's intrinsic floor (ideal quantizer scores the same), not a failure.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
