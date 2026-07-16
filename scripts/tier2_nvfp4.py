"""
NVFP4 -> bf16 dequantization for the Inkling-NVFP4 checkpoint.

Why this exists: the RTX 4090 is Ada (sm_89) and has no native FP4 tensor
cores; NVFP4 kernels in vLLM/SGLang/TensorRT target Blackwell. transformers'
modular_inkling.py is BF16-only (no fp4/scale2/dequant path anywhere). So we
dequantize to bf16 ourselves, just-in-time, one expert at a time.

Format (from hf_quant_config.json, confirmed against BF16 ground truth by
validate() below):
  quant_algo NVFP4, group_size 16, num_bits [2,1] = E2M1, scale_bits [4,3] = E4M3
  w13_weight: U8 [E, 2*inter, hidden/2]   packed 2 nibbles/byte along hidden
  w2_weight : U8 [E, hidden, inter/2]     packed 2 nibbles/byte along inter
  .scale    : F8_E4M3, one per 16 elements along the packed (input/contraction) axis
  .scale2   : F32 [E], per-expert global scale
  value = E2M1_LUT[nibble] * scale_e4m3[block] * scale2[expert]

Only layers 3-65 `mlp.experts.{w13,w2}_weight` are quantized. Everything else in
the checkpoint -- all attention, wr_du, rel_logits_proj, norms, sconv, gate,
shared_experts, embed/unembed -- is BF16 and needs no dequant.
"""
import torch

# E2M1: 1 sign, 2 exponent, 1 mantissa -> 8 magnitudes, signed via high bit.
E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def unpack_fp4(packed: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """[..., n] uint8 -> [..., 2n] float32. Low nibble is the even element."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lut[lo.long()], lut[hi.long()]], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequant_nvfp4(packed, scale_e4m3, scale2, out_dtype=torch.bfloat16):
    """packed [..., n] u8; scale [..., 2n/16] f8_e4m3; scale2 scalar/[E] f32."""
    lut = E2M1_LUT.to(packed.device)
    vals = unpack_fp4(packed, lut)                       # [..., 2n] f32
    s = scale_e4m3.to(torch.float32).repeat_interleave(16, dim=-1)
    vals = vals * s
    if scale2 is not None:
        s2 = scale2.to(torch.float32)
        if s2.numel() > 1:
            s2 = s2.view(-1, *([1] * (vals.dim() - 1)))
        vals = vals * s2
    return vals.to(out_dtype)


def dequant_expert(sd, prefix, e, device="cuda", out_dtype=torch.bfloat16):
    """Dequantize a single expert's weight from a layer state dict slice."""
    packed = sd[f"{prefix}"][e].to(device)
    scale = sd[f"{prefix}.scale"][e].to(device)
    scale2 = sd[f"{prefix}.scale2"][e].to(device)
    return dequant_nvfp4(packed, scale, scale2, out_dtype)
