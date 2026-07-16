"""
Parity test for the #2 fix: prove the compact-rel_logits path reconstructs the
EXACT same position bias as Inkling's original full [1,H,Q,K] gather.

The original InklingRelativeLogits.forward gathers rel_logits into a dense
[1,H,Q,K] bias and masks outside [0, rel_extent). Our replacement returns the
compact [1,H,Q,extent] rel_logits and measuring_attention rebuilds the per-chunk
bias by distance. If these disagree anywhere, the with-bias softmax -- hence the
whole measurement -- is wrong. This checks them bit-for-bit at a size where the
original path still fits.
"""
import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier2_stream as T
from transformers.models.inkling.modeling_inkling import InklingRelativeLogits

torch.manual_seed(0)


def original_bias(proj, rel_extent, relative_states, qpos, kpos):
    """Verbatim semantics of the stock InklingRelativeLogits.forward."""
    rel_logits = (relative_states @ proj).transpose(1, 2)                 # [b,H,q,extent]
    distance = (qpos[:, None] - kpos[None, :])[None, None, :, :]
    gi = distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    bias = rel_logits.gather(-1, gi)
    return bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)  # [b,H,q,K]


def compact_then_chunk_bias(proj, rel_extent, relative_states, qpos, kpos, qchunk):
    """Our path: compact rel_logits, then the per-chunk distance gather from
    measuring_attention, reassembled into a full [1,H,Q,K] for comparison."""
    rel = (relative_states @ proj).transpose(1, 2)          # [1,H,q,extent] (compact)
    H, Q, extent = rel.shape[1], rel.shape[2], rel.shape[3]
    K = kpos.shape[0]
    full = torch.zeros(1, H, Q, K, dtype=rel.dtype)
    for s in range(0, Q, qchunk):
        e = min(s + qchunk, Q)
        d = qpos[s:e][:, None] - kpos[None, :]              # [qc,K]
        rel_chunk = rel[0, :, s:e, :]                       # [H,qc,extent]
        in_ext = (d >= 0) & (d < extent)
        gi = d.clamp(0, extent - 1).unsqueeze(0).expand(H, -1, -1)
        b = torch.gather(rel_chunk, 2, gi).masked_fill(~in_ext.unsqueeze(0), 0.0)
        full[0, :, s:e, :] = b
    return full


def main():
    H, Q, d_rel = 64, 300, 16
    for rel_extent in (512, 1024):
        proj = torch.randn(d_rel, rel_extent)
        rs = torch.randn(1, Q, H, d_rel)
        qpos = torch.arange(Q)
        kpos = torch.arange(Q)
        a = original_bias(proj, rel_extent, rs, qpos, kpos)
        b = compact_then_chunk_bias(proj, rel_extent, rs, qpos, kpos, qchunk=128)
        assert a.shape == b.shape, (a.shape, b.shape)
        maxdiff = (a - b).abs().max().item()
        exact = torch.equal(a, b)
        print(f"rel_extent={rel_extent}: max|orig-compact|={maxdiff:.2e}  bit_exact={exact}")
        assert exact, f"PARITY FAIL at rel_extent={rel_extent}"
    print("PARITY PASS: compact path reproduces the original bias bit-for-bit.")


if __name__ == "__main__":
    main()
