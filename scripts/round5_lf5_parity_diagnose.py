"""Diagnose LF5 CPU parity without computing any registered semantic outcome.

The registered CPU gate initially promoted BF16 capture/checkpoint operands to
FP32 and kept all intermediate results in FP32.  This script tests whether that
changes the reconstructed model by restoring the BF16 *output boundaries* of
the original Q/K projections, K convolution, Q/K RMS norms, relative
projection, and QK matmul while still executing every CPU operator in FP32.

It is diagnostic only: it does not alter a threshold, overwrite a failed
parity report, or write LF5 pair rows.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from round5_offline_attention import (
    DEFAULT_CAPTURE,
    DEFAULT_INPUTS,
    HEAD_DIM,
    N_HEADS,
    OfflineAttention,
    atomic_json,
    residual_short_convolution,
    rms_norm,
    safe_kl,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "analysis" / "round5" / "lf5" / "parity_cpu_boundary_diagnostic.json"


def bf16_boundary(tensor: torch.Tensor) -> torch.Tensor:
    """Round to BF16 and expand back to FP32 for the next CPU operator."""

    return tensor.to(torch.bfloat16).float()


def model_rms_boundary(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 RMS reduction with the two BF16 boundaries in model RMSNorm."""

    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    normalized = bf16_boundary(hidden.float() * torch.rsqrt(variance + 1e-6))
    return bf16_boundary(normalized * weight.float())


@torch.no_grad()
def boundary_rows(
    layer: int,
    qpos: np.ndarray,
    *,
    input_root: Path,
    capture_root: Path,
) -> np.ndarray:
    with OfflineAttention(
        layer,
        backend="cpu",
        input_root=input_root,
        capture_root=capture_root,
    ) as instrument:
        instrument.prepare_text("05_needles")
        assert instrument.hidden is not None
        assert instrument.rvec is not None

        hidden = instrument.hidden.float()
        hidden3 = hidden.unsqueeze(0)

        key_projected = bf16_boundary(F.linear(hidden3, instrument.wk.float()))
        work = key_projected.float()
        convolved = F.conv1d(
            work.transpose(1, 2),
            instrument.k_conv.float(),
            padding=instrument.k_conv.shape[-1] - 1,
            groups=work.shape[-1],
        )[:, :, : work.shape[1]].transpose(1, 2)
        key_projected = bf16_boundary(convolved + work)
        key_states = key_projected.view(
            1, instrument.seq, instrument.n_kv_heads, HEAD_DIM
        )
        key_states = model_rms_boundary(key_states, instrument.k_norm_weight)
        key_states = key_states.transpose(1, 2).contiguous()[0]
        key_states = key_states.repeat_interleave(instrument.groups, dim=0)

        query = bf16_boundary(F.linear(hidden[qpos.tolist()], instrument.wq.float()))
        query = query.view(len(qpos), N_HEADS, HEAD_DIM)
        query = model_rms_boundary(query, instrument.q_norm_weight)

        # The captured FP16 r-vector was converted to BF16 by the original model
        # path before its BF16 relative-table matmul.
        rvec = bf16_boundary(instrument.rvec[qpos.tolist()])
        rel_curves = bf16_boundary(rvec @ instrument.rel_proj.float())

        # Original BF16 QK matmul output is rounded before the FP32 bias add and
        # softmax.  Scaling by 1/128 is exact, but retain the boundary explicitly.
        content = torch.einsum("nhd,hsd->nhs", query, key_states)
        content = bf16_boundary(content) * (1.0 / HEAD_DIM)

        key_pos = torch.arange(instrument.seq)
        output = torch.zeros((len(qpos), N_HEADS, instrument.seq), dtype=torch.float32)
        for index, q_raw in enumerate(qpos):
            q = int(q_raw)
            distance = q - key_pos
            causal = distance >= 0
            if instrument.is_sliding:
                causal &= distance < 512
            in_extent = (distance >= 0) & (distance < instrument.extent)
            gather = distance.clamp(0, instrument.extent - 1).unsqueeze(0).expand(N_HEADS, -1)
            bias = torch.gather(rel_curves[index], 1, gather).masked_fill(
                ~in_extent.unsqueeze(0), 0.0
            )
            logits = (content[index] + bias).masked_fill(
                ~causal.unsqueeze(0), torch.finfo(torch.float32).min
            )
            output[index] = torch.softmax(logits, dim=-1)
        return output.numpy()


@torch.no_grad()
def native_cpu_bf16_rows(
    layer: int,
    qpos: np.ndarray,
    *,
    input_root: Path,
    capture_root: Path,
) -> np.ndarray:
    """Model dtype graph on CPU; diagnostic, not the registered CPU/FP32 path."""

    with OfflineAttention(
        layer,
        backend="cpu",
        input_root=input_root,
        capture_root=capture_root,
    ) as instrument:
        instrument.prepare_text("05_needles")
        assert instrument.hidden is not None
        assert instrument.rvec is not None

        hidden = instrument.hidden.to(torch.bfloat16)
        hidden3 = hidden.unsqueeze(0)
        key_projected = F.linear(hidden3, instrument.wk.to(torch.bfloat16))
        key_projected = residual_short_convolution(key_projected, instrument.k_conv)
        key_states = key_projected.view(
            1, instrument.seq, instrument.n_kv_heads, HEAD_DIM
        )
        key_states = rms_norm(key_states, instrument.k_norm_weight.to(torch.bfloat16))
        key_states = key_states.transpose(1, 2).contiguous()[0]
        key_states = key_states.repeat_interleave(instrument.groups, dim=0)

        query = F.linear(hidden[qpos.tolist()], instrument.wq.to(torch.bfloat16))
        query = query.view(len(qpos), N_HEADS, HEAD_DIM)
        query = rms_norm(query, instrument.q_norm_weight.to(torch.bfloat16))
        rvec = instrument.rvec[qpos.tolist()].to(torch.bfloat16)
        rel_curves = rvec @ instrument.rel_proj.to(torch.bfloat16)
        content = torch.einsum("nhd,hsd->nhs", query, key_states) * (1.0 / HEAD_DIM)

        key_pos = torch.arange(instrument.seq)
        output = torch.zeros((len(qpos), N_HEADS, instrument.seq), dtype=torch.float32)
        for index, q_raw in enumerate(qpos):
            q = int(q_raw)
            distance = q - key_pos
            causal = distance >= 0
            if instrument.is_sliding:
                causal &= distance < 512
            in_extent = (distance >= 0) & (distance < instrument.extent)
            gather = distance.clamp(0, instrument.extent - 1).unsqueeze(0).expand(N_HEADS, -1)
            bias = torch.gather(rel_curves[index], 1, gather).masked_fill(
                ~in_extent.unsqueeze(0), 0.0
            )
            logits = (content[index].float() + bias.float()).masked_fill(
                ~causal.unsqueeze(0), torch.finfo(torch.float32).min
            )
            output[index] = torch.softmax(logits, dim=-1)
        return output.numpy()


@torch.no_grad()
def ordinary_rows(
    layer: int,
    backend: str,
    qpos: np.ndarray,
    *,
    input_root: Path,
    capture_root: Path,
) -> np.ndarray:
    with OfflineAttention(
        layer,
        backend=backend,
        input_root=input_root,
        capture_root=capture_root,
    ) as instrument:
        rows = instrument.rows("05_needles", qpos.tolist(), compact=False)
    return np.stack([row.attention_with_bias for row in rows]).astype(np.float32)


def metrics(original: np.ndarray, offline: np.ndarray) -> dict[str, float | int]:
    delta = np.abs(offline - original)
    argmax_equal = np.argmax(offline, axis=-1) == np.argmax(original, axis=-1)
    row_sum_error = np.abs(offline.sum(axis=-1, dtype=np.float64) - 1.0)
    kl = safe_kl(original, offline)
    return {
        "max_abs_delta": float(delta.max()),
        "p99_abs_delta": float(np.quantile(delta, 0.99)),
        "max_row_sum_error": float(row_sum_error.max()),
        "argmax_agreement": float(argmax_equal.mean()),
        "argmax_mismatches": int(np.count_nonzero(~argmax_equal)),
        "max_kl": float(kl.max()),
        "p99_kl": float(np.quantile(kl, 0.99)),
    }


def stored_fp16(probability: np.ndarray) -> np.ndarray:
    return probability.astype(np.float16).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    needle_path = args.capture_root / f"needlerows_L{args.layer:02d}.npz"
    with np.load(needle_path) as source:
        qpos = source["qpos"].astype(np.int64)
        original = source["rows"].astype(np.float32)

    started = time.time()
    boundary = boundary_rows(
        args.layer,
        qpos,
        input_root=args.input_root,
        capture_root=args.capture_root,
    )
    cpu = ordinary_rows(
        args.layer,
        "cpu",
        qpos,
        input_root=args.input_root,
        capture_root=args.capture_root,
    )
    replay = ordinary_rows(
        args.layer,
        "replay",
        qpos,
        input_root=args.input_root,
        capture_root=args.capture_root,
    )
    native_bf16 = native_cpu_bf16_rows(
        args.layer,
        qpos,
        input_root=args.input_root,
        capture_root=args.capture_root,
    )
    variants = {
        "cpu_fp32_pre_storage": metrics(original, cpu),
        "cpu_fp16_stored": metrics(original, stored_fp16(cpu)),
        "cpu_boundary_emulation_pre_storage": metrics(original, boundary),
        "cpu_boundary_emulation_fp16_stored": metrics(original, stored_fp16(boundary)),
        "cpu_native_bf16_pre_storage_diagnostic": metrics(original, native_bf16),
        "cpu_native_bf16_fp16_stored_diagnostic": metrics(
            original, stored_fp16(native_bf16)
        ),
        "exact_replay_pre_storage_control": metrics(original, replay),
        "exact_replay_fp16_stored_control": metrics(original, stored_fp16(replay)),
    }
    thresholds = {
        "max_abs_delta": 1e-3,
        "max_row_sum_error": 1e-3,
        "argmax_agreement": 1.0,
        "max_kl": 1e-6,
    }
    checks = {}
    for name, result in variants.items():
        checks[name] = {
            key: (result[key] == value if key == "argmax_agreement" else result[key] <= value)
            for key, value in thresholds.items()
        }
    report = {
        "schema_version": 1,
        "kind": "round5_lf5_cpu_boundary_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "layer": args.layer,
        "queries": int(len(qpos)),
        "heads": N_HEADS,
        "operator_backend": "CPU/FP32 with explicit original BF16 output boundaries",
        "diagnostic_only": True,
        "source_sha256": sha256_file(Path(__file__)),
        "input_manifest_sha256": sha256_file(args.input_root / "manifest.json"),
        "needle_rows_sha256": sha256_file(needle_path),
        "thresholds_unchanged": thresholds,
        "variants": variants,
        "checks": checks,
        "would_pass_registered_thresholds": {
            name: bool(all(result.values())) for name, result in checks.items()
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
