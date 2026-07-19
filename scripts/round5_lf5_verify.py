"""Independent LF5 verification and confirmation manifest.

This file deliberately does not import round5_offline_attention or
round5_lf5_analyze. Under Amendment A5 it recomputes six frozen sentinel rows
with a direct CUDA/BF16 implementation from BF16-bit inputs and raw checkpoint
tensors, then independently repeats the registered bracket statistic from the
ragged FP16 row dump. The failed CPU/FP32 gate remains a required negative
provenance record, not a production criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from tier2_stream import ShardReader  # noqa: E402


NVFP4 = ROOT / "nvfp4"
INPUTS = ROOT / "dumps" / "round5" / "attention_inputs"
OLD_CAPTURE = ROOT / "dumps" / "tier2" / "capture"
ROW_DUMP = ROOT / "dumps" / "round5" / "lf5"
LOCI = ROOT / "analysis" / "round5" / "loci.json"
CAPTURE_VALIDATION = ROOT / "analysis" / "round5" / "capture_validation.json"
REPLAY_REPORT = ROOT / "analysis" / "round5" / "lf5" / "parity_replay.json"
CPU_REPORT = ROOT / "analysis" / "round5" / "lf5" / "parity_cpu_sentinels.json"
AMENDMENT_A5 = ROOT / "registrations" / "ROUND5_AMENDMENT_A5.md"
BRACKET_REPORT = ROOT / "analysis" / "round5" / "lf5" / "brackets.json"
ROW_DUMP_VALIDATION = ROOT / "analysis" / "round5" / "lf5" / "row_dump_validation.json"
VERIFY_REPORT = ROOT / "analysis" / "round5" / "lf5" / "verification.json"
CONFIRMATION = ROOT / "analysis" / "round5" / "lf5" / "confirmation.json"
REGISTRATION_COMMIT = "34278b4"
PLAN_COMMIT = "d4e2579"
AMENDMENT_A5_COMMIT = "7bf608d9971997a655a4f9cd46e3bc921ffb74b8"
SENTINEL_LAYERS = [0, 5, 11, 23, 41, 65]
GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
FP16_MIN_POSITIVE = float(np.nextafter(np.float16(0), np.float16(1)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def bf16_bits_to_float32(path: Path) -> np.ndarray:
    words = np.load(path, mmap_mode="r")
    if words.dtype != np.uint16:
        raise TypeError(words.dtype)
    expanded = np.asarray(words, dtype=np.uint32) << np.uint32(16)
    return expanded.view(np.float32)


def rms_norm_numpy(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    return (weight * values * (1.0 / np.sqrt(variance + np.float32(1e-6)))).astype(np.float32)


def k_convolution_numpy(projected: np.ndarray, weight: np.ndarray) -> np.ndarray:
    result = projected.copy()
    taps = weight[:, 0, :]
    for tap in range(4):
        lag = 3 - tap
        if lag:
            result[lag:] += projected[:-lag] * taps[:, tap]
        else:
            result += projected * taps[:, tap]
    return result.astype(np.float32)


def softmax_numpy(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = logits.astype(np.float32, copy=True)
    masked[:, ~valid] = np.finfo(np.float32).min
    maximum = np.max(masked, axis=-1, keepdims=True)
    exponent = np.exp(masked - maximum).astype(np.float32)
    return (exponent / np.sum(exponent, axis=-1, keepdims=True, dtype=np.float32)).astype(np.float32)


def weight_numpy(reader: ShardReader, name: str) -> np.ndarray:
    return reader.get(name, "cpu").float().numpy()


def sentinel_row(layer: int, q: int, reader: ShardReader) -> np.ndarray:
    is_global = layer in GLOBAL_LAYERS
    n_kv = 8 if is_global else 16
    groups = 64 // n_kv
    extent = 1024 if is_global else 512
    prefix = f"model.llm.layers.{layer}.attn."

    hidden = bf16_bits_to_float32(INPUTS / "normalized" / f"attn_in_L{layer:02d}_05_needles.npy")
    wk = weight_numpy(reader, prefix + "wk_dv.weight")
    k_projected = (hidden @ wk.T).astype(np.float32)
    del wk
    k_weight = weight_numpy(reader, prefix + "k_sconv.weight")
    k_projected = k_convolution_numpy(k_projected, k_weight)
    del k_weight
    k_states = k_projected.reshape(len(hidden), n_kv, 128)
    k_norm = weight_numpy(reader, prefix + "k_norm.weight")
    k_states = rms_norm_numpy(k_states, k_norm)

    wq = weight_numpy(reader, prefix + "wq_du.weight")
    query = (hidden[q] @ wq.T).astype(np.float32).reshape(64, 128)
    del wq, hidden
    q_norm = weight_numpy(reader, prefix + "q_norm.weight")
    query = rms_norm_numpy(query, q_norm)

    content = np.empty((64, len(k_states)), dtype=np.float32)
    for head in range(64):
        content[head] = (k_states[:, head // groups, :] @ query[head]) * np.float32(1.0 / 128)

    rvec = np.asarray(
        np.load(OLD_CAPTURE / f"rvec_L{layer:02d}_05_needles.npy", mmap_mode="r")[q],
        dtype=np.float32,
    )
    projection = weight_numpy(reader, prefix + "rel_logits_proj.proj")
    curves = (rvec @ projection).astype(np.float32)
    key_positions = np.arange(len(k_states), dtype=np.int64)
    distance = q - key_positions
    in_extent = (distance >= 0) & (distance < extent)
    gather = np.clip(distance, 0, extent - 1)
    bias = curves[:, gather]
    bias[:, ~in_extent] = 0.0
    valid = distance >= 0
    if not is_global:
        valid &= distance < 512
    return softmax_numpy(content + bias, valid)


def bf16_bits_to_cuda(path: Path) -> torch.Tensor:
    words = np.load(path, mmap_mode="r")
    if words.dtype != np.uint16 or words.ndim != 2:
        raise TypeError((words.dtype, words.shape))
    tensor = torch.from_numpy(np.array(words, dtype=np.uint16, copy=True)).view(torch.bfloat16)
    return tensor.to("cuda")


def rms_norm_cuda(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    input_dtype = values.dtype
    work = values.float()
    variance = work.pow(2).mean(-1, keepdim=True)
    normalized = (work * torch.rsqrt(variance + 1e-6)).to(input_dtype)
    return weight * normalized


@torch.no_grad()
def gpu_sentinel_row(layer: int, q: int, reader: ShardReader) -> np.ndarray:
    """Independent A5 replay path with original full projections/qchunk shape."""

    if not torch.cuda.is_available():
        raise RuntimeError("A5 independent LF5 verification requires CUDA")
    is_global = layer in GLOBAL_LAYERS
    n_kv = 8 if is_global else 16
    groups = 64 // n_kv
    extent = 1024 if is_global else 512
    prefix = f"model.llm.layers.{layer}.attn."

    hidden = bf16_bits_to_cuda(
        INPUTS / "normalized" / f"attn_in_L{layer:02d}_05_needles.npy"
    )
    hidden3 = hidden.unsqueeze(0)
    wk = reader.get(prefix + "wk_dv.weight", "cuda").to(torch.bfloat16)
    key_projected = F.linear(hidden3, wk)
    del wk
    conv_weight = reader.get(prefix + "k_sconv.weight", "cuda").float()
    work = key_projected.float()
    convolved = F.conv1d(
        work.transpose(1, 2),
        conv_weight,
        padding=conv_weight.shape[-1] - 1,
        groups=work.shape[-1],
    )[:, :, : work.shape[1]].transpose(1, 2)
    key_projected = (convolved + work).to(torch.bfloat16)
    del conv_weight, work, convolved
    seq = hidden.shape[0]
    key_states = key_projected.view(1, seq, n_kv, 128)
    k_norm = reader.get(prefix + "k_norm.weight", "cuda").to(torch.bfloat16)
    key_states = rms_norm_cuda(key_states, k_norm).transpose(1, 2).contiguous()
    del key_projected, k_norm

    wq = reader.get(prefix + "wq_du.weight", "cuda").to(torch.bfloat16)
    query_states = F.linear(hidden3, wq).view(1, seq, 64, 128)
    del hidden, hidden3, wq
    q_norm = reader.get(prefix + "q_norm.weight", "cuda").to(torch.bfloat16)
    query_states = rms_norm_cuda(query_states, q_norm).transpose(1, 2).contiguous()
    del q_norm

    rvec_np = np.load(OLD_CAPTURE / f"rvec_L{layer:02d}_05_needles.npy", mmap_mode="r")
    rvec = torch.from_numpy(np.array(rvec_np, copy=True)).to("cuda").to(torch.bfloat16)
    projection = reader.get(prefix + "rel_logits_proj.proj", "cuda").to(torch.bfloat16)
    relative_logits = (rvec.unsqueeze(0) @ projection).transpose(1, 2).contiguous()
    del rvec, projection
    if is_global:
        positions = torch.arange(seq, device="cuda")
        tau = 1.0 + 0.1 * torch.log(((positions + 1).float() / 128000.0).clamp(min=1.0))
        tau = tau.view(1, 1, -1, 1)
        query_states = (query_states.float() * tau).to(torch.bfloat16)
        relative_logits = (relative_logits.float() * tau).to(torch.bfloat16)

    start = (q // 512) * 512
    end = min(start + 512, seq)
    key_states = key_states.repeat_interleave(groups, dim=1)
    content = torch.matmul(
        query_states[:, :, start:end], key_states.transpose(2, 3)
    ) * (1.0 / 128)
    key_positions = torch.arange(seq, device="cuda")
    query_positions = torch.arange(start, end, device="cuda")
    distance = query_positions[:, None] - key_positions[None, :]
    valid = distance >= 0
    if not is_global:
        valid &= distance < 512
    in_extent = (distance >= 0) & (distance < extent)
    gather = distance.clamp(0, extent - 1).unsqueeze(0).expand(64, -1, -1)
    bias = torch.gather(relative_logits[0, :, start:end], 2, gather).masked_fill(
        ~in_extent.unsqueeze(0), 0.0
    )
    logits = (content[0].float() + bias.float()).masked_fill(
        ~valid.unsqueeze(0), torch.finfo(torch.float32).min
    )
    row = torch.softmax(logits, dim=-1)[:, q - start]
    output = row.float().cpu().numpy()
    del key_states, query_states, relative_logits, content, bias, logits, row
    torch.cuda.empty_cache()
    return output


def safe_kl(original: np.ndarray, offline: np.ndarray) -> np.ndarray:
    p = original.astype(np.float64)
    q = offline.astype(np.float64)
    p /= p.sum(axis=-1, keepdims=True)
    q /= q.sum(axis=-1, keepdims=True)
    mask = p > 0
    terms = np.zeros_like(p)
    terms[mask] = p[mask] * (
        np.log(p[mask]) - np.log(np.maximum(q[mask], np.finfo(np.float64).tiny))
    )
    return terms.sum(axis=-1)


def verify_sentinels() -> tuple[list[dict[str, Any]], bool]:
    reader = ShardReader(str(NVFP4))
    results = []
    passed = True
    for layer in SENTINEL_LAYERS:
        with np.load(OLD_CAPTURE / f"needlerows_L{layer:02d}.npz") as capture:
            q = int(capture["qpos"][0])
            original16 = capture["rows"][0].astype(np.float16)
        original = original16.astype(np.float32)
        offline = gpu_sentinel_row(layer, q, reader)
        delta = np.abs(offline - original)
        row_sum_error = np.abs(offline.sum(axis=-1, dtype=np.float64) - 1.0)
        argmax = np.argmax(offline, axis=-1) == np.argmax(original, axis=-1)
        kl = safe_kl(original, offline)
        offline16 = offline.astype(np.float16)
        mismatch = offline16.view(np.uint16) != original16.view(np.uint16)
        checks = {
            "fp16_bitwise_equal": not bool(np.any(mismatch)),
        }
        cpu_tier_diagnostics = {
            "max_abs_delta": float(np.max(delta)) <= 1e-3,
            "max_row_sum_error": float(np.max(row_sum_error)) <= 1e-3,
            "argmax_agreement": bool(np.all(argmax)),
            "max_kl": float(np.max(kl)) <= 1e-6,
        }
        item = {
            "layer": layer,
            "query": q,
            "max_abs_delta": float(np.max(delta)),
            "max_row_sum_error": float(np.max(row_sum_error)),
            "argmax_agreement": float(np.mean(argmax)),
            "max_kl": float(np.max(kl)),
            "mismatch_words": int(np.count_nonzero(mismatch)),
            "backend": "independent_cuda_bf16_replay",
            "checks": checks,
            "cpu_tier_threshold_diagnostics_not_a5_gate": cpu_tier_diagnostics,
            "passed": all(checks.values()),
        }
        results.append(item)
        passed = passed and item["passed"]
        print(json.dumps(item, sort_keys=True), flush=True)
    return results, bool(passed)


class IndependentRows:
    def __init__(self, path: Path):
        with np.load(path / "index.npz") as index:
            qpos = index["qpos"].astype(np.int64)
            self.start = index["support_start"].astype(np.int64)
            self.ptr = index["indptr"].astype(np.int64)
        self.lookup = {int(q): i for i, q in enumerate(qpos)}
        self.values = np.load(path / "attention_with_bias.npy", mmap_mode="r")

    def probabilities(self, q: int, keys: list[int]) -> np.ndarray:
        index = self.lookup[q]
        offsets = self.ptr[index] + np.asarray(keys) - self.start[index]
        return np.asarray(self.values[offsets], dtype=np.float64).mean(axis=1)


def independent_controls(
    q: int,
    matched: int,
    opener: str,
    openers: dict[str, list[int]],
) -> list[int]:
    target = q - matched
    width = max(8, round(0.1 * target))
    result = [
        k for k in openers[opener]
        if k < q and k != matched and abs((q - k) - target) <= width
    ]
    if len(result) >= 8:
        return result
    target_bin = math.floor(math.log2(max(1, target)))
    result = [
        k for k in openers[opener]
        if k < q
        and k != matched
        and math.floor(math.log2(max(1, q - k))) == target_bin
    ]
    return result if len(result) >= 8 else []


def independent_brackets() -> tuple[list[dict[str, Any]], bool]:
    loci = json.loads(LOCI.read_text(encoding="utf-8"))
    main = json.loads(BRACKET_REPORT.read_text(encoding="utf-8"))
    pairs = loci["texts"]["02_code"]["loci"]["bracket_pairs"]
    openers: dict[str, list[int]] = {"(": [], "[": [], "{": []}
    for pair in pairs:
        openers[pair["opener"]].append(int(pair["opener_token_pos"]))
    openers = {key: sorted(set(value)) for key, value in openers.items()}
    main_by_layer = {int(x["layer"]): x for x in main["results"]}
    results = []
    passed = True

    for layer in GLOBAL_LAYERS:
        store = IndependentRows(ROW_DUMP / f"L{layer:02d}_02_code")
        observed_parts = []
        null_inputs = []
        for pair in pairs:
            q = int(pair["token_pos"])
            matched = int(pair["opener_token_pos"])
            controls = independent_controls(q, matched, pair["opener"], openers)
            if not controls:
                continue
            probabilities = np.maximum(
                store.probabilities(q, [matched, *controls]), FP16_MIN_POSITIVE
            )
            observed_parts.append(
                np.log(probabilities[0]) - np.log(np.median(probabilities[1:]))
            )
            null_inputs.append(probabilities)
        observed = float(np.median(observed_parts))
        seed = int(
            hashlib.sha256(f"{REGISTRATION_COMMIT}|LF5|bracket|{layer}".encode()).hexdigest()[:16],
            16,
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        null = np.empty(10000, dtype=np.float64)
        for iteration in range(10000):
            ratios = []
            for values in null_inputs:
                chosen = int(rng.integers(len(values)))
                ratios.append(np.log(values[chosen]) - np.log(np.median(np.delete(values, chosen))))
            null[iteration] = np.median(ratios)
        p_value = float((1 + np.count_nonzero(null >= observed)) / 10001)
        expected = main_by_layer[layer]
        agreement = {
            "eligible_pairs": len(observed_parts) == int(expected["eligible_pairs"]),
            "effect": abs(observed - float(expected["observed_median_log_ratio"])) <= 1e-12,
            "p": p_value == float(expected["p_one_sided"]),
        }
        item = {
            "layer": layer,
            "eligible_pairs": len(observed_parts),
            "observed_median_log_ratio": observed,
            "p_one_sided": p_value,
            "agreement": agreement,
            "passed": all(agreement.values()),
        }
        results.append(item)
        passed = passed and item["passed"]
    return results, bool(passed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-report", type=Path, default=VERIFY_REPORT)
    parser.add_argument("--confirmation", type=Path, default=CONFIRMATION)
    args = parser.parse_args()

    sentinel_results, sentinel_passed = verify_sentinels()
    bracket_results, bracket_passed = independent_brackets()
    verification = {
        "schema_version": 1,
        "kind": "round5_lf5_independent_verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "amendment_a5_commit": AMENDMENT_A5_COMMIT,
        "production_backend": "replay",
        "source_sha256": sha256_file(Path(__file__)),
        "sentinel_layers": SENTINEL_LAYERS,
        "sentinel_results": sentinel_results,
        "sentinel_passed": sentinel_passed,
        "bracket_results": bracket_results,
        "bracket_reference_agreement": bracket_passed,
        "passed": bool(sentinel_passed and bracket_passed),
    }
    atomic_json(args.verify_report, verification)

    capture_validation = json.loads(CAPTURE_VALIDATION.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY_REPORT.read_text(encoding="utf-8"))
    cpu = json.loads(CPU_REPORT.read_text(encoding="utf-8"))
    dump = json.loads((ROW_DUMP / "manifest.json").read_text(encoding="utf-8"))
    dump_validation = json.loads(ROW_DUMP_VALIDATION.read_text(encoding="utf-8"))
    bracket = json.loads(BRACKET_REPORT.read_text(encoding="utf-8"))
    amendment_text = AMENDMENT_A5.read_text(encoding="utf-8")
    cpu_results = cpu.get("results", [])
    gates = {
        "capture_gate": bool(capture_validation.get("passed") and not capture_validation.get("fast_mode")),
        "replay_gate": bool(replay.get("passed") and len(replay.get("results", [])) == 66),
        "a5_amendment_gate": bool(
            "the production LF5 instrument is the **GPU replay backend**" in amendment_text
            and "No threshold is relaxed" in amendment_text
        ),
        "registered_cpu_failure_preserved_gate": bool(
            cpu.get("passed") is False
            and cpu_results
            and cpu_results[0].get("layer") == 0
            and cpu_results[0].get("passed") is False
        ),
        "row_dump_gate": bool(
            dump.get("complete")
            and dump.get("production_backend") == "replay"
            and len(dump.get("groups", {})) == 396
        ),
        "independent_row_dump_validation_gate": bool(
            dump_validation.get("passed") is True
            and dump_validation.get("errors") == []
            and dump_validation.get("production_backend") == "replay"
            and dump_validation.get("groups_checked") == 396
            and dump_validation.get("dump_manifest_sha256")
            == sha256_file(ROW_DUMP / "manifest.json")
        ),
        "bracket_reference_agreement": bracket_passed,
        "independent_replay_reference_gate": sentinel_passed,
    }
    confirmation = {
        "schema_version": 1,
        "kind": "round5_lf5_confirmation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "amendment_a5_commit": AMENDMENT_A5_COMMIT,
        "production_backend": "replay",
        "registered_cpu_gate_passed": False,
        **gates,
        "methodology_passed": all(gates.values()),
        "prediction_passed": bool(bracket.get("prediction_passed")),
        "source_hashes": {
            "capture_validation": sha256_file(CAPTURE_VALIDATION),
            "replay": sha256_file(REPLAY_REPORT),
            "cpu": sha256_file(CPU_REPORT),
            "amendment_a5": sha256_file(AMENDMENT_A5),
            "row_dump_manifest": sha256_file(ROW_DUMP / "manifest.json"),
            "row_dump_validation": sha256_file(ROW_DUMP_VALIDATION),
            "bracket": sha256_file(BRACKET_REPORT),
            "verification": sha256_file(args.verify_report),
        },
    }
    atomic_json(args.confirmation, confirmation)
    print(json.dumps(confirmation, indent=2, sort_keys=True))
    if not confirmation["methodology_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
