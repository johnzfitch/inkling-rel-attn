"""Outcome-blind GPU runner for the seven registered post-R5-D follow-ups.

Production sequence after the source and frozen inputs are committed::

    .venv-tier2/Scripts/python.exe scripts/round5_followup7_runner.py self-test
    .venv-tier2/Scripts/python.exe scripts/round5_followup7_runner.py preflight
    .venv-tier2/Scripts/python.exe scripts/round5_followup7_runner.py batch
    .venv-tier2/Scripts/python.exe scripts/round5_followup7_runner.py fresh

The runner contains no verdict logic. Complete arm directories are immutable;
failed arm directories are preserved and require a public disposition.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import round5_followup7_build as B  # noqa: E402
import round5_r5d_runner as R  # noqa: E402
from tier2_run import build_layer  # noqa: E402
from tier2_stream import ShardReader  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.inkling.modeling_inkling import (  # noqa: E402
    InklingRMSNorm,
    InklingRelativeLogits,
)


REGISTRATION = ROOT / "registrations" / "ROUND5_FOLLOWUP7_PREREG.md"
FROZEN_NPZ = ROOT / "analysis" / "round5" / "followup7" / "frozen_inputs.npz"
FROZEN_MANIFEST = ROOT / "analysis" / "round5" / "followup7" / "frozen_inputs_manifest.json"
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "followup7"
DEFAULT_PREFLIGHT = ROOT / "analysis" / "round5" / "followup7" / "gpu_preflight.json"
PARENT_DUMP = ROOT / "dumps" / "round5" / "r5d"
CORPUS_V2_CAPTURE = ROOT / "dumps" / "round5" / "corpus_v2_corrected_capture"
REGISTERED_COMMIT = "eec3999"
TEXTS = tuple(R.TEXTS)
FRESH_TEXTS = ("07b_slack_multi", "08_math_llm")
SEQ = R.SEQ
HIDDEN = R.HIDDEN
HEADS = R.HEADS
RPERHEAD = R.RPERHEAD
RFLAT = R.RFLAT
QCHUNK = R.QCHUNK
# Match the certified parent readout's GEMM row geometry exactly. Changing the
# row chunk can change the deployed BF16 matmul reduction path.
FULL_CHUNK = 256


@dataclass(frozen=True)
class Arm:
    arm_id: str
    family: str
    kind: str
    layers: tuple[int, ...]
    start_layer: int
    meter_layers: tuple[int, ...]
    texts: tuple[str, ...] = TEXTS


def build_arms() -> list[Arm]:
    arms: list[Arm] = []
    for kind in (
        "d0_off",
        "d1_off",
        "d2_off",
        "d3_off",
        "d1_3_off",
        "restore_d0",
        "restore_d1_3",
        "stencil_only_d0_3",
    ):
        arms.append(Arm(f"{kind}_L29", "F7-1", kind, (29,), 29, (29,)))
    arms.extend(
        [
            Arm("bias_off_L23_L29", "F7-2", "bias_off", (23, 29), 23, (23, 29)),
            Arm("bias_off_L29_L35", "F7-2", "bias_off", (29, 35), 29, (29, 35)),
            Arm("bias_off_L23_L35", "F7-2", "bias_off", (23, 35), 23, (23, 35)),
            Arm("bias_off_L23_L29_L35", "F7-2", "bias_off", (23, 29, 35), 23, (23, 29, 35)),
        ]
    )
    for kind in (
        "r_remove_mean",
        "r_remove_centered",
        "r_remove_carrier_all",
        "r_remove_noncarrier_all",
        "r_remove_carrier_mean",
        "r_remove_noncarrier_mean",
    ):
        arms.append(Arm(f"{kind}_L29", "F7-3", kind, (29,), 29, (29,)))
    for quartile in range(1, 5):
        arms.append(
            Arm(f"head_q{quartile}_off_L29", "F7-4", f"head_q{quartile}_off", (29,), 29, (29,))
        )
    arms.extend(
        [
            Arm("head_top16_only_L29", "F7-4", "head_top16_only", (29,), 29, (29,)),
            Arm("head_top08_stencil_only_L29", "F7-4", "head_top08_stencil_only", (29,), 29, (29,)),
            Arm("head_top16_stencil_only_L29", "F7-4", "head_top16_stencil_only", (29,), 29, (29,)),
            Arm("head_top32_stencil_only_L29", "F7-4", "head_top32_stencil_only", (29,), 29, (29,)),
            Arm(
                "bias_off_L29_patch_query",
                "F7-5",
                "bias_off_patch_query",
                (29,),
                29,
                (29,),
                ("05_needles",),
            ),
            Arm(
                "bias_off_L29_patch_sham",
                "F7-5",
                "bias_off_patch_sham",
                (29,),
                29,
                (29,),
                ("05_needles",),
            ),
            Arm("clock_union_L53", "F7-6", "clock_union", (53,), 53, (53,)),
            Arm("clock_union_L59", "F7-6", "clock_union", (59,), 59, (59,)),
            Arm("clock_union_L53_L59", "F7-6", "clock_union", (53, 59), 53, (53, 59)),
            Arm("clock_pertext_L53_L59", "F7-6", "clock_pertext", (53, 59), 53, (53, 59)),
            Arm("clock_loto_L53_L59", "F7-6", "clock_loto", (53, 59), 53, (53, 59)),
            Arm("clock_sham6_L53_L59", "F7-6", "clock_sham6", (53, 59), 53, (53, 59)),
            Arm("bias_off_L29_fullvocab", "F7-7", "bias_off", (29,), 29, (29,)),
        ]
    )
    if len(arms) != 35 or len({arm.arm_id for arm in arms}) != 35:
        raise AssertionError("follow-up inventory must contain 35 unique arms")
    return arms


ARMS = build_arms()
ARM_BY_ID = {arm.arm_id: arm for arm in ARMS}
_FROZEN: dict[str, np.ndarray] | None = None


def frozen() -> dict[str, np.ndarray]:
    global _FROZEN
    if _FROZEN is None:
        with np.load(FROZEN_NPZ, allow_pickle=False) as values:
            _FROZEN = {name: np.array(values[name], copy=True) for name in values.files}
    return _FROZEN


def critical_git_gate() -> dict[str, Any]:
    if subprocess.run(["git", "merge-base", "--is-ancestor", REGISTERED_COMMIT, "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("follow-up registration is not an ancestor of HEAD")
    head = R.git_output("rev-parse", "HEAD")
    files = [
        "scripts/round5_followup7_build.py",
        "scripts/round5_followup7_runner.py",
        "scripts/round5_followup7_analyze.py",
        "scripts/round5_followup7_verify.py",
        "scripts/round5_r5d_runner.py",
        "scripts/tier2_run.py",
        "scripts/tier2_stream.py",
        "registrations/ROUND5_FOLLOWUP7_PREREG.md",
        "analysis/round5/followup7/frozen_inputs.npz",
        "analysis/round5/followup7/frozen_inputs_manifest.json",
    ]
    records: dict[str, Any] = {}
    for relative in files:
        path = ROOT / relative
        oid = R.git_output("rev-parse", f"{head}:{relative}")
        filtered = R.git_output("hash-object", "--path", relative, relative)
        if oid != filtered:
            raise RuntimeError(f"critical follow-up dependency differs from HEAD: {relative}")
        records[relative] = {
            "worktree_sha256": R.sha256_file(path),
            "git_blob_oid": oid,
            "filtered_worktree_blob_oid": filtered,
        }
    return {"passed": True, "git_head": head, "files": records}


def validate_frozen_inputs() -> dict[str, Any]:
    manifest = R.load_json(FROZEN_MANIFEST)
    if R.sha256_file(FROZEN_NPZ) != manifest.get("npz_sha256"):
        raise RuntimeError("follow-up frozen-input NPZ hash mismatch")
    expected, metadata = B.build_arrays()
    actual = frozen()
    errors: list[str] = []
    if set(actual) != set(expected):
        errors.append("array inventory")
    for name, wanted in expected.items():
        if name not in actual or not np.array_equal(actual[name], wanted):
            errors.append(name)
    if manifest.get("source_sha256") != metadata.get("source_sha256"):
        errors.append("source hashes")
    if errors:
        raise RuntimeError(f"frozen-input rederivation failed: {errors[:8]}")
    return {
        "passed": True,
        "npz_sha256": manifest["npz_sha256"],
        "manifest_sha256": R.sha256_file(FROZEN_MANIFEST),
        "array_count": len(actual),
        "class_counts": manifest["class_counts"],
    }


def reset_active() -> None:
    R.reset_active()
    R._ACTIVE.update(followup_mu=None, followup_g=None, followup_subspace=None, followup_r_pre=None)


def r_transform(relative_states: torch.Tensor, arm: Arm, text: str, layer: int) -> torch.Tensor:
    values = frozen()
    flat = relative_states.float().reshape(relative_states.shape[0], relative_states.shape[1], RFLAT)
    mu = torch.from_numpy(values[f"mu_L{layer:02d}_{text}"]).to(flat.device).view(1, 1, RFLAT)
    if arm.kind.startswith("r_"):
        g = torch.from_numpy(values["carrier_g_L29"]).to(flat.device).view(1, 1, RFLAT)
        carrier_mean = (mu * g).sum(-1, keepdim=True) * g
        if arm.kind == "r_remove_mean":
            transformed = flat - mu
        elif arm.kind == "r_remove_centered":
            transformed = mu.expand_as(flat)
        elif arm.kind == "r_remove_carrier_all":
            transformed = flat - (flat * g).sum(-1, keepdim=True) * g
        elif arm.kind == "r_remove_noncarrier_all":
            transformed = (flat * g).sum(-1, keepdim=True) * g
        elif arm.kind == "r_remove_carrier_mean":
            transformed = flat - carrier_mean
        elif arm.kind == "r_remove_noncarrier_mean":
            transformed = flat - (mu - carrier_mean)
        else:  # pragma: no cover
            raise AssertionError(arm.kind)
    elif arm.kind.startswith("clock_"):
        if arm.kind == "clock_union":
            name = f"clock_union_L{layer:02d}"
        elif arm.kind == "clock_pertext":
            name = f"clock_g_L{layer:02d}_{text}"
        elif arm.kind == "clock_loto":
            name = f"clock_loto_L{layer:02d}_{text}"
        elif arm.kind == "clock_sham6":
            name = f"clock_sham6_L{layer:02d}"
        else:  # pragma: no cover
            raise AssertionError(arm.kind)
        subspace = torch.from_numpy(values[name]).to(flat.device)
        if subspace.ndim == 1:
            subspace = subspace[:, None]
        centered = flat - mu
        transformed = flat - (centered @ subspace) @ subspace.T
    else:
        return relative_states
    return transformed.reshape_as(relative_states).to(relative_states.dtype)


def followup_relative_logits_forward(
    self: Any,
    relative_states: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    del query_positions, key_positions
    arm: Arm | None = R._ACTIVE["arm"]
    layer = int(R._ACTIVE["layer"])
    text = str(R._ACTIVE["text"])
    if arm is not None and layer in arm.layers and (arm.kind.startswith("r_") or arm.kind.startswith("clock_")):
        if arm.kind.startswith("clock_"):
            R._ACTIVE["followup_r_pre"] = relative_states.detach()[0].to("cpu", torch.float16)
        relative_states = r_transform(relative_states, arm, text, layer)
    R._ACTIVE["relative_states"] = relative_states
    return (relative_states @ self.proj).transpose(1, 2)


def head_indices(kind: str) -> np.ndarray:
    values = frozen()
    if kind.startswith("head_q"):
        number = int(kind[len("head_q")])
        return values[f"head_q{number}"].astype(np.int64)
    if "top08" in kind:
        return values["head_top08"].astype(np.int64)
    if "top16" in kind:
        return values["head_top16"].astype(np.int64)
    if "top32" in kind:
        return values["head_top32"].astype(np.int64)
    raise ValueError(kind)


def followup_intervene_bias(
    bias: torch.Tensor, distance: torch.Tensor, arm: Arm | None, layer: int
) -> torch.Tensor:
    if arm is None or layer not in arm.layers:
        return bias
    kind = arm.kind
    if kind in {"bias_off", "bias_off_patch_query", "bias_off_patch_sham"}:
        return torch.zeros_like(bias)
    singleton = {"d0_off": 0, "d1_off": 1, "d2_off": 2, "d3_off": 3}
    if kind in singleton:
        return bias.masked_fill((distance == singleton[kind]).unsqueeze(0), 0.0)
    if kind == "d1_3_off":
        return bias.masked_fill(((distance >= 1) & (distance <= 3)).unsqueeze(0), 0.0)
    if kind in {"restore_d0", "restore_d1_3", "stencil_only_d0_3"}:
        if kind == "restore_d0":
            keep = distance == 0
        elif kind == "restore_d1_3":
            keep = (distance >= 1) & (distance <= 3)
        else:
            keep = (distance >= 0) & (distance <= 3)
        return bias.masked_fill(~keep.unsqueeze(0), 0.0)
    if kind.startswith("head_q") and kind.endswith("_off"):
        result = bias.clone()
        indices = torch.from_numpy(head_indices(kind)).to(bias.device)
        result.index_fill_(0, indices, 0)
        return result
    if kind == "head_top16_only":
        result = torch.zeros_like(bias)
        indices = torch.from_numpy(head_indices(kind)).to(bias.device)
        result[indices] = bias[indices]
        return result
    if kind.startswith("head_top") and kind.endswith("_stencil_only"):
        result = torch.zeros_like(bias)
        indices = torch.from_numpy(head_indices(kind)).to(bias.device)
        keep = ((distance >= 0) & (distance <= 3)).unsqueeze(0)
        selected = bias[indices].masked_fill(~keep, 0.0)
        result[indices] = selected
        return result
    return bias


def configure_attention(config: Any) -> None:
    R.intervene_bias = followup_intervene_bias
    ALL_ATTENTION_FUNCTIONS.register("round5_followup7", R.r5d_attention)
    config._attn_implementation = "round5_followup7"
    InklingRelativeLogits.forward = followup_relative_logits_forward


def toy_gates(device: str) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(0xF70007)
    distance = torch.arange(8, device=device)[None, :].expand(3, -1)
    bias = torch.randn(HEADS, 3, 8, generator=generator, device=device, dtype=torch.bfloat16)
    for name, index in (("d0_off", 0), ("d1_off", 1), ("d2_off", 2), ("d3_off", 3)):
        arm = Arm("toy", "toy", name, (29,), 29, (29,))
        result = followup_intervene_bias(bias, distance, arm, 29)
        if not torch.equal(result[:, :, index], torch.zeros_like(result[:, :, index])):
            raise RuntimeError(f"{name} gate failed")
        retained = [d for d in range(8) if d != index]
        if not torch.equal(result[:, :, retained], bias[:, :, retained]):
            raise RuntimeError(f"{name} isolation gate failed")
    stencil = followup_intervene_bias(
        bias, distance, Arm("toy", "toy", "stencil_only_d0_3", (29,), 29, (29,)), 29
    )
    if not torch.equal(stencil[:, :, :4], bias[:, :, :4]) or torch.count_nonzero(stencil[:, :, 4:]):
        raise RuntimeError("stencil-only gate failed")
    for quartile in range(1, 5):
        kind = f"head_q{quartile}_off"
        result = followup_intervene_bias(
            bias, distance, Arm("toy", "toy", kind, (29,), 29, (29,)), 29
        )
        selected = head_indices(kind)
        other = np.asarray(sorted(set(range(HEADS)) - set(selected.tolist())), dtype=np.int64)
        if torch.count_nonzero(result[selected]) or not torch.equal(result[other], bias[other]):
            raise RuntimeError(f"head quartile isolation failed: {kind}")
    values = frozen()
    source = torch.randn(1, 19, RFLAT, generator=generator, device=device, dtype=torch.bfloat16)
    arm = ARM_BY_ID["r_remove_mean_L29"]
    transformed = r_transform(source.reshape(1, 19, HEADS, RPERHEAD), arm, "01_prose_en", 29)
    if not torch.isfinite(transformed).all() or transformed.dtype != source.dtype:
        raise RuntimeError("r decomposition gate failed")
    for layer in (53, 59):
        u = torch.from_numpy(values[f"clock_union_L{layer:02d}"]).to(device)
        orth = float(torch.max(torch.abs(u.T @ u - torch.eye(6, device=device))).item())
        sham = torch.from_numpy(values[f"clock_sham6_L{layer:02d}"]).to(device)
        overlap = float(torch.max(torch.abs(u.T @ sham)).item())
        if orth > 2e-6 or overlap > 2e-6:
            raise RuntimeError(f"clock subspace gate failed at L{layer}: {orth}, {overlap}")
    query = values["patch_query_positions"]
    sham = values["patch_sham_positions"]
    if query.size != 24 or sham.size != 24 or any(abs(int(a) - int(b)) <= 8 for a in query for b in sham):
        raise RuntimeError("patch position gate failed")
    return {"passed": True, "device": device, "distance_boundaries": True, "head_partition": True,
            "r_decomposition_finite": True, "clock_subspaces": True, "patch_positions": True}


@torch.no_grad()
def compute_full_readout(
    hidden: torch.Tensor,
    ids: np.ndarray,
    final_norm: InklingRMSNorm,
    unembed: torch.Tensor,
    *,
    mup_multiplier: float,
    unpadded_vocab: int,
) -> dict[str, np.ndarray]:
    states = final_norm(hidden) / mup_multiplier
    targets = torch.from_numpy(ids[1:].astype(np.int64)).to(states.device)
    buckets: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "target_logit", "log_normalizer", "target_rank", "log1p_target_rank",
            "top1_id", "top1_probability", "top1_correct", "entropy", "brier", "target_margin"
        )
    }
    for start in range(0, SEQ - 1, FULL_CHUNK):
        stop = min(start + FULL_CHUNK, SEQ - 1)
        logits = torch.nn.functional.linear(states[:, start:stop], unembed)[0]
        logits32 = logits[:, :unpadded_vocab].float()
        target = targets[start:stop]
        selected = logits32.gather(1, target[:, None])[:, 0]
        logz = torch.logsumexp(logits32, dim=-1)
        top_values, top_ids = torch.topk(logits32, k=2, dim=-1)
        top1_id = top_ids[:, 0]
        top1 = top_values[:, 0]
        max_other = torch.where(top1_id == target, top_values[:, 1], top1)
        rank = 1 + torch.sum(logits32 > selected[:, None], dim=-1)
        probabilities = torch.softmax(logits32, dim=-1)
        target_probability = torch.exp(selected - logz)
        entropy = logz - torch.sum(probabilities * logits32, dim=-1)
        brier = torch.sum(probabilities * probabilities, dim=-1) - 2 * target_probability + 1
        values = {
            "target_logit": selected,
            "log_normalizer": logz,
            "target_rank": rank.to(torch.int32),
            "log1p_target_rank": torch.log1p(rank.float()),
            "top1_id": top1_id.to(torch.int32),
            "top1_probability": torch.exp(top1 - logz),
            "top1_correct": (top1_id == target).to(torch.int8),
            "entropy": entropy,
            "brier": brier,
            "target_margin": selected - max_other,
        }
        for name, value in values.items():
            buckets[name].append(value.detach().cpu())
        del logits, logits32, selected, logz, top_values, top_ids, probabilities, values
    output: dict[str, np.ndarray] = {
        "target_position": np.arange(1, SEQ, dtype=np.int32),
        "target_id": ids[1:].astype(np.int32),
    }
    for name, parts in buckets.items():
        value = torch.cat(parts).numpy()
        if value.dtype.kind == "f":
            value = value.astype(np.float32, copy=False)
        output[name] = value
    output["nll"] = (output["log_normalizer"] - output["target_logit"]).astype(np.float32)
    output["log_probability"] = (output["target_logit"] - output["log_normalizer"]).astype(np.float32)
    output["probability"] = np.exp(output["log_probability"].astype(np.float64)).astype(np.float32)
    if any(value.shape != (SEQ - 1,) for value in output.values()):
        raise RuntimeError("invalid full-vocabulary readout shape")
    if any(value.dtype.kind == "f" and not np.isfinite(value).all() for value in output.values()):
        raise RuntimeError("nonfinite full-vocabulary readout")
    return output


DELTA_FIELDS = (
    "target_logit", "log_normalizer", "nll", "log_probability", "probability",
    "log1p_target_rank", "top1_probability", "top1_correct", "entropy", "brier", "target_margin",
)


def add_deltas(values: dict[str, np.ndarray], baseline: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not np.array_equal(values["target_position"], baseline["target_position"]):
        raise RuntimeError("target-position drift")
    if not np.array_equal(values["target_id"], baseline["target_id"]):
        raise RuntimeError("target-ID drift")
    output = dict(values)
    for name in DELTA_FIELDS:
        output[f"delta_{name}"] = (
            values[name].astype(np.float32) - baseline[name].astype(np.float32)
        ).astype(np.float32)
    output["delta_target_rank"] = (
        values["target_rank"].astype(np.int64) - baseline["target_rank"].astype(np.int64)
    ).astype(np.int32)
    return output


def load_full_baseline(root: Path, texts: tuple[str, ...] = TEXTS) -> dict[str, dict[str, np.ndarray]]:
    base = root / "baseline_fullvocab"
    manifest = R.load_json(base / "manifest.json")
    if not manifest.get("complete") or manifest.get("artifact_count") != len(texts):
        raise RuntimeError("full-vocabulary baseline incomplete")
    records = {item["path"]: item for item in manifest["artifacts"]}
    output: dict[str, dict[str, np.ndarray]] = {}
    for text in texts:
        path = base / f"{text}.npz"
        if path.name not in records or R.sha256_file(path) != records[path.name]["sha256"]:
            raise RuntimeError(f"baseline full-vocabulary hash mismatch: {text}")
        with np.load(path, allow_pickle=False) as values:
            output[text] = {name: np.array(values[name], copy=True) for name in values.files}
    return output


def final_modules(config: Any, reader: ShardReader) -> tuple[InklingRMSNorm, torch.Tensor]:
    final_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
    final_norm.weight = torch.nn.Parameter(
        reader.get("model.llm.norm.weight", "cuda").to(torch.bfloat16), requires_grad=False
    )
    final_norm.eval()
    unembed = reader.get("model.llm.unembed.weight", "cuda").to(torch.bfloat16)
    return final_norm, unembed


def save_full_baseline(root: Path, config: Any, reader: ShardReader) -> dict[str, Any]:
    base = root / "baseline_fullvocab"
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(f"refusing to overwrite full-vocabulary baseline: {base}")
    base.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_followup7_fullvocab_baseline",
        "complete": False,
        "created_at_utc": R.utc_now(),
        "artifacts": [],
    }
    R.atomic_json(base / "manifest.json", manifest)
    final_norm, unembed = final_modules(config, reader)
    parent_baseline = R.load_baseline(PARENT_DUMP)
    for text in TEXTS:
        hidden = R.load_bf16_state(R.state_path("hidden_L65", text))
        values = compute_full_readout(
            hidden, R.verified_ids(text), final_norm, unembed,
            mup_multiplier=float(config.logits_mup_width_multiplier),
            unpadded_vocab=int(config.unpadded_vocab_size),
        )
        for name in ("target_logit", "log_normalizer", "nll", "log_probability", "probability"):
            if not np.array_equal(values[name], parent_baseline[text][name]):
                delta = float(np.max(np.abs(values[name] - parent_baseline[text][name])))
                raise RuntimeError(f"baseline full-vocabulary parity failed {text} {name}: {delta}")
        path = base / f"{text}.npz"
        R.atomic_npz(path, **values)
        manifest["artifacts"].append(R.artifact_record(base, path, "followup7_baseline_fullvocab", text=text))
        del hidden
    del final_norm, unembed
    gc.collect()
    torch.cuda.empty_cache()
    manifest["artifact_count"] = len(manifest["artifacts"])
    manifest["complete"] = True
    manifest["completed_at_utc"] = R.utc_now()
    R.atomic_json(base / "manifest.json", manifest)
    return manifest


def expected_arm_artifacts(arm: Arm) -> int:
    count = len(arm.texts)  # token dumps
    count += len(arm.texts) * len(arm.meter_layers)
    if "05_needles" in arm.texts:
        count += 1
    if arm.kind.startswith("clock_"):
        count += len(arm.texts) * len(arm.layers)
    return count


def require_preflight(path: Path, dump: Path) -> dict[str, Any]:
    report = R.load_json(path)
    if report.get("kind") != "round5_followup7_gpu_preflight" or report.get("passed") is not True:
        raise RuntimeError("follow-up preflight absent or failed")
    if report["critical_git"]["git_head"] != R.git_output("rev-parse", "HEAD"):
        raise RuntimeError("HEAD changed after follow-up preflight")
    load_full_baseline(dump)
    return report


@torch.no_grad()
def preflight_command(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    out = args.preflight.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {out}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_followup7_gpu_preflight",
        "passed": False,
        "started_at_utc": R.utc_now(),
        "arm_count": len(ARMS),
    }
    R.atomic_json(out, report)
    started = time.time()
    try:
        report["critical_git"] = critical_git_gate()
        report["parent_critical_git"] = R.critical_git_gate()
        report["frozen_inputs"] = validate_frozen_inputs()
        report["capture"] = R.validate_capture_and_states()
        report["checkpoint"] = R.checkpoint_inventory_gate()
        report["runtime"] = R.runtime_provenance_gate()
        report["stock_attention_parity"] = R.stock_attention_parity_gate()
        report["parent_toy_gates_cpu"] = R.toy_intervention_gates("cpu")
        report["followup_toy_gates_cpu"] = toy_gates("cpu")
        report["followup_toy_gates_gpu"] = toy_gates("cuda")

        config = AutoConfig.from_pretrained(R.NVFP4).text_config
        configure_attention(config)
        reader = ShardReader(str(R.NVFP4))
        baseline_manifest = save_full_baseline(args.dump.resolve(), config, reader)
        report["baseline_fullvocab"] = {
            "passed": True,
            "artifact_count": baseline_manifest["artifact_count"],
            "manifest_sha256": R.sha256_file(args.dump.resolve() / "baseline_fullvocab" / "manifest.json"),
        }

        replay, _ = R.production_layer_gate(
            config, reader, layer_index=29, state_name="hidden_L28", text="06_random", arm=None
        )
        expected = np.load(R.state_path("hidden_L29", "06_random"), allow_pickle=False)
        equal = bool(np.array_equal(R.bf16_bits(replay[0]), expected))
        del replay
        if not equal:
            raise RuntimeError("production L29 no-intervention replay failed")
        report["production_l29_replay"] = {"passed": True, "bitwise_equal": True, "values": SEQ * HIDDEN}

        baseline_state = R.load_bf16_state(R.state_path("hidden_L29", "05_needles"))
        patched = baseline_state.clone()
        positions = torch.from_numpy(frozen()["patch_query_positions"].astype(np.int64)).to("cuda")
        patched[:, positions] = baseline_state[:, positions]
        if not torch.equal(patched, baseline_state):
            raise RuntimeError("baseline identity patch gate failed")
        del patched, baseline_state
        torch.cuda.empty_cache()
        report["baseline_identity_patch"] = {"passed": True, "positions": 24, "bitwise_equal": True}
        report["passed"] = True
        report["completed_at_utc"] = R.utc_now()
        report["elapsed_seconds"] = round(time.time() - started, 3)
        R.atomic_json(out, report)
        print(json.dumps({"passed": True, "preflight": str(out), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["failed_at_utc"] = R.utc_now()
        report["elapsed_seconds"] = round(time.time() - started, 3)
        R.atomic_json(out, report)
        raise
    finally:
        reset_active()


def save_meter(arm_root: Path, meter: R.R5DMeter, arm: Arm, layer: int, text: str, sliding: bool) -> dict[str, Any]:
    values = meter.to_npz()
    R.meter_integrity(values, layer, sliding)
    path = arm_root / "meters" / f"L{layer:02d}_{text}.npz"
    R.atomic_npz(path, **values)
    return R.artifact_record(arm_root, path, "followup7_locus_meter", layer=layer, text=text)


@torch.no_grad()
def run_arm(arm: Arm, args: argparse.Namespace) -> None:
    preflight = require_preflight(args.preflight.resolve(), args.dump.resolve())
    arm_root = args.dump.resolve() / "arms" / arm.arm_id
    manifest_path = arm_root / "manifest.json"
    if manifest_path.exists():
        existing = R.load_json(manifest_path)
        if existing.get("complete") is True:
            print(f"SKIP complete arm {arm.arm_id}", flush=True)
            return
        raise FileExistsError(f"preserving incomplete arm: {arm_root}")
    arm_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_followup7_arm",
        "arm": asdict(arm),
        "complete": False,
        "started_at_utc": R.utc_now(),
        "source_git_head": preflight["critical_git"]["git_head"],
        "preflight_sha256": R.sha256_file(args.preflight.resolve()),
        "frozen_inputs_sha256": R.sha256_file(FROZEN_NPZ),
        "expected_artifact_count": expected_arm_artifacts(arm),
        "artifacts": [],
    }
    R.atomic_json(manifest_path, manifest)
    started = time.time()
    try:
        config = AutoConfig.from_pretrained(R.NVFP4).text_config
        configure_attention(config)
        reader = ShardReader(str(R.NVFP4))
        baseline = load_full_baseline(args.dump.resolve())
        ids_by_text = {text: R.verified_ids(text) for text in arm.texts}
        entry = R.state_name_entering(arm.start_layer)
        hidden = {text: R.load_bf16_state(R.state_path(entry, text)) for text in arm.texts}
        manifest["restart_state"] = entry
        manifest["restart_state_sha256"] = {text: R.sha256_file(R.state_path(entry, text)) for text in arm.texts}
        patch_state = None
        if arm.kind in {"bias_off_patch_query", "bias_off_patch_sham"}:
            patch_state = R.load_bf16_state(R.state_path("hidden_L29", "05_needles"))

        for layer_index in range(arm.start_layer, 66):
            layer_started = time.time()
            layer = build_layer(config, layer_index, reader, "cuda")
            sliding = config.layer_types[layer_index] == "hybrid_sliding"
            try:
                for text in arm.texts:
                    meter = (
                        R.R5DMeter(HEADS, int(config.sliding_window_size) if sliding else SEQ, "cuda")
                        if layer_index in arm.meter_layers
                        else None
                    )
                    reset_active()
                    R._ACTIVE.update(
                        arm=arm,
                        layer=layer_index,
                        text=text,
                        meter=meter,
                        sliding=sliding,
                        window=int(config.sliding_window_size),
                        qchunk=QCHUNK,
                    )
                    hidden[text] = layer(hidden[text], attention_mask=None, conv_mask=None, past_key_values=None)
                    if meter is not None:
                        manifest["artifacts"].append(save_meter(arm_root, meter, arm, layer_index, text, sliding))
                    if arm.kind.startswith("clock_") and layer_index in arm.layers:
                        captured = R._ACTIVE["followup_r_pre"]
                        if captured is None:
                            raise RuntimeError(f"missing pre-clock r-vector {arm.arm_id} L{layer_index} {text}")
                        path = arm_root / "clock" / f"rvec_pre_L{layer_index:02d}_{text}.npy"
                        R.atomic_npy(path, captured.numpy())
                        manifest["artifacts"].append(
                            R.artifact_record(arm_root, path, "followup7_clock_pre_rvec", layer=layer_index, text=text)
                        )
                    del meter
                if layer_index == 29 and patch_state is not None:
                    name = "patch_query_positions" if arm.kind == "bias_off_patch_query" else "patch_sham_positions"
                    positions = torch.from_numpy(frozen()[name].astype(np.int64)).to("cuda")
                    hidden["05_needles"][:, positions] = patch_state[:, positions]
            finally:
                reset_active()
                del layer
            gc.collect()
            torch.cuda.empty_cache()
            manifest["last_completed_layer"] = layer_index
            manifest["elapsed_seconds"] = round(time.time() - started, 3)
            R.atomic_json(manifest_path, manifest)
            print(f"{arm.arm_id} L{layer_index:02d} {time.time() - layer_started:.1f}s", flush=True)

        final_norm, unembed = final_modules(config, reader)
        readouts: dict[str, dict[str, np.ndarray]] = {}
        for text in arm.texts:
            values = compute_full_readout(
                hidden[text], ids_by_text[text], final_norm, unembed,
                mup_multiplier=float(config.logits_mup_width_multiplier),
                unpadded_vocab=int(config.unpadded_vocab_size),
            )
            values = add_deltas(values, baseline[text])
            if arm.arm_id == "bias_off_L29_fullvocab":
                parent_path = PARENT_DUMP / "arms" / "bias_off_L29" / "tokens" / f"{text}.npz"
                with np.load(parent_path, allow_pickle=False) as parent:
                    for name in (
                        "target_logit", "log_normalizer", "nll", "log_probability", "probability",
                        "delta_target_logit", "delta_log_normalizer", "delta_nll",
                        "delta_log_probability", "delta_probability",
                    ):
                        if not np.array_equal(values[name], parent[name]):
                            raise RuntimeError(f"exact-copy bias-off parity failed: {text} {name}")
            path = arm_root / "tokens" / f"{text}.npz"
            R.atomic_npz(path, **values)
            manifest["artifacts"].append(R.artifact_record(arm_root, path, "followup7_token_fullvocab", text=text))
            readouts[text] = values
        if "05_needles" in arm.texts:
            queries, targets, sides = R.needle_indices()
            values = readouts["05_needles"]
            needle = {
                "query_position": queries,
                "target_position": queries + 1,
                "target_id": targets,
                "side_of_seam": sides,
            }
            for name in ("delta_target_logit", "delta_probability", "delta_log_probability", "delta_nll",
                         "delta_log1p_target_rank", "delta_top1_correct", "delta_entropy", "delta_brier"):
                needle[name] = values[name][queries]
            path = arm_root / "needle" / "05_needles.npz"
            R.atomic_npz(path, **needle)
            manifest["artifacts"].append(R.artifact_record(arm_root, path, "followup7_needle_readout", count=24))
        del final_norm, unembed, hidden, patch_state
        gc.collect()
        torch.cuda.empty_cache()

        manifest["artifact_count"] = len(manifest["artifacts"])
        if manifest["artifact_count"] != manifest["expected_artifact_count"]:
            raise RuntimeError(
                f"artifact count {manifest['artifact_count']} != {manifest['expected_artifact_count']}"
            )
        manifest["complete"] = True
        manifest["completed_at_utc"] = R.utc_now()
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        R.atomic_json(manifest_path, manifest)
        print(f"SEALED {arm.arm_id} artifacts={manifest['artifact_count']} elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        manifest["failed_at_utc"] = R.utc_now()
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        R.atomic_json(manifest_path, manifest)
        raise
    finally:
        reset_active()


def verified_fresh_ids(name: str) -> np.ndarray:
    manifest = R.load_json(ROOT / "corpus_v2" / "manifest.json")
    path = ROOT / "corpus_v2" / f"{name}.ids.npy"
    if R.sha256_file(path) != manifest["texts"][name]["ids_sha256"]:
        raise RuntimeError(f"fresh ID binding failed: {name}")
    ids = np.load(path, allow_pickle=False)
    if ids.shape != (SEQ,) or ids.dtype != np.int32:
        raise RuntimeError(f"invalid fresh IDs: {name}")
    return ids


@torch.no_grad()
def fresh_command(args: argparse.Namespace) -> None:
    preflight = require_preflight(args.preflight.resolve(), args.dump.resolve())
    root = args.dump.resolve() / "fresh"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = R.load_json(manifest_path)
        if existing.get("complete"):
            print("SKIP complete fresh-text job", flush=True)
            return
        raise FileExistsError(f"preserving incomplete fresh job: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_followup7_fresh_class_job",
        "complete": False,
        "started_at_utc": R.utc_now(),
        "source_git_head": preflight["critical_git"]["git_head"],
        "artifacts": [],
        "expected_artifact_count": 8,  # 2 prefixes + 2 baseline + 2 arm + 2 meters
    }
    R.atomic_json(manifest_path, manifest)
    started = time.time()
    try:
        config = AutoConfig.from_pretrained(R.NVFP4).text_config
        configure_attention(config)
        reader = ShardReader(str(R.NVFP4))
        ids_by_text = {text: verified_fresh_ids(text) for text in FRESH_TEXTS}
        embed_weight = reader.get("model.llm.embed.weight", "cuda").to(torch.bfloat16)
        embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        embed_norm.weight = torch.nn.Parameter(
            reader.get("model.llm.embed_norm.weight", "cuda").to(torch.bfloat16), requires_grad=False
        )
        embed_norm.eval()
        hidden: dict[str, torch.Tensor] = {}
        for text in FRESH_TEXTS:
            ids_cuda = torch.from_numpy(ids_by_text[text].astype(np.int64)).to("cuda")
            hidden[text] = embed_norm(torch.nn.functional.embedding(ids_cuda, embed_weight).unsqueeze(0))
        del embed_weight, embed_norm
        configure_attention(config)
        for layer_index in range(66):
            layer_started = time.time()
            layer = build_layer(config, layer_index, reader, "cuda")
            sliding = config.layer_types[layer_index] == "hybrid_sliding"
            for text in FRESH_TEXTS:
                reset_active()
                R._ACTIVE.update(arm=None, layer=layer_index, text=text, meter=None, sliding=sliding,
                                 window=int(config.sliding_window_size), qchunk=QCHUNK)
                hidden[text] = layer(hidden[text], attention_mask=None, conv_mask=None, past_key_values=None)
                if layer_index == 28:
                    path = root / "prefix" / f"hidden_L28_{text}.npy"
                    R.atomic_npy(path, R.bf16_bits(hidden[text][0]))
                    manifest["artifacts"].append(R.artifact_record(root, path, "followup7_fresh_prefix", text=text))
            reset_active()
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            manifest["baseline_last_completed_layer"] = layer_index
            R.atomic_json(manifest_path, manifest)
            print(f"fresh baseline L{layer_index:02d} {time.time() - layer_started:.1f}s", flush=True)

        final_norm, unembed = final_modules(config, reader)
        fresh_baseline: dict[str, dict[str, np.ndarray]] = {}
        for text in FRESH_TEXTS:
            values = compute_full_readout(
                hidden[text], ids_by_text[text], final_norm, unembed,
                mup_multiplier=float(config.logits_mup_width_multiplier),
                unpadded_vocab=int(config.unpadded_vocab_size),
            )
            certified_path = CORPUS_V2_CAPTURE / "nll" / f"nll_{text}.npz"
            with np.load(certified_path, allow_pickle=False) as certified:
                if not np.array_equal(values["nll"], certified["nll"]):
                    maximum = float(np.max(np.abs(values["nll"] - certified["nll"])))
                    raise RuntimeError(f"fresh baseline NLL parity failed {text}: {maximum}")
            path = root / "baseline" / f"{text}.npz"
            R.atomic_npz(path, **values)
            manifest["artifacts"].append(R.artifact_record(root, path, "followup7_fresh_baseline_fullvocab", text=text))
            fresh_baseline[text] = values
        del hidden
        gc.collect()
        torch.cuda.empty_cache()

        arm = Arm("fresh_bias_off_L29", "F7-7", "bias_off", (29,), 29, (29,), FRESH_TEXTS)
        hidden = {
            text: R.load_bf16_state(root / "prefix" / f"hidden_L28_{text}.npy") for text in FRESH_TEXTS
        }
        for layer_index in range(29, 66):
            layer_started = time.time()
            layer = build_layer(config, layer_index, reader, "cuda")
            sliding = config.layer_types[layer_index] == "hybrid_sliding"
            for text in FRESH_TEXTS:
                meter = R.R5DMeter(HEADS, SEQ, "cuda") if layer_index == 29 else None
                reset_active()
                R._ACTIVE.update(arm=arm, layer=layer_index, text=text, meter=meter, sliding=sliding,
                                 window=int(config.sliding_window_size), qchunk=QCHUNK)
                hidden[text] = layer(hidden[text], attention_mask=None, conv_mask=None, past_key_values=None)
                if meter is not None:
                    path = root / "meters" / f"L29_{text}.npz"
                    arrays = meter.to_npz()
                    R.meter_integrity(arrays, 29, False)
                    R.atomic_npz(path, **arrays)
                    manifest["artifacts"].append(R.artifact_record(root, path, "followup7_fresh_meter", text=text))
                del meter
            reset_active()
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            manifest["arm_last_completed_layer"] = layer_index
            R.atomic_json(manifest_path, manifest)
            print(f"fresh bias-off L{layer_index:02d} {time.time() - layer_started:.1f}s", flush=True)
        for text in FRESH_TEXTS:
            values = compute_full_readout(
                hidden[text], ids_by_text[text], final_norm, unembed,
                mup_multiplier=float(config.logits_mup_width_multiplier),
                unpadded_vocab=int(config.unpadded_vocab_size),
            )
            values = add_deltas(values, fresh_baseline[text])
            path = root / "bias_off_L29" / f"{text}.npz"
            R.atomic_npz(path, **values)
            manifest["artifacts"].append(R.artifact_record(root, path, "followup7_fresh_biasoff_fullvocab", text=text))
        del hidden, final_norm, unembed
        gc.collect()
        torch.cuda.empty_cache()
        manifest["artifact_count"] = len(manifest["artifacts"])
        if manifest["artifact_count"] != manifest["expected_artifact_count"]:
            raise RuntimeError("fresh artifact count mismatch")
        manifest["complete"] = True
        manifest["completed_at_utc"] = R.utc_now()
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        R.atomic_json(manifest_path, manifest)
        print(f"SEALED fresh job artifacts={manifest['artifact_count']} elapsed={manifest['elapsed_seconds']:.1f}s")
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        manifest["failed_at_utc"] = R.utc_now()
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        R.atomic_json(manifest_path, manifest)
        raise
    finally:
        reset_active()


def batch_command(args: argparse.Namespace) -> None:
    require_preflight(args.preflight.resolve(), args.dump.resolve())
    selection = ARMS
    if args.start_at:
        selection = selection[[arm.arm_id for arm in selection].index(args.start_at):]
    if args.stop_after:
        selection = selection[: [arm.arm_id for arm in selection].index(args.stop_after) + 1]
    for arm in selection:
        run_arm(arm, args)
    payload = {
        "kind": "round5_followup7_batch_status",
        "complete": all(
            (args.dump.resolve() / "arms" / arm.arm_id / "manifest.json").exists()
            and R.load_json(args.dump.resolve() / "arms" / arm.arm_id / "manifest.json").get("complete")
            for arm in ARMS
        ),
        "arms": [arm.arm_id for arm in ARMS],
        "updated_at_utc": R.utc_now(),
    }
    R.atomic_json(args.dump.resolve() / "batch_status.json", payload)


def self_test_command() -> None:
    if len(ARMS) != 35:
        raise AssertionError("arm count")
    validate_frozen_inputs()
    toy_gates("cpu")
    print(json.dumps({"passed": True, "arms": len(ARMS), "families": sorted({arm.family for arm in ARMS})}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    sub.add_parser("list")
    sub.add_parser("preflight")
    one = sub.add_parser("arm")
    one.add_argument("--arm", required=True, choices=sorted(ARM_BY_ID))
    batch = sub.add_parser("batch")
    batch.add_argument("--start-at", choices=[arm.arm_id for arm in ARMS])
    batch.add_argument("--stop-after", choices=[arm.arm_id for arm in ARMS])
    sub.add_parser("fresh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-test":
        self_test_command()
    elif args.command == "list":
        print(json.dumps([asdict(arm) for arm in ARMS], indent=2))
    elif args.command == "preflight":
        preflight_command(args)
    elif args.command == "arm":
        run_arm(ARM_BY_ID[args.arm], args)
    elif args.command == "batch":
        batch_command(args)
    elif args.command == "fresh":
        fresh_command(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
