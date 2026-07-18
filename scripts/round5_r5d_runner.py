"""Outcome-blind GPU runner for the registered Round-5 R5-D campaign.

The runner executes the 67 parent arms in
``ROUND5_R5D_EXECUTION_AMENDMENT.md`` plus the five clock-freeze arms in
``ROUND5_R5D_CLOCK_AMENDMENT.md``.  It deliberately contains no verdict
logic; ``round5_r5d_analyze.py`` owns the frozen readouts and refuses partial
batches.

Production commands, after committing this source and the analyzer::

    .venv-tier2/Scripts/python.exe scripts/round5_r5d_runner.py self-test
    .venv-tier2/Scripts/python.exe scripts/round5_r5d_runner.py preflight
    .venv-tier2/Scripts/python.exe scripts/round5_r5d_runner.py batch

Every arm starts from the certified lossless BF16 D4 state entering its first
intervened layer, propagates through L65, and writes an immutable dump.  A
failed arm is preserved and may only be restarted by explicitly moving its
directory aside; the batch driver resumes at complete arm boundaries.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tokenizers
import torch
import transformers


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import tier2_stream as T  # noqa: E402
from tier2_run import build_layer  # noqa: E402
from tier2_stream import Meter, ShardReader  # noqa: E402

from transformers import AutoConfig  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.inkling.modeling_inkling import (  # noqa: E402
    InklingRMSNorm,
    InklingRelativeLogits,
    eager_attention_forward,
)


NVFP4 = ROOT / "nvfp4"
CORPUS = ROOT / "corpus"
CAPTURE = ROOT / "dumps" / "round5" / "widened_corrected_capture"
CAPTURE_MANIFEST = CAPTURE / "manifest.json"
CAPTURE_VALIDATION = ROOT / "analysis" / "round5" / "widened_capture" / "capture_validation.json"
BASIS_PATH = ROOT / "analysis" / "subspace_anatomy" / "common_bases_top4.npz"
TAIL_PATH = ROOT / "dumps" / "round5" / "r5d_wall_tail_v2" / "tail_tables.npz"
TAIL_REPORT = ROOT / "analysis" / "round5" / "r5d" / "tail_fit_v2.json"
CLOCK_PATH = ROOT / "analysis" / "round5" / "r5d_clock" / "clock_freeze.npz"
CLOCK_MANIFEST = ROOT / "analysis" / "round5" / "r5d_clock" / "clock_freeze_manifest.json"
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "r5d"
DEFAULT_PREFLIGHT = ROOT / "analysis" / "round5" / "r5d" / "gpu_preflight.json"

PARENT_AMENDMENT = ROOT / "registrations" / "ROUND5_R5D_EXECUTION_AMENDMENT.md"
CLOCK_AMENDMENT = ROOT / "registrations" / "ROUND5_R5D_CLOCK_AMENDMENT.md"
TAIL_AMENDMENT = ROOT / "registrations" / "ROUND5_R5D_TAIL_AMENDMENT_A.md"

TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
LAYERS = list(range(66))
SINGLE_LAYERS = [0, 1, 2, 3, 4, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
CLOCK_LAYERS = [53, 59, 65]
SEQ = 8192
HIDDEN = 6144
HEADS = 64
RPERHEAD = 16
RFLAT = HEADS * RPERHEAD
QCHUNK = 512
NLL_CHUNK = 256
CAPTURE_MANIFEST_SHA256 = "2fc2ec9dd4d6b684a473847f6fbd4541d4326b49d7d6456b463ee5722399a75f"
PARENT_COMMIT = "2a48f5b"
CLOCK_COMMIT = "3b48d05"

NEGATIVE_L11_HEADS = (
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 21, 23,
    25, 27, 29, 31, 32, 34, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 57, 60, 62,
)


@dataclass(frozen=True)
class Arm:
    arm_id: str
    kind: str
    layers: tuple[int, ...]
    start_layer: int
    meter_layers: tuple[int, ...]


def build_arm_inventory() -> list[Arm]:
    arms: list[Arm] = []
    for family in ("bias_off", "carrier_out", "near_off", "far_off"):
        for layer in SINGLE_LAYERS:
            arms.append(Arm(f"{family}_L{layer:02d}", family, (layer,), layer, (layer,)))
    arms.extend(
        [
            Arm("wall_heal_global", "wall_heal", tuple(GLOBALS), 5, tuple(GLOBALS)),
            Arm(
                "rising_heads_off_L00_L04",
                "rising_heads_off",
                (0, 1, 2, 3, 4),
                0,
                (0, 1, 2, 3, 4),
            ),
            Arm(
                "negative_seam_heads_off_L11",
                "negative_seam_heads_off",
                (11,),
                11,
                (11,),
            ),
            Arm("clock_freeze_L53", "clock_freeze", (53,), 53, (53,)),
            Arm("clock_freeze_L59", "clock_freeze", (59,), 59, (59,)),
            Arm("clock_freeze_L65", "clock_freeze", (65,), 65, (65,)),
            Arm("clock_freeze_L53_L59", "clock_freeze", (53, 59), 53, (53, 59)),
            Arm("clock_sham_L59", "clock_sham", (59,), 59, (59,)),
        ]
    )
    if len(arms) != 72 or len({arm.arm_id for arm in arms}) != 72:
        raise AssertionError("registered R5-D inventory must contain 72 unique arms")
    return arms


ARMS = build_arm_inventory()
ARM_BY_ID = {arm.arm_id: arm for arm in ARMS}


# Per-forward state read by the compact relative-logit and attention hooks.
_ACTIVE: dict[str, Any] = {
    "arm": None,
    "layer": None,
    "text": None,
    "meter": None,
    "sliding": False,
    "window": 512,
    "qchunk": QCHUNK,
    "clock_direction": None,
    "clock_anchor": None,
    "clock_pre": None,
    "relative_states": None,
    "wall_tail": None,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_from(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def git_blob_sha256(commit: str, relative: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT, stderr=subprocess.STDOUT
    )
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def artifact_record(root: Path, path: Path, kind: str, **metadata: Any) -> dict[str, Any]:
    record = {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    record.update(metadata)
    return record


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_ancestor(commit: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT
    ).returncode:
        raise RuntimeError(f"registered commit is not an ancestor of HEAD: {commit}")


def critical_git_gate() -> dict[str, Any]:
    require_ancestor(PARENT_COMMIT)
    require_ancestor(CLOCK_COMMIT)
    head = git_output("rev-parse", "HEAD")
    critical = [
        "scripts/round5_r5d_runner.py",
        "scripts/round5_r5d_analyze.py",
        "scripts/round5_r5d_tail_v2.py",
        "scripts/round5_clock_freeze_build.py",
        "scripts/tier2_run.py",
        "scripts/tier2_stream.py",
        "scripts/tier2_nvfp4.py",
        "registrations/ROUND5_R5D_EXECUTION_AMENDMENT.md",
        "registrations/ROUND5_R5D_CLOCK_AMENDMENT.md",
        "registrations/ROUND5_R5D_TAIL_AMENDMENT_A.md",
        "analysis/round5/r5d/tail_fit_v2.json",
        "analysis/round5/r5d_clock/clock_freeze.npz",
        "analysis/round5/r5d_clock/clock_freeze_manifest.json",
        "analysis/subspace_anatomy/common_bases_top4.npz",
        "corpus/manifest.json",
        "corpus/tokenizer.json",
        "corpus/05_needles.sidecar.json",
    ]
    records: dict[str, Any] = {}
    for relative in critical:
        current = sha256_file(ROOT / relative)
        committed = git_blob_sha256(head, relative)
        # Git's checkout filter may represent a clean text blob with CRLF on
        # Windows while the canonical object stores LF.  Authenticate the
        # worktree through the same clean filter Git uses, then retain both
        # raw SHA-256 values in provenance instead of falsely treating EOLs as
        # source drift.
        committed_oid = git_output("rev-parse", f"{head}:{relative}")
        worktree_oid = git_output("hash-object", "--path", relative, relative)
        if worktree_oid != committed_oid:
            raise RuntimeError(f"critical dependency differs from HEAD: {relative}")
        records[relative] = {
            "worktree_sha256": current,
            "git_blob_sha256": committed,
            "git_blob_oid": committed_oid,
            "filtered_worktree_blob_oid": worktree_oid,
            "equal_after_git_clean_filter": True,
        }
    return {"passed": True, "git_head": head, "files": records}


def verified_ids(name: str) -> np.ndarray:
    manifest = load_json(CORPUS / "manifest.json")
    path = CORPUS / f"{name}.ids.npy"
    if manifest.get("seq") != SEQ or sha256_file(path) != manifest["texts"][name]["ids_sha256"]:
        raise RuntimeError(f"corpus binding failed: {name}")
    ids = np.load(path, allow_pickle=False)
    if ids.shape != (SEQ,) or ids.dtype != np.int32 or int(ids.min()) < 0:
        raise RuntimeError(f"invalid IDs: {name}, {ids.shape}, {ids.dtype}")
    return ids


def load_bf16_state(path: Path, device: str = "cuda") -> torch.Tensor:
    bits = np.load(path, allow_pickle=False)
    if bits.shape != (SEQ, HIDDEN) or bits.dtype != np.uint16:
        raise RuntimeError(f"invalid D4 state: {path}, {bits.shape}, {bits.dtype}")
    # Copy makes the numpy storage writable and avoids torch's read-only warning.
    tensor = torch.from_numpy(np.array(bits, copy=True)).view(torch.bfloat16)
    return tensor.unsqueeze(0).to(device)


def bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16, got {tensor.dtype}")
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy().copy()


def state_name_entering(layer: int) -> str:
    return "hidden_embed" if layer == 0 else f"hidden_L{layer - 1:02d}"


def state_path(state_name: str, text: str) -> Path:
    return CAPTURE / "states" / f"{state_name}_{text}.npy"


def validate_capture_and_states() -> dict[str, Any]:
    if sha256_file(CAPTURE_MANIFEST) != CAPTURE_MANIFEST_SHA256:
        raise RuntimeError("certified capture manifest hash mismatch")
    validation = load_json(CAPTURE_VALIDATION)
    manifest = load_json(CAPTURE_MANIFEST)
    if (
        not manifest.get("complete")
        or manifest.get("kind") != "round5_d1_widened_a6_capture"
        or manifest.get("artifact_count") != 2324
        or validation.get("capture_manifest_sha256") != CAPTURE_MANIFEST_SHA256
        or not validation.get("D4_satisfied")
        or validation.get("errors")
    ):
        raise RuntimeError("certified D1+D4 validation is not clean")
    records = {item["path"]: item for item in manifest["artifacts"]}
    entry_names = sorted({state_name_entering(arm.start_layer) for arm in ARMS})
    required = [(name, text) for name in entry_names for text in TEXTS]
    required += [("hidden_L65", text) for text in TEXTS]
    checked: list[dict[str, Any]] = []
    for state_name, text in required:
        path = state_path(state_name, text)
        relative = path.relative_to(CAPTURE).as_posix()
        record = records.get(relative)
        if record is None or record.get("kind") != "residual_hidden_state":
            raise RuntimeError(f"missing captured state record: {relative}")
        digest = sha256_file(path)
        bits = np.load(path, allow_pickle=False, mmap_mode="r")
        nonfinite = int((((bits & 0x7F80) == 0x7F80)).sum())
        if (
            digest != record.get("sha256")
            or bits.shape != (SEQ, HIDDEN)
            or bits.dtype != np.uint16
            or nonfinite != 0
        ):
            raise RuntimeError(f"captured state gate failed: {relative}")
        checked.append(
            {
                "path": relative,
                "sha256": digest,
                "shape": [SEQ, HIDDEN],
                "dtype": "uint16(lossless_bfloat16)",
                "nonfinite_bf16_words": nonfinite,
            }
        )
    return {
        "passed": True,
        "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "entry_state_names": entry_names,
        "state_count": len(checked),
        "states": checked,
    }


def validate_frozen_inputs() -> dict[str, Any]:
    tail_report = load_json(TAIL_REPORT)
    if (
        tail_report.get("kind") != "round5_r5d_amendment_a_wall_tail"
        or tail_report.get("passed") is not True
        or tail_report.get("tail_dump", {}).get("sha256") != sha256_file(TAIL_PATH)
    ):
        raise RuntimeError("amended wall-tail binding failed")
    with np.load(TAIL_PATH, allow_pickle=False) as tails:
        if tails.files != [f"L{layer:02d}" for layer in GLOBALS]:
            raise RuntimeError("wall-tail layer inventory mismatch")
        for key in tails.files:
            values = tails[key]
            if values.shape != (RPERHEAD, SEQ - 1024) or values.dtype != np.float32:
                raise RuntimeError(f"invalid wall tail {key}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite wall tail {key}")

    clock_manifest = load_json(CLOCK_MANIFEST)
    if (
        clock_manifest.get("kind") != "round5_r5d_clock_freeze"
        or clock_manifest.get("artifact_sha256") != sha256_file(CLOCK_PATH)
        or clock_manifest.get("capture_manifest_sha256") != CAPTURE_MANIFEST_SHA256
        or clock_manifest.get("builder_source_sha256")
        != sha256_file(SCRIPT_DIR / "round5_clock_freeze_build.py")
    ):
        raise RuntimeError("clock-freeze binding failed")
    for relative_name, expected in clock_manifest["input_sha256"].items():
        if relative_name == "amendment":
            path = CLOCK_AMENDMENT
        elif relative_name.startswith("rvec_"):
            path = CAPTURE / "replay" / relative_name
        elif relative_name.startswith("layer"):
            path = ROOT / "weights" / relative_name
        else:
            raise RuntimeError(f"unknown clock input: {relative_name}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"clock input hash mismatch: {relative_name}")
    with np.load(CLOCK_PATH, allow_pickle=False) as freeze:
        expected_keys = {
            "G_L53", "rbar_L53", "G_L59", "rbar_L59", "G_L65", "rbar_L65", "sham_L59"
        }
        if set(freeze.files) != expected_keys:
            raise RuntimeError("clock-freeze key inventory mismatch")
        for key in expected_keys:
            values = freeze[key]
            if values.shape != (RFLAT,) or values.dtype != np.float64 or not np.isfinite(values).all():
                raise RuntimeError(f"invalid clock input: {key}")
        sham_dot = abs(float(freeze["sham_L59"] @ freeze["G_L59"]))
        if sham_dot > 1e-12:
            raise RuntimeError(f"clock sham is not orthogonal: {sham_dot}")
        norm_error = max(
            abs(float(np.linalg.norm(freeze[key])) - 1.0)
            for key in ("G_L53", "G_L59", "G_L65", "sham_L59")
        )
        if norm_error > 1e-12:
            raise RuntimeError(f"clock direction norm error: {norm_error}")

    with np.load(BASIS_PATH, allow_pickle=False) as basis_dump:
        basis = basis_dump["basis"]
        if basis.shape != (66, 4, HIDDEN) or basis.dtype != np.float32 or not np.isfinite(basis).all():
            raise RuntimeError("invalid certified carrier basis")
        carrier_norm_error = float(np.max(np.abs(np.linalg.norm(basis[:, 0], axis=1) - 1.0)))
        if carrier_norm_error > 2e-5:
            raise RuntimeError(f"carrier basis is not unit normalized: {carrier_norm_error}")
    return {
        "passed": True,
        "tail_report_sha256": sha256_file(TAIL_REPORT),
        "tail_dump_sha256": sha256_file(TAIL_PATH),
        "clock_manifest_sha256": sha256_file(CLOCK_MANIFEST),
        "clock_artifact_sha256": sha256_file(CLOCK_PATH),
        "carrier_basis_sha256": sha256_file(BASIS_PATH),
        "clock_sham_abs_dot_real": sham_dot,
        "clock_direction_max_norm_error": norm_error,
        "carrier_max_norm_error": carrier_norm_error,
    }


class R5DMeter(Meter):
    """Registered distance meter plus per-head attention effective count."""

    def __init__(self, n_heads: int, dmax: int, device: str = "cuda") -> None:
        super().__init__(n_heads, dmax, device)
        self.effective_count_sum = torch.zeros(n_heads, dtype=torch.float64, device=device)
        self.query_count = 0

    def add_chunk(
        self,
        w_with: torch.Tensor,
        w_without: torch.Tensor,
        bias: torch.Tensor,
        content: torch.Tensor,
        q_start: int,
        sliding: bool,
        window: int,
    ) -> None:
        super().add_chunk(w_with, w_without, bias, content, q_start, sliding, window)
        entropy = -(w_with * torch.log(w_with.clamp_min(torch.finfo(w_with.dtype).tiny))).sum(-1)
        self.effective_count_sum += torch.exp(entropy).sum(1).double()
        self.query_count += int(w_with.shape[1])

    def to_npz(self) -> dict[str, np.ndarray]:
        values = super().to_npz()
        values.update(
            effective_count_sum=self.effective_count_sum.cpu().numpy(),
            mean_effective_count=(self.effective_count_sum / max(self.query_count, 1)).cpu().numpy(),
            effective_query_count=np.asarray(self.query_count, dtype=np.int64),
        )
        return values


def freeze_rvec(
    relative_states: torch.Tensor, anchor: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    shape = relative_states.shape
    flat = relative_states.float().reshape(shape[0], shape[1], RFLAT)
    coefficient = ((flat - anchor) * direction).sum(-1, keepdim=True)
    frozen = flat - coefficient * direction
    return frozen.reshape(shape).to(relative_states.dtype)


def r5d_relative_logits_forward(
    self: Any,
    relative_states: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    arm: Arm | None = _ACTIVE["arm"]
    layer = _ACTIVE["layer"]
    if arm is not None and arm.kind in {"clock_freeze", "clock_sham"} and layer in arm.layers:
        _ACTIVE["clock_pre"] = relative_states.detach()[0].to("cpu", torch.float16)
        relative_states = freeze_rvec(
            relative_states, _ACTIVE["clock_anchor"], _ACTIVE["clock_direction"]
        )
    _ACTIVE["relative_states"] = relative_states
    return (relative_states @ self.proj).transpose(1, 2)


def intervene_bias(bias: torch.Tensor, distance: torch.Tensor, arm: Arm | None, layer: int) -> torch.Tensor:
    if arm is None or layer not in arm.layers:
        return bias
    if arm.kind == "bias_off":
        return torch.zeros_like(bias)
    if arm.kind == "near_off":
        return bias.masked_fill((distance < 4).unsqueeze(0), 0.0)
    if arm.kind == "far_off":
        return bias.masked_fill((distance > 128).unsqueeze(0), 0.0)
    if arm.kind == "rising_heads_off":
        return torch.zeros_like(bias)
    if arm.kind == "negative_seam_heads_off":
        result = bias.clone()
        result[list(NEGATIVE_L11_HEADS)] = 0.0
        return result
    return bias


def add_wall_tail(
    bias: torch.Tensor,
    distance: torch.Tensor,
    relative_states: torch.Tensor,
    tail: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    del distance  # contiguous prefill lets us write the triangular tail directly by key index
    maximum_count = min(int(tail.shape[1]), max(0, stop - 1024))
    if maximum_count == 0:
        return bias
    # A monolithic [H,q,7168] tail plus a full gather costs about 1 GB at the
    # final query chunk.  Compute 32 query rows at a time and write only their
    # causal far-field keys.  For absolute query q, keys 0..q-1024 correspond
    # to tail indices q-1024..0 (the reversal below).
    row_chunk = 32
    for row_start in range(start, stop, row_chunk):
        row_stop = min(row_start + row_chunk, stop)
        group_count = min(int(tail.shape[1]), max(0, row_stop - 1024))
        if group_count == 0:
            continue
        logits = (
            relative_states[0, row_start:row_stop] @ tail[:, :group_count]
        ).permute(1, 0, 2)
        for absolute_query in range(row_start, row_stop):
            count = min(int(tail.shape[1]), max(0, absolute_query - 1023))
            if count:
                local_query = absolute_query - row_start
                bias[:, absolute_query - start, :count] = logits[:, local_query, :count].flip(-1)
        del logits
    return bias


def r5d_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    position_bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del attention_mask, dropout, kwargs
    arm: Arm | None = _ACTIVE["arm"]
    layer = int(_ACTIVE["layer"])
    meter: R5DMeter | None = _ACTIVE["meter"]
    sliding = bool(_ACTIVE["sliding"])
    window = int(_ACTIVE["window"])
    qchunk = int(_ACTIVE["qchunk"])
    groups = module.num_key_value_groups
    kx = key.repeat_interleave(groups, dim=1)
    vx = value.repeat_interleave(groups, dim=1)
    heads, sequence, dimension = query.shape[1], query.shape[2], query.shape[3]
    keys = kx.shape[2]
    extent = position_bias.shape[-1] if position_bias is not None else 0
    output = torch.empty(1, heads, sequence, dimension, dtype=query.dtype, device=query.device)
    key_positions = torch.arange(keys, device=query.device)
    negative = torch.finfo(torch.float32).min

    for start in range(0, sequence, qchunk):
        stop = min(start + qchunk, sequence)
        query_positions = torch.arange(start, stop, device=query.device)
        distance = query_positions[:, None] - key_positions[None, :]
        causal = distance >= 0
        if sliding:
            causal &= distance < window
        content = (torch.matmul(query[:, :, start:stop], kx.transpose(2, 3)) * scaling)[0]
        if position_bias is None:
            bias = torch.zeros_like(content)
        else:
            compact = position_bias[0, :, start:stop]
            in_extent = (distance >= 0) & (distance < extent)
            index = distance.clamp(0, extent - 1).unsqueeze(0).expand(heads, -1, -1)
            bias = torch.gather(compact, 2, index).masked_fill(~in_extent.unsqueeze(0), 0.0)
        if arm is not None and arm.kind == "wall_heal" and layer in arm.layers:
            bias = add_wall_tail(
                bias,
                distance,
                _ACTIVE["relative_states"],
                _ACTIVE["wall_tail"],
                start,
                stop,
            )
        bias = intervene_bias(bias, distance, arm, layer)
        masked = ~causal
        content32 = content.float()
        # A6: the deployed model adds content and bias in BF16, including that
        # rounding event, and only then upcasts for the softmax.
        weights = torch.softmax((content + bias).float().masked_fill(masked, negative), dim=-1)
        if meter is not None:
            without = torch.softmax(content32.masked_fill(masked, negative), dim=-1)
            meter.add_chunk(weights, without, bias.float(), content32, start, sliding, window)
            del without
        output[:, :, start:stop] = torch.matmul(weights.to(query.dtype).unsqueeze(0), vx)
        del content, bias, content32, weights
    return output.transpose(1, 2).contiguous(), None


def carrier_project_input(hidden: torch.Tensor, carrier: torch.Tensor, token_chunk: int = 256) -> torch.Tensor:
    result = torch.empty_like(hidden)
    for start in range(0, hidden.shape[1], token_chunk):
        stop = min(start + token_chunk, hidden.shape[1])
        values = hidden[:, start:stop].float()
        coefficient = torch.matmul(values, carrier).unsqueeze(-1)
        result[:, start:stop] = (values - coefficient * carrier).to(hidden.dtype)
    return result


def carrier_pre_hook(carrier: torch.Tensor):
    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(inputs) != 1:
            raise RuntimeError("unexpected r_proj input signature")
        return (carrier_project_input(inputs[0], carrier),)

    return hook


def configure_attention(config: Any) -> None:
    ALL_ATTENTION_FUNCTIONS.register("round5_r5d", r5d_attention)
    config._attn_implementation = "round5_r5d"
    InklingRelativeLogits.forward = r5d_relative_logits_forward


def reset_active() -> None:
    _ACTIVE.update(
        arm=None,
        layer=None,
        text=None,
        meter=None,
        sliding=False,
        window=512,
        qchunk=QCHUNK,
        clock_direction=None,
        clock_anchor=None,
        clock_pre=None,
        relative_states=None,
        wall_tail=None,
    )


def clock_arrays(arm: Arm, layer: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(CLOCK_PATH, allow_pickle=False) as freeze:
        direction_key = "sham_L59" if arm.kind == "clock_sham" else f"G_L{layer}"
        direction = torch.from_numpy(freeze[direction_key].astype(np.float32)).to(device)
        anchor = torch.from_numpy(freeze[f"rbar_L{layer}"].astype(np.float32)).to(device)
    return anchor.view(1, 1, RFLAT), direction.view(1, 1, RFLAT)


def load_carrier(layer: int, device: str = "cuda") -> torch.Tensor:
    with np.load(BASIS_PATH, allow_pickle=False) as dump:
        values = np.array(dump["basis"][layer, 0], copy=True)
    return torch.from_numpy(values).to(device=device, dtype=torch.float32)


def load_wall_tail(layer: int, device: str = "cuda") -> torch.Tensor:
    with np.load(TAIL_PATH, allow_pickle=False) as dump:
        values = np.array(dump[f"L{layer:02d}"], copy=True)
    return torch.from_numpy(values).to(device=device, dtype=torch.bfloat16)


def meter_integrity(values: dict[str, np.ndarray], layer: int, sliding: bool) -> dict[str, Any]:
    expected = float(SEQ)
    mass_error = float(np.max(np.abs(values["mass_with"].sum(1) - expected)))
    without_error = float(np.max(np.abs(values["mass_without"].sum(1) - expected)))
    finite = all(np.isfinite(value).all() for value in values.values())
    if not finite or mass_error > 0.01 or without_error > 0.01:
        raise RuntimeError(
            f"meter integrity failed L{layer:02d}: with={mass_error}, without={without_error}"
        )
    return {
        "passed": True,
        "is_sliding": sliding,
        "max_mass_error_with": mass_error,
        "max_mass_error_without": without_error,
    }


@torch.no_grad()
def compute_readout(
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
    target_logits: list[torch.Tensor] = []
    log_normalizers: list[torch.Tensor] = []
    for start in range(0, SEQ - 1, NLL_CHUNK):
        stop = min(start + NLL_CHUNK, SEQ - 1)
        logits = torch.nn.functional.linear(states[:, start:stop], unembed)[0]
        logits32 = logits[:, :unpadded_vocab].float()
        selected = logits32.gather(1, targets[start:stop, None])[:, 0]
        target_logits.append(selected.cpu())
        log_normalizers.append(torch.logsumexp(logits32, dim=-1).cpu())
        del logits, logits32, selected
    target_logit = torch.cat(target_logits).numpy().astype(np.float32, copy=False)
    log_normalizer = torch.cat(log_normalizers).numpy().astype(np.float32, copy=False)
    nll = (log_normalizer - target_logit).astype(np.float32, copy=False)
    log_probability = (target_logit - log_normalizer).astype(np.float32, copy=False)
    probability = np.exp(log_probability.astype(np.float64)).astype(np.float32)
    if not all(
        np.isfinite(values).all()
        for values in (target_logit, log_normalizer, nll, log_probability, probability)
    ):
        raise RuntimeError("nonfinite final readout")
    return {
        "target_position": np.arange(1, SEQ, dtype=np.int32),
        "target_id": ids[1:].astype(np.int32),
        "target_logit": target_logit,
        "log_normalizer": log_normalizer,
        "nll": nll,
        "log_probability": log_probability,
        "probability": probability,
    }


def save_baseline_calibration(
    dump_root: Path, config: Any, reader: ShardReader
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    baseline_root = dump_root / "baseline"
    if baseline_root.exists() and any(baseline_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite baseline calibration: {baseline_root}")
    baseline_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_r5d_baseline_final_calibration",
        "complete": False,
        "created_at_utc": utc_now(),
        "artifacts": [],
    }
    atomic_json(baseline_root / "manifest.json", manifest)
    final_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
    final_norm.weight = torch.nn.Parameter(
        reader.get("model.llm.norm.weight", "cuda").to(torch.bfloat16), requires_grad=False
    )
    final_norm.eval()
    unembed = reader.get("model.llm.unembed.weight", "cuda").to(torch.bfloat16)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    capture_records = {
        record["path"]: record for record in load_json(CAPTURE_MANIFEST)["artifacts"]
    }
    for text in TEXTS:
        ids = verified_ids(text)
        hidden = load_bf16_state(state_path("hidden_L65", text))
        values = compute_readout(
            hidden,
            ids,
            final_norm,
            unembed,
            mup_multiplier=float(config.logits_mup_width_multiplier),
            unpadded_vocab=int(config.unpadded_vocab_size),
        )
        certified_path = CAPTURE / "nll" / f"nll_{text}.npz"
        certified_relative = certified_path.relative_to(CAPTURE).as_posix()
        certified_record = capture_records.get(certified_relative)
        if (
            certified_record is None
            or certified_record.get("kind") != "next_token_nll"
            or sha256_file(certified_path) != certified_record.get("sha256")
        ):
            raise RuntimeError(f"certified NLL artifact hash mismatch: {text}")
        with np.load(certified_path, allow_pickle=False) as certified:
            if (
                not np.array_equal(values["target_position"], certified["target_position"])
                or not np.array_equal(values["target_id"], certified["target_id"])
                or not np.array_equal(values["nll"], certified["nll"])
            ):
                maximum = float(np.max(np.abs(values["nll"] - certified["nll"])))
                raise RuntimeError(
                    f"baseline NLL is not bitwise equal for {text}; max delta={maximum}"
                )
        path = baseline_root / f"{text}.npz"
        atomic_npz(path, **values)
        manifest["artifacts"].append(
            artifact_record(
                baseline_root,
                path,
                "baseline_final_readout",
                text=text,
                count=SEQ - 1,
                mean_nll=float(np.mean(values["nll"], dtype=np.float64)),
                bitwise_equal_to_certified_nll=True,
            )
        )
        outputs[text] = values
        del hidden
    del unembed, final_norm
    gc.collect()
    torch.cuda.empty_cache()
    manifest["complete"] = True
    manifest["completed_at_utc"] = utc_now()
    manifest["artifact_count"] = len(manifest["artifacts"])
    atomic_json(baseline_root / "manifest.json", manifest)
    return manifest, outputs


def load_baseline(dump_root: Path) -> dict[str, dict[str, np.ndarray]]:
    root = dump_root / "baseline"
    manifest = load_json(root / "manifest.json")
    if not manifest.get("complete") or manifest.get("artifact_count") != len(TEXTS):
        raise RuntimeError("baseline calibration is incomplete")
    records = {item["path"]: item for item in manifest["artifacts"]}
    output: dict[str, dict[str, np.ndarray]] = {}
    for text in TEXTS:
        path = root / f"{text}.npz"
        record = records.get(path.name)
        if record is None or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"baseline calibration hash mismatch: {text}")
        with np.load(path, allow_pickle=False) as values:
            output[text] = {key: np.array(values[key], copy=True) for key in values.files}
    return output


def stock_attention_parity_gate() -> dict[str, Any]:
    class Dummy:
        num_key_value_groups = 2
        training = False

    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xD1A8)
    cases: dict[str, Any] = {}
    reset_active()
    for label, sliding, window, extent in (("global", False, 64, 32), ("sliding", True, 8, 8)):
        heads, kv_heads, sequence, dimension = 4, 2, 17, 8
        query = torch.randn(1, heads, sequence, dimension, device="cuda", dtype=torch.bfloat16, generator=generator)
        key = torch.randn(1, kv_heads, sequence, dimension, device="cuda", dtype=torch.bfloat16, generator=generator)
        value = torch.randn(1, kv_heads, sequence, dimension, device="cuda", dtype=torch.bfloat16, generator=generator)
        compact = torch.randn(1, heads, sequence, extent, device="cuda", dtype=torch.bfloat16, generator=generator)
        positions = torch.arange(sequence, device="cuda")
        distance = positions[:, None] - positions[None, :]
        gather_index = distance.clamp(0, extent - 1)[None, None].expand(1, heads, -1, -1)
        dense_bias = torch.gather(compact, 3, gather_index).masked_fill(
            ~((distance >= 0) & (distance < extent))[None, None], 0.0
        )
        valid = distance >= 0
        if sliding:
            valid &= distance < window
        mask = torch.zeros(1, 1, sequence, sequence, device="cuda", dtype=torch.bfloat16).masked_fill(
            ~valid[None, None], torch.finfo(torch.bfloat16).min
        )
        _ACTIVE.update(layer=0, sliding=sliding, window=window, qchunk=7)
        measured, _ = r5d_attention(Dummy(), query, key, value, None, 1.0 / dimension, position_bias=compact)
        stock, _ = eager_attention_forward(
            Dummy(), query, key, value, mask, 1.0 / dimension, position_bias=dense_bias
        )
        equal = bool(torch.equal(measured, stock))
        maximum = float((measured.float() - stock.float()).abs().max().item())
        if not equal:
            raise RuntimeError(f"stock attention parity failed: {label}, {maximum}")
        cases[label] = {"bitwise_equal": equal, "max_output_delta": maximum}
    reset_active()
    return {"passed": True, "cases": cases}


def legacy_compact_attention_parity_gate(device: str = "cpu") -> dict[str, Any]:
    """Bitwise check against the already-certified A6 compact implementation."""

    class Dummy:
        num_key_value_groups = 2
        training = False

    generator = torch.Generator(device=device)
    generator.manual_seed(0xA672)
    cases: dict[str, Any] = {}
    prior = dict(T._ACTIVE)
    try:
        for label, sliding, window, extent in (
            ("global", False, 32, 19),
            ("sliding", True, 7, 7),
        ):
            heads, kv_heads, sequence, dimension = 4, 2, 23, 8
            query = torch.randn(
                1, heads, sequence, dimension, generator=generator, device=device, dtype=torch.bfloat16
            )
            key = torch.randn(
                1, kv_heads, sequence, dimension, generator=generator, device=device, dtype=torch.bfloat16
            )
            value = torch.randn(
                1, kv_heads, sequence, dimension, generator=generator, device=device, dtype=torch.bfloat16
            )
            compact = torch.randn(
                1, heads, sequence, extent, generator=generator, device=device, dtype=torch.bfloat16
            )
            reset_active()
            _ACTIVE.update(layer=0, sliding=sliding, window=window, qchunk=9)
            ours, _ = r5d_attention(
                Dummy(), query, key, value, None, 1.0 / dimension, position_bias=compact
            )
            T._ACTIVE.update(
                meter=None,
                sliding=sliding,
                window=window,
                qchunk=9,
                needle_qpos=None,
                needle_rows=None,
            )
            legacy, _ = T.measuring_attention(
                Dummy(), query, key, value, None, 1.0 / dimension, position_bias=compact
            )
            equal = bool(torch.equal(ours, legacy))
            maximum = float((ours.float() - legacy.float()).abs().max().item())
            if not equal:
                raise RuntimeError(f"A6 compact parity failed: {label}, {maximum}")
            cases[label] = {"bitwise_equal": equal, "max_output_delta": maximum}
    finally:
        T._ACTIVE.clear()
        T._ACTIVE.update(prior)
        reset_active()
    return {"passed": True, "device": device, "cases": cases}


def toy_intervention_gates(device: str = "cpu") -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(0x5D72)
    distance = torch.arange(0, 140, device=device).view(1, -1).expand(3, -1)
    bias = torch.randn(HEADS, 3, 140, generator=generator, device=device, dtype=torch.float32)
    near = intervene_bias(bias, distance, Arm("toy", "near_off", (7,), 7, (7,)), 7)
    far = intervene_bias(bias, distance, Arm("toy", "far_off", (7,), 7, (7,)), 7)
    if not torch.equal(near[:, :, :4], torch.zeros_like(near[:, :, :4])) or not torch.equal(near[:, :, 4:], bias[:, :, 4:]):
        raise RuntimeError("near-off boundary gate failed")
    if not torch.equal(far[:, :, :129], bias[:, :, :129]) or not torch.equal(far[:, :, 129:], torch.zeros_like(far[:, :, 129:])):
        raise RuntimeError("far-off boundary gate failed")
    selected = intervene_bias(
        bias, distance, Arm("toy", "negative_seam_heads_off", (11,), 11, (11,)), 11
    )
    unselected = sorted(set(range(HEADS)) - set(NEGATIVE_L11_HEADS))
    if not torch.equal(selected[list(NEGATIVE_L11_HEADS)], torch.zeros_like(selected[list(NEGATIVE_L11_HEADS)])):
        raise RuntimeError("selected-head zero gate failed")
    if not torch.equal(selected[unselected], bias[unselected]):
        raise RuntimeError("unselected-head isolation gate failed")

    carrier = torch.randn(HIDDEN, generator=generator, device=device, dtype=torch.float32)
    carrier /= carrier.norm()
    hidden = torch.randn(1, 9, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16)
    source32 = hidden.float()
    projected32 = torch.empty_like(source32)
    for start in range(0, hidden.shape[1], 3):
        stop = min(start + 3, hidden.shape[1])
        chunk = source32[:, start:stop]
        projected32[:, start:stop] = chunk - (chunk @ carrier).unsqueeze(-1) * carrier
    projected = carrier_project_input(hidden, carrier, token_chunk=3)
    carrier_residual = float(torch.max(torch.abs(projected32 @ carrier)).item())
    carrier_bound = float(64 * torch.finfo(torch.float32).eps * source32.norm(dim=-1).max().item())
    if not torch.equal(projected, projected32.to(torch.bfloat16)):
        raise RuntimeError("carrier helper does not implement the frozen FP32-then-BF16 operator")
    if carrier_residual > carrier_bound:
        raise RuntimeError(f"carrier orthogonality gate failed: {carrier_residual} > {carrier_bound}")

    with np.load(CLOCK_PATH, allow_pickle=False) as freeze:
        anchor = torch.from_numpy(freeze["rbar_L59"].astype(np.float32)).to(device).view(1, 1, RFLAT)
        direction = torch.from_numpy(freeze["G_L59"].astype(np.float32)).to(device).view(1, 1, RFLAT)
        sham_dot = abs(float(freeze["sham_L59"] @ freeze["G_L59"]))
    rvec = torch.randn(1, 13, HEADS, RPERHEAD, generator=generator, device=device, dtype=torch.bfloat16)
    flat = rvec.float().reshape(1, 13, RFLAT)
    frozen32 = flat - (((flat - anchor) * direction).sum(-1, keepdim=True) * direction)
    frozen = freeze_rvec(rvec, anchor, direction)
    clock_residual = float(
        torch.max(torch.abs(((frozen32 - anchor) * direction).sum(-1))).item()
    )
    clock_bound = float(
        64 * torch.finfo(torch.float32).eps
        * (frozen32 - anchor).norm(dim=-1).max().item()
    )
    if not torch.equal(frozen, frozen32.reshape_as(rvec).to(torch.bfloat16)):
        raise RuntimeError("clock helper does not implement the frozen FP32-then-BF16 operator")
    if clock_residual > clock_bound or sham_dot > 1e-12:
        raise RuntimeError(
            f"clock operator gate failed: residual={clock_residual}, bound={clock_bound}, sham={sham_dot}"
        )

    rsmall = torch.randn(1, 1100, 2, 3, generator=generator, device=device, dtype=torch.bfloat16)
    tailsmall = torch.randn(3, 76, generator=generator, device=device, dtype=torch.bfloat16)
    qsmall = torch.arange(1020, 1100, device=device)
    ksmall = torch.arange(1100, device=device)
    dsmall = qsmall[:, None] - ksmall[None, :]
    base = torch.randn(2, 80, 1100, generator=generator, device=device, dtype=torch.bfloat16)
    original_base = base.clone()
    healed = add_wall_tail(base, dsmall, rsmall, tailsmall, 1020, 1100)
    learned_mask = dsmall < 1024
    tail_mask = dsmall >= 1024
    if (
        not torch.equal(healed.masked_select(learned_mask.unsqueeze(0)), original_base.masked_select(learned_mask.unsqueeze(0)))
        or not torch.isfinite(healed).all()
        or not bool(torch.any(healed.masked_select(tail_mask.unsqueeze(0)) != original_base.masked_select(tail_mask.unsqueeze(0))))
    ):
        raise RuntimeError("wall-tail preservation/finite gate failed")
    for absolute_query in (1024, 1037, 1099):
        count = absolute_query - 1023
        expected = (
            rsmall[0, absolute_query] @ tailsmall[:, :count]
        ).flip(-1)
        actual = healed[:, absolute_query - 1020, :count]
        if not torch.equal(actual, expected):
            raise RuntimeError(f"wall-tail direct-index gate failed at q={absolute_query}")
    return {
        "passed": True,
        "near_boundary_d4_retained": True,
        "far_boundary_d128_retained": True,
        "selected_head_count": len(NEGATIVE_L11_HEADS),
        "carrier_max_abs_dot": carrier_residual,
        "carrier_fp32_resolution_bound": carrier_bound,
        "clock_max_abs_dot": clock_residual,
        "clock_fp32_resolution_bound": clock_bound,
        "clock_sham_abs_dot_real": sham_dot,
        "wall_below_1024_bitwise_preserved": True,
    }


@torch.no_grad()
def production_layer_gate(
    config: Any,
    reader: ShardReader,
    *,
    layer_index: int,
    state_name: str,
    text: str,
    arm: Arm | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden = load_bf16_state(state_path(state_name, text))
    layer = build_layer(config, layer_index, reader, "cuda")
    is_sliding = config.layer_types[layer_index] == "hybrid_sliding"
    reset_active()
    _ACTIVE.update(
        arm=arm,
        layer=layer_index,
        text=text,
        sliding=is_sliding,
        window=int(config.sliding_window_size),
        qchunk=QCHUNK,
    )
    if arm is not None and arm.kind in {"clock_freeze", "clock_sham"}:
        anchor, direction = clock_arrays(arm, layer_index)
        _ACTIVE.update(clock_anchor=anchor, clock_direction=direction)
    output = layer(hidden, attention_mask=None, conv_mask=None, past_key_values=None)
    clock_pre = _ACTIVE["clock_pre"]
    del layer, hidden
    reset_active()
    gc.collect()
    torch.cuda.empty_cache()
    return output, clock_pre


def checkpoint_inventory_gate() -> dict[str, Any]:
    index_path = NVFP4 / "model.safetensors.index.json"
    index = load_json(index_path)
    capture = load_json(CAPTURE_MANIFEST)
    if sha256_file(index_path) != capture.get("checkpoint_index_sha256"):
        raise RuntimeError("checkpoint index differs from certified capture")
    if sha256_file(NVFP4 / "config.json") != capture.get("config_sha256"):
        raise RuntimeError("checkpoint config differs from certified capture")
    if sha256_file(CORPUS / "tokenizer.json") != capture.get("tokenizer_sha256"):
        raise RuntimeError("tokenizer differs from certified capture")
    indexed = sorted(set(index["weight_map"].values()))
    trunk = [name for name in indexed if name.startswith("model-")]
    nontrunk = [name for name in indexed if not name.startswith("model-")]
    on_disk = sorted(path.name for path in NVFP4.glob("model-*.safetensors"))
    if trunk != on_disk or nontrunk != ["mtp.safetensors"]:
        raise RuntimeError("checkpoint inventory differs from index")
    sizes: dict[str, int] = {}
    for name in trunk:
        size = (NVFP4 / name).stat().st_size
        if size != int(capture["checkpoint_shards"][name]["bytes"]):
            raise RuntimeError(f"checkpoint shard size mismatch: {name}")
        sizes[name] = size
    return {
        "passed": True,
        "index_sha256": sha256_file(index_path),
        "trunk_shards": len(trunk),
        "nontrunk": nontrunk,
        "byte_sizes": sizes,
        "content_hashes_inherited_from_certified_capture": True,
        "certified_capture_validation_rehashed_shards": True,
        "config_sha256": sha256_file(NVFP4 / "config.json"),
        "tokenizer_sha256": sha256_file(CORPUS / "tokenizer.json"),
    }


def runtime_provenance_gate() -> dict[str, Any]:
    capture = load_json(CAPTURE_MANIFEST)
    modules = {
        "numpy": np,
        "tokenizers": tokenizers,
        "torch": torch,
        "transformers": transformers,
    }
    packages: dict[str, Any] = {}
    for name, module in modules.items():
        record = {
            "version": str(module.__version__),
            "module_path": str(Path(module.__file__).resolve()),
        }
        certified = capture["packages"][name]
        if record["version"] != certified["version"]:
            raise RuntimeError(
                f"package version drift for {name}: {record['version']} != {certified['version']}"
            )
        record["certified_capture_version_equal"] = True
        packages[name] = record
    modeling_path = Path(sys.modules[InklingRMSNorm.__module__].__file__).resolve()
    modeling_hash = sha256_file(modeling_path)
    if modeling_hash != capture["modeling_inkling"]["sha256"]:
        raise RuntimeError("installed modeling_inkling source differs from certified capture")
    return {
        "passed": True,
        "packages": packages,
        "modeling_inkling": {"path": str(modeling_path), "sha256": modeling_hash},
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }


def preflight_command(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered R5-D preflight")
    if args.qchunk != QCHUNK:
        raise ValueError("registered qchunk is 512")
    out = args.preflight.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {out}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_r5d_gpu_preflight",
        "started_at_utc": utc_now(),
        "passed": False,
        "arm_count": len(ARMS),
    }
    atomic_json(out, report)
    started = time.time()
    try:
        report["critical_git"] = critical_git_gate()
        report["frozen_inputs"] = validate_frozen_inputs()
        report["capture"] = validate_capture_and_states()
        report["checkpoint"] = checkpoint_inventory_gate()
        report["runtime_provenance"] = runtime_provenance_gate()
        report["toy_interventions_cpu"] = toy_intervention_gates("cpu")
        report["toy_interventions_gpu"] = toy_intervention_gates("cuda")
        report["stock_attention_parity"] = stock_attention_parity_gate()

        config = AutoConfig.from_pretrained(NVFP4).text_config
        if (
            int(config.num_hidden_layers) != 66
            or int(config.hidden_size) != HIDDEN
            or int(config.num_attention_heads) != HEADS
        ):
            raise RuntimeError("unexpected checkpoint configuration")
        configure_attention(config)
        reader = ShardReader(str(NVFP4))
        baseline_manifest, _ = save_baseline_calibration(args.dump.resolve(), config, reader)
        report["baseline_calibration"] = {
            "passed": True,
            "manifest_sha256": sha256_file(args.dump.resolve() / "baseline" / "manifest.json"),
            "artifact_count": baseline_manifest["artifact_count"],
        }

        replay, _ = production_layer_gate(
            config,
            reader,
            layer_index=65,
            state_name="hidden_L64",
            text="06_random",
            arm=None,
        )
        certified_bits = np.load(state_path("hidden_L65", "06_random"), allow_pickle=False)
        replay_bits = bf16_bits(replay[0])
        equal = bool(np.array_equal(replay_bits, certified_bits))
        maximum = float(
            (replay[0].float().cpu() - load_bf16_state(state_path("hidden_L65", "06_random"), "cpu")[0].float()).abs().max().item()
        )
        del replay
        if not equal:
            raise RuntimeError(f"production L65 no-intervention replay failed; max delta={maximum}")
        report["production_l65_replay"] = {
            "passed": True,
            "bitwise_equal": equal,
            "max_abs_delta": maximum,
            "values_compared": int(SEQ * HIDDEN),
        }

        clock_arm = ARM_BY_ID["clock_freeze_L53"]
        _unused, clock_pre = production_layer_gate(
            config,
            reader,
            layer_index=53,
            state_name="hidden_L52",
            text="06_random",
            arm=clock_arm,
        )
        del _unused
        if clock_pre is None:
            raise RuntimeError("clock L53 gate did not capture pre-intervention r-vectors")
        expected_pre = np.load(CAPTURE / "replay" / "rvec_L53_06_random.npy", allow_pickle=False)
        actual_pre = clock_pre.numpy()
        clock_equal = bool(np.array_equal(actual_pre, expected_pre))
        if not clock_equal:
            raise RuntimeError("clock L53 pre-intervention r-vector is not bitwise equal")
        report["clock_l53_locus_identity"] = {
            "passed": True,
            "bitwise_equal": clock_equal,
            "values_compared": int(expected_pre.size),
            "certified_rvec_sha256": sha256_file(CAPTURE / "replay" / "rvec_L53_06_random.npy"),
        }
        report["passed"] = True
        report["completed_at_utc"] = utc_now()
        report["elapsed_seconds"] = round(time.time() - started, 3)
        report["device"] = torch.cuda.get_device_name(0)
        report["cuda"] = torch.version.cuda
        atomic_json(out, report)
        print(json.dumps({"passed": True, "preflight": str(out), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["failed_at_utc"] = utc_now()
        report["elapsed_seconds"] = round(time.time() - started, 3)
        atomic_json(out, report)
        raise
    finally:
        reset_active()


def require_preflight(path: Path, dump_root: Path) -> dict[str, Any]:
    report = load_json(path)
    if report.get("kind") != "round5_r5d_gpu_preflight" or report.get("passed") is not True:
        raise RuntimeError("R5-D GPU preflight is absent or failed")
    current_head = git_output("rev-parse", "HEAD")
    if report["critical_git"]["git_head"] != current_head:
        raise RuntimeError("HEAD changed after GPU preflight")
    baseline = load_json(dump_root / "baseline" / "manifest.json")
    if not baseline.get("complete"):
        raise RuntimeError("baseline calibration is incomplete")
    return report


def save_meter_artifact(
    arm_root: Path,
    meter: R5DMeter,
    *,
    arm: Arm,
    layer: int,
    text: str,
    sliding: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = meter.to_npz()
    integrity = meter_integrity(values, layer, sliding)
    path = arm_root / "meters" / f"L{layer:02d}_{text}.npz"
    metadata = np.asarray(
        json.dumps(
            {
                "arm_id": arm.arm_id,
                "layer": layer,
                "text": text,
                "is_sliding": sliding,
                "seq": SEQ,
                "qchunk": QCHUNK,
            },
            sort_keys=True,
        )
    )
    atomic_npz(path, **values, meta=metadata)
    return (
        artifact_record(
            arm_root,
            path,
            "r5d_locus_meter",
            layer=layer,
            text=text,
            shape=[HEADS, int(values["count"].shape[0])],
        ),
        integrity,
    )


def add_deltas(
    values: dict[str, np.ndarray], baseline: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if not np.array_equal(values["target_position"], baseline["target_position"]):
        raise RuntimeError("target-position drift")
    if not np.array_equal(values["target_id"], baseline["target_id"]):
        raise RuntimeError("target-ID drift")
    output = dict(values)
    for key in ("target_logit", "log_normalizer", "nll", "log_probability", "probability"):
        output[f"delta_{key}"] = (values[key] - baseline[key]).astype(np.float32)
    return output


def needle_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sidecar = load_json(CORPUS / "05_needles.sidecar.json")
    queries = np.asarray([int(entity["token_positions"][1]) for entity in sidecar["entities"]], dtype=np.int32)
    sides = np.asarray([str(entity["side_of_seam"]) for entity in sidecar["entities"]])
    ids = verified_ids("05_needles")
    if queries.shape != (24,) or np.any(queries < 0) or np.any(queries >= SEQ - 1):
        raise RuntimeError("invalid registered needle queries")
    return queries, ids[queries + 1].astype(np.int32), sides


@torch.no_grad()
def run_arm(arm: Arm, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R5-D arms")
    dump_root = args.dump.resolve()
    preflight = require_preflight(args.preflight.resolve(), dump_root)
    arm_root = dump_root / "arms" / arm.arm_id
    manifest_path = arm_root / "manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing.get("complete") is True:
            print(f"SKIP complete arm {arm.arm_id}", flush=True)
            return
        raise FileExistsError(f"preserving incomplete/failed arm: {arm_root}")
    if arm_root.exists() and any(arm_root.iterdir()):
        raise FileExistsError(f"refusing nonempty arm directory: {arm_root}")
    arm_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_r5d_intervention_arm",
        "arm": asdict(arm),
        "complete": False,
        "started_at_utc": utc_now(),
        "source_git_head": preflight["critical_git"]["git_head"],
        "preflight_sha256": sha256_file(args.preflight.resolve()),
        "artifacts": [],
        "expected_artifact_count": (
            7
            + 6 * len(arm.meter_layers)
            + (6 * len(arm.layers) if arm.kind in {"clock_freeze", "clock_sham"} else 0)
        ),
        "meter_integrity": {},
    }
    atomic_json(manifest_path, manifest)
    started = time.time()
    torch.set_grad_enabled(False)
    try:
        config = AutoConfig.from_pretrained(NVFP4).text_config
        configure_attention(config)
        reader = ShardReader(str(NVFP4))
        baseline = load_baseline(dump_root)
        ids_by_text = {text: verified_ids(text) for text in TEXTS}
        entry = state_name_entering(arm.start_layer)
        hidden = {text: load_bf16_state(state_path(entry, text)) for text in TEXTS}
        manifest["restart_state"] = entry
        manifest["restart_state_sha256"] = {
            text: sha256_file(state_path(entry, text)) for text in TEXTS
        }

        for layer_index in range(arm.start_layer, 66):
            layer_started = time.time()
            layer = build_layer(config, layer_index, reader, "cuda")
            sliding = config.layer_types[layer_index] == "hybrid_sliding"
            hook = None
            carrier = None
            wall_tail = None
            anchor = direction = None
            if arm.kind == "carrier_out" and layer_index in arm.layers:
                carrier = load_carrier(layer_index)
                hook = layer.self_attn.r_proj.register_forward_pre_hook(carrier_pre_hook(carrier))
            if arm.kind == "wall_heal" and layer_index in arm.layers:
                wall_tail = load_wall_tail(layer_index)
            if arm.kind in {"clock_freeze", "clock_sham"} and layer_index in arm.layers:
                anchor, direction = clock_arrays(arm, layer_index)
            try:
                for text in TEXTS:
                    meter = (
                        R5DMeter(HEADS, int(config.sliding_window_size) if sliding else SEQ, "cuda")
                        if layer_index in arm.meter_layers
                        else None
                    )
                    reset_active()
                    _ACTIVE.update(
                        arm=arm,
                        layer=layer_index,
                        text=text,
                        meter=meter,
                        sliding=sliding,
                        window=int(config.sliding_window_size),
                        qchunk=args.qchunk,
                        clock_anchor=anchor,
                        clock_direction=direction,
                        wall_tail=wall_tail,
                    )
                    hidden[text] = layer(
                        hidden[text], attention_mask=None, conv_mask=None, past_key_values=None
                    )
                    if meter is not None:
                        record, integrity = save_meter_artifact(
                            arm_root,
                            meter,
                            arm=arm,
                            layer=layer_index,
                            text=text,
                            sliding=sliding,
                        )
                        manifest["artifacts"].append(record)
                        manifest["meter_integrity"][f"L{layer_index:02d}:{text}"] = integrity
                    if arm.kind in {"clock_freeze", "clock_sham"} and layer_index in arm.layers:
                        captured = _ACTIVE["clock_pre"]
                        if captured is None:
                            raise RuntimeError(f"missing pre-intervention clock r-vector L{layer_index:02d} {text}")
                        clock_path = arm_root / "clock" / f"rvec_pre_L{layer_index:02d}_{text}.npy"
                        atomic_npy(clock_path, captured.numpy())
                        manifest["artifacts"].append(
                            artifact_record(
                                arm_root,
                                clock_path,
                                "clock_preintervention_rvec",
                                layer=layer_index,
                                text=text,
                                dtype="float16",
                                shape=[SEQ, HEADS, RPERHEAD],
                            )
                        )
                    del meter
            finally:
                if hook is not None:
                    hook.remove()
                reset_active()
            del layer, carrier, wall_tail, anchor, direction
            gc.collect()
            torch.cuda.empty_cache()
            manifest["last_completed_layer"] = layer_index
            manifest["elapsed_seconds"] = round(time.time() - started, 3)
            atomic_json(manifest_path, manifest)
            print(
                f"{arm.arm_id} L{layer_index:02d} {time.time() - layer_started:.1f}s",
                flush=True,
            )

        final_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        final_norm.weight = torch.nn.Parameter(
            reader.get("model.llm.norm.weight", "cuda").to(torch.bfloat16), requires_grad=False
        )
        final_norm.eval()
        unembed = reader.get("model.llm.unembed.weight", "cuda").to(torch.bfloat16)
        readouts: dict[str, dict[str, np.ndarray]] = {}
        for text in TEXTS:
            values = compute_readout(
                hidden[text],
                ids_by_text[text],
                final_norm,
                unembed,
                mup_multiplier=float(config.logits_mup_width_multiplier),
                unpadded_vocab=int(config.unpadded_vocab_size),
            )
            values = add_deltas(values, baseline[text])
            path = arm_root / "tokens" / f"{text}.npz"
            atomic_npz(path, **values)
            manifest["artifacts"].append(
                artifact_record(
                    arm_root,
                    path,
                    "r5d_token_readout",
                    text=text,
                    count=SEQ - 1,
                )
            )
            finite = bool(
                all(np.isfinite(value).all() for value in values.values() if value.dtype.kind == "f")
            )
            if not finite:
                raise RuntimeError(f"nonfinite token readout: {text}")
            readouts[text] = values
        queries, targets, sides = needle_indices()
        needle_values = readouts["05_needles"]
        # Target position q+1 is stored at zero-based array index q.
        needle = {
            "query_position": queries,
            "target_position": queries + 1,
            "target_id": targets,
            "side_of_seam": sides,
            "delta_target_logit": needle_values["delta_target_logit"][queries],
            "delta_probability": needle_values["delta_probability"][queries],
            "delta_log_probability": needle_values["delta_log_probability"][queries],
            "delta_nll": needle_values["delta_nll"][queries],
        }
        needle_path = arm_root / "needle" / "05_needles.npz"
        atomic_npz(needle_path, **needle)
        manifest["artifacts"].append(
            artifact_record(arm_root, needle_path, "r5d_needle_readout", count=24)
        )
        del unembed, final_norm, hidden
        gc.collect()
        torch.cuda.empty_cache()

        manifest["artifact_count"] = len(manifest["artifacts"])
        if manifest["artifact_count"] != manifest["expected_artifact_count"]:
            raise RuntimeError(
                f"arm artifact count {manifest['artifact_count']} != "
                f"{manifest['expected_artifact_count']}"
            )
        manifest["complete"] = True
        manifest["completed_at_utc"] = utc_now()
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        atomic_json(manifest_path, manifest)
        print(
            f"SEALED {arm.arm_id} artifacts={manifest['artifact_count']} "
            f"elapsed={manifest['elapsed_seconds']:.1f}s",
            flush=True,
        )
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        manifest["failed_at_utc"] = utc_now()
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        atomic_json(manifest_path, manifest)
        raise
    finally:
        reset_active()


def batch_command(args: argparse.Namespace) -> None:
    require_preflight(args.preflight.resolve(), args.dump.resolve())
    selection = ARMS
    if args.start_at:
        if args.start_at not in ARM_BY_ID:
            raise ValueError(f"unknown --start-at arm: {args.start_at}")
        selection = ARMS[[arm.arm_id for arm in ARMS].index(args.start_at) :]
    if args.stop_after:
        if args.stop_after not in {arm.arm_id for arm in selection}:
            raise ValueError(f"unknown/out-of-range --stop-after arm: {args.stop_after}")
        selection = selection[: [arm.arm_id for arm in selection].index(args.stop_after) + 1]
    def write_status() -> dict[str, Any]:
        complete_now: list[str] = []
        for registered in ARMS:
            path = args.dump.resolve() / "arms" / registered.arm_id / "manifest.json"
            if path.exists() and load_json(path).get("complete") is True:
                complete_now.append(registered.arm_id)
        payload = {
            "schema_version": 1,
            "kind": "round5_r5d_batch_status",
            "updated_at_utc": utc_now(),
            "registered_arm_count": len(ARMS),
            "complete_arm_count": len(complete_now),
            "complete_arms": complete_now,
            "status": "COMPLETE" if len(complete_now) == len(ARMS) else "RUNNING",
        }
        atomic_json(args.dump.resolve() / "batch_status.json", payload)
        return payload

    write_status()
    for ordinal, arm in enumerate(selection, start=1):
        print(f"BATCH {ordinal}/{len(selection)} {arm.arm_id}", flush=True)
        run_arm(arm, args)
        write_status()
    status = write_status()
    print(json.dumps(status, indent=2), flush=True)


def self_test_command(_args: argparse.Namespace) -> None:
    inventory = build_arm_inventory()
    starts = sorted({state_name_entering(arm.start_layer) for arm in inventory})
    gates = validate_frozen_inputs()
    toy = toy_intervention_gates("cpu")
    parity = legacy_compact_attention_parity_gate("cpu")
    result = {
        "passed": True,
        "arm_count": len(inventory),
        "family_counts": {
            kind: sum(arm.kind == kind for arm in inventory) for kind in sorted({arm.kind for arm in inventory})
        },
        "restart_states": starts,
        "frozen_inputs": gates,
        "toy_interventions": toy,
        "a6_compact_attention_parity": parity,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def list_command(_args: argparse.Namespace) -> None:
    print(json.dumps([asdict(arm) for arm in ARMS], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--qchunk", type=int, default=QCHUNK)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    subparsers.add_parser("list")
    subparsers.add_parser("preflight")
    arm = subparsers.add_parser("arm")
    arm.add_argument("--arm", required=True, choices=sorted(ARM_BY_ID))
    batch = subparsers.add_parser("batch")
    batch.add_argument("--start-at", choices=[arm.arm_id for arm in ARMS])
    batch.add_argument("--stop-after", choices=[arm.arm_id for arm in ARMS])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.qchunk != QCHUNK:
        raise ValueError("registered qchunk is exactly 512")
    if args.command == "self-test":
        self_test_command(args)
    elif args.command == "list":
        list_command(args)
    elif args.command == "preflight":
        preflight_command(args)
    elif args.command == "arm":
        run_arm(ARM_BY_ID[args.arm], args)
    elif args.command == "batch":
        batch_command(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
