"""D1 widened A6-corrected GPU capture for v1 plus fresh P-e/P-f arms.

Production scope is frozen by ROUND5_CAPTURE_SCOPE_D1.md.  All six v1 texts
receive corrected r-vectors, Tier-2 meters, lossless BF16 normalized attention
inputs, massive-coordinate censuses, LF5 needle rows, and next-token NLL.  The
two fresh paired arms receive r-vectors for P-e/P-f.  Under the registered D4
extension, all eight arms also receive lossless BF16 embedding residuals and
all 66 layer-output residuals.

The production command refuses uncommitted drift in every critical public
source/spec, runs the A8 stock-parity gate, records every checkpoint shard and
package path/version, writes only to a fresh directory, and preserves a failed
manifest on any exception.
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
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tokenizers
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import tier2_stream as T  # noqa: E402
from tier2_run import build_layer  # noqa: E402
from tier2_stream import GLOBAL_LAYERS, Meter, ShardReader, measuring_attention  # noqa: E402

from transformers import AutoConfig  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.inkling.modeling_inkling import (  # noqa: E402
    InklingRMSNorm,
    InklingRelativeLogits,
    eager_attention_forward,
)


NVFP4 = ROOT / "nvfp4"
CORPUS = ROOT / "corpus"
PAIRED = ROOT / "corpus_v2"
TOKENIZER = CORPUS / "tokenizer.json"
PUBLIC_FREEZE = ROOT / "analysis" / "round5" / "pe" / "corpus_freeze.json"
DEFAULT_OUT = ROOT / "dumps" / "round5" / "widened_corrected_capture"
DEFAULT_PREFLIGHT = ROOT / "analysis" / "round5" / "widened_capture" / "preflight.json"

V1_TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
PAIRED_TEXTS = ["09_pe_single_thread", "10_pe_multi_conversation"]
TEXTS = V1_TEXTS + PAIRED_TEXTS
STATE_TEXTS = TEXTS
LAYERS = list(range(66))
SEQ = 8192
HIDDEN_SIZE = 6144
QCHUNK = 512
MASSIVE_THRESHOLD = 30_000.0
EXPECTED_NEEDLE_QUERIES = 24
STATE_SNAPSHOTS_PER_TEXT = len(LAYERS) + 1
STATE_PAYLOAD_BYTES = len(STATE_TEXTS) * STATE_SNAPSHOTS_PER_TEXT * SEQ * HIDDEN_SIZE * 2
ATTENTION_DTYPE_BOUNDARY = "BF16 content+bias add, then FP32 softmax"

D1_COMMIT = "51b8c00fe9b632086d0745221578be452f76f60c"
A5_COMMIT = "7bf608d9971997a655a4f9cd46e3bc921ffb74b8"
A6_COMMIT = "061bb04dffef05eae33f9fffc430394faa4052c5"
A8_COMMIT = "93665e2d75a68d4d2d77e2751c316f9a6665f796"
PE_PF_COMMIT = "71f0ad3efff199a83c333340ddd8c8f9a8d7f228"
D4_COMMIT = "2264ae457b36c75c778fe7874d08fdd0eb84aae5"

CRITICAL_PUBLIC_FILES = [
    "scripts/round5_widened_capture.py",
    "scripts/round5_widened_validate.py",
    "scripts/round5_pe_paired_build.py",
    "scripts/tier2_run.py",
    "scripts/tier2_stream.py",
    "scripts/tier2_nvfp4.py",
    "registrations/ROUND5_CAPTURE_SCOPE_D1.md",
    "registrations/ROUND5_CAPTURE_SCOPE_D4.md",
    "registrations/ROUND5_AMENDMENT_A5.md",
    "registrations/ROUND5_AMENDMENT_A6.md",
    "registrations/ROUND5_AMENDMENT_A8_VALIDATION.md",
    "registrations/ROUND5_CAPTURE_AMENDMENT.md",
    "registrations/ROUND5_APERTURE_CONTEXT_DOSE_PREREG.md",
    "registrations/ROUND5_APERTURE_ANCHOR_PREREG.md",
    "analysis/round5/pe/corpus_freeze.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def git_blob_sha256(commit: str, relative_path: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=ROOT, stderr=subprocess.STDOUT
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


def artifact_record(
    out_root: Path,
    path: Path,
    *,
    kind: str,
    dtype: str,
    shape: tuple[int, ...] | list[int],
    layer: int | None = None,
    text: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(out_root).as_posix(),
        "kind": kind,
        "dtype": dtype,
        "shape": [int(value) for value in shape],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if layer is not None:
        record["layer"] = int(layer)
    if text is not None:
        record["text"] = text
    if extra:
        record.update(extra)
    return record


def package_record(module: Any) -> dict[str, str]:
    return {
        "version": str(module.__version__),
        "module_path": str(Path(module.__file__).resolve()),
    }


def expected_artifact_count() -> int:
    # All-arm rvec + three v1 layer/text products + one needle file/layer +
    # v1 NLL + lossless embedding/layer-output residual states for all arms.
    return (
        len(TEXTS) * len(LAYERS)
        + 3 * len(V1_TEXTS) * len(LAYERS)
        + len(LAYERS)
        + len(V1_TEXTS)
        + len(STATE_TEXTS) * STATE_SNAPSHOTS_PER_TEXT
    )


def load_input_manifests() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    v1 = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    paired = json.loads((PAIRED / "pe_manifest.json").read_text(encoding="utf-8"))
    public = json.loads(PUBLIC_FREEZE.read_text(encoding="utf-8"))
    if v1.get("seq") != SEQ or set(V1_TEXTS) - set(v1.get("texts", {})):
        raise RuntimeError("invalid v1 corpus manifest")
    if (
        paired.get("kind") != "round5_pe_paired_private_corpus"
        or paired.get("complete") is not True
        or paired.get("private") is not True
        or paired.get("seq") != SEQ
        or list(paired.get("arms", {})) != PAIRED_TEXTS
    ):
        raise RuntimeError("invalid private paired-corpus manifest")
    if (
        public.get("kind") != "round5_pe_pf_public_corpus_freeze"
        or public.get("outcome_data_opened") is not False
        or public.get("private_manifest_sha256") != sha256_file(PAIRED / "pe_manifest.json")
        or public.get("private_classes_sha256") != sha256_file(PAIRED / "pe_classes.json")
        or public.get("pair_preservation", {}).get("passed") is not True
        or public.get("pair_preservation", {}).get("intersection_over_union", 0.0) < 0.80
    ):
        raise RuntimeError("invalid public paired-corpus freeze")
    if paired.get("classes", {}).get("sha256") != sha256_file(PAIRED / "pe_classes.json"):
        raise RuntimeError("private paired class freeze is stale")
    builder_hash = sha256_file(SCRIPT_DIR / "round5_pe_paired_build.py")
    if paired.get("builder_source_sha256") != builder_hash or public.get("builder_source_sha256") != builder_hash:
        raise RuntimeError("paired corpus was not built by the current committed builder")
    return v1, paired, public


def verified_ids(
    name: str, v1_manifest: dict[str, Any], paired_manifest: dict[str, Any]
) -> np.ndarray:
    if name in V1_TEXTS:
        path = CORPUS / f"{name}.ids.npy"
        expected = v1_manifest["texts"][name]["ids_sha256"]
    else:
        path = PAIRED / f"{name}.ids.npy"
        expected = paired_manifest["arms"][name]["ids"]["sha256"]
    if sha256_file(path) != expected:
        raise RuntimeError(f"{name} ID hash mismatch")
    ids = np.load(path, allow_pickle=False)
    if ids.shape != (SEQ,) or ids.dtype != np.int32:
        raise RuntimeError(f"invalid IDs for {name}: {ids.shape}, {ids.dtype}")
    return ids


def input_gate() -> dict[str, Any]:
    v1, paired, public = load_input_manifests()
    ids_hashes: dict[str, str] = {}
    for name in TEXTS:
        ids = verified_ids(name, v1, paired)
        if int(ids.min()) < 0:
            raise RuntimeError(f"negative token ID in {name}")
        root = CORPUS if name in V1_TEXTS else PAIRED
        ids_hashes[name] = sha256_file(root / f"{name}.ids.npy")
    for name in PAIRED_TEXTS:
        for suffix, field in (("txt", "text"), ("sidecar.json", "sidecar")):
            path = PAIRED / f"{name}.{suffix}"
            if sha256_file(path) != paired["arms"][name][field]["sha256"]:
                raise RuntimeError(f"paired {field} hash mismatch: {name}")
        sidecar = json.loads((PAIRED / f"{name}.sidecar.json").read_text(encoding="utf-8"))
        if (
            sidecar.get("seq") != SEQ
            or len(sidecar.get("token_message_index", [])) != SEQ
            or len(sidecar.get("token_conversation_index", [])) != SEQ
            or len(sidecar.get("token_offsets", [])) != SEQ
        ):
            raise RuntimeError(f"invalid paired sidecar: {name}")
    classes = json.loads((PAIRED / "pe_classes.json").read_text(encoding="utf-8"))
    if classes.get("pair_preservation", {}).get("passed") is not True:
        raise RuntimeError("private paired-message preservation gate failed")
    return {
        "passed": True,
        "v1_manifest_sha256": sha256_file(CORPUS / "manifest.json"),
        "paired_manifest_sha256": sha256_file(PAIRED / "pe_manifest.json"),
        "paired_classes_sha256": sha256_file(PAIRED / "pe_classes.json"),
        "public_freeze_sha256": sha256_file(PUBLIC_FREEZE),
        "tokenizer_sha256": sha256_file(TOKENIZER),
        "input_ids_sha256": ids_hashes,
        "pair_preservation": public["pair_preservation"],
    }


def critical_git_gate() -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    records: dict[str, Any] = {}
    for relative in CRITICAL_PUBLIC_FILES:
        path = ROOT / relative
        current = sha256_file(path)
        committed = git_blob_sha256(head, relative)
        equal = current == committed
        records[relative] = {"sha256": current, "git_blob_sha256": committed, "equal": equal}
        if not equal:
            raise RuntimeError(f"critical capture dependency is not committed at HEAD: {relative}")
    return {"passed": True, "git_head": head, "files": records}


def checkpoint_shard_records(
    *, hash_contents: bool = True
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    index = json.loads((NVFP4 / "model.safetensors.index.json").read_text(encoding="utf-8"))
    indexed = sorted(set(index["weight_map"].values()))
    trunk = [name for name in indexed if name.startswith("model-")]
    nontrunk = [name for name in indexed if not name.startswith("model-")]
    on_disk = sorted(path.name for path in NVFP4.glob("model-*.safetensors"))
    if trunk != on_disk or nontrunk != ["mtp.safetensors"]:
        raise RuntimeError(
            "checkpoint index/directory inventory mismatch: "
            f"indexed_trunk={len(trunk)}, disk_trunk={len(on_disk)}, nontrunk={nontrunk}"
        )
    records: dict[str, dict[str, Any]] = {}
    for ordinal, name in enumerate(trunk, start=1):
        path = NVFP4 / name
        record: dict[str, Any] = {"bytes": path.stat().st_size}
        if hash_contents:
            print(f"hashing checkpoint shard {ordinal:02d}/{len(trunk)}: {name}", flush=True)
            record["sha256"] = sha256_file(path)
        records[name] = record
    return records, nontrunk


def stock_attention_parity_gate() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the A8 stock-parity gate")

    class Dummy:
        num_key_value_groups = 2
        training = False

    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xD1A8)
    cases: dict[str, Any] = {}
    prior_active = dict(T._ACTIVE)
    prior_capture = dict(T._CAPTURE)
    try:
        for label, sliding, window, extent in (
            ("global", False, 64, 32),
            ("sliding", True, 8, 8),
        ):
            heads, kv_heads, sequence, dimension = 4, 2, 17, 8
            query = torch.randn(
                1,
                heads,
                sequence,
                dimension,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            key = torch.randn(
                1,
                kv_heads,
                sequence,
                dimension,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            value = torch.randn(
                1,
                kv_heads,
                sequence,
                dimension,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            compact = torch.randn(
                1,
                heads,
                sequence,
                extent,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            qpos = torch.arange(sequence, device="cuda")
            kpos = torch.arange(sequence, device="cuda")
            distance = qpos[:, None] - kpos[None, :]
            gather_index = distance.clamp(0, extent - 1)[None, None].expand(
                1, heads, -1, -1
            )
            dense_bias = torch.gather(compact, 3, gather_index).masked_fill(
                ~((distance >= 0) & (distance < extent))[None, None], 0.0
            )
            valid = distance >= 0
            if sliding:
                valid &= distance < window
            mask = torch.zeros(
                1, 1, sequence, sequence, device="cuda", dtype=torch.bfloat16
            ).masked_fill(~valid[None, None], torch.finfo(torch.bfloat16).min)
            scaling = 1.0 / dimension
            T._ACTIVE.update(
                meter=None,
                sliding=sliding,
                window=window,
                qchunk=7,
                needle_qpos=None,
                needle_rows=None,
            )
            T._CAPTURE["enabled"] = False
            measured, _ = measuring_attention(
                Dummy(), query, key, value, None, scaling, position_bias=compact
            )
            stock, _ = eager_attention_forward(
                Dummy(), query, key, value, mask, scaling, position_bias=dense_bias
            )
            equal = bool(torch.equal(measured, stock))
            maximum_delta = float((measured.float() - stock.float()).abs().max().item())
            cases[label] = {
                "bitwise_equal": equal,
                "max_output_delta": maximum_delta,
                "shape": list(measured.shape),
            }
            if not equal:
                raise RuntimeError(f"stock attention parity failed for {label}: {maximum_delta}")
    finally:
        T._ACTIVE.clear()
        T._ACTIVE.update(prior_active)
        T._CAPTURE.clear()
        T._CAPTURE.update(prior_capture)
    return {"passed": True, "cases": cases}


def startup_gate(*, hash_shards: bool = True) -> dict[str, Any]:
    import transformers

    inputs = input_gate()
    git_gate = critical_git_gate()
    parity = stock_attention_parity_gate()
    # Development preflight may skip the expensive content hashes, but it
    # still authenticates the index-versus-directory inventory and byte sizes.
    shards, nontrunk = checkpoint_shard_records(hash_contents=hash_shards)
    modeling_path = Path(sys.modules[InklingRMSNorm.__module__].__file__).resolve()
    return {
        "passed": True,
        "inputs": inputs,
        "critical_git_blobs": git_gate,
        "stock_attention_parity": parity,
        "checkpoint_shards": shards,
        "checkpoint_index_nontrunk_files": nontrunk,
        "packages": {
            "numpy": package_record(np),
            "tokenizers": package_record(tokenizers),
            "torch": package_record(torch),
            "transformers": package_record(transformers),
        },
        "modeling_inkling": {
            "path": str(modeling_path),
            "sha256": sha256_file(modeling_path),
        },
    }


def bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 normalized input, got {tensor.dtype}")
    return tensor.detach().contiguous().to("cpu").view(torch.uint16).numpy().copy()


def save_residual_state(
    out_root: Path,
    path: Path,
    tensor: torch.Tensor,
    *,
    text: str,
    state_name: str,
    state_index: int,
    state_role: str,
    layer: int | None,
) -> dict[str, Any]:
    values = bf16_bits(tensor)
    if values.shape != (SEQ, HIDDEN_SIZE):
        raise RuntimeError(f"invalid residual state shape for {state_name}, {text}: {values.shape}")
    nonfinite = int(
        np.count_nonzero((values & np.uint16(0x7F80)) == np.uint16(0x7F80))
    )
    if nonfinite:
        raise RuntimeError(f"non-finite BF16 residual state for {state_name}, {text}")
    atomic_npy(path, values)
    return artifact_record(
        out_root,
        path,
        kind="residual_hidden_state",
        dtype="bfloat16 payload stored losslessly as uint16",
        shape=values.shape,
        layer=layer,
        text=text,
        extra={
            "state_name": state_name,
            "state_index": state_index,
            "state_role": state_role,
            "nonfinite_bf16_words": nonfinite,
            "d4_commit": D4_COMMIT,
        },
    )


def save_normalized(
    out_root: Path, path: Path, tensor: torch.Tensor, layer: int, text: str
) -> dict[str, Any]:
    values = bf16_bits(tensor)
    if values.shape != (SEQ, HIDDEN_SIZE):
        raise RuntimeError(f"invalid normalized input shape: {values.shape}")
    decoded = torch.from_numpy(values.copy()).view(torch.bfloat16).float()
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError(f"non-finite normalized input at L{layer:02d}, {text}")
    atomic_npy(path, values)
    return artifact_record(
        out_root,
        path,
        kind="normalized_attention_input",
        dtype="bfloat16 payload stored as uint16",
        shape=values.shape,
        layer=layer,
        text=text,
    )


def save_massive(
    out_root: Path, path: Path, hidden: torch.Tensor, layer: int, text: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = hidden.detach()[0]
    indices = torch.nonzero(state.float().abs() > MASSIVE_THRESHOLD, as_tuple=False)
    if len(indices):
        positions = indices[:, 0].to("cpu", torch.int32).numpy()
        channels = indices[:, 1].to("cpu", torch.int32).numpy()
        values = state[indices[:, 0], indices[:, 1]].float().cpu().numpy()
    else:
        positions = np.empty(0, dtype=np.int32)
        channels = np.empty(0, dtype=np.int32)
        values = np.empty(0, dtype=np.float32)
    atomic_npz(path, position=positions, channel=channels, value=values)
    maximum = float(np.max(np.abs(values))) if len(values) else 0.0
    return (
        artifact_record(
            out_root,
            path,
            kind="massive_activation_census",
            dtype="position:int32,channel:int32,value:float32",
            shape=(len(values),),
            layer=layer,
            text=text,
            extra={"threshold_abs_gt": MASSIVE_THRESHOLD, "count": len(values), "max_abs": maximum},
        ),
        {"count": int(len(values)), "max_abs": maximum},
    )


def save_rvec(
    out_root: Path, path: Path, values: np.ndarray, layer: int, text: str
) -> dict[str, Any]:
    values = np.asarray(values)
    if values.shape != (SEQ, 64, 16) or values.dtype != np.float16:
        raise RuntimeError(f"invalid r-vector at L{layer:02d}, {text}: {values.shape}, {values.dtype}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite r-vector at L{layer:02d}, {text}")
    atomic_npy(path, values)
    return artifact_record(
        out_root,
        path,
        kind="rvec",
        dtype="float16",
        shape=values.shape,
        layer=layer,
        text=text,
        extra={"scope": "v1_replay" if text in V1_TEXTS else "fresh_paired"},
    )


def save_meter(
    out_root: Path,
    path: Path,
    meter: Meter,
    *,
    layer: int,
    text: str,
    is_sliding: bool,
    rel_extent: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = meter.to_npz()
    dmax = 512 if is_sliding else SEQ
    expected_shape = (64, dmax)
    for field in (
        "mass_with",
        "mass_without",
        "bias_sum",
        "content_sum",
        "mean_mass_with",
        "mean_mass_without",
        "mean_bias",
        "mean_content",
    ):
        if arrays[field].shape != expected_shape or not np.isfinite(arrays[field]).all():
            raise RuntimeError(f"invalid meter {field} at L{layer:02d}, {text}")
    expected_count = SEQ - np.arange(dmax, dtype=np.float64)
    if not np.array_equal(arrays["count"], expected_count):
        raise RuntimeError(f"meter count geometry mismatch at L{layer:02d}, {text}")
    with_error = float(np.max(np.abs(arrays["mass_with"].sum(axis=1) - SEQ)))
    without_error = float(np.max(np.abs(arrays["mass_without"].sum(axis=1) - SEQ)))
    if with_error > 0.05 or without_error > 0.05:
        raise RuntimeError(
            f"meter mass conservation failed at L{layer:02d}, {text}: "
            f"with={with_error}, without={without_error}"
        )
    meta = {
        "layer": layer,
        "text": text,
        "is_global": layer in GLOBAL_LAYERS,
        "is_sliding": is_sliding,
        "rel_extent": rel_extent,
        "seq": SEQ,
        "qchunk": QCHUNK,
        "n_heads": 64,
        "a6_corrected": True,
    }
    atomic_npz(path, **arrays, meta=np.array(json.dumps(meta, sort_keys=True)))
    integrity = {
        "max_mass_with_error": with_error,
        "max_mass_without_error": without_error,
    }
    return (
        artifact_record(
            out_root,
            path,
            kind="tier2_distance_meter",
            dtype="float64",
            shape=expected_shape,
            layer=layer,
            text=text,
            extra={"dmax": dmax, "integrity": integrity},
        ),
        integrity,
    )


def save_needle_rows(
    out_root: Path,
    path: Path,
    rows: dict[int, torch.Tensor],
    expected_qpos: list[int],
    layer: int,
) -> dict[str, Any]:
    qpos = sorted(rows)
    if qpos != expected_qpos or len(qpos) != EXPECTED_NEEDLE_QUERIES:
        raise RuntimeError(f"needle query inventory mismatch at L{layer:02d}")
    values = np.stack([rows[position].numpy() for position in qpos]).astype(np.float16)
    if values.shape != (EXPECTED_NEEDLE_QUERIES, 64, SEQ) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid needle rows at L{layer:02d}: {values.shape}")
    row_sum_error = float(
        np.max(np.abs(values.astype(np.float32).sum(axis=-1, dtype=np.float64) - 1.0))
    )
    if row_sum_error > 0.05:
        raise RuntimeError(f"needle-row mass error at L{layer:02d}: {row_sum_error}")
    atomic_npz(path, qpos=np.asarray(qpos, dtype=np.int64), rows=values)
    return artifact_record(
        out_root,
        path,
        kind="lf5_needle_rows",
        dtype="qpos:int64,rows:float16",
        shape=values.shape,
        layer=layer,
        text="05_needles",
        extra={"max_row_sum_error_after_fp16": row_sum_error, "backend": "CUDA/BF16 replay"},
    )


@torch.no_grad()
def compute_nll(
    hidden: torch.Tensor,
    ids: np.ndarray,
    final_norm: InklingRMSNorm,
    unembed: torch.Tensor,
    *,
    mup_multiplier: float,
    unpadded_vocab: int,
    token_chunk: int,
) -> np.ndarray:
    states = final_norm(hidden) / mup_multiplier
    targets = torch.from_numpy(ids[1:].astype(np.int64)).to(states.device)
    if int(targets.max()) >= unpadded_vocab:
        raise RuntimeError("target token exceeds unpadded vocabulary")
    losses: list[torch.Tensor] = []
    for start in range(0, SEQ - 1, token_chunk):
        stop = min(start + token_chunk, SEQ - 1)
        logits = torch.nn.functional.linear(states[:, start:stop, :], unembed)[0]
        logits32 = logits[:, :unpadded_vocab].float()
        target_logits = logits32.gather(1, targets[start:stop, None])[:, 0]
        losses.append((torch.logsumexp(logits32, dim=-1) - target_logits).cpu())
        del logits, logits32, target_logits
    result = torch.cat(losses).numpy().astype(np.float32, copy=False)
    if result.shape != (SEQ - 1,) or not np.isfinite(result).all():
        raise RuntimeError("invalid NLL output")
    return result


def provenance(
    config: Any, args: argparse.Namespace, gate: dict[str, Any]
) -> dict[str, Any]:
    return {
        "registration_commits": {
            "D1": D1_COMMIT,
            "A5": A5_COMMIT,
            "A6": A6_COMMIT,
            "A8": A8_COMMIT,
            "P-e/P-f": PE_PF_COMMIT,
            "D4": D4_COMMIT,
        },
        "git_head": gate["critical_git_blobs"]["git_head"],
        "git_branch": git_output("branch", "--show-current"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "source_sha256": {
            Path(relative).name: record["sha256"]
            for relative, record in gate["critical_git_blobs"]["files"].items()
            if relative.startswith("scripts/")
        },
        "registration_sha256": {
            relative: record["sha256"]
            for relative, record in gate["critical_git_blobs"]["files"].items()
            if relative.startswith("registrations/")
        },
        "public_freeze_sha256": gate["inputs"]["public_freeze_sha256"],
        "input_manifest_sha256": {
            "corpus": gate["inputs"]["v1_manifest_sha256"],
            "corpus_v2/pe_manifest.json": gate["inputs"]["paired_manifest_sha256"],
            "corpus_v2/pe_classes.json": gate["inputs"]["paired_classes_sha256"],
        },
        "input_ids_sha256": gate["inputs"]["input_ids_sha256"],
        "tokenizer_sha256": gate["inputs"]["tokenizer_sha256"],
        "checkpoint_index_sha256": sha256_file(NVFP4 / "model.safetensors.index.json"),
        "checkpoint_shards": gate["checkpoint_shards"],
        "checkpoint_index_nontrunk_files": gate["checkpoint_index_nontrunk_files"],
        "config_sha256": sha256_file(NVFP4 / "config.json"),
        "packages": gate["packages"],
        "modeling_inkling": gate["modeling_inkling"],
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "stock_attention_parity": gate["stock_attention_parity"],
        "attention_dtype_boundary": ATTENTION_DTYPE_BOUNDARY,
        "texts": TEXTS,
        "v1_texts": V1_TEXTS,
        "paired_texts": PAIRED_TEXTS,
        "state_texts": STATE_TEXTS,
        "layers": LAYERS,
        "seq": SEQ,
        "qchunk": args.qchunk,
        "nll_token_chunk": args.nll_token_chunk,
        "capture_features": {
            "v1": ["rvec", "meter", "normalized_attention_input", "massive_census", "nll"],
            "05_needles_addition": ["lf5_needle_rows"],
            "paired": ["rvec"],
            "all_arms_addition": ["lossless_bf16_residual_states"],
            "residual_hidden_states": True,
            "D4_satisfied": True,
        },
        "d4_state_contract": {
            "registration_commit": D4_COMMIT,
            "confirmatory_texts": V1_TEXTS,
            "secondary_descriptive_texts": PAIRED_TEXTS,
            "snapshots_per_text": STATE_SNAPSHOTS_PER_TEXT,
            "artifact_count": len(STATE_TEXTS) * STATE_SNAPSHOTS_PER_TEXT,
            "raw_payload_bytes": STATE_PAYLOAD_BYTES,
            "shape": [SEQ, HIDDEN_SIZE],
            "storage_dtype": "uint16",
            "logical_dtype": "bfloat16",
            "state_sequence": ["hidden_embed"]
            + [f"hidden_L{layer:02d}" for layer in LAYERS],
            "layer_semantics": (
                "hidden_embed enters L0; hidden_Lxx is the residual output of Lxx; "
                "rotation attributable to L compares its input with hidden_Lxx"
            ),
            "clipping_or_imputation": False,
        },
        "model": {
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "num_attention_heads": int(config.num_attention_heads),
            "unpadded_vocab_size": int(config.unpadded_vocab_size),
            "logits_mup_width_multiplier": float(config.logits_mup_width_multiplier),
        },
    }


def needle_queries() -> list[int]:
    sidecar = json.loads((CORPUS / "05_needles.sidecar.json").read_text(encoding="utf-8"))
    positions = sorted(
        {
            int(entity["token_positions"][1])
            for entity in sidecar["entities"]
            if len(entity.get("token_positions", [])) >= 2
            and int(entity["token_positions"][1]) < SEQ
        }
    )
    if len(positions) != EXPECTED_NEEDLE_QUERIES:
        raise RuntimeError(f"expected 24 needle queries, found {len(positions)}")
    return positions


def capture_command(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the D1 widened capture")
    if args.qchunk != QCHUNK:
        raise ValueError("registered qchunk is 512")
    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_d1_widened_a6_capture",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "production_capture": True,
        "expected_artifact_count": expected_artifact_count(),
        "artifacts": [],
        "startup_gate": {"passed": False},
        "massive_summary": {},
        "meter_integrity": {},
        "nll_summary": {},
    }
    atomic_json(manifest_path, manifest)
    started = time.time()
    torch.set_grad_enabled(False)
    try:
        gate = startup_gate(hash_shards=True)
        manifest["startup_gate"] = gate
        config = AutoConfig.from_pretrained(NVFP4).text_config
        if int(config.num_hidden_layers) != len(LAYERS):
            raise RuntimeError("unexpected checkpoint layer count")
        ALL_ATTENTION_FUNCTIONS.register("tier2_measure", measuring_attention)
        config._attn_implementation = "tier2_measure"
        InklingRelativeLogits.forward = T.compact_relative_logits_forward
        manifest.update(provenance(config, args, gate))
        atomic_json(manifest_path, manifest)

        v1_manifest, paired_manifest, _public = load_input_manifests()
        ids_by_text = {
            name: verified_ids(name, v1_manifest, paired_manifest) for name in TEXTS
        }
        queries = needle_queries()
        reader = ShardReader(str(NVFP4))
        embed_weight = reader.get("model.llm.embed.weight", "cuda").to(torch.bfloat16)
        embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        embed_norm.weight = torch.nn.Parameter(
            reader.get("model.llm.embed_norm.weight", "cuda").to(torch.bfloat16),
            requires_grad=False,
        )
        embed_norm.eval()
        hidden: dict[str, torch.Tensor] = {}
        for name in TEXTS:
            ids_cuda = torch.from_numpy(ids_by_text[name].astype(np.int64)).to("cuda")
            hidden[name] = embed_norm(
                torch.nn.functional.embedding(ids_cuda, embed_weight).unsqueeze(0)
            )
            embed_state_path = out_root / "states" / f"hidden_embed_{name}.npy"
            manifest["artifacts"].append(
                save_residual_state(
                    out_root,
                    embed_state_path,
                    hidden[name][0],
                    text=name,
                    state_name="hidden_embed",
                    state_index=0,
                    state_role="normalized_embedding_entering_L0",
                    layer=None,
                )
            )
        del embed_weight, embed_norm
        torch.cuda.empty_cache()
        print(f"embedded {len(TEXTS)} arms at seq={SEQ}", flush=True)

        T._CAPTURE["enabled"] = True
        for layer_index in LAYERS:
            layer_started = time.time()
            layer = build_layer(config, layer_index, reader, "cuda")
            is_sliding = config.layer_types[layer_index] == "hybrid_sliding"
            rel_extent = config.sliding_window_size if is_sliding else config.rel_extent
            normalized_holder: dict[str, torch.Tensor] = {}
            active_text: dict[str, str | None] = {"name": None}

            def capture_norm(
                _module: torch.nn.Module, _inputs: Any, output: torch.Tensor
            ) -> None:
                name = active_text["name"]
                if name in V1_TEXTS:
                    if "value" in normalized_holder:
                        raise RuntimeError("input-layernorm hook fired twice before save")
                    normalized_holder["value"] = output.detach()[0].to("cpu")

            hook = layer.input_layernorm.register_forward_hook(capture_norm)
            manifest["massive_summary"][f"L{layer_index:02d}"] = {}
            manifest["meter_integrity"][f"L{layer_index:02d}"] = {}
            try:
                for name in TEXTS:
                    active_text["name"] = name
                    normalized_holder.clear()
                    meter = (
                        Meter(64, config.sliding_window_size if is_sliding else SEQ, "cuda")
                        if name in V1_TEXTS
                        else None
                    )
                    qpos = queries if name == "05_needles" else None
                    T._ACTIVE.update(
                        meter=meter,
                        sliding=is_sliding,
                        window=config.sliding_window_size,
                        qchunk=args.qchunk,
                        needle_qpos=qpos,
                        needle_rows=({} if qpos else None),
                    )
                    T._CAPTURE["rvec"] = None
                    hidden[name] = layer(
                        hidden[name],
                        attention_mask=None,
                        conv_mask=None,
                        past_key_values=None,
                    )

                    state_name = f"hidden_L{layer_index:02d}"
                    state_path = out_root / "states" / f"{state_name}_{name}.npy"
                    manifest["artifacts"].append(
                        save_residual_state(
                            out_root,
                            state_path,
                            hidden[name][0],
                            text=name,
                            state_name=state_name,
                            state_index=layer_index + 1,
                            state_role="layer_output_residual",
                            layer=layer_index,
                        )
                    )

                    captured = T._CAPTURE["rvec"]
                    if captured is None:
                        raise RuntimeError(f"missing r-vector at L{layer_index:02d}, {name}")
                    rvec_dir = "replay" if name in V1_TEXTS else "paired"
                    rvec_path = out_root / rvec_dir / f"rvec_L{layer_index:02d}_{name}.npy"
                    manifest["artifacts"].append(
                        save_rvec(out_root, rvec_path, captured.numpy(), layer_index, name)
                    )

                    if name in V1_TEXTS:
                        if "value" not in normalized_holder:
                            raise RuntimeError(
                                f"input-layernorm hook did not fire at L{layer_index:02d}, {name}"
                            )
                        normalized_path = (
                            out_root
                            / "normalized"
                            / f"attn_in_L{layer_index:02d}_{name}.npy"
                        )
                        manifest["artifacts"].append(
                            save_normalized(
                                out_root,
                                normalized_path,
                                normalized_holder.pop("value"),
                                layer_index,
                                name,
                            )
                        )
                        meter_path = (
                            out_root
                            / "meters"
                            / f"layer{layer_index:02d}_{name}_s{SEQ}.npz"
                        )
                        meter_record, integrity = save_meter(
                            out_root,
                            meter_path,
                            meter,
                            layer=layer_index,
                            text=name,
                            is_sliding=is_sliding,
                            rel_extent=int(rel_extent),
                        )
                        manifest["artifacts"].append(meter_record)
                        manifest["meter_integrity"][f"L{layer_index:02d}"][name] = integrity
                        massive_path = (
                            out_root
                            / "massive"
                            / f"massive_L{layer_index:02d}_{name}.npz"
                        )
                        massive_record, massive_summary = save_massive(
                            out_root, massive_path, hidden[name], layer_index, name
                        )
                        manifest["artifacts"].append(massive_record)
                        manifest["massive_summary"][f"L{layer_index:02d}"][name] = massive_summary

                    if name == "05_needles":
                        row_path = out_root / "replay" / f"needlerows_L{layer_index:02d}.npz"
                        manifest["artifacts"].append(
                            save_needle_rows(
                                out_root,
                                row_path,
                                T._ACTIVE["needle_rows"],
                                queries,
                                layer_index,
                            )
                        )
                    T._ACTIVE["meter"] = None
                    T._ACTIVE["needle_rows"] = None
                    T._CAPTURE["rvec"] = None
                    del meter
            finally:
                hook.remove()
                active_text["name"] = None
                T._ACTIVE["meter"] = None
                T._ACTIVE["needle_rows"] = None
                T._CAPTURE["rvec"] = None
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            manifest["last_completed_layer"] = layer_index
            manifest["elapsed_seconds"] = round(time.time() - started, 3)
            atomic_json(manifest_path, manifest)
            print(
                f"layer {layer_index:02d} {'G' if layer_index in GLOBAL_LAYERS else '.'} "
                f"{time.time() - layer_started:.1f}s",
                flush=True,
            )

        T._CAPTURE["enabled"] = False
        final_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        final_norm.weight = torch.nn.Parameter(
            reader.get("model.llm.norm.weight", "cuda").to(torch.bfloat16),
            requires_grad=False,
        )
        final_norm.eval()
        unembed = reader.get("model.llm.unembed.weight", "cuda").to(torch.bfloat16)
        for name in V1_TEXTS:
            nll = compute_nll(
                hidden[name],
                ids_by_text[name],
                final_norm,
                unembed,
                mup_multiplier=float(config.logits_mup_width_multiplier),
                unpadded_vocab=int(config.unpadded_vocab_size),
                token_chunk=args.nll_token_chunk,
            )
            nll_path = out_root / "nll" / f"nll_{name}.npz"
            atomic_npz(
                nll_path,
                target_position=np.arange(1, SEQ, dtype=np.int32),
                target_id=ids_by_text[name][1:].astype(np.int32),
                nll=nll,
            )
            manifest["artifacts"].append(
                artifact_record(
                    out_root,
                    nll_path,
                    kind="next_token_nll",
                    dtype="target_position:int32,target_id:int32,nll:float32",
                    shape=nll.shape,
                    text=name,
                )
            )
            manifest["nll_summary"][name] = {
                "count": len(nll),
                "mean": float(np.mean(nll, dtype=np.float64)),
                "median": float(np.median(nll)),
                "min": float(np.min(nll)),
                "max": float(np.max(nll)),
                "finite": bool(np.isfinite(nll).all()),
            }
        del unembed, final_norm

        uniform = math.log(int(config.unpadded_vocab_size))
        prose = manifest["nll_summary"]["01_prose_en"]["mean"]
        random = manifest["nll_summary"]["06_random"]["mean"]
        nll_gate = {
            "uniform_nll": uniform,
            "all_finite": all(item["finite"] for item in manifest["nll_summary"].values()),
            "prose_positive_below_uniform": 0.0 < prose < uniform,
            "random_above_prose": random > prose,
        }
        nll_gate["passed"] = bool(
            nll_gate["all_finite"]
            and nll_gate["prose_positive_below_uniform"]
            and nll_gate["random_above_prose"]
        )
        manifest["nll_gate"] = nll_gate
        if not nll_gate["passed"]:
            raise RuntimeError(f"NLL integrity gate failed: {nll_gate}")

        if len(manifest["artifacts"]) != expected_artifact_count():
            raise RuntimeError(
                f"artifact count {len(manifest['artifacts'])} != {expected_artifact_count()}"
            )
        manifest["artifact_count"] = len(manifest["artifacts"])
        manifest["lf5_handoff"] = {
            "backend": "replay",
            "amendment_a5_commit": A5_COMMIT,
            "input_root": ".",
            "capture_root": "replay",
            "needle_rows_recaptured": True,
            "normalized_inputs_recaptured": True,
            "parity_required_after_independent_capture_validation": True,
            "parity_command": (
                "python scripts/round5_offline_attention.py parity --backend replay "
                "--input-root dumps/round5/widened_corrected_capture "
                "--capture-root dumps/round5/widened_corrected_capture/replay "
                "--layers all --report analysis/round5/widened_capture/lf5_replay_parity.json"
            ),
        }
        manifest["complete"] = True
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_seconds"] = round(time.time() - started, 3)
        atomic_json(manifest_path, manifest)
        print(
            f"DONE artifacts={manifest['artifact_count']} wall={manifest['wall_seconds'] / 60:.2f} min",
            flush=True,
        )
    except Exception as exc:
        T._CAPTURE["enabled"] = False
        T._CAPTURE["rvec"] = None
        T._ACTIVE["meter"] = None
        T._ACTIVE["needle_rows"] = None
        manifest["complete"] = False
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = repr(exc)
        manifest["traceback"] = traceback.format_exc()
        manifest["wall_seconds"] = round(time.time() - started, 3)
        atomic_json(manifest_path, manifest)
        raise


def preflight_command(args: argparse.Namespace) -> None:
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite preflight report: {args.report}")
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_d1_widened_preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(Path(__file__)),
        "passed": False,
    }
    try:
        report["startup_gate"] = startup_gate(hash_shards=not args.skip_shard_hashes)
        report["passed"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
    report["wall_seconds"] = round(time.time() - started, 3)
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def self_test() -> None:
    original = torch.tensor([0.0, -0.0, 1.0, -2.5, 123.0], dtype=torch.bfloat16)
    payload = bf16_bits(original)
    restored = torch.from_numpy(payload.copy()).view(torch.bfloat16)
    if not torch.equal(original.view(torch.uint16), restored.view(torch.uint16)):
        raise AssertionError("BF16 bit round-trip failed")
    if expected_artifact_count() != 2324:
        raise AssertionError(f"unexpected production artifact count: {expected_artifact_count()}")
    if STATE_PAYLOAD_BYTES != 53_955_526_656:
        raise AssertionError(f"unexpected D4 payload size: {STATE_PAYLOAD_BYTES}")
    meter = Meter(2, 4, "cpu")
    with_bias = torch.tensor([[[1.0]], [[1.0]]], dtype=torch.float32)
    meter.add_chunk(
        with_bias,
        with_bias,
        torch.zeros_like(with_bias),
        torch.zeros_like(with_bias),
        0,
        False,
        4,
    )
    arrays = meter.to_npz()
    if arrays["count"].tolist() != [1.0, 0.0, 0.0, 0.0]:
        raise AssertionError("meter geometry self-test failed")
    with tempfile.TemporaryDirectory(prefix="inkling-d1-runner-") as temporary:
        path = Path(temporary) / "record.json"
        atomic_json(path, {"ok": True})
        if json.loads(path.read_text(encoding="utf-8")) != {"ok": True}:
            raise AssertionError("atomic JSON self-test failed")
    print("widened capture runner self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--out", type=Path, default=DEFAULT_OUT)
    capture.add_argument("--qchunk", type=int, default=QCHUNK)
    capture.add_argument("--nll-token-chunk", type=int, default=256)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--report", type=Path, default=DEFAULT_PREFLIGHT)
    preflight.add_argument("--skip-shard-hashes", action="store_true")
    subparsers.add_parser("check-inputs")
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "capture":
        capture_command(args)
    elif args.command == "preflight":
        preflight_command(args)
    elif args.command == "check-inputs":
        print(json.dumps(input_gate(), indent=2, sort_keys=True))
    else:
        self_test()


if __name__ == "__main__":
    main()
