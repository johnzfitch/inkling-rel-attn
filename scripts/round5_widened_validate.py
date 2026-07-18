"""Independent validation for the D1 widened corrected capture.

This module intentionally imports neither the paired builder nor the capture
runner.  Before GPU execution, ``check-inputs`` independently rederives the
paired boundaries, doses, P-f classes, controls, and pairing gate.  After the
pass, ``validate`` authenticates historical public sources through Git blobs,
rehashes checkpoint shards and every output, and validates all array-level
contracts before any outcome analysis or LF5 replay parity is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tokenizers
import torch
import transformers
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
PAIRED = ROOT / "corpus_v2"
NVFP4 = ROOT / "nvfp4"
PUBLIC_FREEZE = ROOT / "analysis" / "round5" / "pe" / "corpus_freeze.json"
DEFAULT_CAPTURE = ROOT / "dumps" / "round5" / "widened_corrected_capture"
DEFAULT_REPORT = ROOT / "analysis" / "round5" / "widened_capture" / "capture_validation.json"

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
LAYERS = list(range(66))
GLOBALS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}
SEQ = 8192
BIN_SIZE = 256
QCHUNK = 512
MASSIVE_THRESHOLD = 30_000.0
EXPECTED_NEEDLE_QUERIES = 24
ATTENTION_DTYPE_BOUNDARY = "BF16 content+bias add, then FP32 softmax"

D1_COMMIT = "51b8c00fe9b632086d0745221578be452f76f60c"
A5_COMMIT = "7bf608d9971997a655a4f9cd46e3bc921ffb74b8"
A6_COMMIT = "061bb04dffef05eae33f9fffc430394faa4052c5"
A8_COMMIT = "93665e2d75a68d4d2d77e2751c316f9a6665f796"
PE_PF_COMMIT = "71f0ad3efff199a83c333340ddd8c8f9a8d7f228"

CRITICAL_PUBLIC_FILES = [
    "scripts/round5_widened_capture.py",
    "scripts/round5_widened_validate.py",
    "scripts/round5_pe_paired_build.py",
    "scripts/tier2_run.py",
    "scripts/tier2_stream.py",
    "scripts/tier2_nvfp4.py",
    "registrations/ROUND5_CAPTURE_SCOPE_D1.md",
    "registrations/ROUND5_AMENDMENT_A5.md",
    "registrations/ROUND5_AMENDMENT_A6.md",
    "registrations/ROUND5_AMENDMENT_A8_VALIDATION.md",
    "registrations/ROUND5_CAPTURE_AMENDMENT.md",
    "registrations/ROUND5_APERTURE_CONTEXT_DOSE_PREREG.md",
    "registrations/ROUND5_APERTURE_ANCHOR_PREREG.md",
    "analysis/round5/pe/corpus_freeze.json",
]


def sha256_file(path: Path, *, chunk: int = 5 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_sha256_file(path: Path) -> tuple[str | None, str | None]:
    """Hash current-checkout state without turning A8 drift into a failure."""
    try:
        return sha256_file(path), None
    except Exception as exc:
        return None, repr(exc)


def git_blob_sha256(commit: str, relative_path: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=ROOT, stderr=subprocess.STDOUT
    )
    return hashlib.sha256(payload).hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def package_record(module: Any) -> dict[str, str]:
    return {
        "version": str(module.__version__),
        "module_path": str(Path(module.__file__).resolve()),
    }


def expected_artifact_count() -> int:
    return len(TEXTS) * len(LAYERS) + 3 * len(V1_TEXTS) * len(LAYERS) + len(LAYERS) + len(V1_TEXTS)


def expected_paths() -> dict[str, str]:
    result: dict[str, str] = {}
    for layer in LAYERS:
        for text in V1_TEXTS:
            result[f"replay/rvec_L{layer:02d}_{text}.npy"] = "rvec"
            result[f"normalized/attn_in_L{layer:02d}_{text}.npy"] = "normalized_attention_input"
            result[f"meters/layer{layer:02d}_{text}_s{SEQ}.npz"] = "tier2_distance_meter"
            result[f"massive/massive_L{layer:02d}_{text}.npz"] = "massive_activation_census"
        for text in PAIRED_TEXTS:
            result[f"paired/rvec_L{layer:02d}_{text}.npy"] = "rvec"
        result[f"replay/needlerows_L{layer:02d}.npz"] = "lf5_needle_rows"
    for text in V1_TEXTS:
        result[f"nll/nll_{text}.npz"] = "next_token_nll"
    return result


def safe_artifact(capture: Path, relative: str) -> Path:
    candidate = (capture / relative).resolve()
    candidate.relative_to(capture.resolve())
    return candidate


def seed_for(label: str, ids_hash: str) -> int:
    payload = f"{PE_PF_COMMIT}|{PE_PF_COMMIT}|{ids_hash}|{label}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def stratified_random_mask(
    reference: list[int], excluded: set[int], *, seed: int
) -> list[int]:
    reference_array = np.asarray(sorted(set(reference)), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    selected: list[int] = []
    for start in range(0, SEQ, BIN_SIZE):
        stop = min(start + BIN_SIZE, SEQ)
        count = int(np.count_nonzero((reference_array >= start) & (reference_array < stop)))
        if not count:
            continue
        candidates = np.asarray(
            [position for position in range(start, stop) if position not in excluded],
            dtype=np.int64,
        )
        if len(candidates) < count:
            raise RuntimeError("insufficient random-control candidates")
        selected.extend(int(value) for value in rng.choice(candidates, size=count, replace=False))
    selected.sort()
    return selected


def independent_boundaries(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    messages = {int(item["start_token"]): item for item in sidecar["messages"]}
    conversation_starts = [int(value) for value in sidecar["conversation_start_tokens"]]
    higher = set(conversation_starts[1:])
    result: list[dict[str, Any]] = []
    for boundary_value in sidecar["message_start_tokens"]:
        boundary = int(boundary_value)
        if boundary == 0:
            continue
        if boundary in higher:
            opener = max(value for value in conversation_starts if value < boundary)
            kind = "higher_scope"
        else:
            opener = max(value for value in conversation_starts if value <= boundary)
            kind = "ordinary_message"
        result.append(
            {
                "token": boundary,
                "pair_id": messages[boundary]["pair_id"],
                "boundary_type": kind,
                "segment_open_token": opener,
                "retired_context_dose": boundary - opener,
            }
        )
    return result


def independent_classes(
    name: str,
    ids: np.ndarray,
    text: str,
    sidecar: dict[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    fragments = tokenizer.decode_batch(
        [[int(token_id)] for token_id in ids], skip_special_tokens=False
    )
    offsets = [(int(a), int(b)) for a, b in sidecar["token_offsets"]]
    starts = [int(value) for value in sidecar["message_start_tokens"]]
    start_set = set(starts)
    current = 0
    message_start_for_token: list[int] = []
    for position in range(SEQ):
        if position in start_set:
            current = position
        message_start_for_token.append(current)
    url_spans = [(match.start(), match.end()) for match in re.finditer(r"https?://\S+", text)]
    urls = [
        position
        for position, (start, stop) in enumerate(offsets)
        if stop > start and any(start >= left and stop <= right for left, right in url_spans)
    ]
    gratitude_words = {"thank", "thanks", "thx", "ty"}
    gratitude: list[int] = []
    proper: list[int] = []
    for position, fragment in enumerate(fragments):
        folded = fragment.strip().casefold()
        if folded in gratitude_words or folded.startswith("thank"):
            gratitude.append(position)
        previous = fragments[position - 1] if position else ""
        after_terminal = bool(
            previous.endswith("\n") or re.search(r"[.!?][\"')\]]*\s*$", previous)
        )
        message_offset = position - message_start_for_token[position]
        if (
            re.fullmatch(r" ?[A-Z][a-z]{2,}", fragment)
            and message_offset not in (0, 1, 2)
            and not after_terminal
        ):
            proper.append(position)
    colons = [
        start + 1 for start in starts if start + 1 < SEQ and fragments[start + 1].strip() == ":"
    ]
    boundaries = independent_boundaries(sidecar)
    ordinary = [item["token"] for item in boundaries if item["boundary_type"] == "ordinary_message"]
    higher = [item["token"] for item in boundaries if item["boundary_type"] == "higher_scope"]
    classes = {
        "url_tokens": sorted(set(urls)),
        "proper_noun_proxy": sorted(set(proper)),
        "gratitude_tokens": sorted(set(gratitude)),
        "label_colons": sorted(set(colons)),
        "ordinary_message_starts": ordinary,
        "higher_scope_starts": higher,
    }
    excluded = set().union(*(set(values) for values in classes.values()))
    ids_hash = hashlib.sha256(ids.tobytes()).hexdigest()
    seeds = {key: seed_for(f"{name}|{key}", ids_hash) for key in classes}
    masks = {
        key: stratified_random_mask(values, excluded, seed=seeds[key])
        for key, values in classes.items()
    }
    return {
        "boundaries": boundaries,
        "classes": classes,
        "counts": {key: len(values) for key, values in classes.items()},
        "voided_below_8": {
            key: len(classes[key]) < 8
            for key in ("url_tokens", "proper_noun_proxy", "gratitude_tokens", "label_colons")
        },
        "random_seeds": seeds,
        "random_masks": masks,
    }


def validate_paired_inputs() -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    manifest_path = PAIRED / "pe_manifest.json"
    classes_path = PAIRED / "pe_classes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(classes_path.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC_FREEZE.read_text(encoding="utf-8"))
    if (
        manifest.get("kind") != "round5_pe_paired_private_corpus"
        or manifest.get("complete") is not True
        or manifest.get("private") is not True
        or manifest.get("seq") != SEQ
    ):
        errors.append("invalid private paired manifest contract")
    if public.get("kind") != "round5_pe_pf_public_corpus_freeze":
        errors.append("invalid public paired freeze kind")
    if public.get("outcome_data_opened") is not False:
        errors.append("public paired freeze is not marked pre-outcome")
    inventory = public.get("source_inventory", {})
    if (
        inventory.get("nonempty_channels") != 19
        or inventory.get("human_chars") != 1_008_273
        or inventory.get("selected_human_chars") != 807_211
        or inventory.get("selected_human_messages") != 11_599
        or inventory.get("selected_distinct_users") != 15
    ):
        errors.append("public paired source inventory differs from preregistration")
    if public.get("private_manifest_sha256") != sha256_file(manifest_path):
        errors.append("public freeze does not bind private manifest")
    if public.get("private_classes_sha256") != sha256_file(classes_path):
        errors.append("public freeze does not bind private classes")
    if manifest.get("classes", {}).get("sha256") != sha256_file(classes_path):
        errors.append("private manifest does not bind classes")
    tokenizer_path = CORPUS / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if manifest.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        errors.append("paired tokenizer hash mismatch")

    sidecars: dict[str, dict[str, Any]] = {}
    ids_by_arm: dict[str, np.ndarray] = {}
    independent: dict[str, Any] = {}
    pair_positions: dict[str, dict[str, int]] = {}
    for name in PAIRED_TEXTS:
        ids_path = PAIRED / f"{name}.ids.npy"
        text_path = PAIRED / f"{name}.txt"
        sidecar_path = PAIRED / f"{name}.sidecar.json"
        records = manifest.get("arms", {}).get(name, {})
        for field, path in (("ids", ids_path), ("text", text_path), ("sidecar", sidecar_path)):
            if records.get(field, {}).get("sha256") != sha256_file(path):
                errors.append(f"paired {name} {field} hash mismatch")
        ids = np.load(ids_path, allow_pickle=False)
        text = text_path.read_text(encoding="utf-8")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        ids_by_arm[name] = ids
        sidecars[name] = sidecar
        if ids.shape != (SEQ,) or ids.dtype != np.int32:
            errors.append(f"invalid paired IDs: {name}")
        token_messages = [int(value) for value in sidecar.get("token_message_index", [])]
        token_conversations = [int(value) for value in sidecar.get("token_conversation_index", [])]
        starts = [0] + [
            position
            for position in range(1, len(token_messages))
            if token_messages[position] != token_messages[position - 1]
        ]
        conversation_starts = [0] + [
            position
            for position in range(1, len(token_conversations))
            if token_conversations[position] != token_conversations[position - 1]
        ]
        if (
            len(token_messages) != SEQ
            or len(token_conversations) != SEQ
            or len(sidecar.get("token_offsets", [])) != SEQ
            or starts != sidecar.get("message_start_tokens")
            or conversation_starts != sidecar.get("conversation_start_tokens")
        ):
            errors.append(f"paired transition sidecar mismatch: {name}")
        pair_positions[name] = {
            item["pair_id"]: int(item["start_token"]) for item in sidecar["messages"]
        }
        independent[name] = independent_classes(name, ids, text, sidecar, tokenizer)
        published_arm = frozen.get("arms", {}).get(name, {})
        for field in (
            "boundaries",
            "classes",
            "counts",
            "voided_below_8",
            "random_seeds",
            "random_masks",
        ):
            if independent[name][field] != published_arm.get(field):
                errors.append(f"paired independent {field} mismatch: {name}")
        public_arm = public.get("arms", {}).get(name, {})
        if public_arm.get("class_counts") != independent[name]["counts"]:
            errors.append(f"public paired class counts mismatch: {name}")
        if public_arm.get("voided_below_8") != independent[name]["voided_below_8"]:
            errors.append(f"public paired void disposition mismatch: {name}")

    union = set(pair_positions[PAIRED_TEXTS[0]]) | set(pair_positions[PAIRED_TEXTS[1]])
    intersection = set(pair_positions[PAIRED_TEXTS[0]]) & set(pair_positions[PAIRED_TEXTS[1]])
    preservation = len(intersection) / max(len(union), 1)
    expected_pairing = {
        "single_message_starts": len(pair_positions[PAIRED_TEXTS[0]]),
        "multi_message_starts": len(pair_positions[PAIRED_TEXTS[1]]),
        "intersection": len(intersection),
        "union": len(union),
        "intersection_over_union": preservation,
        "minimum": 0.80,
        "passed": preservation >= 0.80,
    }
    if frozen.get("pair_preservation") != expected_pairing:
        errors.append("private paired-preservation record mismatch")
    if public.get("pair_preservation") != expected_pairing:
        errors.append("public paired-preservation record mismatch")
    deltas = [
        pair_positions[PAIRED_TEXTS[1]][pair_id]
        - pair_positions[PAIRED_TEXTS[0]][pair_id]
        for pair_id in sorted(intersection)
    ]
    alignment = {
        "paired_count": len(deltas),
        "exact_position_matches": int(np.count_nonzero(np.asarray(deltas) == 0)),
        "mismatched_positions": int(np.count_nonzero(np.asarray(deltas) != 0)),
        "median_multi_minus_single": float(np.median(deltas)),
        "min_multi_minus_single": int(min(deltas)),
        "max_multi_minus_single": int(max(deltas)),
    }
    if frozen.get("pair_position_alignment") != alignment:
        errors.append("private paired-position alignment record mismatch")
    if public.get("pair_position_alignment") != alignment:
        errors.append("public paired-position alignment record mismatch")
    ordinary = {
        name: {
            item["pair_id"]
            for item in independent[name]["boundaries"]
            if item["boundary_type"] == "ordinary_message"
        }
        for name in PAIRED_TEXTS
    }
    paired_ordinary = ordinary[PAIRED_TEXTS[0]] & ordinary[PAIRED_TEXTS[1]]
    if public.get("paired_ordinary_message_count") != len(paired_ordinary):
        errors.append("public paired-ordinary count mismatch")
    summary = {
        "private_manifest_sha256": sha256_file(manifest_path),
        "private_classes_sha256": sha256_file(classes_path),
        "public_freeze_sha256": sha256_file(PUBLIC_FREEZE),
        "pair_preservation": expected_pairing,
        "pair_position_alignment": alignment,
        "paired_ordinary_messages": len(paired_ordinary),
        "class_counts": {name: independent[name]["counts"] for name in PAIRED_TEXTS},
        "all_pf_classes_nonvoid": all(
            not value
            for name in PAIRED_TEXTS
            for value in independent[name]["voided_below_8"].values()
        ),
    }
    return errors, summary


def needle_queries() -> list[int]:
    sidecar = json.loads((CORPUS / "05_needles.sidecar.json").read_text(encoding="utf-8"))
    return sorted(
        {
            int(entity["token_positions"][1])
            for entity in sidecar["entities"]
            if len(entity.get("token_positions", [])) >= 2
            and int(entity["token_positions"][1]) < SEQ
        }
    )


def validate_public_provenance(
    manifest: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    capture_head = str(manifest.get("git_head", ""))
    current_drift: list[str] = []
    historical: dict[str, Any] = {}
    gate_files = (
        manifest.get("startup_gate", {})
        .get("critical_git_blobs", {})
        .get("files", {})
    )
    for relative in CRITICAL_PUBLIC_FILES:
        try:
            historical_hash = git_blob_sha256(capture_head, relative)
        except Exception as exc:
            errors.append(f"capture Git-blob lookup failed for {relative}: {exc}")
            continue
        captured_hash = gate_files.get(relative, {}).get("sha256")
        if captured_hash != historical_hash:
            errors.append(f"historical public-source mismatch: {relative}")
        # A8 authenticates the capture against its historical Git tree. A
        # later checkout may move or remove a public path; that is drift to
        # report, not grounds to reject otherwise authentic bytes.
        current_hash, current_error = diagnostic_sha256_file(ROOT / relative)
        if current_hash != historical_hash:
            current_drift.append(relative)
        record = {
            "captured_sha256": captured_hash,
            "git_blob_sha256": historical_hash,
            "current_sha256": current_hash,
        }
        if current_error is not None:
            record["current_error"] = current_error
        historical[relative] = record
    return {
        "capture_git_head": capture_head,
        "capture_git_tree_verified": not any(
            error.startswith("historical public-source mismatch")
            or error.startswith("capture Git-blob lookup failed")
            for error in errors
        ),
        "current_checkout_drift": current_drift,
        "files": historical,
    }


def validate_checkpoint(
    manifest: dict[str, Any], errors: list[str], *, rehash_shards: bool
) -> dict[str, Any]:
    index_path = NVFP4 / "model.safetensors.index.json"
    if manifest.get("checkpoint_index_sha256") != sha256_file(index_path):
        errors.append("checkpoint index hash mismatch")
    if manifest.get("config_sha256") != sha256_file(NVFP4 / "config.json"):
        errors.append("checkpoint config hash mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = sorted(set(index["weight_map"].values()))
    trunk = [name for name in indexed if name.startswith("model-")]
    nontrunk = [name for name in indexed if not name.startswith("model-")]
    on_disk = sorted(path.name for path in NVFP4.glob("model-*.safetensors"))
    records = manifest.get("checkpoint_shards", {})
    if trunk != on_disk or nontrunk != ["mtp.safetensors"] or set(records) != set(trunk):
        errors.append("checkpoint shard inventory mismatch")
    elif rehash_shards:
        for ordinal, name in enumerate(trunk, start=1):
            path = NVFP4 / name
            print(f"independently hashing shard {ordinal:02d}/{len(trunk)}: {name}", flush=True)
            record = records[name]
            if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(
                path, chunk=3 << 20
            ):
                errors.append(f"checkpoint shard mismatch: {name}")
    return {"trunk_shards": len(trunk), "nontrunk": nontrunk, "rehash_shards": rehash_shards}


def validate_runtime(manifest: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    packages = {
        "numpy": package_record(np),
        "tokenizers": package_record(tokenizers),
        "torch": package_record(torch),
        "transformers": package_record(transformers),
    }
    if manifest.get("packages") != packages:
        errors.append("runtime package versions/module paths differ")
    modeling_path = Path(transformers.models.inkling.modeling_inkling.__file__).resolve()
    modeling = {"path": str(modeling_path), "sha256": sha256_file(modeling_path)}
    if manifest.get("modeling_inkling") != modeling:
        errors.append("stock modeling_inkling source differs")
    return {"packages": packages, "modeling_inkling": modeling}


def validate_inputs_against_capture(
    manifest: dict[str, Any], errors: list[str], paired_summary: dict[str, Any]
) -> dict[str, Any]:
    v1_manifest_path = CORPUS / "manifest.json"
    paired_manifest_path = PAIRED / "pe_manifest.json"
    paired_classes_path = PAIRED / "pe_classes.json"
    expected_manifests = {
        "corpus": sha256_file(v1_manifest_path),
        "corpus_v2/pe_manifest.json": sha256_file(paired_manifest_path),
        "corpus_v2/pe_classes.json": sha256_file(paired_classes_path),
    }
    if manifest.get("input_manifest_sha256") != expected_manifests:
        errors.append("capture input-manifest hashes differ")
    if manifest.get("public_freeze_sha256") != sha256_file(PUBLIC_FREEZE):
        errors.append("capture public-freeze hash differs")
    v1 = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_manifest_path.read_text(encoding="utf-8"))
    ids_by_text: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for text in TEXTS:
        root = CORPUS if text in V1_TEXTS else PAIRED
        path = root / f"{text}.ids.npy"
        hashes[text] = sha256_file(path)
        expected = (
            v1["texts"][text]["ids_sha256"]
            if text in V1_TEXTS
            else paired["arms"][text]["ids"]["sha256"]
        )
        if hashes[text] != expected:
            errors.append(f"current input ID hash mismatch: {text}")
        ids = np.load(path, allow_pickle=False)
        if ids.shape != (SEQ,) or ids.dtype != np.int32:
            errors.append(f"current input ID shape/dtype mismatch: {text}")
        ids_by_text[text] = ids
    if manifest.get("input_ids_sha256") != hashes:
        errors.append("capture input-ID hashes differ")
    if manifest.get("tokenizer_sha256") != sha256_file(CORPUS / "tokenizer.json"):
        errors.append("capture tokenizer hash differs")
    return {"ids": ids_by_text, "hashes": hashes, "paired": paired_summary}


def validate_meter(
    path: Path, layer: int, text: str, errors: list[str]
) -> dict[str, float]:
    is_sliding = layer not in GLOBALS
    dmax = 512 if is_sliding else SEQ
    required = {
        "mass_with",
        "mass_without",
        "bias_sum",
        "content_sum",
        "count",
        "mean_mass_with",
        "mean_mass_without",
        "mean_bias",
        "mean_content",
        "meta",
    }
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            errors.append(f"meter field inventory mismatch: L{layer:02d} {text}")
            return {"with": float("inf"), "without": float("inf")}
        arrays = {field: data[field] for field in required if field != "meta"}
        try:
            meta = json.loads(str(data["meta"]))
        except Exception:
            errors.append(f"invalid meter metadata: L{layer:02d} {text}")
            meta = {}
    expected_meta = {
        "layer": layer,
        "text": text,
        "is_global": layer in GLOBALS,
        "is_sliding": is_sliding,
        "rel_extent": 512 if is_sliding else 1024,
        "seq": SEQ,
        "qchunk": QCHUNK,
        "n_heads": 64,
        "a6_corrected": True,
    }
    if meta != expected_meta:
        errors.append(f"meter metadata mismatch: L{layer:02d} {text}")
    expected_shape = (64, dmax)
    for field in required - {"meta", "count"}:
        if arrays[field].shape != expected_shape or arrays[field].dtype != np.float64:
            errors.append(f"meter shape/dtype mismatch {field}: L{layer:02d} {text}")
        if not np.isfinite(arrays[field]).all():
            errors.append(f"non-finite meter {field}: L{layer:02d} {text}")
    count_expected = SEQ - np.arange(dmax, dtype=np.float64)
    if arrays["count"].dtype != np.float64 or not np.array_equal(arrays["count"], count_expected):
        errors.append(f"meter count mismatch: L{layer:02d} {text}")
    denominator = np.maximum(arrays["count"], 1.0)
    for summed, mean in (
        ("mass_with", "mean_mass_with"),
        ("mass_without", "mean_mass_without"),
        ("bias_sum", "mean_bias"),
        ("content_sum", "mean_content"),
    ):
        if not np.array_equal(arrays[mean], arrays[summed] / denominator):
            errors.append(f"meter derived mean mismatch {mean}: L{layer:02d} {text}")
    with_error = float(np.max(np.abs(arrays["mass_with"].sum(axis=1) - SEQ)))
    without_error = float(np.max(np.abs(arrays["mass_without"].sum(axis=1) - SEQ)))
    if with_error > 0.05 or without_error > 0.05:
        errors.append(f"meter mass conservation failed: L{layer:02d} {text}")
    return {"with": with_error, "without": without_error}


def validate_artifacts(
    capture: Path,
    manifest: dict[str, Any],
    ids_by_text: dict[str, np.ndarray],
    errors: list[str],
) -> dict[str, Any]:
    expected = expected_paths()
    records = manifest.get("artifacts", [])
    by_path = {record.get("path"): record for record in records}
    if len(records) != len(by_path) or set(by_path) != set(expected):
        errors.append("capture artifact path inventory differs")
    queries = needle_queries()
    if len(queries) != EXPECTED_NEEDLE_QUERIES:
        errors.append("needle query count differs")
    meter_errors: list[float] = []
    normalized_nonfinite = 0
    meter_integrity: dict[str, dict[str, dict[str, float]]] = {
        f"L{layer:02d}": {} for layer in LAYERS
    }
    massive_summary: dict[str, dict[str, dict[str, float | int]]] = {
        f"L{layer:02d}": {} for layer in LAYERS
    }
    nll_summary: dict[str, dict[str, float | int | bool]] = {}
    for ordinal, relative in enumerate(sorted(expected), start=1):
        record = by_path.get(relative, {})
        try:
            path = safe_artifact(capture, relative)
        except Exception:
            errors.append(f"unsafe artifact path: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        if record.get("kind") != expected[relative]:
            errors.append(f"artifact kind mismatch: {relative}")
        if record.get("bytes") != path.stat().st_size:
            errors.append(f"artifact byte count mismatch: {relative}")
        if record.get("sha256") != sha256_file(path):
            errors.append(f"artifact SHA-256 mismatch: {relative}")

        if expected[relative] == "rvec":
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (SEQ, 64, 16) or values.dtype != np.float16:
                errors.append(f"invalid r-vector shape/dtype: {relative}")
            elif not np.isfinite(values).all():
                errors.append(f"non-finite r-vector: {relative}")
        elif expected[relative] == "normalized_attention_input":
            bits = np.load(path, mmap_mode="r", allow_pickle=False)
            if bits.shape != (SEQ, 6144) or bits.dtype != np.uint16:
                errors.append(f"invalid BF16 payload shape/dtype: {relative}")
            else:
                nonfinite = int(np.count_nonzero((bits & np.uint16(0x7F80)) == np.uint16(0x7F80)))
                normalized_nonfinite += nonfinite
                if nonfinite:
                    errors.append(f"non-finite BF16 normalized input: {relative}")
        elif expected[relative] == "tier2_distance_meter":
            match = re.fullmatch(r"meters/layer(\d{2})_(.+)_s8192\.npz", relative)
            if not match:
                errors.append(f"invalid meter path: {relative}")
            else:
                integrity = validate_meter(path, int(match.group(1)), match.group(2), errors)
                meter_errors.extend(integrity.values())
                meter_integrity[f"L{int(match.group(1)):02d}"][match.group(2)] = {
                    "max_mass_with_error": integrity["with"],
                    "max_mass_without_error": integrity["without"],
                }
        elif expected[relative] == "massive_activation_census":
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"position", "channel", "value"}:
                    errors.append(f"massive census field mismatch: {relative}")
                    continue
                position, channel, value = data["position"], data["channel"], data["value"]
            if (
                position.dtype != np.int32
                or channel.dtype != np.int32
                or value.dtype != np.float32
                or not (len(position) == len(channel) == len(value))
                or np.any(position < 0)
                or np.any(position >= SEQ)
                or np.any(channel < 0)
                or np.any(channel >= 6144)
                or not np.isfinite(value).all()
                or np.any(np.abs(value) <= MASSIVE_THRESHOLD)
            ):
                errors.append(f"invalid massive census: {relative}")
            if len(set(zip(position.tolist(), channel.tolist()))) != len(position):
                errors.append(f"duplicate massive coordinates: {relative}")
            match = re.fullmatch(r"massive/massive_L(\d{2})_(.+)\.npz", relative)
            if match:
                massive_summary[f"L{int(match.group(1)):02d}"][match.group(2)] = {
                    "count": int(len(value)),
                    "max_abs": float(np.max(np.abs(value))) if len(value) else 0.0,
                }
        elif expected[relative] == "lf5_needle_rows":
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"qpos", "rows"}:
                    errors.append(f"needle-row field mismatch: {relative}")
                    continue
                qpos, rows = data["qpos"], data["rows"]
            if (
                qpos.dtype != np.int64
                or qpos.tolist() != queries
                or rows.dtype != np.float16
                or rows.shape != (EXPECTED_NEEDLE_QUERIES, 64, SEQ)
                or not np.isfinite(rows).all()
            ):
                errors.append(f"invalid LF5 needle rows: {relative}")
            elif float(np.max(np.abs(rows.astype(np.float32).sum(axis=-1) - 1.0))) > 0.05:
                errors.append(f"LF5 needle-row mass conservation failed: {relative}")
        elif expected[relative] == "next_token_nll":
            text = relative.removeprefix("nll/nll_").removesuffix(".npz")
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"target_position", "target_id", "nll"}:
                    errors.append(f"NLL field mismatch: {relative}")
                    continue
                position, target, nll = data["target_position"], data["target_id"], data["nll"]
            if (
                position.dtype != np.int32
                or not np.array_equal(position, np.arange(1, SEQ, dtype=np.int32))
                or target.dtype != np.int32
                or not np.array_equal(target, ids_by_text[text][1:])
                or nll.dtype != np.float32
                or nll.shape != (SEQ - 1,)
                or not np.isfinite(nll).all()
            ):
                errors.append(f"invalid NLL artifact: {relative}")
            nll_summary[text] = {
                "count": int(len(nll)),
                "mean": float(np.mean(nll, dtype=np.float64)),
                "median": float(np.median(nll)),
                "min": float(np.min(nll)),
                "max": float(np.max(nll)),
                "finite": bool(np.isfinite(nll).all()),
            }
        if ordinal % 64 == 0:
            print(f"validated {ordinal}/{len(expected)} artifacts", flush=True)
    if manifest.get("meter_integrity") != meter_integrity:
        errors.append("manifest meter-integrity summary differs from artifacts")
    if manifest.get("massive_summary") != massive_summary:
        errors.append("manifest massive-coordinate summary differs from artifacts")
    if manifest.get("nll_summary") != nll_summary:
        errors.append("manifest NLL summary differs from artifacts")
    if set(nll_summary) == set(V1_TEXTS):
        uniform = math.log(int(manifest.get("model", {}).get("unpadded_vocab_size", 0)))
        rebuilt_nll_gate = {
            "uniform_nll": uniform,
            "all_finite": all(bool(item["finite"]) for item in nll_summary.values()),
            "prose_positive_below_uniform": 0.0 < float(nll_summary["01_prose_en"]["mean"]) < uniform,
            "random_above_prose": float(nll_summary["06_random"]["mean"])
            > float(nll_summary["01_prose_en"]["mean"]),
        }
        rebuilt_nll_gate["passed"] = bool(
            rebuilt_nll_gate["all_finite"]
            and rebuilt_nll_gate["prose_positive_below_uniform"]
            and rebuilt_nll_gate["random_above_prose"]
        )
        if manifest.get("nll_gate") != rebuilt_nll_gate:
            errors.append("manifest NLL integrity gate differs from artifacts")
    else:
        rebuilt_nll_gate = {"passed": False, "reason": "incomplete NLL inventory"}
    return {
        "artifact_count": len(expected),
        "normalized_nonfinite_words": normalized_nonfinite,
        "maximum_meter_mass_error": max(meter_errors, default=0.0),
        "nll_gate": rebuilt_nll_gate,
    }


def validate_capture_command(args: argparse.Namespace) -> None:
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {args.report}")
    capture = args.capture.resolve()
    manifest_path = capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("kind") != "round5_d1_widened_a6_capture" or manifest.get("schema_version") != 1:
        errors.append("wrong widened capture kind/schema")
    if manifest.get("complete") is not True or manifest.get("production_capture") is not True:
        errors.append("capture is not a complete production output")
    if (
        manifest.get("artifact_count") != expected_artifact_count()
        or manifest.get("expected_artifact_count") != expected_artifact_count()
        or len(manifest.get("artifacts", [])) != expected_artifact_count()
    ):
        errors.append("wrong widened artifact count")
    if (
        manifest.get("texts") != TEXTS
        or manifest.get("v1_texts") != V1_TEXTS
        or manifest.get("paired_texts") != PAIRED_TEXTS
        or manifest.get("layers") != LAYERS
        or manifest.get("seq") != SEQ
        or manifest.get("qchunk") != QCHUNK
    ):
        errors.append("wrong registered capture scope")
    registrations = manifest.get("registration_commits", {})
    expected_registrations = {
        "D1": D1_COMMIT,
        "A5": A5_COMMIT,
        "A6": A6_COMMIT,
        "A8": A8_COMMIT,
        "P-e/P-f": PE_PF_COMMIT,
    }
    if registrations != expected_registrations:
        errors.append("wrong registration commit boundaries")
    if manifest.get("attention_dtype_boundary") != ATTENTION_DTYPE_BOUNDARY:
        errors.append("wrong attention dtype boundary")
    features = manifest.get("capture_features", {})
    if features.get("residual_hidden_states") is not False or features.get("D4_satisfied") is not False:
        errors.append("D4 scope was silently broadened")
    startup = manifest.get("startup_gate", {})
    parity = manifest.get("stock_attention_parity", {})
    if startup.get("passed") is not True or parity.get("passed") is not True:
        errors.append("A8 startup/parity gate did not pass")
    if set(parity.get("cases", {})) != {"global", "sliding"} or any(
        case.get("bitwise_equal") is not True or case.get("max_output_delta") != 0.0
        for case in parity.get("cases", {}).values()
    ):
        errors.append("stock-parity cases are invalid")
    handoff = manifest.get("lf5_handoff", {})
    if (
        handoff.get("backend") != "replay"
        or handoff.get("input_root") != "."
        or handoff.get("capture_root") != "replay"
        or handoff.get("needle_rows_recaptured") is not True
        or handoff.get("normalized_inputs_recaptured") is not True
        or handoff.get("parity_required_after_independent_capture_validation") is not True
    ):
        errors.append("LF5 production handoff is incomplete")

    paired_errors, paired_summary = validate_paired_inputs()
    errors.extend(paired_errors)
    provenance = validate_public_provenance(manifest, errors)
    checkpoint = validate_checkpoint(manifest, errors, rehash_shards=not args.skip_shard_rehash)
    runtime = validate_runtime(manifest, errors)
    inputs = validate_inputs_against_capture(manifest, errors, paired_summary)
    artifacts = validate_artifacts(capture, manifest, inputs["ids"], errors)

    report = {
        "schema_version": 1,
        "kind": "round5_d1_widened_independent_capture_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(Path(__file__)),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "capture_git_head": manifest.get("git_head"),
        "historical_source_validation": provenance,
        "checkpoint_validation": checkpoint,
        "runtime_validation": runtime,
        "input_validation": {
            "hashes": inputs["hashes"],
            "paired": paired_summary,
        },
        "artifact_validation": artifacts,
        "lf5_replay_parity_ready": not errors,
        "D4_satisfied": False,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def check_inputs_command() -> None:
    errors, summary = validate_paired_inputs()
    v1 = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    if v1.get("seq") != SEQ:
        errors.append("invalid v1 sequence length")
    for text in V1_TEXTS:
        path = CORPUS / f"{text}.ids.npy"
        if v1.get("texts", {}).get(text, {}).get("ids_sha256") != sha256_file(path):
            errors.append(f"v1 ID hash mismatch: {text}")
        values = np.load(path, allow_pickle=False)
        if values.shape != (SEQ,) or values.dtype != np.int32:
            errors.append(f"v1 ID shape/dtype mismatch: {text}")
    result = {
        "kind": "round5_d1_widened_independent_input_check",
        "paired": summary,
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def self_test() -> None:
    finite = np.asarray([0x0000, 0x3F80, 0x7F7F], dtype=np.uint16)
    nonfinite = np.asarray([0x7F80, 0x7FC0, 0xFF80], dtype=np.uint16)
    if np.any((finite & np.uint16(0x7F80)) == np.uint16(0x7F80)):
        raise AssertionError("BF16 finite-bit self-test failed")
    if not np.all((nonfinite & np.uint16(0x7F80)) == np.uint16(0x7F80)):
        raise AssertionError("BF16 nonfinite-bit self-test failed")
    if expected_artifact_count() != 1788 or len(expected_paths()) != 1788:
        raise AssertionError("artifact inventory self-test failed")
    reference = [1, 3, 260]
    first = stratified_random_mask(reference, set(reference), seed=9)
    second = stratified_random_mask(reference, set(reference), seed=9)
    if first != second or set(first) & set(reference):
        raise AssertionError("random-control self-test failed")
    with tempfile.TemporaryDirectory(prefix="inkling-d1-validator-") as temporary:
        root = Path(temporary)
        (root / "ok").write_text("ok", encoding="utf-8")
        missing_hash, missing_error = diagnostic_sha256_file(root / "missing")
        if missing_hash is not None or missing_error is None:
            raise AssertionError("missing current-source drift was not diagnostic")
        if safe_artifact(root, "ok") != (root / "ok").resolve():
            raise AssertionError("safe-artifact self-test failed")
        try:
            safe_artifact(root, "../escape")
        except ValueError:
            pass
        else:
            raise AssertionError("path traversal was not rejected")
    print("widened capture validator self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    validate.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    validate.add_argument("--skip-shard-rehash", action="store_true")
    subparsers.add_parser("check-inputs")
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "validate":
        validate_capture_command(args)
    elif args.command == "check-inputs":
        check_inputs_command()
    else:
        self_test()


if __name__ == "__main__":
    main()
