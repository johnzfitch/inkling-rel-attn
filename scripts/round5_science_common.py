"""Shared, outcome-neutral utilities for certified Round 5 dump science."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "dumps" / "round5" / "widened_corrected_capture"
VALIDATION = ROOT / "analysis" / "round5" / "widened_capture" / "capture_validation.json"
MANIFEST = CAPTURE / "manifest.json"
MANIFEST_SHA256 = "2fc2ec9dd4d6b684a473847f6fbd4541d4326b49d7d6456b463ee5722399a75f"
PLAN_COMMIT = "88dd002"
PLAN_PATH = ROOT / "registrations" / "ROUND5_DUMP_SCIENCE_EXECUTION_PLAN.md"
TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
GLOBAL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(name: str) -> int:
    payload = f"{MANIFEST_SHA256}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    """Decode a uint16 BF16 bit payload exactly into float32."""
    values = np.asarray(words)
    if values.dtype != np.uint16:
        raise TypeError(f"expected uint16 BF16 words, got {values.dtype}")
    expanded = np.asarray(values, dtype=np.uint32) << np.uint32(16)
    return expanded.view(np.float32)


def require_certified_capture() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError("certified capture manifest hash changed")
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    artifacts = validation.get("artifact_validation", {})
    if (
        validation.get("kind") != "round5_d1_widened_independent_capture_validation"
        or not validation.get("passed")
        or validation.get("errors")
        or not validation.get("D4_satisfied")
        or validation.get("capture_manifest_sha256") != MANIFEST_SHA256
        or artifacts.get("artifact_count") != 2324
        or artifacts.get("state_artifact_count") != 536
        or artifacts.get("state_nonfinite_words") != 0
    ):
        raise RuntimeError("widened capture validation is missing, failed, or stale")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        not manifest.get("complete")
        or manifest.get("artifact_count") != 2324
        or len(manifest.get("artifacts", [])) != 2324
    ):
        raise RuntimeError("capture manifest is incomplete")
    return validation, manifest


def artifact_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {record["path"]: record for record in manifest["artifacts"]}
    if len(records) != len(manifest["artifacts"]):
        raise RuntimeError("duplicate artifact paths in capture manifest")
    return records


def provenance(script_path: Path) -> dict[str, Any]:
    return {
        "capture_manifest_sha256": MANIFEST_SHA256,
        "capture_validation_sha256": sha256_file(VALIDATION),
        "execution_plan_commit": PLAN_COMMIT,
        "execution_plan_sha256": sha256_file(PLAN_PATH),
        "script": script_path.relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(script_path),
    }


def median_absolute_deviation(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    return np.median(np.abs(values - median), axis=axis)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    flat = p.ravel()
    order = np.argsort(flat, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        (flat.size - np.arange(flat.size)) * flat[order]
    )
    adjusted = np.empty_like(flat)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.reshape(p.shape)


def refuse_existing(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing outcome artifacts: " + ", ".join(existing))


def self_test_common() -> None:
    source = np.array([0.0, -0.0, 1.0, -2.5, 123.0], dtype=np.float32)
    # Round to BF16 by retaining the high word, then test the exact decoder.
    words = source.view(np.uint32).astype(np.uint32)
    words = (words >> np.uint32(16)).astype(np.uint16)
    decoded = bf16_words_to_float32(words)
    expected = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.array_equal(decoded.view(np.uint32), expected.view(np.uint32)):
        raise AssertionError("BF16 decoder failed")
    p = np.array([0.01, 0.04, 0.03])
    if not np.allclose(holm_adjust(p), [0.03, 0.06, 0.06]):
        raise AssertionError("Holm adjustment failed")

