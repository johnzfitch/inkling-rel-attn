"""Build the outcome-blind frozen inputs for the seven Round-5 follow-ups.

The builder reads only registered corpora, certified baseline r-vectors,
weights, sidecars, and the preregistration. It refuses overwrite; ``check``
recomputes every array and hash in memory and compares them to the sealed file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from corpus_v2_freeze_classes import PRONOUNS, normalized_word


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "registrations" / "ROUND5_FOLLOWUP7_PREREG.md"
CAPTURE = ROOT / "dumps" / "round5" / "widened_corrected_capture"
CORPUS = ROOT / "corpus"
CORPUS_V2 = ROOT / "corpus_v2"
DEFAULT_ROOT = ROOT / "analysis" / "round5" / "followup7"
DEFAULT_NPZ = DEFAULT_ROOT / "frozen_inputs.npz"
DEFAULT_MANIFEST = DEFAULT_ROOT / "frozen_inputs_manifest.json"
REGISTERED_COMMIT = "eec3999"
TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
SEQ = 8192
HEADS = 64
RPERHEAD = 16
RFLAT = HEADS * RPERHEAD
BLOCK = 64
BLOCK_START = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def seed(registration_sha: str, label: str) -> int:
    digest = hashlib.sha256(f"{registration_sha}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def verified_rvec(layer: int, text: str, sources: dict[str, str]) -> np.ndarray:
    path = CAPTURE / "replay" / f"rvec_L{layer:02d}_{text}.npy"
    sources[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    values = np.load(path, allow_pickle=False)
    if values.shape != (SEQ, HEADS, RPERHEAD) or values.dtype != np.float16:
        raise RuntimeError(f"invalid certified r-vector: {path}, {values.shape}, {values.dtype}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"nonfinite certified r-vector: {path}")
    return values.astype(np.float32).reshape(SEQ, RFLAT)


def clock_direction(values: np.ndarray) -> np.ndarray:
    blocks = values[BLOCK_START:].reshape(-1, BLOCK, RFLAT).mean(axis=1, dtype=np.float64)
    if blocks.shape != (127, RFLAT):
        raise RuntimeError(f"invalid clock block geometry: {blocks.shape}")
    midpoint = np.arange(BLOCK_START, SEQ, BLOCK, dtype=np.float64) + (BLOCK - 1) / 2
    x = np.log1p(midpoint)
    xc = x - x.mean()
    slope = (xc @ (blocks - blocks.mean(axis=0))) / float(xc @ xc)
    norm = float(np.linalg.norm(slope))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("degenerate clock direction")
    return (slope / norm).astype(np.float64)


def orthonormal_columns(matrix: np.ndarray, expected_rank: int) -> tuple[np.ndarray, np.ndarray]:
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    if rank != expected_rank:
        raise RuntimeError(f"clock union rank {rank} != {expected_rank}; amendment required")
    basis = u[:, :expected_rank]
    # SVD signs are deterministic only after an explicit convention.
    for column in range(expected_rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    error = float(np.max(np.abs(basis.T @ basis - np.eye(expected_rank))))
    if error > 1e-12:
        raise RuntimeError(f"non-orthonormal SVD basis: {error}")
    return basis.astype(np.float64), singular.astype(np.float64)


def sham_basis(
    real_basis: np.ndarray, registration_sha: str, label: str, dimension: int
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed(registration_sha, label)))
    candidate = rng.standard_normal((RFLAT, dimension + 8), dtype=np.float64)
    candidate -= real_basis @ (real_basis.T @ candidate)
    q, _ = np.linalg.qr(candidate, mode="reduced")
    sham = q[:, :dimension]
    for column in range(dimension):
        pivot = int(np.argmax(np.abs(sham[:, column])))
        if sham[pivot, column] < 0:
            sham[:, column] *= -1
    overlap = float(np.max(np.abs(real_basis.T @ sham)))
    orthogonality = float(np.max(np.abs(sham.T @ sham - np.eye(dimension))))
    if overlap > 1e-12 or orthogonality > 1e-12:
        raise RuntimeError(f"invalid sham basis: overlap={overlap}, orth={orthogonality}")
    return sham.astype(np.float64)


def verified_ids(root: Path, name: str, manifest: dict[str, Any], sources: dict[str, str]) -> np.ndarray:
    path = root / f"{name}.ids.npy"
    digest = sha256_file(path)
    sources[path.relative_to(ROOT).as_posix()] = digest
    if digest != manifest["texts"][name]["ids_sha256"]:
        raise RuntimeError(f"ID binding failed: {name}")
    values = np.load(path, allow_pickle=False)
    if values.shape != (SEQ,) or values.dtype != np.int32:
        raise RuntimeError(f"invalid token IDs: {name}")
    return values


def build_arrays() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if subprocess.run(["git", "merge-base", "--is-ancestor", REGISTERED_COMMIT, "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError(f"registration commit is not an ancestor: {REGISTERED_COMMIT}")
    registration_sha = sha256_file(REGISTRATION)
    sources: dict[str, str] = {REGISTRATION.relative_to(ROOT).as_posix(): registration_sha}
    arrays: dict[str, np.ndarray] = {}

    # L29 text means and carrier image.
    l29_values: dict[str, np.ndarray] = {}
    for text in TEXTS:
        values = verified_rvec(29, text, sources)
        l29_values[text] = values
        arrays[f"mu_L29_{text}"] = values.mean(axis=0, dtype=np.float64).astype(np.float32)

    basis_path = ROOT / "analysis" / "subspace_anatomy" / "common_bases_top4.npz"
    wr_path = ROOT / "weights" / "layer29_wr_du.npy"
    proj_path = ROOT / "weights" / "layer29_rel_logits_proj.npy"
    for path in (basis_path, wr_path, proj_path):
        sources[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    with np.load(basis_path, allow_pickle=False) as loaded:
        hidden_carrier = np.asarray(loaded["basis"][29, 0], dtype=np.float64)
    wr = np.asarray(np.load(wr_path, allow_pickle=False), dtype=np.float64)
    if wr.shape != (RFLAT, 6144) or hidden_carrier.shape != (6144,):
        raise RuntimeError("invalid L29 carrier dependencies")
    carrier_image = wr @ hidden_carrier
    carrier_image /= np.linalg.norm(carrier_image)
    arrays["carrier_g_L29"] = carrier_image.astype(np.float32)

    # Outcome-blind head ranking from the six baseline realized kernels.
    proj = np.asarray(np.load(proj_path, allow_pickle=False), dtype=np.float64)
    if proj.shape != (RPERHEAD, 1024):
        raise RuntimeError(f"invalid L29 projection bank: {proj.shape}")
    scores = np.empty((len(TEXTS), HEADS), dtype=np.float64)
    for row, text in enumerate(TEXTS):
        kernel = arrays[f"mu_L29_{text}"].astype(np.float64).reshape(HEADS, RPERHEAD) @ proj
        scores[row] = kernel[:, 1:4].mean(axis=1) - kernel[:, 0]
    median_scores = np.median(scores, axis=0)
    order = np.lexsort((np.arange(HEADS), -median_scores)).astype(np.int16)
    arrays["head_stencil_score_by_text"] = scores.astype(np.float32)
    arrays["head_stencil_score_median"] = median_scores.astype(np.float32)
    arrays["head_order"] = order
    for quartile in range(4):
        arrays[f"head_q{quartile + 1}"] = order[quartile * 16 : (quartile + 1) * 16]
    for count in (8, 16, 32):
        arrays[f"head_top{count:02d}"] = order[:count]
    if sorted(order.tolist()) != list(range(HEADS)):
        raise RuntimeError("head ranking is not a partition")

    # Text-specific, full-union, LOTO, and sham clocks.
    clock_singular: dict[str, list[float]] = {}
    for layer in (53, 59):
        directions: list[np.ndarray] = []
        for text in TEXTS:
            values = verified_rvec(layer, text, sources)
            arrays[f"mu_L{layer:02d}_{text}"] = values.mean(axis=0, dtype=np.float64).astype(np.float32)
            direction = clock_direction(values)
            arrays[f"clock_g_L{layer:02d}_{text}"] = direction.astype(np.float32)
            directions.append(direction)
        matrix = np.stack(directions, axis=1)
        union, singular = orthonormal_columns(matrix, 6)
        arrays[f"clock_union_L{layer:02d}"] = union.astype(np.float32)
        arrays[f"clock_direction_matrix_L{layer:02d}"] = matrix.astype(np.float32)
        clock_singular[f"L{layer:02d}"] = singular.tolist()
        for held_out, text in enumerate(TEXTS):
            loto, _ = orthonormal_columns(np.delete(matrix, held_out, axis=1), 5)
            arrays[f"clock_loto_L{layer:02d}_{text}"] = loto.astype(np.float32)
        arrays[f"clock_sham6_L{layer:02d}"] = sham_basis(
            union, registration_sha, f"clock_sham6_L{layer:02d}", 6
        ).astype(np.float32)

    # Exact needle and seeded sham patch positions.
    sidecar_path = CORPUS / "05_needles.sidecar.json"
    sources[sidecar_path.relative_to(ROOT).as_posix()] = sha256_file(sidecar_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    queries = np.asarray(
        [int(entity["token_positions"][1]) for entity in sidecar["entities"]], dtype=np.int32
    )
    if queries.shape != (24,) or len(set(queries.tolist())) != 24:
        raise RuntimeError("invalid needle query inventory")
    excluded: set[int] = set()
    for query in queries:
        excluded.update(range(max(256, int(query) - 8), min(8191, int(query) + 9)))
    eligible = np.asarray([p for p in range(256, 8191) if p not in excluded], dtype=np.int32)
    rng = np.random.Generator(np.random.PCG64(seed(registration_sha, "needle_patch_sham")))
    sham_positions = np.sort(rng.choice(eligible, size=24, replace=False)).astype(np.int32)
    arrays["patch_query_positions"] = np.sort(queries)
    arrays["patch_sham_positions"] = sham_positions

    # Fresh class positions.
    v2_manifest_path = CORPUS_V2 / "manifest.json"
    depth_path = CORPUS_V2 / "depth_classes.json"
    tokenizer_path = CORPUS / "tokenizer.json"
    for path in (v2_manifest_path, depth_path, tokenizer_path):
        sources[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    v2_manifest = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    depth = json.loads(depth_path.read_text(encoding="utf-8"))
    arrays["class_07b_first_content"] = np.asarray(depth["classes"]["first_content"], dtype=np.int32)
    arrays["class_07b_pronouns"] = np.asarray(depth["classes"]["pronouns"], dtype=np.int32)

    math = "08_math_llm"
    math_ids = verified_ids(CORPUS_V2, math, v2_manifest, sources)
    math_sidecar_path = CORPUS_V2 / f"{math}.sidecar.json"
    sources[math_sidecar_path.relative_to(ROOT).as_posix()] = sha256_file(math_sidecar_path)
    math_sidecar = json.loads(math_sidecar_path.read_text(encoding="utf-8"))
    starts = np.asarray(sorted(int(value) for value in math_sidecar["unit_start_tokens"]), dtype=np.int32)
    if starts.size != int(math_sidecar["n_units_used"]) or starts.size != 10:
        raise RuntimeError("unexpected math unit-start inventory")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    fragments = tokenizer.decode_batch([[int(token)] for token in math_ids], skip_special_tokens=False)
    pronouns = np.asarray(
        [position for position, fragment in enumerate(fragments) if normalized_word(fragment) in PRONOUNS],
        dtype=np.int32,
    )
    if pronouns.size == 0:
        raise RuntimeError("empty math pronoun class")
    arrays["class_08_unit_starts"] = starts
    arrays["class_08_pronouns"] = pronouns

    for name, values in arrays.items():
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite frozen array: {name}")
    metadata = {
        "schema_version": 1,
        "kind": "round5_followup7_frozen_inputs",
        "registration_commit": REGISTERED_COMMIT,
        "registration_sha256": registration_sha,
        "source_git_head_at_build": git("rev-parse", "HEAD"),
        "source_sha256": sources,
        "texts": TEXTS,
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "head_order": order.astype(int).tolist(),
        "clock_singular_values": clock_singular,
        "patch_sham_seed": seed(registration_sha, "needle_patch_sham"),
        "class_counts": {
            "07b_first_content": int(arrays["class_07b_first_content"].size),
            "07b_pronouns": int(arrays["class_07b_pronouns"].size),
            "08_unit_starts": int(arrays["class_08_unit_starts"].size),
            "08_pronouns": int(arrays["class_08_pronouns"].size),
        },
    }
    return arrays, metadata


def build(args: argparse.Namespace) -> None:
    if args.npz.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite frozen follow-up inputs")
    arrays, metadata = build_arrays()
    atomic_npz(args.npz, arrays)
    metadata["created_at_utc"] = utc_now()
    metadata["npz_sha256"] = sha256_file(args.npz)
    metadata["array_count"] = len(arrays)
    atomic_json(args.manifest, metadata)
    print(json.dumps({"npz": str(args.npz), "sha256": metadata["npz_sha256"], "arrays": len(arrays)}, indent=2))


def check(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256_file(args.npz) != manifest["npz_sha256"]:
        raise RuntimeError("frozen NPZ hash mismatch")
    expected, metadata = build_arrays()
    errors: list[str] = []
    with np.load(args.npz, allow_pickle=False) as actual:
        if set(actual.files) != set(expected):
            errors.append("array inventory mismatch")
        for name, wanted in expected.items():
            if name not in actual.files or not np.array_equal(actual[name], wanted):
                errors.append(f"array mismatch: {name}")
    for key in ("registration_sha256", "source_sha256", "head_order", "clock_singular_values", "class_counts"):
        if manifest.get(key) != metadata.get(key):
            errors.append(f"manifest mismatch: {key}")
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps({"passed": True, "arrays": len(expected), "npz_sha256": manifest["npz_sha256"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "build":
        build(arguments)
    else:
        check(arguments)
