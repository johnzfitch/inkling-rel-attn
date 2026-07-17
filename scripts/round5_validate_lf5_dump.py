"""Independent integrity validator for the A5 LF5 ragged production dump."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "lf5"
DEFAULT_REPORT = ROOT / "analysis" / "round5" / "lf5" / "row_dump_validation.json"
COMPONENTS = [
    "content_logits",
    "positional_bias",
    "total_logits",
    "attention_with_bias",
    "attention_without_bias",
]
GLOBAL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
    os.replace(temporary, path)


def all_finite(array: np.ndarray, block_rows: int = 1 << 18) -> bool:
    return all(
        np.isfinite(array[start : start + block_rows]).all()
        for start in range(0, len(array), block_rows)
    )


def all_nonnegative(array: np.ndarray, block_rows: int = 1 << 18) -> bool:
    return all(
        np.all(array[start : start + block_rows] >= 0)
        for start in range(0, len(array), block_rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    manifest_path = args.dump / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not manifest.get("complete"):
        errors.append("root manifest is incomplete")
    if manifest.get("production_backend") != "replay":
        errors.append("root manifest is not A5 replay production")
    groups = manifest.get("groups", {})
    if len(groups) != 396:
        errors.append(f"expected 396 groups, found {len(groups)}")

    rows_checked = 0
    values_checked = 0
    files_hashed = 0
    max_row_sum_error = 0.0
    for key, root_group in sorted(groups.items()):
        directory = args.dump / key
        group_path = directory / "group_manifest.json"
        if not group_path.exists():
            errors.append(f"{key}: missing group manifest")
            continue
        group = json.loads(group_path.read_text(encoding="utf-8"))
        if group != root_group:
            errors.append(f"{key}: root/group manifest mismatch")
        layer = int(group.get("layer", -1))
        text = str(group.get("text", ""))
        if key != f"L{layer:02d}_{text}":
            errors.append(f"{key}: key/metadata mismatch")
        if group.get("backend") != "replay" or group.get("dtype") != "float16":
            errors.append(f"{key}: backend/dtype mismatch")

        recorded_files = group.get("files", {})
        expected_names = {"index.npz", *(f"{name}.npy" for name in COMPONENTS)}
        if set(recorded_files) != expected_names:
            errors.append(f"{key}: file set mismatch")
        for name, record in recorded_files.items():
            path = directory / name
            if not path.exists():
                errors.append(f"{key}: missing {name}")
                continue
            if path.stat().st_size != record.get("bytes"):
                errors.append(f"{key}: byte count mismatch for {name}")
            if sha256_file(path) != record.get("sha256"):
                errors.append(f"{key}: SHA-256 mismatch for {name}")
            files_hashed += 1

        with np.load(directory / "index.npz") as index:
            qpos = index["qpos"].astype(np.int64)
            starts = index["support_start"].astype(np.int64)
            stops = index["support_stop"].astype(np.int64)
            indptr = index["indptr"].astype(np.int64)
        expected_qpos = np.asarray(manifest["queries"][text], dtype=np.int64)
        if not np.array_equal(qpos, expected_qpos):
            errors.append(f"{key}: frozen query positions mismatch")
        expected_starts = np.array(
            [0 if layer in GLOBAL_LAYERS else max(0, int(q) - 511) for q in qpos],
            dtype=np.int64,
        )
        expected_stops = qpos + 1
        expected_indptr = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(expected_stops - expected_starts)]
        )
        if not np.array_equal(starts, expected_starts) or not np.array_equal(stops, expected_stops):
            errors.append(f"{key}: support mismatch")
        if not np.array_equal(indptr, expected_indptr):
            errors.append(f"{key}: indptr mismatch")

        total_entries = int(indptr[-1])
        arrays: dict[str, np.ndarray] = {}
        for component in COMPONENTS:
            array = np.load(directory / f"{component}.npy", mmap_mode="r")
            arrays[component] = array
            if array.dtype != np.float16 or array.shape != (total_entries, 64):
                errors.append(f"{key}: {component} shape/dtype={array.shape}/{array.dtype}")
                continue
            if not all_finite(array):
                errors.append(f"{key}: {component} contains non-finite values")
            if component.startswith("attention_") and not all_nonnegative(array):
                errors.append(f"{key}: {component} contains negative probability")
            values_checked += int(array.size)

        for component in ("attention_with_bias", "attention_without_bias"):
            array = arrays[component]
            for row_index in range(len(qpos)):
                lo, hi = int(indptr[row_index]), int(indptr[row_index + 1])
                sums = array[lo:hi].sum(axis=0, dtype=np.float64)
                row_error = float(np.max(np.abs(sums - 1.0)))
                max_row_sum_error = max(max_row_sum_error, row_error)
                if row_error > 1e-3:
                    errors.append(
                        f"{key}: {component} q={int(qpos[row_index])} row-sum error={row_error}"
                    )
        rows_checked += int(len(qpos))
        print(f"validated {key}", flush=True)

    report = {
        "schema_version": 1,
        "kind": "round5_lf5_row_dump_independent_validation",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dump_manifest_sha256": sha256_file(manifest_path),
        "production_backend": manifest.get("production_backend"),
        "groups_checked": len(groups),
        "files_hashed": files_hashed,
        "rows_checked": rows_checked,
        "values_checked": values_checked,
        "max_fp16_row_sum_error": max_row_sum_error,
        "errors": errors,
        "passed": not errors and len(groups) == 396 and files_hashed == 396 * 6,
        "source_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
