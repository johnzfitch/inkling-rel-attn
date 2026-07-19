"""Independent integrity validator for the Round-5 amended capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = ROOT / "dumps" / "round5" / "attention_inputs"
OLD_CAPTURE = ROOT / "dumps" / "tier2" / "capture"


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


def fp16_bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return (
        a.dtype == b.dtype == np.float16
        and a.shape == b.shape
        and np.array_equal(a.view(np.uint16), b.view(np.uint16))
    )


def validate(args: argparse.Namespace) -> dict[str, Any]:
    capture = args.capture.resolve()
    manifest_path = capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    artifacts = manifest.get("artifacts", [])
    kinds = Counter(item.get("kind") for item in artifacts)

    if not manifest.get("complete"):
        errors.append("manifest is not complete")
    if manifest.get("full_registered_capture"):
        expected = {
            "normalized_attention_input": 396,
            "rvec_replay": 396,
            "massive_activation_census": 396,
            "needle_rows_replay": 66,
            "next_token_nll": 6,
        }
        for kind, count in expected.items():
            if kinds[kind] != count:
                errors.append(f"{kind}: expected {count}, found {kinds[kind]}")

    rel_paths = [item["path"] for item in artifacts]
    if len(rel_paths) != len(set(rel_paths)):
        errors.append("duplicate artifact paths in manifest")

    normalized_checked = 0
    replay_checked = 0
    massive_checked = 0
    nll_checked = 0
    for item in artifacts:
        path = capture / item["path"]
        if not path.exists():
            errors.append(f"missing artifact: {item['path']}")
            continue
        if path.stat().st_size != item["bytes"]:
            errors.append(f"byte-count mismatch: {item['path']}")
        if not args.fast and sha256_file(path) != item["sha256"]:
            errors.append(f"SHA-256 mismatch: {item['path']}")

        kind = item["kind"]
        if kind == "normalized_attention_input":
            array = np.load(path, mmap_mode="r")
            if array.dtype != np.uint16 or array.shape != (8192, 6144):
                errors.append(f"normalized shape/dtype mismatch: {item['path']}")
            else:
                flat = array.reshape(-1)
                for start in range(0, flat.size, 8 << 20):
                    words = np.asarray(flat[start : start + (8 << 20)])
                    if np.any((words & np.uint16(0x7F80)) == np.uint16(0x7F80)):
                        errors.append(f"non-finite BF16 payload: {item['path']}")
                        break
            normalized_checked += 1
        elif kind == "rvec_replay":
            actual = np.load(path, mmap_mode="r")
            original = np.load(Path(item["original_path"]), mmap_mode="r")
            if not fp16_bits_equal(actual, original):
                errors.append(f"rvec replay mismatch: {item['path']}")
            replay_checked += 1
        elif kind == "needle_rows_replay":
            with np.load(path) as actual, np.load(Path(item["original_path"])) as original:
                if not np.array_equal(actual["qpos"], original["qpos"]):
                    errors.append(f"needle qpos mismatch: {item['path']}")
                if not fp16_bits_equal(actual["rows"], original["rows"]):
                    errors.append(f"needle replay mismatch: {item['path']}")
            replay_checked += 1
        elif kind == "massive_activation_census":
            with np.load(path) as census:
                if set(census.files) != {"position", "channel", "value"}:
                    errors.append(f"massive census keys mismatch: {item['path']}")
                elif not (
                    len(census["position"]) == len(census["channel"]) == len(census["value"])
                ):
                    errors.append(f"massive census length mismatch: {item['path']}")
            massive_checked += 1
        elif kind == "next_token_nll":
            with np.load(path) as nll:
                if nll["nll"].shape != (8191,) or not np.isfinite(nll["nll"]).all():
                    errors.append(f"invalid NLL: {item['path']}")
            nll_checked += 1

    if manifest.get("full_registered_capture") and not manifest.get("nll_gate", {}).get("passed"):
        errors.append("NLL gate did not pass")

    return {
        "schema_version": 1,
        "kind": "round5_capture_independent_validation",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "fast_mode": bool(args.fast),
        "artifact_counts": dict(sorted(kinds.items())),
        "normalized_checked": normalized_checked,
        "replay_checked": replay_checked,
        "massive_checked": massive_checked,
        "nll_checked": nll_checked,
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fast", action="store_true", help="skip full artifact re-hashing")
    args = parser.parse_args()
    report = validate(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        atomic_json(args.report, report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
