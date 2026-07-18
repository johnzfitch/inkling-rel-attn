"""Independent second-analyst verification for Round 5 LF1.

This script intentionally imports no LF1 producer code.  It reconstructs the
mode-0 curve from each raw projection bank through the left Gram eigensystem
(rather than the producer's direct SVD), rebuilds every registered pip test,
and independently evaluates the disclosed max-across-rows diagnostic.

Usage:
    python scripts/round5_lf1_verify.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DUMP = ROOT / "dumps" / "round5" / "lf1" / "lf1_pips.npz"
MANIFEST = DUMP.with_name("manifest.json")
RESULT = ROOT / "analysis" / "round5" / "lf1" / "lf1_pips.json"
OUTPUT = ROOT / "analysis" / "round5" / "lf1" / "verification.json"

GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
GLOBAL_SET = set(GLOBALS)
HALF_WINDOW = 8
MARGIN = 16
POWERS = (16, 32, 64, 128, 256, 512, 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def independent_mode0(bank: np.ndarray) -> np.ndarray:
    """Leading right mode via eigendecomposition of A A^T, not direct SVD."""
    a = np.asarray(bank, dtype=np.float64)
    eigenvalues, left = np.linalg.eigh(a @ a.T)
    singular = float(np.sqrt(max(eigenvalues[-1], 0.0)))
    if singular == 0.0:
        raise ValueError("zero leading singular value")
    curve = a.T @ left[:, -1] / singular
    curve /= np.linalg.norm(curve)
    if float(np.mean(curve[:32])) < 0.0:
        curve = -curve
    return curve


def independent_pips(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    out = np.full(curve.shape, np.nan, dtype=np.float64)
    for distance in range(MARGIN, len(curve) - MARGIN):
        neighbors = np.r_[
            curve[distance - HALF_WINDOW : distance],
            curve[distance + 1 : distance + HALF_WINDOW + 1],
        ]
        center = float(np.median(neighbors))
        scale = max(float(np.median(np.abs(neighbors - center))), 1e-12)
        out[distance] = abs(float(curve[distance]) - center) / scale
    return out


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment implemented locally for audit independence."""
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [0.0] * len(pvalues)
    monotone = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        monotone = max(monotone, (total - rank) * pvalues[index])
        adjusted[index] = min(monotone, 1.0)
    return adjusted


def finite_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[mask] - right[mask]))) if np.any(mask) else 0.0


def add_close_check(
    errors: list[str], label: str, actual: float, expected: float, tolerance: float = 1e-10
) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        errors.append(f"{label}: recomputed={actual!r}, published={expected!r}")


def build_family(curves: dict[int, np.ndarray], layers: list[int]) -> list[dict]:
    tests: list[dict] = []
    for layer in layers:
        pips = curves[layer]
        eligible = np.isfinite(pips)
        for distance in POWERS:
            if MARGIN <= distance < len(pips) - MARGIN:
                statistic = float(pips[distance])
                pvalue = float(np.mean(pips[eligible] >= statistic))
                tests.append(
                    {
                        "layer": layer,
                        "d": distance,
                        "pip": statistic,
                        "p": pvalue,
                        "peeked_band_step": bool(distance == 512 and layer in GLOBAL_SET),
                    }
                )
    for test, adjusted in zip(tests, holm_adjust([test["p"] for test in tests])):
        test["p_holm"] = adjusted
        test["significant"] = bool(adjusted < 0.05)
    return tests


def build_row_max_family(row_curves: dict[int, np.ndarray], layers: list[int]) -> list[dict]:
    tests: list[dict] = []
    for layer in layers:
        rows = row_curves[layer]
        combined = np.full(rows.shape[1], np.nan, dtype=np.float64)
        combined[MARGIN : rows.shape[1] - MARGIN] = np.max(
            rows[:, MARGIN : rows.shape[1] - MARGIN], axis=0
        )
        eligible = np.isfinite(combined)
        for distance in POWERS:
            if MARGIN <= distance < combined.size - MARGIN:
                statistic = float(combined[distance])
                pvalue = float(np.mean(combined[eligible] >= statistic))
                tests.append(
                    {
                        "layer": layer,
                        "d": distance,
                        "max_row_pip": statistic,
                        "p": pvalue,
                    }
                )
    for test, adjusted in zip(tests, holm_adjust([test["p"] for test in tests])):
        test["p_holm"] = adjusted
        test["significant"] = bool(adjusted < 0.05)
    return tests


def compare_test_tables(
    errors: list[str], label: str, recomputed: list[dict], published: list[dict]
) -> None:
    if len(recomputed) != len(published):
        errors.append(f"{label}: test count {len(recomputed)} != {len(published)}")
        return
    published_by_key = {(row["layer"], row["d"]): row for row in published}
    for row in recomputed:
        key = (row["layer"], row["d"])
        if key not in published_by_key:
            errors.append(f"{label}: missing published test {key}")
            continue
        other = published_by_key[key]
        for field in ("pip", "p", "p_holm"):
            add_close_check(errors, f"{label} {key} {field}", row[field], other[field])
        for field in ("peeked_band_step", "significant"):
            if row[field] != other[field]:
                errors.append(
                    f"{label} {key} {field}: recomputed={row[field]!r}, "
                    f"published={other[field]!r}"
                )


def main() -> None:
    manifest = load_json(MANIFEST)
    result = load_json(RESULT)
    errors: list[str] = []

    if manifest.get("kind") != "round5_lf1_pip_dump" or manifest.get("schema_version") != 1:
        errors.append("unexpected LF1 dump-manifest kind or schema")
    if result.get("kind") != "round5_lf1_pips" or result.get("schema_version") != 1:
        errors.append("unexpected LF1 published-result kind or schema")
    expected_method = {"half_window": HALF_WINDOW, "margin": MARGIN, "curve": "mode0 of proj; rows secondary"}
    if manifest.get("method") != expected_method:
        errors.append("LF1 manifest method differs from the frozen verifier assumptions")

    dump_hash = sha256_file(DUMP)
    if dump_hash != manifest["dump_sha256"]:
        errors.append("dump SHA-256 does not match the manifest")
    if dump_hash != result["dump_sha256"]:
        errors.append("dump SHA-256 does not match the published result")

    curves: dict[int, np.ndarray] = {}
    row_curves: dict[int, np.ndarray] = {}
    max_mode0_dump_difference = 0.0
    max_rows_dump_difference = 0.0
    verified_weight_hashes = 0

    with np.load(DUMP, allow_pickle=False) as dumped:
        for layer in range(66):
            name = f"L{layer:02d}"
            weight_path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
            weight_hash = sha256_file(weight_path)
            expected_hash = manifest["input_sha256"].get(weight_path.name)
            if weight_hash != expected_hash:
                errors.append(f"{weight_path.name}: weight SHA-256 mismatch")
            else:
                verified_weight_hashes += 1

            bank = np.load(weight_path, allow_pickle=False).astype(np.float64)
            mode_pips = independent_pips(independent_mode0(bank))
            rows = np.stack([independent_pips(row) for row in bank])
            curves[layer] = mode_pips
            row_curves[layer] = rows

            dumped_mode = dumped[f"mode0_{name}"]
            dumped_rows = dumped[f"rows_{name}"]
            max_mode0_dump_difference = max(
                max_mode0_dump_difference, finite_max_abs(mode_pips, dumped_mode)
            )
            max_rows_dump_difference = max(
                max_rows_dump_difference, finite_max_abs(rows, dumped_rows)
            )
            if not np.allclose(mode_pips, dumped_mode, rtol=0.0, atol=1e-8, equal_nan=True):
                errors.append(f"{name}: independently rebuilt mode-0 pip series differs from dump")
            if not np.allclose(rows, dumped_rows, rtol=0.0, atol=1e-10, equal_nan=True):
                errors.append(f"{name}: independently rebuilt row pip bank differs from dump")

    global_tests = build_family(curves, GLOBALS)
    local_layers = [layer for layer in range(66) if layer not in GLOBAL_SET]
    local_tests = build_family(curves, local_layers)
    compare_test_tables(errors, "global", global_tests, result["global_tests"])
    compare_test_tables(errors, "local", local_tests, result["local_tests"])

    global_survivors = [row for row in global_tests if row["significant"]]
    local_survivors = [row for row in local_tests if row["significant"]]
    unpeeked_survivors = [row for row in global_survivors if not row["peeked_band_step"]]
    published_survivor_keys = {
        (row["layer"], row["d"]) for row in result.get("global_survivors", [])
    }
    rebuilt_survivor_keys = {(row["layer"], row["d"]) for row in global_survivors}
    if published_survivor_keys != rebuilt_survivor_keys:
        errors.append("published global-survivor summary differs from the full test table")
    if result.get("n_local_tests") != len(local_tests):
        errors.append("published local-test count differs from the full test table")
    if bool(not unpeeked_survivors) != result["prediction"]["passed"]:
        errors.append("registered prediction decision differs from published result")

    raw_global = build_row_max_family(row_curves, GLOBALS)
    raw_local = build_row_max_family(row_curves, local_layers)
    raw_global_survivors = [row for row in raw_global if row["significant"]]
    raw_local_survivors = [row for row in raw_local if row["significant"]]
    raw_best = min(raw_global + raw_local, key=lambda row: (row["p"], row["layer"], row["d"]))

    report = {
        "kind": "round5_lf1_independent_verification",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(Path(__file__)),
        "independence": {
            "producer_imports": [],
            "mode0_rederivation": "leading right mode from eig(A A^T), not direct SVD",
            "statistics": "pip, empirical p, and Holm procedures reimplemented locally",
            "source_object": "66 original rel_logits_proj weight banks",
        },
        "inputs": {
            "dump": {"path": str(DUMP.relative_to(ROOT)), "sha256": dump_hash},
            "manifest_sha256": sha256_file(MANIFEST),
            "published_result_sha256": sha256_file(RESULT),
            "weight_hashes_verified": f"{verified_weight_hashes}/66",
        },
        "dump_rederivation": {
            "max_abs_mode0_pip_difference": max_mode0_dump_difference,
            "max_abs_row_pip_difference": max_rows_dump_difference,
        },
        "registered_readout": {
            "global_tests": len(global_tests),
            "local_tests": len(local_tests),
            "global_survivors": global_survivors,
            "local_survivors": local_survivors,
            "unpeeked_global_survivors": unpeeked_survivors,
            "prediction_passed": bool(not unpeeked_survivors),
        },
        "raw_bank_diagnostic": {
            "status": "unregistered secondary diagnostic; does not decide LF1",
            "method": "max pip over 16 raw rows at each distance; empirical distance null; Holm within global/local families",
            "global_tests": raw_global,
            "local_tests": raw_local,
            "global_survivors": raw_global_survivors,
            "local_survivors": raw_local_survivors,
            "best_raw_p": raw_best,
        },
        "errors": errors,
        "passed": not errors,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "registered_survivors": len(global_survivors) + len(local_survivors),
                "raw_bank_survivors": len(raw_global_survivors) + len(raw_local_survivors),
                "best_raw_bank_test": raw_best,
                "errors": errors,
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
