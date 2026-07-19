"""Independent second-analyst verification for Round 5 LF7.

The final LF7 dump is self-contained.  This verifier imports no LF7 producer
code and reconstructs its hidden-side distance matrices from the stored bases
with the projector-overlap identity

    d(A, B)^2 = 1 - ||A^T B||_F^2 / k,

rather than the producer's singular-value calculation.  It also reconstructs
the fixed-seed sketch Grams from dumped bases and spectra.  The curve-side
matrix has no equivalent basis payload, so only its published bookkeeping is
checked; its parent-neighborhood observation remains explicitly unpromoted.

Usage:
    python scripts/round5_lf7_verify.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "dumps" / "round5" / "lf7" / "lf7_dump.npz"
MANIFEST = DUMP.with_name("manifest.json")
RESULT = ROOT / "analysis" / "round5" / "lf7" / "lf7_parentage.json"
OUTPUT = ROOT / "analysis" / "round5" / "lf7" / "verification.json"

ENERGY = 0.90
K_FIXED = 64
SKETCH_SEED = 0x1F7
SKETCH_WIDTH = 256
FORK_PERCENTILE = 5.0
DEEP_MIN = 30
MIN_DEEP_PARENTS = 6
NEIGHBORHOOD_SPAN = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parent_table(
    matrix: np.ndarray, names: list[str], trunk: list[int], mtps: list[int], largest: bool = False
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for mtp_index in mtps:
        values = matrix[mtp_index, trunk]
        offset = int(np.argmax(values) if largest else np.argmin(values))
        trunk_index = trunk[offset]
        result[names[mtp_index]] = {
            "parent": names[trunk_index],
            "depth": trunk_index,
            "distance": float(values[offset]),
        }
    return result


def matrix_null(matrix: np.ndarray, trunk: list[int]) -> np.ndarray:
    return np.asarray(
        [matrix[left, right] for offset, left in enumerate(trunk) for right in trunk[offset + 1 :]],
        dtype=np.float64,
    )


def close_or_error(
    errors: list[str], label: str, actual: float, expected: float, tolerance: float = 1e-7
) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        errors.append(f"{label}: recomputed={actual!r}, published={expected!r}")


def compare_parents(
    errors: list[str], label: str, recomputed: dict[str, dict], published: dict[str, dict]
) -> None:
    if set(recomputed) != set(published):
        errors.append(f"{label}: unit names differ")
        return
    for name, row in recomputed.items():
        other = published[name]
        for field in ("parent", "depth"):
            if row[field] != other[field]:
                errors.append(
                    f"{label} {name} {field}: recomputed={row[field]!r}, published={other[field]!r}"
                )
        close_or_error(errors, f"{label} {name} value", row["distance"], other["distance"])


def main() -> None:
    manifest = load_json(MANIFEST)
    published = load_json(RESULT)
    errors: list[str] = []

    if manifest.get("kind") != "round5_lf7_parentage_dump" or manifest.get("schema_version") != 1:
        errors.append("unexpected LF7 dump-manifest kind or schema")
    if published.get("kind") != "round5_lf7_parentage" or published.get("schema_version") != 1:
        errors.append("unexpected LF7 published-result kind or schema")
    frozen_method = manifest.get("method_frozen", {})
    for field, expected in {
        "energy": ENERGY,
        "k_fixed": K_FIXED,
        "sketch_seed": SKETCH_SEED,
        "curve_window": 512,
    }.items():
        if frozen_method.get(field) != expected:
            errors.append(f"LF7 manifest {field} differs from frozen verifier assumption")

    print("hashing self-contained LF7 dump", flush=True)
    dump_hash = sha256_file(DUMP)
    if dump_hash != manifest["dump_sha256"] or dump_hash != published["dump_sha256"]:
        errors.append("LF7 dump SHA-256 disagrees with manifest or result")

    with np.load(DUMP, allow_pickle=False) as dumped:
        names = [str(value) for value in dumped["names"]]
        if names != [f"L{layer:02d}" for layer in range(66)] + [f"M{index}" for index in range(8)]:
            errors.append("unexpected LF7 unit order")
        trunk = [index for index, name in enumerate(names) if name.startswith("L")]
        mtps = [index for index, name in enumerate(names) if name.startswith("M")]
        spectra = dumped["spectra"].astype(np.float64)
        stored_k90 = dumped["k90"].astype(np.int64)
        energy_fraction = np.cumsum(spectra * spectra, axis=1) / np.sum(
            spectra * spectra, axis=1, keepdims=True
        )
        rebuilt_k90 = np.asarray(
            [int(np.searchsorted(row, ENERGY) + 1) for row in energy_fraction], dtype=np.int64
        )
        if not np.array_equal(rebuilt_k90, stored_k90):
            errors.append("energy-90% ranks do not rederive from stored spectra")

        bases: list[np.ndarray] = []
        for index, name in enumerate(names):
            basis = dumped[f"basis_{name}"].astype(np.float64)
            if basis.shape != (6144, 1024):
                errors.append(f"{name}: basis shape is {basis.shape}, expected (6144, 1024)")
            bases.append(basis)
            if (index + 1) % 8 == 0 or index + 1 == len(names):
                print(f"loaded bases {index + 1}/{len(names)}", flush=True)

        size = len(names)
        rebuilt_primary = np.zeros((size, size), dtype=np.float64)
        rebuilt_fixed = np.zeros((size, size), dtype=np.float64)
        for left_index in range(size):
            left = bases[left_index]
            for right_index in range(left_index + 1, size):
                rank = int(min(rebuilt_k90[left_index], rebuilt_k90[right_index]))
                cross_width = max(rank, K_FIXED)
                cross = left[:, :cross_width].T @ bases[right_index][:, :cross_width]
                primary_overlap = float(np.sum(cross[:rank, :rank] ** 2))
                fixed_overlap = float(np.sum(cross[:K_FIXED, :K_FIXED] ** 2))
                primary_distance = np.sqrt(max(1.0 - primary_overlap / rank, 0.0))
                fixed_distance = np.sqrt(max(1.0 - fixed_overlap / K_FIXED, 0.0))
                rebuilt_primary[left_index, right_index] = primary_distance
                rebuilt_primary[right_index, left_index] = primary_distance
                rebuilt_fixed[left_index, right_index] = fixed_distance
                rebuilt_fixed[right_index, left_index] = fixed_distance
            if (left_index + 1) % 6 == 0 or left_index + 1 == size:
                print(f"rederived hidden distances {left_index + 1}/{size}", flush=True)

        stored_primary = dumped["d_primary"].astype(np.float64)
        stored_fixed = dumped["d_fixed"].astype(np.float64)
        max_primary_difference = float(np.max(np.abs(rebuilt_primary - stored_primary)))
        max_fixed_difference = float(np.max(np.abs(rebuilt_fixed - stored_fixed)))
        if not np.allclose(rebuilt_primary, stored_primary, rtol=0.0, atol=1e-7):
            errors.append("projector-identity primary matrix differs from dumped SVD matrix")
        if not np.allclose(rebuilt_fixed, stored_fixed, rtol=0.0, atol=1e-7):
            errors.append("projector-identity fixed-k matrix differs from dumped SVD matrix")

        print("rederiving fixed-seed sketch Grams", flush=True)
        rng = np.random.default_rng(SKETCH_SEED)
        projection = rng.standard_normal((6144, SKETCH_WIDTH)).astype(np.float64)
        grams: list[np.ndarray] = []
        for index, (basis, singular_values) in enumerate(zip(bases, spectra)):
            coordinates = basis.T @ projection
            weighted = singular_values[:, None] * coordinates
            gram = weighted.T @ weighted
            grams.append((gram / np.linalg.norm(gram)).astype(np.float32))
            if (index + 1) % 8 == 0 or index + 1 == size:
                print(f"rederived sketches {index + 1}/{size}", flush=True)
        rebuilt_sketch = np.eye(size, dtype=np.float64)
        for left_index in range(size):
            for right_index in range(left_index + 1, size):
                similarity = float(
                    np.sum(
                        grams[left_index].astype(np.float64)
                        * grams[right_index].astype(np.float64)
                    )
                )
                rebuilt_sketch[left_index, right_index] = similarity
                rebuilt_sketch[right_index, left_index] = similarity
        stored_sketch = dumped["s_sketch"].astype(np.float64)
        max_sketch_difference = float(np.max(np.abs(rebuilt_sketch - stored_sketch)))
        if not np.allclose(rebuilt_sketch, stored_sketch, rtol=0.0, atol=2e-6):
            errors.append("basis/spectrum sketch reconstruction differs from dumped sketch matrix")

        primary = parent_table(rebuilt_primary, names, trunk, mtps)
        fixed = parent_table(rebuilt_fixed, names, trunk, mtps)
        sketch = parent_table(rebuilt_sketch, names, trunk, mtps, largest=True)
        null = matrix_null(rebuilt_primary, trunk)
        fork_threshold = float(np.percentile(null, FORK_PERCENTILE))
        for name, row in primary.items():
            row["null_percentile"] = float(100.0 * np.mean(null <= row["distance"]))
            row["fork"] = bool(row["distance"] < fork_threshold)
            row["sketch_agrees"] = bool(row["parent"] == sketch[name]["parent"])

        compare_parents(errors, "primary", primary, published["primary_parents"])
        compare_parents(errors, "fixed-k", fixed, published["fixed_k_parents"])
        compare_parents(errors, "sketch", sketch, published["sketch_parents"])
        for name, row in primary.items():
            other = published["primary_parents"][name]
            close_or_error(
                errors, f"primary {name} null percentile", row["null_percentile"], other["null_percentile"]
            )
            for field in ("fork", "sketch_agrees"):
                if row[field] != other[field]:
                    errors.append(f"primary {name} {field} decision differs")

        rebuilt_null_summary = {
            "n_pairs": int(null.size),
            "p05": fork_threshold,
            "median": float(np.median(null)),
            "min": float(np.min(null)),
            "max": float(np.max(null)),
        }
        for field, value in rebuilt_null_summary.items():
            if field == "n_pairs":
                if value != published["trunk_null"][field]:
                    errors.append("trunk-null pair count differs")
            else:
                close_or_error(errors, f"trunk null {field}", value, published["trunk_null"][field])

        depths = [row["depth"] for row in primary.values()]
        clause_a = bool(
            sum(depth >= DEEP_MIN for depth in depths) >= MIN_DEEP_PARENTS
            and float(np.median(depths)) >= DEEP_MIN
        )
        clause_b = bool(max(depths) - min(depths) <= NEIGHBORHOOD_SPAN)
        prediction_passed = bool(clause_a and clause_b)
        sketch_agreement = sum(row["sketch_agrees"] for row in primary.values())
        promotion_gate_met = bool(sketch_agreement >= 6)
        no_fork_confirmed = bool(not any(row["fork"] for row in primary.values()))
        for field, value in (
            ("clause_a_deep_cluster", clause_a),
            ("clause_b_shared_neighborhood", clause_b),
            ("passed", prediction_passed),
        ):
            if value != published["prediction"][field]:
                errors.append(f"registered prediction field {field} differs")
        if f"{sketch_agreement}/8" != published["sketch_agreement"]:
            errors.append("sketch-agreement count differs")
        if promotion_gate_met != published["promotion_gate"]["met"]:
            errors.append("parent-identity promotion-gate decision differs")

        # The curve matrix is checked only as a finished readout.  Its source
        # bases were not included in the final dump, and the published mode-0
        # rederivation already failed to confirm its neighborhood.
        curve_error_count_before = len(errors)
        stored_curve = dumped["d_curve"].astype(np.float64)
        curve = parent_table(stored_curve, names, trunk, mtps)
        curve_null = matrix_null(stored_curve, trunk)
        for name, row in curve.items():
            row["null_percentile"] = float(100.0 * np.mean(curve_null <= row["distance"]))
            row["below_trunk_minimum"] = bool(row["distance"] < np.min(curve_null))
        compare_parents(errors, "curve bookkeeping", curve, published["curve_parents"])
        for name, row in curve.items():
            other = published["curve_parents"][name]
            close_or_error(
                errors,
                f"curve bookkeeping {name} null percentile",
                row["null_percentile"],
                other["null_percentile"],
            )
            if row["below_trunk_minimum"] != other["below_trunk_minimum"]:
                errors.append(f"curve bookkeeping {name} below-null-minimum decision differs")
        rebuilt_curve_null = {
            "median": float(np.median(curve_null)),
            "p05": float(np.percentile(curve_null, 5)),
            "min": float(np.min(curve_null)),
        }
        for field, value in rebuilt_curve_null.items():
            close_or_error(
                errors, f"curve null {field}", value, published["curve_trunk_null"][field]
            )
        curve_bookkeeping_matches = len(errors) == curve_error_count_before

    report = {
        "kind": "round5_lf7_independent_verification",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(Path(__file__)),
        "independence": {
            "producer_imports": [],
            "primary_rederivation": "projector Frobenius-overlap identity; no cross-matrix SVD",
            "sketch_rederivation": "R^T V diag(s^2) V^T R from dumped bases/spectra and fixed seed",
            "source_object": "self-contained LF7 dump",
        },
        "inputs": {
            "dump": {"path": str(DUMP.relative_to(ROOT)), "sha256": dump_hash},
            "manifest_sha256": sha256_file(MANIFEST),
            "published_result_sha256": sha256_file(RESULT),
        },
        "matrix_rederivation": {
            "k90_exact_match": bool(np.array_equal(rebuilt_k90, stored_k90)),
            "max_abs_primary_difference": max_primary_difference,
            "max_abs_fixed_k_difference": max_fixed_difference,
            "max_abs_sketch_difference": max_sketch_difference,
        },
        "hidden_side_readout": {
            "primary_parents": primary,
            "fixed_k_parents": fixed,
            "sketch_parents": sketch,
            "trunk_null": rebuilt_null_summary,
            "sketch_agreement": f"{sketch_agreement}/8",
            "registered_prediction_passed": prediction_passed,
            "no_fork_confirmed": no_fork_confirmed,
            "parent_identity_promotion_gate_met": promotion_gate_met,
        },
        "curve_side_disposition": {
            "basis_rederived": False,
            "finished_matrix_bookkeeping_matches": curve_bookkeeping_matches,
            "claim_promoted": False,
            "reason": "curve bases absent; published mode-0 rederivation has 1/8 agreement and saturated null",
        },
        "certification_scope": (
            "hidden-side no-fork verdict only; does not certify a parent identity "
            "or the curve-side L47/L51 neighborhood"
        ),
        "errors": errors,
        "passed": not errors,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "no_fork_confirmed": no_fork_confirmed,
                "registered_prediction_passed": prediction_passed,
                "sketch_agreement": f"{sketch_agreement}/8",
                "primary_matrix_max_abs_difference": max_primary_difference,
                "errors": errors,
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
