"""Independent second-analyst verification for Round 5 LF2.

No LF2 producer or sensitivity module is imported.  The script rebuilds the
observed log-curves from original projection weights, implements continuous
hinge regression through residualized regressors, regenerates the registered
IID null, regenerates both circular-block nulls, and checks every published
decision plus the audit's rise/decay classification.

Usage:
    python scripts/round5_lf2_verify.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DUMP = ROOT / "dumps" / "round5" / "lf2" / "lf2_knees.npz"
MANIFEST = DUMP.with_name("manifest.json")
SCALES = ROOT / "analysis" / "round5" / "lf2" / "corpus_scales.json"
RESULT = ROOT / "analysis" / "round5" / "lf2" / "lf2_knees.json"
SENSITIVITY = ROOT / "analysis" / "round5" / "lf2" / "lf2_block_sensitivity.json"
OUTPUT = ROOT / "analysis" / "round5" / "lf2" / "verification.json"

GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
BREAKPOINTS = np.arange(16, 393, 2, dtype=np.int64)
IID_SURROGATES = 2000
IID_SEED = 0x1F2
BLOCK_SURROGATES = 2000
BLOCK_SEED = 0x1F2B


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def independent_mode0(bank: np.ndarray) -> np.ndarray:
    a = np.asarray(bank, dtype=np.float64)
    values, vectors = np.linalg.eigh(a @ a.T)
    singular = np.sqrt(max(float(values[-1]), 0.0))
    curve = a.T @ vectors[:, -1] / singular
    curve /= np.linalg.norm(curve)
    if float(np.mean(curve[:32])) < 0.0:
        curve = -curve
    return curve


def two_exponential(distance: np.ndarray, parameters: list[float]) -> np.ndarray:
    a1, r1, a2, r2 = parameters
    return a1 * np.exp(-r1 * distance) + a2 * np.exp(-r2 * distance)


class HingeScanner:
    """Fast nested-regression scan using residualized hinge columns.

    The producer solves a three-column least-squares problem at every hinge.
    Here the base linear trend is projected out once.  The incremental sum of
    squares for a candidate is (y_resid dot hinge)^2 / ||hinge_resid||^2,
    which is algebraically equivalent but independently implemented.
    """

    def __init__(self, distance: np.ndarray):
        self.distance = np.asarray(distance, dtype=np.float64)
        self.base = np.column_stack([self.distance, np.ones_like(self.distance)])
        self.base_inverse = np.linalg.inv(self.base.T @ self.base)
        denominators = []
        indices = []
        for breakpoint in BREAKPOINTS:
            hinge = np.maximum(self.distance - float(breakpoint), 0.0)
            residual = hinge - self.base @ (self.base_inverse @ (self.base.T @ hinge))
            denominators.append(float(residual @ residual))
            indices.append(int(np.searchsorted(self.distance, breakpoint, side="right")))
        self.denominators = np.asarray(denominators)
        self.indices = np.asarray(indices)

    def scan_many(self, curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.atleast_2d(np.asarray(curves, dtype=np.float64))
        coefficients = (values @ self.base) @ self.base_inverse
        residuals = values - coefficients @ self.base.T
        base_sse = np.sum(residuals * residuals, axis=1)

        suffix = np.cumsum(residuals[:, ::-1], axis=1)[:, ::-1]
        suffix_weighted = np.cumsum(
            (residuals * self.distance[None, :])[:, ::-1], axis=1
        )[:, ::-1]
        numerators = np.empty((len(values), len(BREAKPOINTS)), dtype=np.float64)
        for column, (breakpoint, index) in enumerate(zip(BREAKPOINTS, self.indices)):
            numerators[:, column] = (
                suffix_weighted[:, index] - float(breakpoint) * suffix[:, index]
            )
        increments = numerators * numerators / self.denominators[None, :]
        best_index = np.argmax(increments, axis=1)
        improvements = increments[np.arange(len(values)), best_index] / np.maximum(
            base_sse, 1e-300
        )
        return improvements, BREAKPOINTS[best_index]

    def fit_slopes(self, curve: np.ndarray, breakpoint: int) -> tuple[float, float]:
        hinge = np.maximum(self.distance - float(breakpoint), 0.0)
        design = np.column_stack([self.distance, hinge, np.ones_like(self.distance)])
        coefficients = np.linalg.solve(design.T @ design, design.T @ curve)
        return float(coefficients[0]), float(coefficients[0] + coefficients[1])


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def circular_block_draws(
    residuals: np.ndarray, block_size: int, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Vectorized equivalent of sequential circular block sampling."""
    n = residuals.size
    blocks_per_draw = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(count, blocks_per_draw))
    offsets = np.arange(block_size, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return residuals[indices.reshape(count, -1)[:, :n]]


def close_or_error(
    errors: list[str], label: str, actual: float, expected: float, tolerance: float = 1e-10
) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        errors.append(f"{label}: recomputed={actual!r}, published={expected!r}")


def compare_row(
    errors: list[str], label: str, recomputed: dict, published: dict
) -> None:
    for field in ("improvement", "p", "p_holm"):
        close_or_error(errors, f"{label} {field}", recomputed[field], published[field])
    for field in ("breakpoint", "significant"):
        if recomputed[field] != published[field]:
            errors.append(
                f"{label} {field}: recomputed={recomputed[field]!r}, "
                f"published={published[field]!r}"
            )


def main() -> None:
    manifest = load_json(MANIFEST)
    scales = load_json(SCALES)
    published = load_json(RESULT)
    sensitivity = load_json(SENSITIVITY)
    errors: list[str] = []

    if manifest.get("kind") != "round5_lf2_knee_dump" or manifest.get("schema_version") != 1:
        errors.append("unexpected LF2 dump-manifest kind or schema")
    if published.get("kind") != "round5_lf2_knees" or published.get("schema_version") != 1:
        errors.append("unexpected LF2 published-result kind or schema")
    if (
        sensitivity.get("kind") != "round5_lf2_block_sensitivity"
        or sensitivity.get("schema_version") != 1
    ):
        errors.append("unexpected LF2 sensitivity kind or schema")
    frozen_dump_settings = {
        "surrogates": IID_SURROGATES,
        "seed": IID_SEED,
        "window": [8, 400],
        "break_range": [16, 392],
    }
    for field, expected in frozen_dump_settings.items():
        if manifest.get(field) != expected:
            errors.append(f"LF2 manifest {field} differs from frozen verifier assumption")
    if sensitivity.get("surrogates") != BLOCK_SURROGATES:
        errors.append("LF2 sensitivity surrogate count differs from frozen verifier assumption")
    if published.get("frozen_ranges") != scales.get("frozen_ranges"):
        errors.append("published LF2 scale ranges differ from frozen scales artifact")

    dump_hash = sha256_file(DUMP)
    if dump_hash != manifest["dump_sha256"] or dump_hash != published["dump_sha256"]:
        errors.append("LF2 dump SHA-256 disagrees with manifest or result")
    if dump_hash != sensitivity["dump_sha256"]:
        errors.append("LF2 sensitivity artifact names a different dump")
    if sha256_file(SCALES) != published["scales_sha256"]:
        errors.append("frozen scale SHA-256 disagrees with published result")

    sent_lo, sent_hi = scales["frozen_ranges"]["sentence"]
    para_lo, para_hi = scales["frozen_ranges"]["paragraph"]
    iid_rng = np.random.default_rng(IID_SEED)
    observed_rows: list[dict] = []
    layer_material: dict[int, dict] = {}
    max_y_difference = 0.0
    max_smooth_difference = 0.0
    max_iid_null_difference = 0.0
    iid_break_mismatches = 0
    verified_weight_hashes = 0

    with np.load(DUMP, allow_pickle=False) as dumped:
        distance = dumped["d"].astype(np.float64)
        scanner = HingeScanner(distance)

        for layer in GLOBALS:
            label = f"L{layer:02d}"
            weight_path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
            if sha256_file(weight_path) == manifest["input_sha256"].get(weight_path.name):
                verified_weight_hashes += 1
            else:
                errors.append(f"{weight_path.name}: input hash mismatch")

            curve = np.abs(independent_mode0(np.load(weight_path, allow_pickle=False)))[8:401]
            floor = 1e-6 * float(curve.max())
            rebuilt_y = np.log10(np.maximum(curve, floor))
            y = dumped[f"y_{label}"].astype(np.float64)
            max_y_difference = max(max_y_difference, float(np.max(np.abs(rebuilt_y - y))))
            if not np.allclose(rebuilt_y, y, rtol=0.0, atol=1e-8):
                errors.append(f"{label}: raw-weight log curve differs from dump")

            observed_manifest = manifest["observed"][label]
            rebuilt_smooth = np.log10(
                np.maximum(two_exponential(distance, observed_manifest["two_exp_params"]), floor)
            )
            smooth = dumped[f"smooth_{label}"].astype(np.float64)
            max_smooth_difference = max(
                max_smooth_difference, float(np.max(np.abs(rebuilt_smooth - smooth)))
            )
            if not np.allclose(rebuilt_smooth, smooth, rtol=0.0, atol=1e-9):
                errors.append(f"{label}: smooth two-exponential curve differs from manifest fit")

            observed_improvement, observed_break = scanner.scan_many(y)
            improvement = float(observed_improvement[0])
            breakpoint = int(observed_break[0])
            close_or_error(
                errors, f"{label} observed improvement", improvement, observed_manifest["improvement"]
            )
            if breakpoint != observed_manifest["breakpoint"]:
                errors.append(
                    f"{label} breakpoint: recomputed={breakpoint}, "
                    f"manifest={observed_manifest['breakpoint']}"
                )

            residuals = y - smooth
            iid_surrogates = smooth[None, :] + iid_rng.choice(
                residuals, size=(IID_SURROGATES, residuals.size), replace=True
            )
            regenerated_null, regenerated_breaks = scanner.scan_many(iid_surrogates)
            dumped_null = dumped[f"null_imp_{label}"].astype(np.float64)
            dumped_breaks = dumped[f"null_break_{label}"].astype(np.int64)
            max_iid_null_difference = max(
                max_iid_null_difference,
                float(np.max(np.abs(regenerated_null - dumped_null))),
            )
            iid_break_mismatches += int(np.sum(regenerated_breaks != dumped_breaks))
            if not np.allclose(regenerated_null, dumped_null, rtol=0.0, atol=1e-9):
                errors.append(f"{label}: regenerated IID null improvements differ from dump")
            if not np.array_equal(regenerated_breaks, dumped_breaks):
                errors.append(f"{label}: regenerated IID null breakpoints differ from dump")

            pvalue = float((1 + np.sum(dumped_null >= improvement)) / (len(dumped_null) + 1))
            pre_slope, post_slope = scanner.fit_slopes(y, breakpoint)
            lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
            observed_rows.append(
                {
                    "layer": layer,
                    "improvement": improvement,
                    "breakpoint": breakpoint,
                    "p": pvalue,
                    "pre_slope": pre_slope,
                    "post_slope": post_slope,
                    "residual_lag1": lag1,
                }
            )
            layer_material[layer] = {
                "y": y,
                "smooth": smooth,
                "residuals": residuals,
                "improvement": improvement,
                "breakpoint": breakpoint,
            }

        for row, adjusted in zip(observed_rows, holm_adjust([row["p"] for row in observed_rows])):
            row["p_holm"] = adjusted
            row["significant"] = bool(adjusted < 0.05)
            row["sentence_scale"] = bool(
                row["significant"] and sent_lo <= row["breakpoint"] <= sent_hi
            )
            row["paragraph_scale"] = bool(
                row["significant"] and para_lo <= row["breakpoint"] <= para_hi
            )
            row["rise_to_decay_crest"] = bool(
                row["paragraph_scale"] and row["pre_slope"] > 0.0 and row["post_slope"] < 0.0
            )

        published_by_layer = {row["layer"]: row for row in published["layers"]}
        for row in observed_rows:
            compare_row(errors, f"IID L{row['layer']:02d}", row, published_by_layer[row["layer"]])
            for field in ("sentence_scale", "paragraph_scale"):
                if row[field] != published_by_layer[row["layer"]][field]:
                    errors.append(f"IID L{row['layer']:02d} {field} decision differs")

        block_results: dict[str, list[dict]] = {}
        for block_size in (16, 32):
            rng = np.random.default_rng(BLOCK_SEED + block_size)
            rows: list[dict] = []
            for layer in GLOBALS:
                material = layer_material[layer]
                resampled = circular_block_draws(
                    material["residuals"], block_size, rng, BLOCK_SURROGATES
                )
                null_improvement, _ = scanner.scan_many(material["smooth"][None, :] + resampled)
                pvalue = float(
                    (1 + np.sum(null_improvement >= material["improvement"]))
                    / (BLOCK_SURROGATES + 1)
                )
                rows.append(
                    {
                        "layer": layer,
                        "improvement": material["improvement"],
                        "breakpoint": material["breakpoint"],
                        "p": pvalue,
                    }
                )
            for row, adjusted in zip(rows, holm_adjust([row["p"] for row in rows])):
                row["p_holm"] = adjusted
                row["significant"] = bool(adjusted < 0.05)
                row["paragraph_scale"] = bool(
                    row["significant"] and para_lo <= row["breakpoint"] <= para_hi
                )

            published_block = {
                row["layer"]: row for row in sensitivity["results"][f"block_{block_size}"]
            }
            for row in rows:
                compare_row(
                    errors,
                    f"block-{block_size} L{row['layer']:02d}",
                    row,
                    published_block[row["layer"]],
                )
            block_results[f"block_{block_size}"] = rows

        for layer in GLOBALS:
            close_or_error(
                errors,
                f"L{layer:02d} residual lag-1",
                next(row["residual_lag1"] for row in observed_rows if row["layer"] == layer),
                sensitivity["residual_lag1_autocorr"][f"L{layer:02d}"],
            )

    iid_paragraph = [row for row in observed_rows if row["paragraph_scale"]]
    crests = [row for row in observed_rows if row["rise_to_decay_crest"]]
    block_paragraph = {
        key: [row["layer"] for row in rows if row["paragraph_scale"]]
        for key, rows in block_results.items()
    }
    n_sentence = sum(row["sentence_scale"] for row in observed_rows)
    if n_sentence != published["n_sentence_knees"]:
        errors.append("sentence-knee count differs from published result")
    if len(iid_paragraph) != published["n_paragraph_knees"]:
        errors.append("paragraph-knee count differs from published result")
    expected_prediction = {
        "sentence_clause": bool(n_sentence >= 6),
        "paragraph_clause": bool(len(iid_paragraph) == 0),
        "passed": bool(n_sentence >= 6 and len(iid_paragraph) == 0),
    }
    for field, value in expected_prediction.items():
        if published["prediction"].get(field) != value:
            errors.append(f"published LF2 prediction field {field} differs")

    report = {
        "kind": "round5_lf2_independent_verification",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(Path(__file__)),
        "independence": {
            "producer_imports": [],
            "curve_rederivation": "leading mode from eig(A A^T) on original weight banks",
            "hinge_rederivation": "nested-regression increment from residualized hinge columns",
            "null_rederivation": "registered IID and audit block nulls regenerated from fixed seeds",
        },
        "inputs": {
            "dump": {"path": str(DUMP.relative_to(ROOT)), "sha256": dump_hash},
            "manifest_sha256": sha256_file(MANIFEST),
            "scales_sha256": sha256_file(SCALES),
            "published_result_sha256": sha256_file(RESULT),
            "block_sensitivity_sha256": sha256_file(SENSITIVITY),
            "weight_hashes_verified": f"{verified_weight_hashes}/11",
        },
        "dump_rederivation": {
            "max_abs_log_curve_difference": max_y_difference,
            "max_abs_smooth_curve_difference": max_smooth_difference,
            "max_abs_iid_null_improvement_difference": max_iid_null_difference,
            "iid_null_breakpoint_mismatches": iid_break_mismatches,
        },
        "iid_readout": {
            "layers": observed_rows,
            "sentence_knees": n_sentence,
            "paragraph_knees": len(iid_paragraph),
            "paragraph_layers": [row["layer"] for row in iid_paragraph],
            "median_paragraph_breakpoint": float(
                np.median([row["breakpoint"] for row in iid_paragraph])
            ),
            "rise_to_decay_crests": len(crests),
            "crest_layers": [row["layer"] for row in crests],
            "registered_prediction_passed": bool(
                n_sentence >= 6 and len(iid_paragraph) == 0
            ),
        },
        "block_sensitivity": {
            "results": block_results,
            "paragraph_layers": block_paragraph,
            "common_paragraph_survivors": sorted(
                set(block_paragraph["block_16"]) & set(block_paragraph["block_32"])
            ),
        },
        "errors": errors,
        "passed": not errors,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "iid_paragraph_hinges": len(iid_paragraph),
                "block_16_paragraph_hinges": len(block_paragraph["block_16"]),
                "block_32_paragraph_hinges": len(block_paragraph["block_32"]),
                "rise_to_decay_crests": len(crests),
                "errors": errors,
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
