"""Registered LF3 absolute-position-counter test on certified r-vectors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from round5_science_common import (
    CAPTURE,
    ROOT,
    artifact_index,
    atomic_json,
    atomic_npz,
    provenance,
    refuse_existing,
    require_certified_capture,
    self_test_common,
    sha256_file,
)


TEXTS = ["06_random", "01_prose_en"]
DEFAULT_OUT = ROOT / "analysis" / "round5" / "lf3"


def correlations_with_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    z = np.asarray(vector, dtype=np.float64)
    z = z - z.mean()
    x_sum = x.sum(axis=0, dtype=np.float64)
    x_square_sum = np.sum(x * x, axis=0, dtype=np.float64)
    x_ss = np.maximum(x_square_sum - x_sum * x_sum / x.shape[0], 0.0)
    denominator = np.sqrt(float(z @ z) * x_ss)
    numerator = z @ x
    if np.any(denominator <= 0):
        raise RuntimeError("constant r-vector coordinate in position correlation")
    return numerator / denominator


def tail_test(block_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if block_means.shape[0] != 127:
        raise ValueError(block_means.shape)
    midpoint = 64.0 + np.arange(127, dtype=np.float64) * 64.0 + 31.5
    regressors = np.stack([midpoint, np.log1p(midpoint)])
    centered_x = np.asarray(block_means, dtype=np.float64)
    centered_x -= centered_x.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.sum(centered_x * centered_x, axis=0, dtype=np.float64))
    if np.any(x_norm <= 0):
        raise RuntimeError("constant tail block coordinate")
    observed = np.empty((2, centered_x.shape[1]), dtype=np.float64)
    null_maximum = np.empty(126, dtype=np.float64)
    shifted_regressors = np.empty((2, 126, 127), dtype=np.float64)
    for regressor_index, regressor in enumerate(regressors):
        centered = regressor - regressor.mean()
        observed[regressor_index] = (centered @ centered_x) / (
            np.linalg.norm(centered) * x_norm
        )
        shifted_regressors[regressor_index] = np.stack(
            [np.roll(centered, shift) for shift in range(1, 127)]
        )
    null_maximum.fill(0.0)
    for regressor_index in range(2):
        z = shifted_regressors[regressor_index]
        correlations = (z @ centered_x) / (
            np.linalg.norm(z, axis=1)[:, None] * x_norm[None, :]
        )
        null_maximum = np.maximum(null_maximum, np.max(np.abs(correlations), axis=1))
    return observed, null_maximum


def block_displacement_and_bos_rank(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(matrix, dtype=np.float64).reshape(128, 64, -1)
    block_mean = x.mean(axis=1)
    total = block_mean.sum(axis=0)
    other_mean = (total[None, :] - block_mean) / 127.0
    displacement = np.linalg.norm(block_mean - other_mean, axis=1)
    less = int(np.sum(displacement < displacement[0]))
    equal = int(np.sum(displacement == displacement[0]))
    percentile = (less + 0.5 * equal) / 128.0
    return displacement, float(percentile)


def run(out: Path) -> None:
    report_path = out / "lf3.json"
    dump_path = out / "lf3_arrays.npz"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, dump_path, results_path)
    _, manifest = require_certified_capture()
    records = artifact_index(manifest)
    full_correlation = np.empty((2, 66, 1024), dtype=np.float64)
    tail_correlation = np.empty((2, 66, 2, 1024), dtype=np.float64)
    layer_null = np.empty((2, 66, 126), dtype=np.float64)
    block_displacement = np.empty((2, 66, 128), dtype=np.float64)
    bos_percentile = np.empty((2, 66), dtype=np.float64)
    input_hashes: dict[str, str] = {}
    token_index = np.arange(8192, dtype=np.float64)

    for text_index, text in enumerate(TEXTS):
        for layer in range(66):
            path = CAPTURE / "replay" / f"rvec_L{layer:02d}_{text}.npy"
            relative = path.relative_to(CAPTURE).as_posix()
            record = records.get(relative)
            if record is None or record.get("kind") != "rvec":
                raise RuntimeError(f"r-vector is not bound by manifest: {relative}")
            input_hashes[relative] = record["sha256"]
            rvec = np.load(path, mmap_mode="r")
            if rvec.shape != (8192, 64, 16) or rvec.dtype != np.float16:
                raise RuntimeError(f"invalid r-vector: {path}")
            flat = np.asarray(rvec, dtype=np.float64).reshape(8192, 1024)
            full_correlation[text_index, layer] = correlations_with_vector(flat, token_index)
            tail_blocks = flat[64:].reshape(127, 64, 1024).mean(axis=1)
            observed, null = tail_test(tail_blocks)
            tail_correlation[text_index, layer] = observed
            layer_null[text_index, layer] = null
            displacement, percentile = block_displacement_and_bos_rank(flat)
            block_displacement[text_index, layer] = displacement
            bos_percentile[text_index, layer] = percentile
            print(f"LF3 {text} L{layer:02d}/65", flush=True)

    text_reports: dict[str, Any] = {}
    for text_index, text in enumerate(TEXTS):
        observed_max = float(np.max(np.abs(tail_correlation[text_index])))
        global_null = np.max(layer_null[text_index], axis=0)
        pvalue = float((1 + np.sum(global_null >= observed_max)) / 127.0)
        median_bos = float(np.median(bos_percentile[text_index]))
        text_reports[text] = {
            "global_tail_max_abs_correlation": observed_max,
            "search_wide_circular_shift_p": pvalue,
            "no_global_counter": bool(pvalue > 0.05),
            "median_bos_displacement_percentile": median_bos,
            "bos_transient_present": bool(median_bos >= 0.95),
            "largest_all_token_linear_correlation": float(
                np.max(np.abs(full_correlation[text_index]))
            ),
        }
    primary = text_reports["06_random"]
    prediction_passed = bool(primary["no_global_counter"] and primary["bos_transient_present"])
    report = {
        "schema_version": 1,
        "kind": "round5_lf3_absolute_position_counter",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "primary_text": "06_random",
        "control_text": "01_prose_en",
        "registered_prediction_passed": prediction_passed,
        "by_text": text_reports,
        "position_explained_variance_spectrum": {
            text: [
                np.sort(np.square(full_correlation[text_index, layer]))[::-1].tolist()
                for layer in range(66)
            ]
            for text_index, text in enumerate(TEXTS)
        },
        "input_rvec_sha256": input_hashes,
        "provenance": provenance(Path(__file__)),
    }
    atomic_npz(
        dump_path,
        full_linear_correlation=full_correlation,
        tail_linear_log_correlation=tail_correlation,
        layerwise_shift_null_maximum=layer_null,
        block_displacement=block_displacement,
        bos_displacement_percentile=bos_percentile,
    )
    report["dump_sha256"] = sha256_file(dump_path)
    atomic_json(report_path, report)
    results = [
        "# LF3 — absolute-position counter",
        "",
        "**Status: answered from the certified corrected capture; independent re-derivation pending.**",
        "",
        f"- Registered prediction passed: **{str(prediction_passed).lower()}**.",
        f"- Random tail: max |r| `{primary['global_tail_max_abs_correlation']:.6g}`, search-wide circular-shift p `{primary['search_wide_circular_shift_p']:.6g}`; no global counter = **{str(primary['no_global_counter']).lower()}**.",
        f"- Random BOS localization: median first-block displacement percentile `{primary['median_bos_displacement_percentile']:.6g}`; transient present = **{str(primary['bos_transient_present']).lower()}**.",
        f"- Prose control tail p `{text_reports['01_prose_en']['search_wide_circular_shift_p']:.6g}`, BOS percentile `{text_reports['01_prose_en']['median_bos_displacement_percentile']:.6g}`.",
        "",
        "Full correlations, position-explained spectra, search-wide nulls, controls, and source hashes are in `lf3.json` and `lf3_arrays.npz`.",
        "",
    ]
    results_path.write_text("\n".join(results), encoding="utf-8", newline="\n")
    print(json.dumps({"prediction_passed": prediction_passed, "by_text": text_reports}, indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(8192, 8))
    matrix[:, 0] += np.arange(8192) * 0.01
    correlation = correlations_with_vector(matrix, np.arange(8192))
    if correlation[0] < 0.99:
        raise AssertionError(correlation)
    tail = matrix[64:].reshape(127, 64, 8).mean(axis=1)
    observed, null = tail_test(tail)
    if observed.shape != (2, 8) or null.shape != (126,) or not np.isfinite(null).all():
        raise AssertionError((observed.shape, null.shape))
    matrix[:64, 1:] += 100
    displacement, percentile = block_displacement_and_bos_rank(matrix)
    if displacement.shape != (128,) or percentile < 0.99:
        raise AssertionError((displacement, percentile))
    print("round5_lf3_absolute_counter self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        run(args.out)


if __name__ == "__main__":
    main()
