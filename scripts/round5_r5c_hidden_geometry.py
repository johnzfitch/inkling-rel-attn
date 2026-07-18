"""Registered broad R5-C hidden-state geometry on certified D4 states."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from round5_science_common import (
    CAPTURE,
    GLOBAL_LAYERS,
    ROOT,
    TEXTS,
    artifact_index,
    atomic_json,
    atomic_npz,
    bf16_words_to_float32,
    provenance,
    refuse_existing,
    require_certified_capture,
    self_test_common,
    sha256_file,
)


DEFAULT_OUT = ROOT / "analysis" / "round5" / "r5c_hidden_geometry"
BASIS_PATH = ROOT / "analysis" / "subspace_anatomy" / "common_bases_top4.npz"
SAMPLE_INDICES = np.asarray(
    [block * 256 + offset for block in range(32) for offset in (31, 95, 159, 223)],
    dtype=np.int64,
)


def state_path(state_name: str, text: str) -> Path:
    return CAPTURE / "states" / f"{state_name}_{text}.npy"


def load_state(path: Path) -> np.ndarray:
    words = np.load(path, mmap_mode="r")
    if words.shape != (8192, 6144) or words.dtype != np.uint16:
        raise RuntimeError(f"invalid state payload: {path}")
    return bf16_words_to_float32(words)


def full_state_metrics(
    input_state: np.ndarray,
    output_state: np.ndarray,
    carrier: np.ndarray,
    chunk_rows: int = 512,
) -> dict[str, Any]:
    norms = np.empty(8192, dtype=np.float64)
    rotations = np.empty(8192, dtype=np.float64)
    coordinate_sum = np.zeros(6144, dtype=np.float64)
    square_sum = 0.0
    carrier_sum = 0.0
    carrier_square_sum = 0.0
    carrier32 = np.asarray(carrier, dtype=np.float32)
    for start in range(0, 8192, chunk_rows):
        stop = start + chunk_rows
        x = output_state[start:stop]
        y = input_state[start:stop]
        xx = np.sum(x * x, axis=1, dtype=np.float64)
        yy = np.sum(y * y, axis=1, dtype=np.float64)
        xy = np.sum(x * y, axis=1, dtype=np.float64)
        norms[start:stop] = np.sqrt(xx)
        rotations[start:stop] = 1.0 - xy / np.sqrt(xx * yy)
        coordinate_sum += x.sum(axis=0, dtype=np.float64)
        square_sum += float(xx.sum(dtype=np.float64))
        projected = x @ carrier32
        carrier_sum += float(projected.sum(dtype=np.float64))
        carrier_square_sum += float(np.sum(projected * projected, dtype=np.float64))
    mean = coordinate_sum / 8192
    total_variance = square_sum / 8192 - float(mean @ mean)
    carrier_variance = carrier_square_sum / 8192 - (carrier_sum / 8192) ** 2
    if total_variance <= 0 or carrier_variance < -1e-8:
        raise RuntimeError((total_variance, carrier_variance))
    bos_mean = output_state[:64].mean(axis=0, dtype=np.float64)
    tail_mean = output_state[64:].mean(axis=0, dtype=np.float64)
    return {
        "norm_mean": float(norms.mean()),
        "norm_median": float(np.median(norms)),
        "rotation_mean": float(rotations.mean()),
        "rotation_median": float(np.median(rotations)),
        "total_hidden_variance": float(total_variance),
        "carrier_variance": float(max(carrier_variance, 0.0)),
        "carrier_variance_share": float(max(carrier_variance, 0.0) / total_variance),
        "bos_tail_mean_displacement": float(np.linalg.norm(bos_mean - tail_mean)),
    }


def sample_geometry(output_state: np.ndarray, carrier: np.ndarray) -> dict[str, float]:
    sample = np.asarray(output_state[SAMPLE_INDICES], dtype=np.float64)
    sample -= sample.mean(axis=0, keepdims=True)
    gram = sample @ sample.T
    gram = (gram + gram.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    square = float(eigenvalues @ eigenvalues)
    if total <= 0 or square <= 0:
        raise RuntimeError("degenerate sampled state covariance")
    participation_ratio = total * total / square

    squared_distance = (
        np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram
    )
    squared_distance = np.maximum(squared_distance, 0.0)
    np.fill_diagonal(squared_distance, np.inf)
    nearest = np.partition(squared_distance, 9, axis=1)[:, :10]
    nearest.sort(axis=1)
    distances = np.sqrt(nearest)
    if np.any(distances <= 0) or not np.isfinite(distances).all():
        raise RuntimeError("zero or non-finite k-NN sample distance")
    tk = distances[:, 9]
    denominators = np.mean(np.log(tk[:, None] / distances[:, :9]), axis=1)
    if np.any(denominators <= 0):
        raise RuntimeError("degenerate Levina-Bickel denominator")
    knn_dimension = float(np.median(1.0 / denominators))

    order = np.argsort(eigenvalues)[::-1]
    positive = order[eigenvalues[order] > eigenvalues[order[0]] * 1e-12]
    top = positive[:32]
    sample_carrier = sample @ np.asarray(carrier, dtype=np.float64)
    coefficients = (eigenvectors[:, top].T @ sample_carrier) / np.sqrt(eigenvalues[top])
    cos_squared = float(np.clip(coefficients @ coefficients, 0.0, 1.0))
    angle_degrees = float(np.degrees(np.arccos(np.sqrt(cos_squared))))
    return {
        "participation_ratio": float(participation_ratio),
        "knn_dimension_k10": knn_dimension,
        "carrier_angle_to_top32_degrees": angle_degrees,
        "carrier_top32_projection_share": cos_squared,
    }


def flip_discontinuity(values: np.ndarray, name: str) -> dict[str, Any]:
    median_curve = np.median(values, axis=1)
    rows = []
    for layer in range(1, 66):
        if (layer in GLOBAL_LAYERS) != ((layer - 1) in GLOBAL_LAYERS):
            continue
        per_text_change = values[layer] - values[layer - 1]
        change = float(median_curve[layer] - median_curve[layer - 1])
        direction = np.sign(change)
        agreement = int(np.sum(np.sign(per_text_change) == direction)) if direction else 0
        rows.append(
            {
                "destination_layer": layer,
                "change_in_six_text_median": change,
                "absolute_change": abs(change),
                "direction_agreement_count": agreement,
                "inside_flip_band": 13 <= layer <= 28,
            }
        )
    rows.sort(key=lambda row: row["absolute_change"], reverse=True)
    maximum = rows[0]
    unique = len(rows) == 1 or maximum["absolute_change"] > rows[1]["absolute_change"]
    detected = bool(
        unique and maximum["inside_flip_band"] and maximum["direction_agreement_count"] >= 4
    )
    return {
        "metric": name,
        "unique_maximum": unique,
        "detected": detected,
        "maximum": maximum,
        "ranked_same_scope_changes": rows,
    }


def exact_global_rotation_test(rotation: np.ndarray) -> dict[str, Any]:
    globals_sorted = sorted(GLOBAL_LAYERS)
    locals_sorted = [layer for layer in range(66) if layer not in GLOBAL_LAYERS]
    differences = np.median(rotation[globals_sorted], axis=0) - np.median(
        rotation[locals_sorted], axis=0
    )
    observed = float(np.mean(differences))
    null = np.asarray(
        [np.mean(differences * np.asarray(signs)) for signs in itertools.product((-1, 1), repeat=6)],
        dtype=np.float64,
    )
    pvalue = float(np.mean(null >= observed))
    neighbors: dict[str, Any] = {}
    for layer in globals_sorted:
        local_neighbors = [
            candidate
            for candidate in (layer - 1, layer + 1)
            if 0 <= candidate < 66 and candidate not in GLOBAL_LAYERS
        ]
        contrasts = rotation[layer] - np.mean(rotation[local_neighbors], axis=0)
        neighbors[str(layer)] = {
            "local_neighbors": local_neighbors,
            "per_text_contrast": contrasts.tolist(),
            "median_contrast": float(np.median(contrasts)),
        }
    return {
        "per_text_global_minus_local_median": differences.tolist(),
        "observed_mean_difference": observed,
        "exact_one_sided_sign_flip_p": pvalue,
        "passed": bool(observed > 0 and pvalue <= 0.05),
        "neighbor_local_diagnostics": neighbors,
    }


def run(out: Path) -> None:
    report_path = out / "geometry.json"
    dump_path = out / "geometry_arrays.npz"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, dump_path, results_path)
    _, manifest = require_certified_capture()
    records = artifact_index(manifest)
    with np.load(BASIS_PATH) as basis_dump:
        basis = np.asarray(basis_dump["basis"][:, 0], dtype=np.float32)
        live_share = np.asarray(basis_dump["live_share"][:, 0], dtype=np.float64)
    basis_norm = np.linalg.norm(basis, axis=1)
    if not np.allclose(basis_norm, 1.0, rtol=0, atol=2e-6):
        raise RuntimeError("communal carrier basis is not unit normalized")
    basis = basis / basis_norm[:, None]
    norms_mean = np.empty((66, 6), dtype=np.float64)
    norms_median = np.empty((66, 6), dtype=np.float64)
    rotation_mean = np.empty((66, 6), dtype=np.float64)
    rotation_median = np.empty((66, 6), dtype=np.float64)
    total_variance = np.empty((66, 6), dtype=np.float64)
    carrier_variance = np.empty((66, 6), dtype=np.float64)
    carrier_share = np.empty((66, 6), dtype=np.float64)
    carrier_angle = np.empty((66, 6), dtype=np.float64)
    carrier_bulk_share = np.empty((66, 6), dtype=np.float64)
    participation_ratio = np.empty((66, 6), dtype=np.float64)
    knn_dimension = np.empty((66, 6), dtype=np.float64)
    bos_displacement = np.empty((66, 6), dtype=np.float64)
    input_hashes: dict[str, str] = {}

    for text_index, text in enumerate(TEXTS):
        input_name = "hidden_embed"
        input_path = state_path(input_name, text)
        input_state = load_state(input_path)
        input_hashes[input_path.relative_to(CAPTURE).as_posix()] = records[
            input_path.relative_to(CAPTURE).as_posix()
        ]["sha256"]
        for layer in range(66):
            output_path = state_path(f"hidden_L{layer:02d}", text)
            relative = output_path.relative_to(CAPTURE).as_posix()
            record = records.get(relative)
            if record is None or record.get("kind") != "residual_hidden_state":
                raise RuntimeError(f"state is not bound by manifest: {relative}")
            input_hashes[relative] = record["sha256"]
            output_state = load_state(output_path)
            full = full_state_metrics(input_state, output_state, basis[layer])
            sampled = sample_geometry(output_state, basis[layer])
            norms_mean[layer, text_index] = full["norm_mean"]
            norms_median[layer, text_index] = full["norm_median"]
            rotation_mean[layer, text_index] = full["rotation_mean"]
            rotation_median[layer, text_index] = full["rotation_median"]
            total_variance[layer, text_index] = full["total_hidden_variance"]
            carrier_variance[layer, text_index] = full["carrier_variance"]
            carrier_share[layer, text_index] = full["carrier_variance_share"]
            bos_displacement[layer, text_index] = full["bos_tail_mean_displacement"]
            participation_ratio[layer, text_index] = sampled["participation_ratio"]
            knn_dimension[layer, text_index] = sampled["knn_dimension_k10"]
            carrier_angle[layer, text_index] = sampled["carrier_angle_to_top32_degrees"]
            carrier_bulk_share[layer, text_index] = sampled["carrier_top32_projection_share"]
            input_state = output_state
            print(f"R5-C geometry {text} L{layer:02d}/65", flush=True)

    flip_pr = flip_discontinuity(participation_ratio, "participation_ratio")
    flip_rotation = flip_discontinuity(rotation_median, "median_rotation")
    global_rotation = exact_global_rotation_test(rotation_median)
    layer_median_carrier = np.median(carrier_share, axis=1)
    carrier_pass = bool(np.all(layer_median_carrier < 0.01))
    max_cell_index = np.unravel_index(np.argmax(carrier_share), carrier_share.shape)
    verdicts = {
        "carrier_hidden_variance_below_1pct_every_layer_median": carrier_pass,
        "flip_discontinuity_detected_by_either_metric": bool(
            flip_pr["detected"] or flip_rotation["detected"]
        ),
        "global_layers_rotate_more": global_rotation["passed"],
    }
    report = {
        "schema_version": 1,
        "kind": "round5_r5c_hidden_geometry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "texts": TEXTS,
        "sample_indices": SAMPLE_INDICES.tolist(),
        "registered_verdicts": verdicts,
        "carrier_prediction": {
            "layer_median_hidden_variance_share": layer_median_carrier.tolist(),
            "maximum_cell": {
                "layer": int(max_cell_index[0]),
                "text": TEXTS[int(max_cell_index[1])],
                "share": float(carrier_share[max_cell_index]),
            },
            "median_live_positional_read_share": float(np.median(live_share)),
            "per_layer_live_positional_read_share": live_share.tolist(),
        },
        "flip_band": {
            "participation_ratio": flip_pr,
            "median_rotation": flip_rotation,
        },
        "global_rotation": global_rotation,
        "arrays": {
            "norm_mean": norms_mean.tolist(),
            "norm_median": norms_median.tolist(),
            "rotation_mean": rotation_mean.tolist(),
            "rotation_median": rotation_median.tolist(),
            "total_hidden_variance": total_variance.tolist(),
            "carrier_variance": carrier_variance.tolist(),
            "carrier_variance_share": carrier_share.tolist(),
            "carrier_angle_to_top32_degrees": carrier_angle.tolist(),
            "carrier_top32_projection_share": carrier_bulk_share.tolist(),
            "participation_ratio": participation_ratio.tolist(),
            "knn_dimension_k10": knn_dimension.tolist(),
            "bos_tail_mean_displacement": bos_displacement.tolist(),
        },
        "input_state_sha256": input_hashes,
        "carrier_basis_sha256": sha256_file(BASIS_PATH),
        "provenance": provenance(Path(__file__)),
    }
    atomic_npz(
        dump_path,
        norm_mean=norms_mean,
        norm_median=norms_median,
        rotation_mean=rotation_mean,
        rotation_median=rotation_median,
        total_hidden_variance=total_variance,
        carrier_variance=carrier_variance,
        carrier_variance_share=carrier_share,
        carrier_angle_to_top32_degrees=carrier_angle,
        carrier_top32_projection_share=carrier_bulk_share,
        participation_ratio=participation_ratio,
        knn_dimension_k10=knn_dimension,
        bos_tail_mean_displacement=bos_displacement,
        sample_indices=SAMPLE_INDICES,
        live_positional_read_share=live_share,
    )
    report["dump_sha256"] = sha256_file(dump_path)
    atomic_json(report_path, report)
    results = [
        "# R5-C hidden-state geometry",
        "",
        "**Status: answered from the certified corrected D4 capture; independent re-derivation pending.**",
        "",
        f"- Narrow protected carrier (<1% at every layer median): **{str(carrier_pass).lower()}**; maximum cell `{carrier_share[max_cell_index]:.6g}` at L{max_cell_index[0]} / {TEXTS[max_cell_index[1]]}; median positional read-energy share `{np.median(live_share):.6g}`.",
        f"- Flip-band geometry discontinuity: **{str(verdicts['flip_discontinuity_detected_by_either_metric']).lower()}** (PR max at L{flip_pr['maximum']['destination_layer']}; rotation max at L{flip_rotation['maximum']['destination_layer']}).",
        f"- Global layers rotate more: **{str(global_rotation['passed']).lower()}** (mean paired contrast `{global_rotation['observed_mean_difference']:.6g}`, exact one-sided p `{global_rotation['exact_one_sided_sign_flip_p']:.6g}`).",
        "",
        "The full layer/text geometry, discontinuity rankings, controls, and input hashes are in `geometry.json` and `geometry_arrays.npz`.",
        "",
    ]
    results_path.write_text("\n".join(results), encoding="utf-8", newline="\n")
    print(json.dumps(verdicts, indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    rng = np.random.default_rng(0)
    state = rng.normal(size=(8192, 6144)).astype(np.float32)
    carrier = np.zeros(6144, dtype=np.float32)
    carrier[0] = 1.0
    metrics = full_state_metrics(state, state, carrier, chunk_rows=1024)
    if abs(metrics["rotation_median"]) > 1e-12:
        raise AssertionError(metrics["rotation_median"])
    geometry = sample_geometry(state, carrier)
    if not all(np.isfinite(list(geometry.values()))):
        raise AssertionError(geometry)
    values = np.tile(np.arange(66, dtype=np.float64)[:, None], (1, 6))
    values[20:] += 100
    flip = flip_discontinuity(values, "synthetic")
    if not flip["detected"] or flip["maximum"]["destination_layer"] != 20:
        raise AssertionError(flip)
    rotation = np.zeros((66, 6), dtype=np.float64)
    rotation[sorted(GLOBAL_LAYERS)] = 1
    test = exact_global_rotation_test(rotation)
    if not test["passed"] or test["exact_one_sided_sign_flip_p"] != 1 / 64:
        raise AssertionError(test)
    print("round5_r5c_hidden_geometry self-test passed")


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
