"""Registered R5-B cross-text dispersion of realized live bias curves."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
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
    provenance,
    refuse_existing,
    require_certified_capture,
    self_test_common,
    sha256_file,
)


DEFAULT_OUT = ROOT / "analysis" / "round5" / "r5b"
WEIGHTS = ROOT / "weights"
TEXT_PAIRS = list(combinations(range(6), 2))


def normalized_curve_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    numerator = np.linalg.norm(left - right, axis=-1)
    denominator = 0.5 * (
        np.linalg.norm(left, axis=-1) + np.linalg.norm(right, axis=-1)
    )
    if np.any(denominator <= 0):
        raise RuntimeError("zero live-bias curve norm")
    return numerator / denominator


def unique_arg_extreme(values: np.ndarray, maximum: bool) -> tuple[int, bool]:
    order = np.argsort(values)
    if maximum:
        order = order[::-1]
    winner = int(order[0])
    unique = bool(values[order[0]] != values[order[1]])
    return winner, unique


def run(out: Path) -> None:
    report_path = out / "r5b.json"
    dump_path = out / "r5b_arrays.npz"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, dump_path, results_path)
    _, manifest = require_certified_capture()
    records = artifact_index(manifest)
    mean_r = np.empty((66, 6, 64, 16), dtype=np.float64)
    pair_distance = np.empty((66, 15, 64), dtype=np.float64)
    text_centrality = np.empty((66, 6, 64), dtype=np.float64)
    layer_dispersion = np.empty(66, dtype=np.float64)
    input_hashes: dict[str, str] = {}
    projection_hashes: dict[str, str] = {}

    for layer in range(66):
        extent = 1024 if layer in GLOBAL_LAYERS else 512
        projection_path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        projection = np.asarray(np.load(projection_path), dtype=np.float64)
        if projection.shape != (16, extent) or not np.isfinite(projection).all():
            raise RuntimeError(f"invalid projection: {projection_path}")
        projection_hashes[projection_path.relative_to(ROOT).as_posix()] = sha256_file(projection_path)
        curves = np.empty((6, 64, extent), dtype=np.float64)
        for text_index, text in enumerate(TEXTS):
            path = CAPTURE / "replay" / f"rvec_L{layer:02d}_{text}.npy"
            relative = path.relative_to(CAPTURE).as_posix()
            record = records.get(relative)
            if record is None or record.get("kind") != "rvec":
                raise RuntimeError(f"r-vector is not bound by manifest: {relative}")
            input_hashes[relative] = record["sha256"]
            rvec = np.load(path, mmap_mode="r")
            if rvec.shape != (8192, 64, 16) or rvec.dtype != np.float16:
                raise RuntimeError(f"invalid r-vector: {path}")
            mean_r[layer, text_index] = np.asarray(rvec, dtype=np.float64).mean(axis=0)
            curves[text_index] = mean_r[layer, text_index] @ projection
        for pair_index, (left, right) in enumerate(TEXT_PAIRS):
            pair_distance[layer, pair_index] = normalized_curve_distance(
                curves[left], curves[right]
            )
        layer_dispersion[layer] = float(np.median(pair_distance[layer]))
        layer_mean_curve = curves.mean(axis=0)
        for text_index in range(6):
            text_centrality[layer, text_index] = normalized_curve_distance(
                curves[text_index], layer_mean_curve
            )
        print(f"R5-B L{layer:02d}/65 dispersion={layer_dispersion[layer]:.6g}", flush=True)

    maximum_layer, maximum_unique = unique_arg_extreme(layer_dispersion, maximum=True)
    early_median = float(np.median(layer_dispersion[0:6]))
    mid_median = float(np.median(layer_dispersion[23:48]))
    mid_globals = [23, 29, 35, 41, 47]
    l65_below_mid_globals = bool(
        all(layer_dispersion[65] < layer_dispersion[layer] for layer in mid_globals)
    )
    depth_passed = bool(
        maximum_unique
        and 23 <= maximum_layer <= 47
        and mid_median > early_median
        and l65_below_mid_globals
    )
    centrality_summary = np.median(text_centrality, axis=(0, 2))
    nearest_index, nearest_unique = unique_arg_extreme(centrality_summary, maximum=False)
    farthest_index, farthest_unique = unique_arg_extreme(centrality_summary, maximum=True)
    ordering_passed = bool(
        nearest_unique
        and farthest_unique
        and TEXTS[nearest_index] == "06_random"
        and TEXTS[farthest_index] == "02_code"
    )
    verdicts = {
        "registered_depth_profile": depth_passed,
        "random_nearest_code_farthest": ordering_passed,
        "all_registered_predictions": bool(depth_passed and ordering_passed),
    }
    report = {
        "schema_version": 1,
        "kind": "round5_r5b_prompt_dependence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "registered_verdicts": verdicts,
        "depth_profile": {
            "layer_dispersion": layer_dispersion.tolist(),
            "unique_maximum_layer": maximum_layer,
            "maximum_unique": maximum_unique,
            "early_L0_5_median": early_median,
            "mid_L23_47_median": mid_median,
            "L65": float(layer_dispersion[65]),
            "L65_below_every_mid_global": l65_below_mid_globals,
        },
        "text_centrality": {
            "median_over_layers_and_heads": {
                text: float(centrality_summary[index]) for index, text in enumerate(TEXTS)
            },
            "nearest": TEXTS[nearest_index],
            "nearest_unique": nearest_unique,
            "farthest": TEXTS[farthest_index],
            "farthest_unique": farthest_unique,
        },
        "text_pairs": [[TEXTS[left], TEXTS[right]] for left, right in TEXT_PAIRS],
        "input_rvec_sha256": input_hashes,
        "projection_sha256": projection_hashes,
        "provenance": provenance(Path(__file__)),
    }
    atomic_npz(
        dump_path,
        mean_r=mean_r,
        pairwise_normalized_curve_distance=pair_distance,
        text_distance_to_layer_mean=text_centrality,
        layer_dispersion=layer_dispersion,
        text_centrality_summary=centrality_summary,
    )
    report["dump_sha256"] = sha256_file(dump_path)
    atomic_json(report_path, report)
    results = [
        "# R5-B — prompt dependence of realized transport",
        "",
        "**Status: answered from the certified corrected r-vectors; independent re-derivation pending.**",
        "",
        f"- Registered depth profile: **{str(depth_passed).lower()}**; unique maximum L{maximum_layer}, early median `{early_median:.6g}`, mid median `{mid_median:.6g}`, L65 `{layer_dispersion[65]:.6g}`.",
        f"- Random nearest / code farthest: **{str(ordering_passed).lower()}**; observed nearest `{TEXTS[nearest_index]}`, farthest `{TEXTS[farthest_index]}`.",
        "- Median centrality: "
        + ", ".join(f"{text} `{centrality_summary[index]:.6g}`" for index, text in enumerate(TEXTS))
        + ".",
        "",
        "Full headwise pair distances, mean r-vectors, layer profile, controls, and source hashes are in `r5b.json` and `r5b_arrays.npz`.",
        "",
    ]
    results_path.write_text("\n".join(results), encoding="utf-8", newline="\n")
    print(json.dumps(verdicts, indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    a = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    b = np.asarray([[0.0, 1.0], [0.0, 1.0]])
    distance = normalized_curve_distance(a, b)
    expected = np.asarray([np.sqrt(2.0), 2.0 / 3.0])
    if not np.allclose(distance, expected):
        raise AssertionError((distance, expected))
    if unique_arg_extreme(np.asarray([1.0, 3.0, 2.0]), True) != (1, True):
        raise AssertionError("maximum helper failed")
    if unique_arg_extreme(np.asarray([1.0, 1.0, 2.0]), False)[1]:
        raise AssertionError("tie helper failed")
    print("round5_r5b_prompt_dependence self-test passed")


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
