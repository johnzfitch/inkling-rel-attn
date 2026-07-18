"""Registered LF9 long-range attention bandwidth budget from Tier-2 meters."""

from __future__ import annotations

import argparse
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
    provenance,
    refuse_existing,
    require_certified_capture,
    self_test_common,
    sha256_file,
)


DEFAULT_OUT = ROOT / "analysis" / "round5" / "lf9"
CONDITIONS = ["with_bias", "without_bias"]


def distribution_metrics(mass: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(mass, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 64 or values.shape[1] < 258:
        raise ValueError(values.shape)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("invalid distance mass")
    total = values.sum(axis=1, dtype=np.float64)
    if np.any(total <= 0):
        raise RuntimeError("zero total attention mass")
    probability = values / total[:, None]
    far_share = probability[:, 257:].sum(axis=1, dtype=np.float64)
    entropy = -np.sum(
        np.where(probability > 0, probability * np.log(np.maximum(probability, 1e-300)), 0.0),
        axis=1,
        dtype=np.float64,
    )
    far_mass = values[:, 257:]
    far_total = far_mass.sum(axis=1, dtype=np.float64)
    if np.any(far_total <= 0):
        raise RuntimeError("zero far-field attention mass")
    far_probability = far_mass / far_total[:, None]
    far_entropy = -np.sum(
        np.where(
            far_probability > 0,
            far_probability * np.log(np.maximum(far_probability, 1e-300)),
            0.0,
        ),
        axis=1,
        dtype=np.float64,
    )
    return far_share, entropy, np.exp(entropy), far_entropy, np.exp(far_entropy)


def unique_extreme(values: dict[int, float], maximum: bool) -> tuple[int, bool]:
    ordered = sorted(values.items(), key=lambda pair: pair[1], reverse=maximum)
    return ordered[0][0], bool(len(ordered) == 1 or ordered[0][1] != ordered[1][1])


def run(out: Path) -> None:
    report_path = out / "lf9.json"
    dump_path = out / "lf9_arrays.npz"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, dump_path, results_path)
    _, manifest = require_certified_capture()
    records = artifact_index(manifest)
    shape = (66, 6, 2, 64)
    far_share = np.empty(shape, dtype=np.float64)
    entropy = np.empty(shape, dtype=np.float64)
    effective_count = np.empty(shape, dtype=np.float64)
    far_entropy = np.empty(shape, dtype=np.float64)
    far_effective_count = np.empty(shape, dtype=np.float64)
    input_hashes: dict[str, str] = {}

    for layer in range(66):
        for text_index, text in enumerate(TEXTS):
            path = CAPTURE / "meters" / f"layer{layer:02d}_{text}_s8192.npz"
            relative = path.relative_to(CAPTURE).as_posix()
            record = records.get(relative)
            if record is None or record.get("kind") != "tier2_distance_meter":
                raise RuntimeError(f"meter is not bound by manifest: {relative}")
            input_hashes[relative] = record["sha256"]
            with np.load(path, allow_pickle=False) as meter:
                sources = [meter["mass_with"], meter["mass_without"]]
                for condition_index, source in enumerate(sources):
                    metrics = distribution_metrics(source)
                    far_share[layer, text_index, condition_index] = metrics[0]
                    entropy[layer, text_index, condition_index] = metrics[1]
                    effective_count[layer, text_index, condition_index] = metrics[2]
                    far_entropy[layer, text_index, condition_index] = metrics[3]
                    far_effective_count[layer, text_index, condition_index] = metrics[4]
        print(f"LF9 L{layer:02d}/65", flush=True)

    head_median = np.median(far_share, axis=3)
    aggregate_far_share = np.median(head_median, axis=1)
    paired_effect = np.median(
        np.median(far_share[:, :, 0] - far_share[:, :, 1], axis=2), axis=1
    )
    aggregate_entropy = np.median(np.median(entropy, axis=3), axis=1)
    aggregate_effective = np.median(np.median(effective_count, axis=3), axis=1)
    aggregate_far_entropy = np.median(np.median(far_entropy, axis=3), axis=1)
    aggregate_far_effective = np.median(np.median(far_effective_count, axis=3), axis=1)

    global_with = {layer: float(aggregate_far_share[layer, 0]) for layer in GLOBAL_LAYERS}
    peak_layer, peak_unique = unique_extreme(global_with, maximum=True)
    minimum_layer, minimum_unique = unique_extreme(global_with, maximum=False)
    depth_passed = bool(
        peak_unique and 23 <= peak_layer <= 47 and minimum_unique and minimum_layer == 65
    )
    positive_layers = [23, 29, 35, 41, 47]
    negative_layers = [5, 65]
    direction_passed = bool(
        all(paired_effect[layer] > 0 for layer in positive_layers)
        and all(paired_effect[layer] < 0 for layer in negative_layers)
    )
    verdicts = {
        "mid_depth_global_peak_and_L65_unique_minimum": depth_passed,
        "registered_bias_directions": direction_passed,
        "all_registered_predictions": bool(depth_passed and direction_passed),
    }
    report = {
        "schema_version": 1,
        "kind": "round5_lf9_bandwidth_budget",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "registered_verdicts": verdicts,
        "depth_extrema": {
            "global_with_bias_far_share": {str(k): v for k, v in sorted(global_with.items())},
            "peak_layer": peak_layer,
            "peak_unique": peak_unique,
            "minimum_layer": minimum_layer,
            "minimum_unique": minimum_unique,
        },
        "bias_far_share_effect": {
            str(layer): float(paired_effect[layer]) for layer in range(66)
        },
        "aggregate_head_then_text_median": {
            "far_share": aggregate_far_share.tolist(),
            "distance_entropy_nats": aggregate_entropy.tolist(),
            "distance_effective_count": aggregate_effective.tolist(),
            "conditional_far_entropy_nats": aggregate_far_entropy.tolist(),
            "conditional_far_effective_count": aggregate_far_effective.tolist(),
        },
        "input_meter_sha256": input_hashes,
        "provenance": provenance(Path(__file__)),
    }
    atomic_npz(
        dump_path,
        far_share=far_share,
        entropy=entropy,
        effective_count=effective_count,
        conditional_far_entropy=far_entropy,
        conditional_far_effective_count=far_effective_count,
        aggregate_far_share=aggregate_far_share,
        aggregate_entropy=aggregate_entropy,
        aggregate_effective_count=aggregate_effective,
        aggregate_conditional_far_entropy=aggregate_far_entropy,
        aggregate_conditional_far_effective_count=aggregate_far_effective,
        paired_bias_far_share_effect=paired_effect,
    )
    report["dump_sha256"] = sha256_file(dump_path)
    atomic_json(report_path, report)
    results = [
        "# LF9 — long-range bandwidth budget",
        "",
        "**Status: answered from the certified corrected Tier-2 meters; independent re-derivation pending.**",
        "",
        f"- Mid-depth global peak plus unique L65 minimum: **{str(depth_passed).lower()}** (peak L{peak_layer}, minimum L{minimum_layer}).",
        f"- Registered bias directions: **{str(direction_passed).lower()}**.",
        "- Paired with-minus-without far-share effects at registered layers: "
        + ", ".join(f"L{layer} `{paired_effect[layer]:+.6g}`" for layer in positive_layers + negative_layers)
        + ".",
        "",
        "Full head/text/layer shares, entropies, effective counts, contrasts, and source hashes are in `lf9.json` and `lf9_arrays.npz`.",
        "",
    ]
    results_path.write_text("\n".join(results), encoding="utf-8", newline="\n")
    print(json.dumps(verdicts, indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    mass = np.ones((64, 512), dtype=np.float64)
    far, entropy, effective, far_h, far_effective = distribution_metrics(mass)
    expected_far = (512 - 257) / 512
    if not np.allclose(far, expected_far):
        raise AssertionError(far)
    if not np.allclose(entropy, np.log(512)) or not np.allclose(effective, 512):
        raise AssertionError((entropy, effective))
    if not np.allclose(far_h, np.log(255)) or not np.allclose(far_effective, 255):
        raise AssertionError((far_h, far_effective))
    if unique_extreme({1: 2.0, 2: 1.0}, True) != (1, True):
        raise AssertionError("unique extreme failed")
    if unique_extreme({1: 2.0, 2: 2.0}, True)[1]:
        raise AssertionError("tie handling failed")
    print("round5_lf9_bandwidth_budget self-test passed")


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
