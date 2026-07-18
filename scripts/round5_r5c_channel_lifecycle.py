"""Registered R5-C channel-4786/3290 lifecycle on certified D4 states."""

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
    TEXTS,
    artifact_index,
    atomic_json,
    atomic_npz,
    bf16_words_to_float32,
    deterministic_seed,
    median_absolute_deviation,
    provenance,
    refuse_existing,
    require_certified_capture,
    sha256_file,
    self_test_common,
)


CHANNELS = [4786, 3290]
LAYERS = list(range(66))
STAT_NAMES = [
    "mean",
    "median",
    "rms",
    "mad",
    "abs_q90",
    "abs_q95",
    "abs_q99",
    "abs_q999",
    "abs_max",
    "hidden_variance_share",
    "coverage_abs_gt_30000",
]
DEFAULT_OUT = ROOT / "analysis" / "round5" / "r5c_channel_lifecycle"


def state_path(layer: int, text: str) -> Path:
    return CAPTURE / "states" / f"hidden_L{layer:02d}_{text}.npy"


def all_coordinate_variance(path: Path, chunk_rows: int = 256) -> float:
    words = np.load(path, mmap_mode="r")
    if words.shape != (8192, 6144) or words.dtype != np.uint16:
        raise RuntimeError(f"invalid state payload: {path}")
    coordinate_sum = np.zeros(6144, dtype=np.float64)
    square_sum = 0.0
    count = 0
    for start in range(0, 8192, chunk_rows):
        block = bf16_words_to_float32(words[start : start + chunk_rows])
        coordinate_sum += block.sum(axis=0, dtype=np.float64)
        square_sum += float(np.square(block, dtype=np.float64).sum(dtype=np.float64))
        count += block.shape[0]
    mean = coordinate_sum / count
    total = square_sum / count - float(mean @ mean)
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(f"invalid total hidden variance: {path}: {total}")
    return total


def channel_statistics(values: np.ndarray, total_variance: float) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    absolute = np.abs(x)
    median = float(np.median(x))
    return [
        float(np.mean(x)),
        median,
        float(np.sqrt(np.mean(x * x))),
        float(np.median(np.abs(x - median))),
        *[float(value) for value in np.quantile(absolute, [0.90, 0.95, 0.99, 0.999])],
        float(np.max(absolute)),
        float(np.var(x) / total_variance),
        float(np.mean(absolute > 30000.0)),
    ]


def broadcast_onsets(coverage: np.ndarray) -> dict[str, Any]:
    # coverage: layer, text, channel
    result: dict[str, Any] = {}
    for channel_index, channel in enumerate(CHANNELS):
        candidates = []
        onset = None
        for layer in range(23, 29):
            sustained = []
            for text_index, text in enumerate(TEXTS):
                trajectory = coverage[:, text_index, channel_index]
                changes = np.diff(trajectory)
                passed = bool(np.all(changes[layer - 1 : layer + 2] > 0))
                sustained.append(passed)
            row = {
                "layer": layer,
                "supporting_texts": [text for text, passed in zip(TEXTS, sustained) if passed],
                "support_count": int(sum(sustained)),
                "passes_four_of_six": bool(sum(sustained) >= 4),
            }
            candidates.append(row)
            if onset is None and row["passes_four_of_six"]:
                onset = layer
        result[str(channel)] = {
            "onset_layer": onset,
            "present": onset is not None,
            "candidates": candidates,
        }
    result["both_channels_present"] = bool(
        all(result[str(channel)]["present"] for channel in CHANNELS)
    )
    return result


def handoff_bootstrap(block_means: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    # block_means: state(35..44), text, channel, block
    replicates = 5000
    interval = np.empty((len(TEXTS), 9, 2, 2), dtype=np.float64)
    by_text: dict[str, Any] = {}
    for text_index, text in enumerate(TEXTS):
        rng = np.random.default_rng(deterministic_seed(f"r5c-handoff:{text}"))
        sampled_changes = np.empty((replicates, 9, 2), dtype=np.float64)
        sharp = np.zeros(replicates, dtype=bool)
        for start in range(0, replicates, 250):
            stop = min(start + 250, replicates)
            indices = rng.integers(0, 32, size=(stop - start, 32))
            # state, channel, replicate
            means = np.empty((10, 2, stop - start), dtype=np.float64)
            source = block_means[:, text_index]
            for state_index in range(10):
                for channel_index in range(2):
                    means[state_index, channel_index] = source[
                        state_index, channel_index, indices
                    ].mean(axis=1)
            changes = np.diff(means, axis=0).transpose(2, 0, 1)
            sampled_changes[start:stop] = changes
            target = changes[:, 4, :]
            other_max = np.max(
                np.abs(np.concatenate([changes[:, :4, :], changes[:, 5:, :]], axis=1)),
                axis=1,
            )
            sharp[start:stop] = np.all(np.abs(target) > other_max, axis=1) & (
                target[:, 0] * target[:, 1] < 0
            )
        interval[text_index] = np.quantile(sampled_changes, [0.025, 0.975], axis=0).transpose(1, 2, 0)
        fraction = float(np.mean(sharp))
        by_text[text] = {
            "bootstrap_sharp_fraction": fraction,
            "supports_sharp_at_95pct": bool(fraction >= 0.95),
        }
    support_count = sum(row["supports_sharp_at_95pct"] for row in by_text.values())
    report = {
        "statistic": "signed_mean_activation",
        "bootstrap_replicates": replicates,
        "by_text": by_text,
        "support_count": int(support_count),
        "verdict": "sharp" if support_count >= 5 else "gradual/mixed",
    }
    return report, interval


def run(out: Path) -> None:
    report_path = out / "lifecycle.json"
    dump_path = out / "lifecycle_arrays.npz"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, dump_path, results_path)
    _, manifest = require_certified_capture()
    manifest_records = artifact_index(manifest)

    stats = np.empty((66, len(TEXTS), 2, len(STAT_NAMES)), dtype=np.float64)
    block_means = np.empty((10, len(TEXTS), 2, 32), dtype=np.float64)
    input_hashes: dict[str, str] = {}
    for layer in LAYERS:
        for text_index, text in enumerate(TEXTS):
            path = state_path(layer, text)
            relative = path.relative_to(CAPTURE).as_posix()
            record = manifest_records.get(relative)
            if record is None or record.get("kind") != "residual_hidden_state":
                raise RuntimeError(f"state is not bound by manifest: {relative}")
            input_hashes[relative] = record["sha256"]
            words = np.load(path, mmap_mode="r")
            channel_values = bf16_words_to_float32(words[:, CHANNELS]).astype(np.float64)
            total_variance = all_coordinate_variance(path)
            for channel_index in range(2):
                stats[layer, text_index, channel_index] = channel_statistics(
                    channel_values[:, channel_index], total_variance
                )
            if 35 <= layer <= 44:
                block_means[layer - 35, text_index] = channel_values.reshape(32, 256, 2).mean(axis=1).T
        print(f"R5-C lifecycle L{layer:02d}/65", flush=True)

    stat_index = {name: index for index, name in enumerate(STAT_NAMES)}
    coverage = stats[..., stat_index["coverage_abs_gt_30000"]]
    broadcast = broadcast_onsets(coverage)
    handoff, handoff_interval = handoff_bootstrap(block_means)
    cross_text = {
        "range": np.ptp(stats, axis=1),
        "mad": median_absolute_deviation(stats, axis=1),
    }

    summary = {
        "schema_version": 1,
        "kind": "round5_r5c_channel_lifecycle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "channels_zero_indexed": CHANNELS,
        "texts": TEXTS,
        "layers": LAYERS,
        "statistics": STAT_NAMES,
        "broadcast_onset": broadcast,
        "handoff_L39_to_L40": handoff,
        "registered_verdicts": {
            "broadcast_both_channels": broadcast["both_channels_present"],
            "handoff": handoff["verdict"],
        },
        "per_layer_text_channel": stats.tolist(),
        "cross_text_range_by_layer_channel_stat": cross_text["range"].tolist(),
        "cross_text_mad_by_layer_channel_stat": cross_text["mad"].tolist(),
        "input_state_sha256": input_hashes,
        "provenance": provenance(Path(__file__)),
    }
    atomic_npz(
        dump_path,
        stats=stats,
        block_means=block_means,
        handoff_change_ci95=handoff_interval,
        cross_text_range=cross_text["range"],
        cross_text_mad=cross_text["mad"],
        channels=np.asarray(CHANNELS),
    )
    summary["dump_sha256"] = sha256_file(dump_path)
    atomic_json(report_path, summary)

    lines = [
        "# R5-C channel 4786/3290 lifecycle",
        "",
        "**Status: answered from the certified corrected D4 capture; independent re-derivation pending.**",
        "",
        f"- Channel 4786 broadcast onset: `{broadcast['4786']['onset_layer']}`.",
        f"- Channel 3290 broadcast onset: `{broadcast['3290']['onset_layer']}`.",
        f"- Both registered onsets present: **{str(broadcast['both_channels_present']).lower()}**.",
        f"- L39/40 registered handoff classification: **{handoff['verdict']}** ({handoff['support_count']}/6 texts support sharp at the frozen 95% bootstrap rule).",
        "",
        "Full per-text trajectories, cross-text dispersion, source hashes, and bootstrap intervals are in `lifecycle.json` and `lifecycle_arrays.npz`.",
        "",
    ]
    results_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(summary["registered_verdicts"], indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    coverage = np.zeros((66, 6, 2), dtype=np.float64)
    for layer in range(23, 31):
        coverage[layer:, :4, 0] += 1.0
    result = broadcast_onsets(coverage)
    if result["4786"]["onset_layer"] != 23:
        raise AssertionError(result)
    x = np.arange(8192, dtype=np.float64)
    stats = channel_statistics(x, float(np.var(x) * 3))
    if len(stats) != len(STAT_NAMES) or not np.isclose(stats[-2], 1 / 3):
        raise AssertionError(stats)
    print("round5_r5c_channel_lifecycle self-test passed")


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
