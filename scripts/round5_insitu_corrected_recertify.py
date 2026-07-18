"""Mechanical A6 re-certification of provisional in-situ Round 5 findings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu
from tokenizers import Tokenizer

from round5_science_common import (
    CAPTURE,
    GLOBAL_LAYERS,
    MANIFEST_SHA256,
    ROOT,
    TEXTS,
    artifact_index,
    atomic_json,
    provenance,
    refuse_existing,
    require_certified_capture,
    self_test_common,
    sha256_file,
)


DEFAULT_OUT = ROOT / "analysis" / "round5" / "insitu_corrected"
PARITY = ROOT / "analysis" / "round5" / "widened_capture" / "lf5_replay_parity.json"
CORPUS = ROOT / "corpus"
WEIGHTS = ROOT / "weights"
OLD_TIER2 = ROOT / "analysis" / "tier2" / "tier2_findings.json"
OLD_NEEDLES = ROOT / "analysis" / "needles" / "needle_results.json"
OLD_SUMMARY = ROOT / "analysis" / "revised_mechanisms" / "revised_mechanism_summary.json"
GLOBALS = sorted(GLOBAL_LAYERS)
ECHO_TEXTS = TEXTS[:3]


def require_parity() -> dict[str, Any]:
    report = json.loads(PARITY.read_text(encoding="utf-8"))
    if (
        report.get("kind") != "round5_lf5_parity"
        or report.get("backend") != "replay"
        or not report.get("passed")
        or report.get("input_manifest_sha256") != MANIFEST_SHA256
        or len(report.get("results", [])) != 66
        or not all(row.get("bitwise_equal") and row.get("passed") for row in report["results"])
    ):
        raise RuntimeError("LF5 replay parity is missing, failed, or stale")
    return report


def load_meter(layer: int, text: str) -> dict[str, np.ndarray]:
    path = CAPTURE / "meters" / f"layer{layer:02d}_{text}_s8192.npz"
    with np.load(path, allow_pickle=False) as meter:
        return {name: np.asarray(meter[name]) for name in meter.files if name != "meta"}


def seam_analysis(meters: dict[tuple[int, str], dict[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in GLOBALS:
        per_text = {}
        for text in TEXTS:
            meter = meters[(layer, text)]
            with_mass = meter["mean_mass_with"]
            without_mass = meter["mean_mass_without"]
            mean_bias = meter["mean_bias"]
            inside, outside = slice(1008, 1024), slice(1024, 1040)
            with_step = with_mass[:, inside].mean(1) - with_mass[:, outside].mean(1)
            without_step = without_mass[:, inside].mean(1) - without_mass[:, outside].mean(1)
            attributable = with_step - without_step
            per_text[text] = {
                "bias_attrib_step_mean": float(attributable.mean()),
                "without_step_mean": float(without_step.mean()),
                "heads_positive_frac": float(np.mean(attributable > 0)),
                "bias_in_mean": float(mean_bias[:, inside].mean()),
                "bias_out_max": float(np.max(np.abs(mean_bias[:, outside]))),
            }
        result[str(layer)] = per_text
    return result


def needle_setup() -> tuple[list[dict[str, Any]], dict[str, int]]:
    sidecar = json.loads((CORPUS / "05_needles.sidecar.json").read_text(encoding="utf-8"))
    entities = [entity for entity in sidecar["entities"] if len(entity["token_positions"]) >= 2]
    tokenizer = Tokenizer.from_file(str(CORPUS / "tokenizer.json"))
    widths = {
        entity["codeword"]: len(tokenizer.encode(" " + entity["codeword"]).ids)
        for entity in entities
    }
    return entities, widths


def needle_analysis(
    entities: list[dict[str, Any]], widths: dict[str, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for layer in GLOBALS:
        row_path = CAPTURE / "replay" / f"needlerows_L{layer:02d}.npz"
        with np.load(row_path) as row_dump:
            qpos = np.asarray(row_dump["qpos"])
            rows = np.asarray(row_dump["rows"], dtype=np.float64)
        projection = np.asarray(
            np.load(WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"), dtype=np.float64
        )
        rvec = np.load(
            CAPTURE / "replay" / f"rvec_L{layer:02d}_05_needles.npy", mmap_mode="r"
        )
        qpos_index = {int(query): index for index, query in enumerate(qpos)}
        keys = np.arange(8192)
        per = []
        for entity in entities:
            p0, query = map(int, entity["token_positions"][:2])
            if query not in qpos_index:
                continue
            row = rows[qpos_index[query]]
            width = widths[entity["codeword"]]
            window = slice(p0, p0 + width)
            distance = query - keys
            bias_by_distance = np.asarray(rvec[query], dtype=np.float64) @ projection
            bias = np.zeros_like(row)
            active = (distance >= 0) & (distance < projection.shape[1])
            bias[:, active] = bias_by_distance[:, distance[active]]
            causal = keys <= query
            without = row * np.exp(-bias)
            without[:, ~causal] = 0.0
            without /= without.sum(axis=1, keepdims=True) + 1e-300
            band = np.zeros(8192, dtype=bool)
            lo, hi = max(0, p0 - 128), min(query, p0 + 128 + width)
            band[lo:hi] = True
            band[window] = False
            band[~causal] = False

            def metrics(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                intro = weight[:, window].sum(axis=1)
                baseline = weight[:, band].mean(axis=1) * width + 1e-12
                return intro, intro / baseline

            with_intro, with_ratio = metrics(row)
            without_intro, without_ratio = metrics(without)
            per.append(
                {
                    "cw": entity["codeword"],
                    "dist": int(query - p0),
                    "side": entity["side_of_seam"],
                    "with_mean": float(with_intro.mean()),
                    "with_max": float(with_intro.max()),
                    "wo_mean": float(without_intro.mean()),
                    "wo_max": float(without_intro.max()),
                    "with_ratio_max": float(with_ratio.max()),
                    "wo_ratio_max": float(without_ratio.max()),
                    "argmax_head_with": int(with_intro.argmax()),
                    "argmax_head_wo": int(without_intro.argmax()),
                }
            )
        results[str(layer)] = per
        below = [row for row in per if row["side"] == "below"]
        above = [row for row in per if row["side"] == "above"]
        pvalue = float(
            mannwhitneyu(
                [row["with_max"] for row in below],
                [row["with_max"] for row in above],
                alternative="two-sided",
            ).pvalue
        )
        summary[str(layer)] = {
            "mann_whitney_two_sided_p": pvalue,
            "median_best_head_mass_below": float(np.median([row["with_max"] for row in below])),
            "median_best_head_mass_above": float(np.median([row["with_max"] for row in above])),
            "median_with_without_multiplier_below": float(
                np.median([row["with_max"] / (row["wo_max"] + 1e-300) for row in below])
            ),
            "median_with_without_multiplier_above": float(
                np.median([row["with_max"] / (row["wo_max"] + 1e-300) for row in above])
            ),
        }
    return results, summary


def echo_analysis(meters: dict[tuple[int, str], dict[str, np.ndarray]]) -> dict[str, Any]:
    deltas = np.zeros((11, 3), dtype=np.float64)
    head_fraction = np.zeros((11, 3), dtype=np.float64)
    curves: dict[tuple[int, str], np.ndarray] = {}
    for layer_index, layer in enumerate(GLOBALS):
        for text_index, text in enumerate(ECHO_TEXTS):
            bias = meters[(layer, text)]["mean_bias"]
            curves[(layer, text)] = bias.mean(axis=0)
            per_head = bias[:, 512:528].mean(axis=1) - bias[:, 496:512].mean(axis=1)
            deltas[layer_index, text_index] = per_head.mean()
            head_fraction[layer_index, text_index] = np.mean(per_head > 0)
    controls = [128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896]
    consistent = []
    for boundary in controls:
        count = 0
        for layer in GLOBALS:
            values = [
                curves[(layer, text)][boundary : boundary + 16].mean()
                - curves[(layer, text)][boundary - 16 : boundary].mean()
                for text in ECHO_TEXTS
            ]
            count += int(np.all(np.asarray(values) > 0))
        consistent.append(count)
    return {
        "positive_layer_text_cells": int(np.sum(deltas > 0)),
        "layer_text_cells_total": int(deltas.size),
        "step_delta": deltas.tolist(),
        "step_delta_min": float(deltas.min()),
        "step_delta_max": float(deltas.max()),
        "head_positive_fraction_min": float(head_fraction.min()),
        "head_positive_fraction_max": float(head_fraction.max()),
        "control_boundaries": controls,
        "layers_consistently_positive_across_three_texts": consistent,
    }


def aggregate_profile(
    meters: dict[tuple[int, str], dict[str, np.ndarray]], layer: int, field: str
) -> np.ndarray:
    return np.mean(
        [meters[(layer, text)][field].mean(axis=0) for text in TEXTS], axis=0
    )


def wall_analysis(meters: dict[tuple[int, str], dict[str, np.ndarray]]) -> dict[str, Any]:
    edge = {
        str(layer): float(
            np.mean(
                [meters[(layer, text)]["mean_bias"][:, 1008:1024].mean() for text in TEXTS]
            )
        )
        for layer in GLOBALS
    }
    bias65 = aggregate_profile(meters, 65, "mean_bias")
    with65 = aggregate_profile(meters, 65, "mean_mass_with")
    without65 = aggregate_profile(meters, 65, "mean_mass_without")
    inside, outside = slice(1008, 1024), slice(1024, 1040)
    return {
        "edge_bias_by_global_layer": edge,
        "L65_near_bias_d0_7": float(bias65[:8].mean()),
        "L65_edge_bias_d1008_1023": float(bias65[inside].mean()),
        "L65_with_bias_mass_inside_outside_ratio": float(
            with65[inside].mean() / with65[outside].mean()
        ),
        "L65_without_bias_mass_inside_outside_ratio": float(
            without65[inside].mean() / without65[outside].mean()
        ),
        "L65_without_bias_inside_mass": float(without65[inside].mean()),
        "L65_without_bias_outside_mass": float(without65[outside].mean()),
    }


def heartbeat_analysis(
    meters: dict[tuple[int, str], dict[str, np.ndarray]]
) -> dict[str, Any]:
    target = 2334
    controls = np.r_[
        np.arange(target - 160, target - 40), np.arange(target + 40, target + 161)
    ]
    ratios = {}
    heads = {}
    for layer in GLOBALS:
        without = meters[(layer, "03_templated")]["mean_mass_without"]
        background = without[:, controls].mean(axis=1)
        per_head = without[:, target] / (background + 1e-300)
        ratio = float(np.percentile(per_head, 97.5))
        ratios[str(layer)] = ratio
        heads[str(layer)] = int(np.abs(per_head - ratio).argmin())
    return {
        "target_distance": target,
        "high_head_ratio_97_5pct_by_layer": ratios,
        "representative_head_by_layer": heads,
        "all_global_ratios_above_one": bool(all(value > 1 for value in ratios.values())),
        "note": "Computed from mean_mass_without; positional bias is exactly zero at d=2334.",
    }


def run(out: Path) -> None:
    report_path = out / "insitu_corrected.json"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, results_path)
    _, manifest = require_certified_capture()
    parity = require_parity()
    records = artifact_index(manifest)
    input_hashes: dict[str, str] = {}
    meters: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for layer in GLOBALS:
        for text in TEXTS:
            relative = f"meters/layer{layer:02d}_{text}_s8192.npz"
            record = records.get(relative)
            if record is None or record.get("kind") != "tier2_distance_meter":
                raise RuntimeError(f"meter is not bound by manifest: {relative}")
            input_hashes[relative] = record["sha256"]
            meters[(layer, text)] = load_meter(layer, text)
        print(f"corrected in-situ meters L{layer:02d}", flush=True)
    for layer in GLOBALS:
        relative = f"replay/needlerows_L{layer:02d}.npz"
        if records.get(relative, {}).get("kind") != "lf5_needle_rows":
            raise RuntimeError(f"needle rows are not bound by manifest: {relative}")
        input_hashes[relative] = records[relative]["sha256"]
        rvec_relative = f"replay/rvec_L{layer:02d}_05_needles.npy"
        if records.get(rvec_relative, {}).get("kind") != "rvec":
            raise RuntimeError(f"needle r-vector is not bound by manifest: {rvec_relative}")
        input_hashes[rvec_relative] = records[rvec_relative]["sha256"]

    seam = seam_analysis(meters)
    entities, widths = needle_setup()
    needles, needle_summary = needle_analysis(entities, widths)
    echo = echo_analysis(meters)
    wall = wall_analysis(meters)
    heartbeat = heartbeat_analysis(meters)

    significant_needle_layers = [
        int(layer)
        for layer, row in needle_summary.items()
        if row["mann_whitney_two_sided_p"] < 0.05
    ]
    seam_bias_positive = bool(
        all(
            np.mean([row["bias_attrib_step_mean"] for row in per_text.values()]) > 0
            for per_text in seam.values()
        )
    )
    seam_outside_zero = bool(
        all(row["bias_out_max"] == 0 for per_text in seam.values() for row in per_text.values())
    )
    needle_no_flip = significant_needle_layers == [5, 65]
    echo_no_flip = echo["positive_layer_text_cells"] == 33
    wall_retained = bool(
        max(wall["edge_bias_by_global_layer"], key=wall["edge_bias_by_global_layer"].get)
        == "65"
        and wall["L65_with_bias_mass_inside_outside_ratio"]
        > wall["L65_without_bias_mass_inside_outside_ratio"]
    )
    heartbeat_retained = heartbeat["all_global_ratios_above_one"]
    no_flip = bool(
        seam_bias_positive
        and seam_outside_zero
        and needle_no_flip
        and echo_no_flip
        and wall_retained
        and heartbeat_retained
    )
    report = {
        "schema_version": 1,
        "kind": "round5_a6_corrected_insitu_recertification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "capture_manifest_sha256": MANIFEST_SHA256,
        "lf5_replay_parity_sha256": sha256_file(PARITY),
        "lf5_production_replay_bitwise_equal": True,
        "lf5_values_compared": sum(row["elements"] for row in parity["results"]),
        "registered_a6_no_flip_expectation_confirmed": no_flip,
        "component_verdicts": {
            "seam_bias_attributable_step_positive_every_global_layer": seam_bias_positive,
            "seam_bias_exactly_zero_outside_extent": seam_outside_zero,
            "needle_significant_only_L5_L65": needle_no_flip,
            "echo_33_of_33_positive": echo_no_flip,
            "L65_terminal_wall_retained": wall_retained,
            "heartbeat_beyond_horizon_retained": heartbeat_retained,
        },
        "seam": seam,
        "needle_results": needles,
        "needle_summary": needle_summary,
        "needle_significant_layers_p_lt_0p05": significant_needle_layers,
        "global_512_echo": echo,
        "terminal_wall": wall,
        "heartbeat_content_only": heartbeat,
        "historical_source_sha256": {
            "tier2": sha256_file(OLD_TIER2),
            "needles": sha256_file(OLD_NEEDLES),
            "revised_summary": sha256_file(OLD_SUMMARY),
        },
        "input_capture_sha256": input_hashes,
        "corpus_sidecar_sha256": sha256_file(CORPUS / "05_needles.sidecar.json"),
        "tokenizer_sha256": sha256_file(CORPUS / "tokenizer.json"),
        "projection_sha256": {
            str(layer): sha256_file(WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy")
            for layer in GLOBALS
        },
        "provenance": provenance(Path(__file__)),
    }
    atomic_json(report_path, report)
    lines = [
        "# A6-corrected in-situ mechanical re-certification",
        "",
        "**Status: answered; independent re-derivation pending.**",
        "",
        f"- Registered A6 no-verdict-flip expectation: **{str(no_flip).lower()}**.",
        f"- d=1024 seam retains a positive bias-attributable step at every global layer, with exact zero outside the extent: **{str(seam_bias_positive and seam_outside_zero).lower()}**.",
        f"- Needle seam deficit significant only at L5/L65: **{str(needle_no_flip).lower()}**; observed p<0.05 layers `{significant_needle_layers}`.",
        f"- d=512 echo: **{echo['positive_layer_text_cells']}/33** positive layer/text cells.",
        f"- L65 terminal wall retained: **{str(wall_retained).lower()}**; with-bias inside/outside `{wall['L65_with_bias_mass_inside_outside_ratio']:.6g}x`, content-only `{wall['L65_without_bias_mass_inside_outside_ratio']:.6g}x`.",
        f"- Heartbeat induction beyond d=1024 retained: **{str(heartbeat_retained).lower()}**; minimum 97.5th-percentile head ratio `{min(heartbeat['high_head_ratio_97_5pct_by_layer'].values()):.6g}`.",
        f"- LF5 production replay remains bitwise across `{report['lf5_values_compared']:,}` attention values.",
        "",
    ]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(report["component_verdicts"], indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    mass = np.ones((64, 8192), dtype=np.float64)
    zero = np.zeros_like(mass)
    meters = {
        (layer, text): {
            "mean_mass_with": mass,
            "mean_mass_without": mass,
            "mean_bias": zero,
        }
        for layer in GLOBALS
        for text in TEXTS
    }
    echo = echo_analysis(meters)
    if echo["positive_layer_text_cells"] != 0:
        raise AssertionError(echo)
    heartbeat = heartbeat_analysis(meters)
    if not all(np.isclose(value, 1.0) for value in heartbeat["high_head_ratio_97_5pct_by_layer"].values()):
        raise AssertionError(heartbeat)
    print("round5_insitu_corrected_recertify self-test passed")


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
