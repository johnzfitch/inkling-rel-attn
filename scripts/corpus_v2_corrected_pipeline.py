"""A6-corrected corpus-v2 and depth-resolved aperture readouts.

Run in order: ``novelty`` -> ``compute`` -> ``analyze``. The script writes only
fresh corrected directories; provisional v2.0 artifacts are never modified.
P-v2-1..4 reuse their original definitions verbatim. P-d1..4 use the fixed
bands and band-median statistic in ``ROUND5_DEPTH_RESOLVED_PREREG.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

import corpus_v2_pipeline as legacy
from corpus_v2_freeze_classes import build_manifest as rebuild_v20_classes


ROOT = Path(__file__).resolve().parents[1]
CORPUS_V2 = ROOT / "corpus_v2"
CAPTURE = ROOT / "dumps" / "round5" / "corpus_v2_corrected_capture"
CAPTURE_VALIDATION = (
    ROOT / "analysis" / "round5" / "corpus_v2_corrected" / "capture_validation.json"
)
CLASSES_V20 = CORPUS_V2 / "classes.json"
CLASSES_DEPTH = CORPUS_V2 / "depth_classes.json"
WEIGHTS = ROOT / "weights"
LF4_INPUT_VALIDATION = ROOT / "analysis" / "round5" / "lf4" / "input_validation.json"
APERTURE_DUMP = ROOT / "dumps" / "round5" / "corpus_v2_corrected_aperture"
REPORT_DIR = ROOT / "analysis" / "round5" / "corpus_v2_corrected"
NOVELTY_REPORT = REPORT_DIR / "novelty.json"
READOUT_REPORT = REPORT_DIR / "readouts.json"
DEPTH_REPORT = REPORT_DIR / "depth_readouts.json"

TEXTS = ["07_slack_human", "08_math_llm", "07b_slack_multi", "01_prose_en"]
V20_TEXTS = ["07_slack_human", "08_math_llm"]
MATH = "08_math_llm"
SLACK_V21 = "07b_slack_multi"
PROSE = "01_prose_en"
MID_GLOBALS = [23, 29, 35, 41, 47]
SHALLOW_GLOBALS = [17, 23, 29]
DEEP_GLOBALS = [35, 41, 47, 53, 59]
TERMINAL_GLOBALS = [53, 59]
APERTURE_LAYERS = sorted(set(MID_GLOBALS + SHALLOW_GLOBALS + DEEP_GLOBALS))
PERMUTATIONS = 10000
BIN_SIZE = 256
REGISTRATION_COMMIT = "7fb84ab4e5d164bc759c32e38b0e58803abd5299"
A6_COMMIT = "061bb04dffef05eae33f9fffc430394faa4052c5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def seed_for_depth(label: str) -> int:
    return int(hashlib.sha256(f"{A6_COMMIT}|P-d|{label}".encode()).hexdigest()[:16], 16)


def require_capture_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    validation = json.loads(CAPTURE_VALIDATION.read_text(encoding="utf-8"))
    manifest_path = CAPTURE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        validation.get("kind") != "corpus_v2_a6_capture_independent_validation"
        or validation.get("passed") is not True
        or validation.get("errors") != []
        or validation.get("capture_manifest_sha256") != sha256_file(manifest_path)
        or manifest.get("kind") != "corpus_v2_a6_corrected_capture"
        or manifest.get("complete") is not True
        or manifest.get("artifact_count") != 268
        or manifest.get("amendment_a6_commit") != A6_COMMIT
        or manifest.get("stock_attention_parity", {}).get("passed") is not True
    ):
        raise RuntimeError("corrected capture validation is missing, failed, or stale")
    return validation, manifest


def require_classes() -> tuple[dict[str, Any], dict[str, Any]]:
    original = json.loads(CLASSES_V20.read_text(encoding="utf-8"))
    rebuilt = rebuild_v20_classes()
    if (
        original.get("kind") != "corpus_v2_private_class_freeze"
        or original.get("classes") != rebuilt.get("classes")
        or original.get("counts") != rebuilt.get("counts")
        or original.get("inputs", {}).get("slack_ids_sha256")
        != sha256_file(CORPUS_V2 / "07_slack_human.ids.npy")
        or original.get("inputs", {}).get("slack_sidecar_sha256")
        != sha256_file(CORPUS_V2 / "07_slack_human.sidecar.json")
    ):
        raise RuntimeError("original corpus-v2 class freeze is stale")

    depth = json.loads(CLASSES_DEPTH.read_text(encoding="utf-8"))
    if (
        depth.get("kind") != "round5_a6_depth_private_class_freeze"
        or depth.get("amendment_a6_commit") != A6_COMMIT
        or depth.get("inputs", {}).get("slack_ids_sha256")
        != sha256_file(CORPUS_V2 / "07b_slack_multi.ids.npy")
        or depth.get("inputs", {}).get("slack_sidecar_sha256")
        != sha256_file(CORPUS_V2 / "07b_slack_multi.sidecar.json")
        or depth.get("inputs", {}).get("math_ids_sha256")
        != sha256_file(CORPUS_V2 / "08_math_llm.ids.npy")
    ):
        raise RuntimeError("A6 depth class freeze is stale")
    expected = {"speaker_labels", "pronouns", "first_content", "sentence_starts"}
    if set(depth.get("classes", {})) != expected:
        raise RuntimeError("unexpected A6 depth classes")
    for name, positions in depth["classes"].items():
        if (
            len(positions) != depth["counts"][name]
            or len(positions) != len(set(positions))
            or any(int(position) < 0 or int(position) >= 8192 for position in positions)
        ):
            raise RuntimeError(f"invalid A6 depth class: {name}")
    return original, depth


def load_nll(text: str) -> np.ndarray:
    path = CAPTURE / "nll" / f"nll_{text}.npz"
    with np.load(path, allow_pickle=False) as data:
        positions = data["target_position"]
        nll = data["nll"]
    if not np.array_equal(positions, np.arange(1, 8192, dtype=np.int32)):
        raise RuntimeError(f"invalid NLL alignment for {text}")
    if nll.shape != (8191,) or nll.dtype != np.float32 or not np.isfinite(nll).all():
        raise RuntimeError(f"invalid NLL values for {text}")
    return nll


def require_novelty() -> dict[str, Any]:
    report = json.loads(NOVELTY_REPORT.read_text(encoding="utf-8"))
    if (
        report.get("kind") != "corpus_v2_a6_corrected_novelty_gate"
        or report.get("eligible_for_aperture") is not True
        or report.get("capture_validation_sha256") != sha256_file(CAPTURE_VALIDATION)
    ):
        raise RuntimeError("corrected novelty gate is missing, contaminated, or stale")
    return report


def novelty_command(_args: argparse.Namespace) -> None:
    validation, manifest = require_capture_validation()
    if NOVELTY_REPORT.exists():
        raise FileExistsError(f"refusing to overwrite outcome report: {NOVELTY_REPORT}")
    means = {text: float(np.mean(load_nll(text), dtype=np.float64)) for text in TEXTS}
    registered = {text: means[text] for text in V20_TEXTS}
    contaminated = [text for text, mean in registered.items() if mean < 1.0]
    thresholds_pass = all(mean >= 1.5 for mean in registered.values())
    ordering_pass = registered["07_slack_human"] > registered["08_math_llm"]
    report = {
        "schema_version": 1,
        "kind": "corpus_v2_a6_corrected_novelty_gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "amendment_a6_commit": A6_COMMIT,
        "amendment_a7_execution_sha256": sha256_file(
            ROOT / "ROUND5_AMENDMENT_A7_EXECUTION.md"
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "legacy_dependency_sha256": sha256_file(ROOT / "scripts" / "corpus_v2_pipeline.py"),
        "capture_manifest_sha256": sha256_file(CAPTURE / "manifest.json"),
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "capture_validation_passed": bool(validation["passed"]),
        "capture_git_head": manifest["git_head"],
        "count_per_arm": 8191,
        "mean_nll": means,
        "registered_gate_arms": V20_TEXTS,
        "replacement_threshold": 1.0,
        "prediction_threshold": 1.5,
        "contaminated_arms": contaminated,
        "eligible_for_aperture": not contaminated,
        "both_means_ge_1_5": thresholds_pass,
        "slack_gt_math_llm": ordering_pass,
        "prediction_passed": bool(thresholds_pass and ordering_pass),
        "nonregistered_descriptive_arms": [SLACK_V21, PROSE],
    }
    atomic_json(NOVELTY_REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if contaminated:
        raise SystemExit(2)


def compute_command(args: argparse.Namespace) -> None:
    validation, capture_manifest = require_capture_validation()
    novelty = require_novelty()
    original_classes, depth_classes = require_classes()
    weight_validation = json.loads(LF4_INPUT_VALIDATION.read_text(encoding="utf-8"))
    if weight_validation.get("passed") is not True:
        raise RuntimeError("Round-5 LF4 projection validation did not pass")
    if APERTURE_DUMP.exists() and any(APERTURE_DUMP.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty dump: {APERTURE_DUMP}")
    APERTURE_DUMP.mkdir(parents=True, exist_ok=True)
    manifest_path = APERTURE_DUMP / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "corpus_v2_a6_corrected_aperture_dump",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "registration_commit": REGISTRATION_COMMIT,
        "amendment_a6_commit": A6_COMMIT,
        "amendment_a7_execution_sha256": sha256_file(
            ROOT / "ROUND5_AMENDMENT_A7_EXECUTION.md"
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "legacy_dependency_sha256": sha256_file(ROOT / "scripts" / "corpus_v2_pipeline.py"),
        "capture_manifest_sha256": sha256_file(CAPTURE / "manifest.json"),
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "capture_validation_passed": bool(validation["passed"]),
        "capture_git_head": capture_manifest["git_head"],
        "novelty_report_sha256": sha256_file(NOVELTY_REPORT),
        "eligible_for_aperture": bool(novelty["eligible_for_aperture"]),
        "v20_classes_sha256": sha256_file(CLASSES_V20),
        "depth_classes_sha256": sha256_file(CLASSES_DEPTH),
        "v20_class_counts": original_classes["counts"],
        "depth_class_counts": depth_classes["counts"],
        "lf4_input_validation_sha256": sha256_file(LF4_INPUT_VALIDATION),
        "layers": APERTURE_LAYERS,
        "texts": TEXTS,
        "files": {},
    }
    atomic_json(manifest_path, manifest)
    for layer in APERTURE_LAYERS:
        projection_path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        projection = np.load(projection_path, allow_pickle=False)
        expected = weight_validation["records"][f"L{layer:02d}_projection"]["sha256"]
        if sha256_file(projection_path) != expected:
            raise RuntimeError(f"stale projection at L{layer:02d}")
        for text in TEXTS:
            key = f"L{layer:02d}_{text}"
            rvec_path = CAPTURE / "rvec" / f"rvec_{key}.npy"
            rvec = np.load(rvec_path, mmap_mode="r")
            started = time.time()
            arrays = legacy.aperture_blocked(rvec, projection, block_tokens=args.block_tokens)
            output = APERTURE_DUMP / f"aperture_{key}.npz"
            atomic_npz(output, **arrays)
            manifest["files"][key] = {
                "path": output.relative_to(APERTURE_DUMP).as_posix(),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "rvec_sha256": sha256_file(rvec_path),
                "projection_sha256": sha256_file(projection_path),
                "elapsed_seconds": round(time.time() - started, 3),
            }
            manifest["last_completed"] = key
            atomic_json(manifest_path, manifest)
            print(f"{key}: {manifest['files'][key]['elapsed_seconds']:.2f}s", flush=True)
    expected_files = len(APERTURE_LAYERS) * len(TEXTS)
    if len(manifest["files"]) != expected_files:
        raise RuntimeError("incomplete corrected aperture dump")
    manifest["complete"] = True
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)


def require_aperture_dump() -> dict[str, Any]:
    path = APERTURE_DUMP / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("kind") != "corpus_v2_a6_corrected_aperture_dump"
        or manifest.get("complete") is not True
        or len(manifest.get("files", {})) != len(APERTURE_LAYERS) * len(TEXTS)
        or manifest.get("capture_validation_sha256") != sha256_file(CAPTURE_VALIDATION)
        or manifest.get("novelty_report_sha256") != sha256_file(NOVELTY_REPORT)
        or manifest.get("source_sha256") != sha256_file(Path(__file__))
    ):
        raise RuntimeError("corrected aperture dump is incomplete or stale")
    for record in manifest["files"].values():
        artifact = APERTURE_DUMP / record["path"]
        if sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"aperture artifact hash mismatch: {artifact}")
    return manifest


def averaged_scores(text: str, layers: list[int], bin_size: int) -> np.ndarray:
    values = []
    for layer in layers:
        with np.load(APERTURE_DUMP / f"aperture_L{layer:02d}_{text}.npz") as data:
            values.append(legacy.midrank_percentiles(data["aperture_full"], bin_size))
    return np.mean(values, axis=0, dtype=np.float64)


def per_layer_scores(text: str, layers: list[int]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for layer in layers:
        with np.load(APERTURE_DUMP / f"aperture_L{layer:02d}_{text}.npz") as data:
            result[layer] = legacy.midrank_percentiles(data["aperture_full"], BIN_SIZE)
    return result


def band_permutation_test(
    scores: dict[int, np.ndarray],
    positions: list[int],
    *,
    direction: Literal["positive", "negative"],
    seed: int,
) -> dict[str, Any]:
    selected = np.asarray(sorted(set(int(value) for value in positions)), dtype=np.int64)
    if len(selected) == 0:
        raise ValueError("empty depth class")
    layers = sorted(scores)
    layer_effects = {
        str(layer): float(np.median(scores[layer][selected]) - 0.5) for layer in layers
    }
    observed = float(np.median(list(layer_effects.values())))
    blocks: list[np.ndarray] = []
    counts: list[int] = []
    for start in range(0, 8192, BIN_SIZE):
        stop = min(start + BIN_SIZE, 8192)
        blocks.append(np.arange(start, stop, dtype=np.int64))
        counts.append(int(np.count_nonzero((selected >= start) & (selected < stop))))
    rng = np.random.Generator(np.random.PCG64(seed))
    null = np.empty(PERMUTATIONS, dtype=np.float64)
    for iteration in range(PERMUTATIONS):
        sampled = np.concatenate(
            [
                rng.choice(block, size=count, replace=False)
                for block, count in zip(blocks, counts)
                if count
            ]
        )
        null[iteration] = np.median(
            [float(np.median(scores[layer][sampled]) - 0.5) for layer in layers]
        )
    exceed = (
        int(np.count_nonzero(null >= observed))
        if direction == "positive"
        else int(np.count_nonzero(null <= observed))
    )
    return {
        "count": int(len(selected)),
        "layers": layers,
        "direction": direction,
        "effect": observed,
        "per_layer_effects": layer_effects,
        "permutations": PERMUTATIONS,
        "seed": seed,
        "p": float((1 + exceed) / (PERMUTATIONS + 1)),
        "null_quantiles": {
            "q025": float(np.quantile(null, 0.025)),
            "q50": float(np.quantile(null, 0.5)),
            "q975": float(np.quantile(null, 0.975)),
        },
    }


def v20_readout(
    novelty: dict[str, Any], original_classes: dict[str, Any], aperture_manifest: dict[str, Any]
) -> dict[str, Any]:
    scores_512 = {text: averaged_scores(text, MID_GLOBALS, 512) for text in V20_TEXTS}
    nll = {text: load_nll(text) for text in V20_TEXTS}
    p2_primary = {
        text: legacy.binned_spearman(scores_512[text], nll[text]) for text in V20_TEXTS
    }
    p2_cross = {
        "slack_aperture_math_nll": legacy.binned_spearman(
            scores_512["07_slack_human"], nll["08_math_llm"]
        ),
        "math_aperture_slack_nll": legacy.binned_spearman(
            scores_512["08_math_llm"], nll["07_slack_human"]
        ),
    }
    p2_control_passed = not any(item["passed"] for item in p2_cross.values())
    slack_scores = averaged_scores("07_slack_human", MID_GLOBALS, 256)
    math_scores = averaged_scores("08_math_llm", MID_GLOBALS, 256)
    specifications = [
        ("message_starts", "positive", "P-v2-3"),
        ("pronouns", "negative", "P-v2-4"),
        ("function_words", "negative", "P-v2-4"),
    ]
    primary: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for name, direction, prediction in specifications:
        positions = [int(value) for value in original_classes["classes"][name]]
        item = legacy.permutation_test(
            slack_scores,
            positions,
            direction=direction,
            seed=legacy.seed_for(f"primary|{name}"),
        )
        item.update(name=name, prediction=prediction, text="07_slack_human")
        primary.append(item)
        control = legacy.permutation_test(
            math_scores,
            positions,
            direction=direction,
            seed=legacy.seed_for(f"cross-control|{name}"),
        )
        control.update(
            name=name,
            prediction=prediction,
            text="08_math_llm",
            mask_source="07_slack_human",
        )
        controls.append(control)
    primary_adjusted = legacy.holm_adjust([item["p"] for item in primary])
    control_adjusted = legacy.holm_adjust([item["p"] for item in controls])
    for item, adjusted in zip(primary, primary_adjusted):
        item["p_holm"] = adjusted
        sign_ok = item["effect"] > 0 if item["direction"] == "positive" else item["effect"] < 0
        item["prediction_passed"] = bool(sign_ok and adjusted < 0.05)
    for item, adjusted in zip(controls, control_adjusted):
        item["p_holm"] = adjusted
        item["passed_as_null"] = bool(adjusted >= 0.05)
    primary_by_name = {item["name"]: item for item in primary}
    control_by_name = {item["name"]: item for item in controls}
    p3_prediction = bool(primary_by_name["message_starts"]["prediction_passed"])
    p3_control = bool(control_by_name["message_starts"]["passed_as_null"])
    p4_prediction = bool(
        primary_by_name["pronouns"]["prediction_passed"]
        and primary_by_name["function_words"]["prediction_passed"]
    )
    p4_control = bool(
        control_by_name["pronouns"]["passed_as_null"]
        and control_by_name["function_words"]["passed_as_null"]
    )
    return {
        "schema_version": 1,
        "kind": "corpus_v2_a6_corrected_registered_readouts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "amendment_a6_commit": A6_COMMIT,
        "amendment_a7_execution_sha256": sha256_file(
            ROOT / "ROUND5_AMENDMENT_A7_EXECUTION.md"
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "legacy_dependency_sha256": sha256_file(ROOT / "scripts" / "corpus_v2_pipeline.py"),
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "novelty_report_sha256": sha256_file(NOVELTY_REPORT),
        "novelty_eligible": bool(novelty["eligible_for_aperture"]),
        "classes_sha256": sha256_file(CLASSES_V20),
        "class_counts": original_classes["counts"],
        "aperture_manifest_sha256": sha256_file(APERTURE_DUMP / "manifest.json"),
        "aperture_source_sha256": aperture_manifest["source_sha256"],
        "mid_global_layers": MID_GLOBALS,
        "permutations": PERMUTATIONS,
        "p_v2_2": {
            "primary": p2_primary,
            "cross_arm_controls": p2_cross,
            "prediction_passed": all(item["passed"] for item in p2_primary.values()),
            "control_passed": p2_control_passed,
        },
        "class_primary": primary,
        "class_cross_arm_controls": controls,
        "p_v2_3": {"prediction_passed": p3_prediction, "control_passed": p3_control},
        "p_v2_4": {"prediction_passed": p4_prediction, "control_passed": p4_control},
        "all_true_null_controls_passed": bool(p2_control_passed and p3_control and p4_control),
        "prediction_summary": {
            "P-v2-1": bool(novelty["prediction_passed"]),
            "P-v2-2": all(item["passed"] for item in p2_primary.values()),
            "P-v2-3": p3_prediction,
            "P-v2-4": p4_prediction,
        },
    }


def depth_readout(depth_classes: dict[str, Any]) -> dict[str, Any]:
    all_scores = {
        text: per_layer_scores(text, APERTURE_LAYERS) for text in (SLACK_V21, PROSE, MATH)
    }
    specifications = [
        ("P-d1", SLACK_V21, "speaker_labels", DEEP_GLOBALS, "positive"),
        ("P-d2", SLACK_V21, "pronouns", SHALLOW_GLOBALS, "negative"),
        ("P-d3", SLACK_V21, "first_content", DEEP_GLOBALS, "negative"),
        ("P-d4", PROSE, "sentence_starts", DEEP_GLOBALS, "positive"),
    ]
    primary: list[dict[str, Any]] = []
    crossed: list[dict[str, Any]] = []
    random_source: list[dict[str, Any]] = []
    random_math: list[dict[str, Any]] = []
    for prediction, text, class_name, target_layers, direction in specifications:
        positions = [int(value) for value in depth_classes["classes"][class_name]]
        primary_item = band_permutation_test(
            {layer: all_scores[text][layer] for layer in target_layers},
            positions,
            direction=direction,
            seed=seed_for_depth(f"primary|{prediction}"),
        )
        shallow_effect = float(
            np.median(
                [np.median(all_scores[text][layer][positions]) - 0.5 for layer in SHALLOW_GLOBALS]
            )
        )
        deep_effect = float(
            np.median(
                [np.median(all_scores[text][layer][positions]) - 0.5 for layer in DEEP_GLOBALS]
            )
        )
        terminal_effect = float(
            np.median(
                [np.median(all_scores[text][layer][positions]) - 0.5 for layer in TERMINAL_GLOBALS]
            )
        )
        if prediction == "P-d1":
            threshold_pass = deep_effect >= 0.10 and shallow_effect <= 0.0
        elif prediction == "P-d2":
            threshold_pass = shallow_effect <= -0.05 and abs(terminal_effect) < 0.03
        elif prediction == "P-d3":
            threshold_pass = deep_effect <= -0.05
        else:
            threshold_pass = deep_effect >= 0.05
        primary_item.update(
            prediction=prediction,
            text=text,
            class_name=class_name,
            shallow_band_effect=shallow_effect,
            deep_band_effect=deep_effect,
            terminal_band_effect=terminal_effect,
            threshold_passed=bool(threshold_pass),
        )
        primary.append(primary_item)

        control = band_permutation_test(
            {layer: all_scores[MATH][layer] for layer in target_layers},
            positions,
            direction=direction,
            seed=seed_for_depth(f"cross-math|{prediction}"),
        )
        control.update(prediction=prediction, text=MATH, mask_source=text)
        crossed.append(control)

        source_random_positions = depth_classes["random_masks"][prediction][text]
        source_random = band_permutation_test(
            {layer: all_scores[text][layer] for layer in target_layers},
            source_random_positions,
            direction=direction,
            seed=seed_for_depth(f"random-source|{prediction}"),
        )
        source_random.update(prediction=prediction, text=text)
        random_source.append(source_random)

        math_random_positions = depth_classes["random_masks"][prediction][MATH]
        math_random_item = band_permutation_test(
            {layer: all_scores[MATH][layer] for layer in target_layers},
            math_random_positions,
            direction=direction,
            seed=seed_for_depth(f"random-math|{prediction}"),
        )
        math_random_item.update(prediction=prediction, text=MATH)
        random_math.append(math_random_item)

    for family, field in (
        (primary, "prediction_passed"),
        (crossed, "passed_as_null"),
        (random_source, "passed_as_null"),
        (random_math, "passed_as_null"),
    ):
        adjusted = legacy.holm_adjust([item["p"] for item in family])
        for item, p_holm in zip(family, adjusted):
            item["p_holm"] = p_holm
            if field == "prediction_passed":
                item[field] = bool(item["threshold_passed"] and p_holm < 0.05)
            else:
                item[field] = bool(p_holm >= 0.05)

    return {
        "schema_version": 1,
        "kind": "round5_a6_depth_resolved_readouts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_a6_commit": A6_COMMIT,
        "amendment_a7_execution_sha256": sha256_file(
            ROOT / "ROUND5_AMENDMENT_A7_EXECUTION.md"
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "depth_prereg_sha256": sha256_file(ROOT / "ROUND5_DEPTH_RESOLVED_PREREG.md"),
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "aperture_manifest_sha256": sha256_file(APERTURE_DUMP / "manifest.json"),
        "depth_classes_sha256": sha256_file(CLASSES_DEPTH),
        "class_counts": depth_classes["counts"],
        "bands": {
            "shallow_global": SHALLOW_GLOBALS,
            "deep_global": DEEP_GLOBALS,
            "terminal_global": TERMINAL_GLOBALS,
        },
        "band_statistic": "median of per-layer effects",
        "position_bin": BIN_SIZE,
        "permutations": PERMUTATIONS,
        "holm_family": ["P-d1", "P-d2", "P-d3", "P-d4"],
        "primary": primary,
        "crossed_math_controls": crossed,
        "random_position_controls_source": random_source,
        "random_position_controls_math": random_math,
        "prediction_summary": {item["prediction"]: item["prediction_passed"] for item in primary},
        "threshold_summary": {item["prediction"]: item["threshold_passed"] for item in primary},
        "all_true_null_controls_passed": bool(
            all(item["passed_as_null"] for family in (crossed, random_source, random_math) for item in family)
        ),
    }


def analyze_command(_args: argparse.Namespace) -> None:
    require_capture_validation()
    novelty = require_novelty()
    original_classes, depth_classes = require_classes()
    aperture_manifest = require_aperture_dump()
    for report in (READOUT_REPORT, DEPTH_REPORT):
        if report.exists():
            raise FileExistsError(f"refusing to overwrite outcome report: {report}")
    v20 = v20_readout(novelty, original_classes, aperture_manifest)
    depth = depth_readout(depth_classes)
    atomic_json(READOUT_REPORT, v20)
    atomic_json(DEPTH_REPORT, depth)
    print(json.dumps(v20["prediction_summary"], indent=2, sort_keys=True))
    print(json.dumps(depth["prediction_summary"], indent=2, sort_keys=True))
    print(f"v2 controls passed={v20['all_true_null_controls_passed']}")
    print(f"depth controls passed={depth['all_true_null_controls_passed']}")


def self_test() -> None:
    legacy.self_test()
    values = np.linspace(0.0, 1.0, 8192, endpoint=False)
    scores = {17: values, 23: values, 29: values}
    positions = list(range(240, 8192, 256))
    first = band_permutation_test(
        scores, positions, direction="positive", seed=seed_for_depth("self-test")
    )
    second = band_permutation_test(
        scores, positions, direction="positive", seed=seed_for_depth("self-test")
    )
    if first != second or first["effect"] <= 0:
        raise AssertionError("depth permutation self-test failed")
    print("corrected pipeline self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("novelty")
    compute = subparsers.add_parser("compute")
    compute.add_argument("--block-tokens", type=int, default=32)
    subparsers.add_parser("analyze")
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "novelty":
        novelty_command(args)
    elif args.command == "compute":
        compute_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
