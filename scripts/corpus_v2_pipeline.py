"""Dump-first corpus-v2 novelty and registered aperture readouts.

Commands run in order: novelty -> compute -> analyze. Raw classes, captures,
and aperture arrays remain private/gitignored; only compact reports are public.
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


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus_v2"
CAPTURE = ROOT / "dumps" / "round5" / "corpus_v2_capture"
CAPTURE_VALIDATION = ROOT / "analysis" / "round5" / "corpus_v2" / "capture_validation.json"
CLASSES = CORPUS / "classes.json"
WEIGHTS = ROOT / "weights"
LF4_INPUT_VALIDATION = ROOT / "analysis" / "round5" / "lf4" / "input_validation.json"
APERTURE_DUMP = ROOT / "dumps" / "round5" / "corpus_v2_aperture"
NOVELTY_REPORT = ROOT / "analysis" / "round5" / "corpus_v2" / "novelty.json"
READOUT_REPORT = ROOT / "analysis" / "round5" / "corpus_v2" / "readouts.json"
TEXTS = ["07_slack_human", "08_math_llm"]
MID_GLOBALS = [23, 29, 35, 41, 47]
REGISTRATION_COMMIT = "7fb84ab4e5d164bc759c32e38b0e58803abd5299"
PUBLIC_BOUNDARY_COMMIT = "65b220c2d185829dfc4c8e617a67e673d2fa9cd2"
PERMUTATIONS = 10000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def seed_for(label: str) -> int:
    digest = hashlib.sha256(
        f"{REGISTRATION_COMMIT}|corpus-v2|{label}".encode()
    ).hexdigest()
    return int(digest[:16], 16)


def require_capture_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    validation = json.loads(CAPTURE_VALIDATION.read_text(encoding="utf-8"))
    manifest_path = CAPTURE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        validation.get("kind") != "corpus_v2_capture_independent_validation"
        or validation.get("passed") is not True
        or validation.get("errors") != []
        or validation.get("capture_manifest_sha256") != sha256_file(manifest_path)
        or manifest.get("complete") is not True
        or manifest.get("artifact_count") != 134
        or manifest.get("public_boundary_commit") != PUBLIC_BOUNDARY_COMMIT
    ):
        raise RuntimeError("corpus-v2 capture validation is missing, failed, or stale")
    return validation, manifest


def require_classes() -> dict[str, Any]:
    classes = json.loads(CLASSES.read_text(encoding="utf-8"))
    if (
        classes.get("kind") != "corpus_v2_private_class_freeze"
        or classes.get("public_boundary_commit") != PUBLIC_BOUNDARY_COMMIT
        or classes.get("inputs", {}).get("private_manifest_sha256")
        != sha256_file(CORPUS / "manifest.json")
        or classes.get("inputs", {}).get("slack_ids_sha256")
        != sha256_file(CORPUS / "07_slack_human.ids.npy")
        or classes.get("inputs", {}).get("slack_sidecar_sha256")
        != sha256_file(CORPUS / "07_slack_human.sidecar.json")
    ):
        raise RuntimeError("private class freeze is missing or stale")
    expected = {"message_starts", "pronouns", "function_words"}
    if set(classes.get("classes", {})) != expected:
        raise RuntimeError("private class freeze has unexpected classes")
    for name in expected:
        positions = [int(value) for value in classes["classes"][name]]
        if (
            not positions
            or len(positions) != len(set(positions))
            or any(value < 0 or value >= 8192 for value in positions)
            or len(positions) != int(classes["counts"][name])
        ):
            raise RuntimeError(f"invalid frozen class: {name}")
    return classes


def require_novelty() -> dict[str, Any]:
    report = json.loads(NOVELTY_REPORT.read_text(encoding="utf-8"))
    if (
        report.get("kind") != "corpus_v2_novelty_gate"
        or report.get("eligible_for_aperture") is not True
        or report.get("capture_validation_sha256") != sha256_file(CAPTURE_VALIDATION)
    ):
        raise RuntimeError("novelty gate is missing, contaminated, or stale")
    return report


def load_nll(text: str) -> np.ndarray:
    with np.load(CAPTURE / "nll" / f"nll_{text}.npz") as data:
        target_position = data["target_position"]
        nll = data["nll"]
    if not np.array_equal(target_position, np.arange(1, 8192, dtype=np.int32)):
        raise RuntimeError(f"invalid target-position alignment for {text}")
    if nll.shape != (8191,) or nll.dtype != np.float32 or not np.isfinite(nll).all():
        raise RuntimeError(f"invalid NLL array for {text}")
    return nll


def novelty_command(_args: argparse.Namespace) -> None:
    validation, manifest = require_capture_validation()
    if NOVELTY_REPORT.exists():
        raise FileExistsError(f"refusing to overwrite outcome report: {NOVELTY_REPORT}")
    means = {
        text: float(np.mean(load_nll(text), dtype=np.float64)) for text in TEXTS
    }
    contaminated = [text for text, mean in means.items() if mean < 1.0]
    thresholds_pass = all(mean >= 1.5 for mean in means.values())
    ordering_pass = means["07_slack_human"] > means["08_math_llm"]
    report = {
        "schema_version": 1,
        "kind": "corpus_v2_novelty_gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        "source_sha256": sha256_file(Path(__file__)),
        "capture_manifest_sha256": sha256_file(CAPTURE / "manifest.json"),
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "capture_validation_passed": bool(validation["passed"]),
        "capture_git_head": manifest["git_head"],
        "count_per_arm": 8191,
        "mean_nll": means,
        "replacement_threshold": 1.0,
        "prediction_threshold": 1.5,
        "contaminated_arms": contaminated,
        "eligible_for_aperture": not contaminated,
        "both_means_ge_1_5": thresholds_pass,
        "slack_gt_math_llm": ordering_pass,
        "prediction_passed": bool(thresholds_pass and ordering_pass),
    }
    atomic_json(NOVELTY_REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if contaminated:
        raise SystemExit(2)


def aperture_blocked(
    rvec: np.ndarray, projection: np.ndarray, *, block_tokens: int
) -> dict[str, np.ndarray]:
    if rvec.shape != (8192, 64, 16) or rvec.dtype != np.float16:
        raise ValueError((rvec.shape, rvec.dtype))
    if projection.shape != (16, 1024):
        raise ValueError(projection.shape)
    numerator = np.zeros(8192, dtype=np.float64)
    denominator = np.zeros(8192, dtype=np.float64)
    projection64 = np.asarray(projection, dtype=np.float64)
    for start in range(0, 8192, block_tokens):
        stop = min(start + block_tokens, 8192)
        coefficients = np.asarray(rvec[start:stop], dtype=np.float64)
        curves = coefficients.reshape(-1, 16) @ projection64
        absolute = np.abs(curves).reshape(stop - start, 64, 1024)
        denominator[start:stop] = absolute.sum(axis=(1, 2), dtype=np.float64)
        numerator[start:stop] = absolute[:, :, 129:].sum(
            axis=(1, 2), dtype=np.float64
        )
    if np.any(denominator <= 0):
        raise RuntimeError("zero aperture denominator")
    aperture = numerator / denominator
    if not np.isfinite(aperture).all() or np.any(aperture < 0) or np.any(aperture > 1):
        raise RuntimeError("invalid aperture")
    return {
        "aperture_full": aperture,
        "full_numerator": numerator,
        "full_denominator": denominator,
    }


def compute_command(args: argparse.Namespace) -> None:
    validation, capture_manifest = require_capture_validation()
    novelty = require_novelty()
    classes = require_classes()
    weight_validation = json.loads(LF4_INPUT_VALIDATION.read_text(encoding="utf-8"))
    if weight_validation.get("passed") is not True:
        raise RuntimeError("Round-5 LF4 weight validation did not pass")
    if APERTURE_DUMP.exists() and any(APERTURE_DUMP.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty dump: {APERTURE_DUMP}")
    APERTURE_DUMP.mkdir(parents=True, exist_ok=True)
    manifest_path = APERTURE_DUMP / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "corpus_v2_aperture_dump",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "registration_commit": REGISTRATION_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        "source_sha256": sha256_file(Path(__file__)),
        "capture_manifest_sha256": sha256_file(CAPTURE / "manifest.json"),
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "capture_validation_passed": bool(validation["passed"]),
        "capture_git_head": capture_manifest["git_head"],
        "novelty_report_sha256": sha256_file(NOVELTY_REPORT),
        "eligible_for_aperture": bool(novelty["eligible_for_aperture"]),
        "classes_sha256": sha256_file(CLASSES),
        "class_counts": classes["counts"],
        "lf4_input_validation_sha256": sha256_file(LF4_INPUT_VALIDATION),
        "layers": MID_GLOBALS,
        "texts": TEXTS,
        "files": {},
    }
    atomic_json(manifest_path, manifest)
    for layer in MID_GLOBALS:
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
            arrays = aperture_blocked(rvec, projection, block_tokens=args.block_tokens)
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
    if len(manifest["files"]) != 10:
        raise RuntimeError("incomplete aperture dump")
    manifest["complete"] = True
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)


def midrank_percentiles(values: np.ndarray, bin_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for start in range(0, len(values), bin_size):
        stop = min(start + bin_size, len(values))
        block = values[start:stop]
        order = np.argsort(block, kind="mergesort")
        sorted_values = block[order]
        ranks = np.empty(len(block), dtype=np.float64)
        cursor = 0
        while cursor < len(block):
            end = cursor + 1
            while end < len(block) and sorted_values[end] == sorted_values[cursor]:
                end += 1
            ranks[order[cursor:end]] = ((cursor + 1) + end) / 2.0
            cursor = end
        result[start:stop] = (ranks - 0.5) / len(block)
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("invalid Spearman inputs")
    x_rank = midrank_percentiles(np.asarray(x, dtype=np.float64), len(x))
    y_rank = midrank_percentiles(np.asarray(y, dtype=np.float64), len(y))
    x_centered = x_rank - np.mean(x_rank, dtype=np.float64)
    y_centered = y_rank - np.mean(y_rank, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered, dtype=np.float64)
        * np.sum(y_centered * y_centered, dtype=np.float64)
    )
    if denominator == 0:
        raise RuntimeError("constant input in Spearman correlation")
    return float(np.sum(x_centered * y_centered, dtype=np.float64) / denominator)


def averaged_scores(text: str, bin_size: int) -> np.ndarray:
    scores = []
    for layer in MID_GLOBALS:
        with np.load(APERTURE_DUMP / f"aperture_L{layer:02d}_{text}.npz") as data:
            aperture = data["aperture_full"]
        scores.append(midrank_percentiles(aperture, bin_size))
    return np.mean(scores, axis=0, dtype=np.float64)


def binned_spearman(aperture_scores: np.ndarray, nll: np.ndarray) -> dict[str, Any]:
    correlations = []
    for start in range(0, 8192, 512):
        stop = min(start + 512, 8192)
        positions = np.arange(max(1, start), stop, dtype=np.int64)
        correlations.append(spearman(aperture_scores[positions], nll[positions - 1]))
    positive = int(np.count_nonzero(np.asarray(correlations) > 0))
    median = float(np.median(correlations))
    return {
        "bin_size": 512,
        "bin_correlations": correlations,
        "median_spearman": median,
        "positive_bins": positive,
        "total_bins": 16,
        "passed": bool(median > 0 and positive >= 12),
    }


def permutation_test(
    scores: np.ndarray,
    positions: list[int],
    *,
    direction: Literal["positive", "negative"],
    seed: int,
) -> dict[str, Any]:
    selected_positions = np.asarray(sorted(set(positions)), dtype=np.int64)
    if len(selected_positions) == 0:
        raise ValueError("empty registered class")
    observed = float(np.median(scores[selected_positions]) - 0.5)
    blocks = []
    counts = []
    for start in range(0, 8192, 256):
        stop = min(start + 256, 8192)
        blocks.append(np.arange(start, stop, dtype=np.int64))
        counts.append(
            int(np.count_nonzero(
                (selected_positions >= start) & (selected_positions < stop)
            ))
        )
    rng = np.random.Generator(np.random.PCG64(seed))
    null = np.empty(PERMUTATIONS, dtype=np.float64)
    for iteration in range(PERMUTATIONS):
        choices = [
            rng.choice(block, size=count, replace=False)
            for block, count in zip(blocks, counts)
            if count
        ]
        null[iteration] = float(np.median(scores[np.concatenate(choices)]) - 0.5)
    exceed = (
        np.count_nonzero(null >= observed)
        if direction == "positive"
        else np.count_nonzero(null <= observed)
    )
    return {
        "count": int(len(selected_positions)),
        "direction": direction,
        "effect": observed,
        "permutations": PERMUTATIONS,
        "seed": seed,
        "p": float((1 + exceed) / (PERMUTATIONS + 1)),
        "null_quantiles": {
            "q025": float(np.quantile(null, 0.025)),
            "q50": float(np.quantile(null, 0.5)),
            "q975": float(np.quantile(null, 0.975)),
        },
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def require_aperture_dump() -> dict[str, Any]:
    manifest_path = APERTURE_DUMP / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or len(manifest.get("files", {})) != 10
        or manifest.get("capture_validation_sha256") != sha256_file(CAPTURE_VALIDATION)
        or manifest.get("novelty_report_sha256") != sha256_file(NOVELTY_REPORT)
        or manifest.get("classes_sha256") != sha256_file(CLASSES)
        or manifest.get("source_sha256") != sha256_file(Path(__file__))
    ):
        raise RuntimeError("aperture dump is incomplete or stale")
    for record in manifest["files"].values():
        path = APERTURE_DUMP / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"aperture artifact hash mismatch: {path}")
    return manifest


def analyze_command(_args: argparse.Namespace) -> None:
    require_capture_validation()
    novelty = require_novelty()
    classes = require_classes()
    aperture_manifest = require_aperture_dump()
    if READOUT_REPORT.exists():
        raise FileExistsError(f"refusing to overwrite outcome report: {READOUT_REPORT}")

    scores_512 = {text: averaged_scores(text, 512) for text in TEXTS}
    nll = {text: load_nll(text) for text in TEXTS}
    p2_primary = {
        text: binned_spearman(scores_512[text], nll[text]) for text in TEXTS
    }
    p2_cross = {
        "slack_aperture_math_nll": binned_spearman(
            scores_512["07_slack_human"], nll["08_math_llm"]
        ),
        "math_aperture_slack_nll": binned_spearman(
            scores_512["08_math_llm"], nll["07_slack_human"]
        ),
    }
    p2_control_passed = not any(item["passed"] for item in p2_cross.values())

    slack_scores = averaged_scores("07_slack_human", 256)
    math_scores = averaged_scores("08_math_llm", 256)
    specifications = [
        ("message_starts", "positive", "P-v2-3"),
        ("pronouns", "negative", "P-v2-4"),
        ("function_words", "negative", "P-v2-4"),
    ]
    primary = []
    controls = []
    for name, direction, prediction in specifications:
        positions = [int(value) for value in classes["classes"][name]]
        item = permutation_test(
            slack_scores,
            positions,
            direction=direction,
            seed=seed_for(f"primary|{name}"),
        )
        item.update(name=name, prediction=prediction, text="07_slack_human")
        primary.append(item)
        control = permutation_test(
            math_scores,
            positions,
            direction=direction,
            seed=seed_for(f"cross-control|{name}"),
        )
        control.update(
            name=name,
            prediction=prediction,
            text="08_math_llm",
            mask_source="07_slack_human",
        )
        controls.append(control)

    primary_adjusted = holm_adjust([item["p"] for item in primary])
    control_adjusted = holm_adjust([item["p"] for item in controls])
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
    report = {
        "schema_version": 1,
        "kind": "corpus_v2_registered_readouts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        "source_sha256": sha256_file(Path(__file__)),
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "novelty_report_sha256": sha256_file(NOVELTY_REPORT),
        "novelty_eligible": bool(novelty["eligible_for_aperture"]),
        "classes_sha256": sha256_file(CLASSES),
        "class_counts": classes["counts"],
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
        "p_v2_3": {
            "prediction_passed": p3_prediction,
            "control_passed": p3_control,
        },
        "p_v2_4": {
            "prediction_passed": p4_prediction,
            "control_passed": p4_control,
        },
        "all_true_null_controls_passed": bool(
            p2_control_passed and p3_control and p4_control
        ),
        "prediction_summary": {
            "P-v2-1": bool(novelty["prediction_passed"]),
            "P-v2-2": all(item["passed"] for item in p2_primary.values()),
            "P-v2-3": p3_prediction,
            "P-v2-4": p4_prediction,
        },
    }
    atomic_json(READOUT_REPORT, report)
    print(json.dumps(report["prediction_summary"], indent=2, sort_keys=True))
    print(f"true-null controls passed={report['all_true_null_controls_passed']}")


def self_test() -> None:
    values = np.array([3.0, 1.0, 1.0, 4.0, 2.0], dtype=np.float64)
    ranks = midrank_percentiles(values, len(values))
    expected = np.array([0.7, 0.2, 0.2, 0.9, 0.5])
    if not np.allclose(ranks, expected, atol=0, rtol=0):
        raise AssertionError((ranks, expected))
    if abs(spearman(np.arange(8), np.arange(8)) - 1.0) > 1e-15:
        raise AssertionError("Spearman identity failed")
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    if not np.allclose(adjusted, [0.03, 0.06, 0.06]):
        raise AssertionError(adjusted)
    print("self-test passed")


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
