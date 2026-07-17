"""Independent validation for the A6/A7 corrected capture and readouts.

This module intentionally imports neither capture nor analysis code. The first
command validates raw artifacts before outcomes are opened. ``confirm`` then
rederives every registered statistic and decision from dumps and private class
freezes, including P-v2 and P-d decision booleans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tokenizers
import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
CORPUS_V2 = ROOT / "corpus_v2"
CORPUS_V1 = ROOT / "corpus"
NVFP4 = ROOT / "nvfp4"
CAPTURE = ROOT / "dumps" / "round5" / "corpus_v2_corrected_capture"
APERTURE = ROOT / "dumps" / "round5" / "corpus_v2_corrected_aperture"
REPORT_DIR = ROOT / "analysis" / "round5" / "corpus_v2_corrected"
CAPTURE_VALIDATION = REPORT_DIR / "capture_validation.json"
NOVELTY = REPORT_DIR / "novelty.json"
READOUTS = REPORT_DIR / "readouts.json"
DEPTH_READOUTS = REPORT_DIR / "depth_readouts.json"
VERIFICATION = REPORT_DIR / "verification.json"
CLASSES_V20 = CORPUS_V2 / "classes.json"
CLASSES_DEPTH = CORPUS_V2 / "depth_classes.json"

TEXTS = ["07_slack_human", "08_math_llm", "07b_slack_multi", "01_prose_en"]
TEXT_ROOT = {
    "07_slack_human": CORPUS_V2,
    "08_math_llm": CORPUS_V2,
    "07b_slack_multi": CORPUS_V2,
    "01_prose_en": CORPUS_V1,
}
V20_TEXTS = ["07_slack_human", "08_math_llm"]
SLACK_V21 = "07b_slack_multi"
MATH = "08_math_llm"
PROSE = "01_prose_en"
LAYERS = list(range(66))
MID_GLOBALS = [23, 29, 35, 41, 47]
SHALLOW = [17, 23, 29]
DEEP = [35, 41, 47, 53, 59]
TERMINAL = [53, 59]
APERTURE_LAYERS = sorted(set(MID_GLOBALS + SHALLOW + DEEP))
SEQ = 8192
BIN_SIZE = 256
PERMUTATIONS = 10000
REGISTRATION_COMMIT = "7fb84ab4e5d164bc759c32e38b0e58803abd5299"
A6_COMMIT = "061bb04dffef05eae33f9fffc430394faa4052c5"


def sha256_file(path: Path, *, chunk: int = 5 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def package_record(module: Any) -> dict[str, str]:
    return {"version": str(module.__version__), "module_path": str(Path(module.__file__).resolve())}


def safe_artifact(relative: str) -> Path:
    candidate = (CAPTURE / relative).resolve()
    candidate.relative_to(CAPTURE.resolve())
    return candidate


def validate_capture_command(args: argparse.Namespace) -> None:
    if CAPTURE_VALIDATION.exists():
        raise FileExistsError(f"refusing to overwrite validation: {CAPTURE_VALIDATION}")
    errors: list[str] = []
    manifest_path = CAPTURE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "corpus_v2_a6_corrected_capture":
        errors.append("wrong capture kind")
    if manifest.get("complete") is not True or manifest.get("production_capture") is not True:
        errors.append("capture is not a complete production output")
    if manifest.get("artifact_count") != 268 or len(manifest.get("artifacts", [])) != 268:
        errors.append("wrong artifact count")
    if manifest.get("texts") != TEXTS or manifest.get("layers") != LAYERS or manifest.get("seq") != SEQ:
        errors.append("wrong registered texts/layers/sequence")
    if manifest.get("amendment_a6_commit") != A6_COMMIT:
        errors.append("wrong A6 boundary")
    if manifest.get("attention_dtype_boundary") != "BF16 content+bias add, then FP32 softmax":
        errors.append("wrong attention dtype boundary")
    parity = manifest.get("stock_attention_parity", {})
    if parity.get("passed") is not True or any(
        case.get("bitwise_equal") is not True or case.get("max_output_delta") != 0.0
        for case in parity.get("cases", {}).values()
    ) or set(parity.get("cases", {})) != {"global", "sliding"}:
        errors.append("stock attention parity did not pass")

    expected_specs = {
        "spec_sha256": ROOT / "CORPUS_V2_SPEC.md",
        "amendment_a1_sha256": ROOT / "CORPUS_V2_AMENDMENT_A1.md",
        "amendment_a6_sha256": ROOT / "ROUND5_AMENDMENT_A6.md",
        "amendment_a7_execution_sha256": ROOT / "ROUND5_AMENDMENT_A7_EXECUTION.md",
        "depth_prereg_sha256": ROOT / "ROUND5_DEPTH_RESOLVED_PREREG.md",
        "execution_plan_sha256": ROOT / "CORPUS_V2_EXECUTION_PLAN.md",
        "tokenizer_sha256": CORPUS_V1 / "tokenizer.json",
        "checkpoint_index_sha256": NVFP4 / "model.safetensors.index.json",
        "config_sha256": NVFP4 / "config.json",
    }
    for field, path in expected_specs.items():
        if manifest.get(field) != sha256_file(path):
            errors.append(f"stale provenance field: {field}")

    source_hashes = manifest.get("source_sha256", {})
    for name in ("corpus_v2_capture.py", "tier2_run.py", "tier2_stream.py", "tier2_nvfp4.py"):
        if source_hashes.get(name) != sha256_file(ROOT / "scripts" / name):
            errors.append(f"stale capture source: {name}")

    packages = {
        "numpy": package_record(np),
        "tokenizers": package_record(tokenizers),
        "torch": package_record(torch),
        "transformers": package_record(transformers),
    }
    if manifest.get("packages") != packages:
        errors.append("runtime package versions/module paths differ")
    modeling_path = Path(
        transformers.models.inkling.modeling_inkling.__file__
    ).resolve()
    if manifest.get("modeling_inkling_sha256") != sha256_file(modeling_path):
        errors.append("stock modeling_inkling source differs")

    input_manifests = {
        "corpus_v2": CORPUS_V2 / "manifest.json",
        "corpus": CORPUS_V1 / "manifest.json",
    }
    for name, path in input_manifests.items():
        if manifest.get("input_manifest_sha256", {}).get(name) != sha256_file(path):
            errors.append(f"input manifest differs: {name}")
    ids_by_text: dict[str, np.ndarray] = {}
    for text in TEXTS:
        path = TEXT_ROOT[text] / f"{text}.ids.npy"
        if manifest.get("input_ids_sha256", {}).get(text) != sha256_file(path):
            errors.append(f"input IDs differ: {text}")
        ids = np.load(path, allow_pickle=False)
        if ids.shape != (SEQ,) or ids.dtype != np.int32:
            errors.append(f"invalid input IDs: {text}")
        ids_by_text[text] = ids

    index = json.loads((NVFP4 / "model.safetensors.index.json").read_text(encoding="utf-8"))
    indexed_files = sorted(set(index["weight_map"].values()))
    shards = [name for name in indexed_files if name.startswith("model-")]
    nontrunk = [name for name in indexed_files if not name.startswith("model-")]
    shard_records = manifest.get("checkpoint_shards", {})
    if (
        len(shards) != 33
        or nontrunk != ["mtp.safetensors"]
        or manifest.get("checkpoint_index_nontrunk_files") != nontrunk
        or set(shard_records) != set(shards)
    ):
        errors.append("checkpoint shard inventory differs")
    else:
        for ordinal, name in enumerate(shards, start=1):
            path = NVFP4 / name
            record = shard_records[name]
            if record.get("bytes") != path.stat().st_size or not isinstance(record.get("sha256"), str):
                errors.append(f"bad checkpoint shard record: {name}")
                continue
            if args.rehash_shards:
                print(f"independently hashing shard {ordinal:02d}/33: {name}", flush=True)
                if record["sha256"] != sha256_file(path, chunk=3 << 20):
                    errors.append(f"checkpoint shard hash mismatch: {name}")

    expected_paths = {
        f"rvec/rvec_L{layer:02d}_{text}.npy"
        for layer in LAYERS
        for text in TEXTS
    } | {f"nll/nll_{text}.npz" for text in TEXTS}
    records = manifest.get("artifacts", [])
    record_paths = [record.get("path") for record in records]
    if len(record_paths) != len(set(record_paths)) or set(record_paths) != expected_paths:
        errors.append("artifact path inventory differs")
    for ordinal, record in enumerate(records, start=1):
        relative = str(record.get("path"))
        try:
            path = safe_artifact(relative)
        except Exception:
            errors.append(f"unsafe artifact path: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        if record.get("bytes") != path.stat().st_size:
            errors.append(f"artifact size mismatch: {relative}")
        if sha256_file(path) != record.get("sha256"):
            errors.append(f"artifact hash mismatch: {relative}")
        if ordinal % 32 == 0:
            print(f"validated {ordinal}/{len(records)} artifacts", flush=True)
        if relative.startswith("rvec/"):
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (SEQ, 64, 16) or values.dtype != np.float16:
                errors.append(f"invalid r-vector shape/dtype: {relative}")
            elif not np.isfinite(values).all():
                errors.append(f"non-finite r-vector: {relative}")
        else:
            text = str(record.get("text"))
            with np.load(path, allow_pickle=False) as data:
                positions = data["target_position"]
                target_ids = data["target_id"]
                nll = data["nll"]
            if (
                not np.array_equal(positions, np.arange(1, SEQ, dtype=np.int32))
                or not np.array_equal(target_ids, ids_by_text[text][1:])
                or nll.shape != (SEQ - 1,)
                or nll.dtype != np.float32
                or not np.isfinite(nll).all()
            ):
                errors.append(f"invalid NLL artifact: {relative}")

    report = {
        "schema_version": 1,
        "kind": "corpus_v2_a6_capture_independent_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(Path(__file__)),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "capture_git_head": manifest.get("git_head"),
        "artifact_count": len(records),
        "checkpoint_shards_recorded": len(shard_records),
        "checkpoint_shards_rehashed": bool(args.rehash_shards),
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(CAPTURE_VALIDATION, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def midrank(values: np.ndarray, bin_size: int) -> np.ndarray:
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
    xr = midrank(x, len(x))
    yr = midrank(y, len(y))
    xc = xr - xr.mean(dtype=np.float64)
    yc = yr - yr.mean(dtype=np.float64)
    denominator = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
    return float(np.sum(xc * yc) / denominator)


def averaged_scores(text: str, layers: list[int], bin_size: int) -> np.ndarray:
    scores = []
    for layer in layers:
        with np.load(APERTURE / f"aperture_L{layer:02d}_{text}.npz") as data:
            scores.append(midrank(data["aperture_full"], bin_size))
    return np.mean(scores, axis=0, dtype=np.float64)


def layer_scores(text: str) -> dict[int, np.ndarray]:
    result = {}
    for layer in APERTURE_LAYERS:
        with np.load(APERTURE / f"aperture_L{layer:02d}_{text}.npz") as data:
            result[layer] = midrank(data["aperture_full"], BIN_SIZE)
    return result


def load_nll(text: str) -> np.ndarray:
    with np.load(CAPTURE / "nll" / f"nll_{text}.npz") as data:
        return np.asarray(data["nll"], dtype=np.float32)


def binned_spearman(scores: np.ndarray, nll: np.ndarray) -> dict[str, Any]:
    correlations = []
    for start in range(0, SEQ, 512):
        stop = min(start + 512, SEQ)
        positions = np.arange(max(1, start), stop, dtype=np.int64)
        correlations.append(spearman(scores[positions], nll[positions - 1]))
    median = float(np.median(correlations))
    positive = int(np.count_nonzero(np.asarray(correlations) > 0))
    return {
        "bin_correlations": correlations,
        "median_spearman": median,
        "positive_bins": positive,
        "total_bins": 16,
        "passed": bool(median > 0 and positive >= 12),
    }


def seed_v20(label: str) -> int:
    return int(hashlib.sha256(f"{REGISTRATION_COMMIT}|corpus-v2|{label}".encode()).hexdigest()[:16], 16)


def seed_depth(label: str) -> int:
    return int(hashlib.sha256(f"{A6_COMMIT}|P-d|{label}".encode()).hexdigest()[:16], 16)


def permutation(
    scores: np.ndarray,
    positions: list[int],
    *,
    direction: Literal["positive", "negative"],
    seed: int,
) -> dict[str, float]:
    selected = np.asarray(sorted(set(int(value) for value in positions)), dtype=np.int64)
    observed = float(np.median(scores[selected]) - 0.5)
    blocks, counts = [], []
    for start in range(0, SEQ, BIN_SIZE):
        stop = min(start + BIN_SIZE, SEQ)
        blocks.append(np.arange(start, stop, dtype=np.int64))
        counts.append(int(np.count_nonzero((selected >= start) & (selected < stop))))
    rng = np.random.Generator(np.random.PCG64(seed))
    exceed = 0
    for _ in range(PERMUTATIONS):
        sampled = np.concatenate(
            [rng.choice(block, size=count, replace=False) for block, count in zip(blocks, counts) if count]
        )
        value = float(np.median(scores[sampled]) - 0.5)
        exceed += int(value >= observed if direction == "positive" else value <= observed)
    return {"effect": observed, "p": float((1 + exceed) / (PERMUTATIONS + 1))}


def band_permutation(
    scores: dict[int, np.ndarray],
    positions: list[int],
    *,
    direction: Literal["positive", "negative"],
    seed: int,
) -> dict[str, Any]:
    selected = np.asarray(sorted(set(int(value) for value in positions)), dtype=np.int64)
    layers = sorted(scores)
    per_layer = {str(layer): float(np.median(scores[layer][selected]) - 0.5) for layer in layers}
    observed = float(np.median(list(per_layer.values())))
    blocks, counts = [], []
    for start in range(0, SEQ, BIN_SIZE):
        stop = min(start + BIN_SIZE, SEQ)
        blocks.append(np.arange(start, stop, dtype=np.int64))
        counts.append(int(np.count_nonzero((selected >= start) & (selected < stop))))
    rng = np.random.Generator(np.random.PCG64(seed))
    exceed = 0
    for _ in range(PERMUTATIONS):
        sampled = np.concatenate(
            [rng.choice(block, size=count, replace=False) for block, count in zip(blocks, counts) if count]
        )
        value = float(
            np.median([np.median(scores[layer][sampled]) - 0.5 for layer in layers])
        )
        exceed += int(value >= observed if direction == "positive" else value <= observed)
    return {
        "effect": observed,
        "per_layer_effects": per_layer,
        "p": float((1 + exceed) / (PERMUTATIONS + 1)),
    }


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def compare(errors: list[str], label: str, actual: Any, expected: Any, tolerance: float = 1e-12) -> None:
    if isinstance(expected, (float, np.floating)):
        if not np.isclose(float(actual), float(expected), rtol=0, atol=tolerance):
            errors.append(f"{label}: {actual!r} != {expected!r}")
    elif actual != expected:
        errors.append(f"{label}: {actual!r} != {expected!r}")


def confirm_command(_args: argparse.Namespace) -> None:
    if VERIFICATION.exists():
        raise FileExistsError(f"refusing to overwrite verification: {VERIFICATION}")
    errors: list[str] = []
    validation = json.loads(CAPTURE_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("passed") is not True or validation.get("capture_manifest_sha256") != sha256_file(CAPTURE / "manifest.json"):
        errors.append("capture validation gate failed or is stale")
    aperture_manifest = json.loads((APERTURE / "manifest.json").read_text(encoding="utf-8"))
    if aperture_manifest.get("complete") is not True or len(aperture_manifest.get("files", {})) != 32:
        errors.append("aperture dump incomplete")
    else:
        for record in aperture_manifest["files"].values():
            if sha256_file(APERTURE / record["path"]) != record["sha256"]:
                errors.append(f"aperture hash mismatch: {record['path']}")

    novelty = json.loads(NOVELTY.read_text(encoding="utf-8"))
    means = {text: float(np.mean(load_nll(text), dtype=np.float64)) for text in TEXTS}
    for text, value in means.items():
        compare(errors, f"novelty mean {text}", novelty["mean_nll"][text], value)
    novelty_pass = bool(
        means["07_slack_human"] >= 1.5
        and means["08_math_llm"] >= 1.5
        and means["07_slack_human"] > means["08_math_llm"]
    )
    contaminated = [text for text in V20_TEXTS if means[text] < 1.0]
    compare(errors, "novelty decision", novelty["prediction_passed"], novelty_pass)
    compare(errors, "novelty contamination", novelty["contaminated_arms"], contaminated)

    report = json.loads(READOUTS.read_text(encoding="utf-8"))
    classes = json.loads(CLASSES_V20.read_text(encoding="utf-8"))
    scores512 = {text: averaged_scores(text, MID_GLOBALS, 512) for text in V20_TEXTS}
    nll = {text: load_nll(text) for text in V20_TEXTS}
    p2 = {text: binned_spearman(scores512[text], nll[text]) for text in V20_TEXTS}
    p2_cross = {
        "slack_aperture_math_nll": binned_spearman(scores512["07_slack_human"], nll[MATH]),
        "math_aperture_slack_nll": binned_spearman(scores512[MATH], nll["07_slack_human"]),
    }
    for text, independent in p2.items():
        stored = report["p_v2_2"]["primary"][text]
        compare(errors, f"P-v2-2 median {text}", stored["median_spearman"], independent["median_spearman"])
        compare(errors, f"P-v2-2 positive {text}", stored["positive_bins"], independent["positive_bins"])
        compare(errors, f"P-v2-2 decision {text}", stored["passed"], independent["passed"])
        for index, value in enumerate(independent["bin_correlations"]):
            compare(errors, f"P-v2-2 bin {text}/{index}", stored["bin_correlations"][index], value)
    for name, independent in p2_cross.items():
        stored = report["p_v2_2"]["cross_arm_controls"][name]
        compare(errors, f"P-v2-2 control median {name}", stored["median_spearman"], independent["median_spearman"])
        compare(errors, f"P-v2-2 control decision {name}", stored["passed"], independent["passed"])

    slack_scores = averaged_scores("07_slack_human", MID_GLOBALS, 256)
    math_scores = averaged_scores(MATH, MID_GLOBALS, 256)
    specs = [("message_starts", "positive"), ("pronouns", "negative"), ("function_words", "negative")]
    v20_primary, v20_controls = [], []
    for name, direction in specs:
        positions = classes["classes"][name]
        v20_primary.append(
            permutation(slack_scores, positions, direction=direction, seed=seed_v20(f"primary|{name}"))
        )
        v20_controls.append(
            permutation(math_scores, positions, direction=direction, seed=seed_v20(f"cross-control|{name}"))
        )
    primary_holm = holm([item["p"] for item in v20_primary])
    control_holm = holm([item["p"] for item in v20_controls])
    v20_decisions: dict[str, bool] = {}
    v20_nulls: dict[str, bool] = {}
    for index, (name, direction) in enumerate(specs):
        stored_primary = next(item for item in report["class_primary"] if item["name"] == name)
        stored_control = next(item for item in report["class_cross_arm_controls"] if item["name"] == name)
        independent = v20_primary[index]
        independent_control = v20_controls[index]
        decision = bool(
            (independent["effect"] > 0 if direction == "positive" else independent["effect"] < 0)
            and primary_holm[index] < 0.05
        )
        null_pass = bool(control_holm[index] >= 0.05)
        v20_decisions[name] = decision
        v20_nulls[name] = null_pass
        for field, value in (("effect", independent["effect"]), ("p", independent["p"]), ("p_holm", primary_holm[index])):
            compare(errors, f"v2 class {name} {field}", stored_primary[field], value)
        compare(errors, f"v2 class {name} decision", stored_primary["prediction_passed"], decision)
        for field, value in (("effect", independent_control["effect"]), ("p", independent_control["p"]), ("p_holm", control_holm[index])):
            compare(errors, f"v2 control {name} {field}", stored_control[field], value)
        compare(errors, f"v2 control {name} decision", stored_control["passed_as_null"], null_pass)
    v20_summary = {
        "P-v2-1": novelty_pass,
        "P-v2-2": all(item["passed"] for item in p2.values()),
        "P-v2-3": v20_decisions["message_starts"],
        "P-v2-4": v20_decisions["pronouns"] and v20_decisions["function_words"],
    }
    compare(errors, "v2 prediction summary", report["prediction_summary"], v20_summary)
    v20_controls_passed = bool(
        not any(item["passed"] for item in p2_cross.values())
        and all(v20_nulls.values())
    )
    compare(errors, "v2 controls summary", report["all_true_null_controls_passed"], v20_controls_passed)

    depth_report = json.loads(DEPTH_READOUTS.read_text(encoding="utf-8"))
    depth_classes = json.loads(CLASSES_DEPTH.read_text(encoding="utf-8"))
    scores = {text: layer_scores(text) for text in (SLACK_V21, PROSE, MATH)}
    depth_specs = [
        ("P-d1", SLACK_V21, "speaker_labels", DEEP, "positive"),
        ("P-d2", SLACK_V21, "pronouns", SHALLOW, "negative"),
        ("P-d3", SLACK_V21, "first_content", DEEP, "negative"),
        ("P-d4", PROSE, "sentence_starts", DEEP, "positive"),
    ]
    families: dict[str, list[dict[str, Any]]] = {
        "primary": [], "cross": [], "random_source": [], "random_math": []
    }
    thresholds: list[bool] = []
    for prediction, text, class_name, target_layers, direction in depth_specs:
        positions = depth_classes["classes"][class_name]
        primary = band_permutation(
            {layer: scores[text][layer] for layer in target_layers},
            positions, direction=direction, seed=seed_depth(f"primary|{prediction}")
        )
        shallow_effect = float(np.median([np.median(scores[text][layer][positions]) - 0.5 for layer in SHALLOW]))
        deep_effect = float(np.median([np.median(scores[text][layer][positions]) - 0.5 for layer in DEEP]))
        terminal_effect = float(np.median([np.median(scores[text][layer][positions]) - 0.5 for layer in TERMINAL]))
        if prediction == "P-d1":
            threshold = deep_effect >= 0.10 and shallow_effect <= 0.0
        elif prediction == "P-d2":
            threshold = shallow_effect <= -0.05 and abs(terminal_effect) < 0.03
        elif prediction == "P-d3":
            threshold = deep_effect <= -0.05
        else:
            threshold = deep_effect >= 0.05
        primary.update(shallow=shallow_effect, deep=deep_effect, terminal=terminal_effect)
        families["primary"].append(primary)
        thresholds.append(bool(threshold))
        families["cross"].append(
            band_permutation(
                {layer: scores[MATH][layer] for layer in target_layers},
                positions, direction=direction, seed=seed_depth(f"cross-math|{prediction}")
            )
        )
        families["random_source"].append(
            band_permutation(
                {layer: scores[text][layer] for layer in target_layers},
                depth_classes["random_masks"][prediction][text],
                direction=direction, seed=seed_depth(f"random-source|{prediction}")
            )
        )
        families["random_math"].append(
            band_permutation(
                {layer: scores[MATH][layer] for layer in target_layers},
                depth_classes["random_masks"][prediction][MATH],
                direction=direction, seed=seed_depth(f"random-math|{prediction}")
            )
        )
    adjusted = {name: holm([item["p"] for item in family]) for name, family in families.items()}
    stored_families = {
        "primary": depth_report["primary"],
        "cross": depth_report["crossed_math_controls"],
        "random_source": depth_report["random_position_controls_source"],
        "random_math": depth_report["random_position_controls_math"],
    }
    depth_summary: dict[str, bool] = {}
    all_depth_controls = True
    for family_name, family in families.items():
        for index, independent in enumerate(family):
            prediction = depth_specs[index][0]
            stored = stored_families[family_name][index]
            compare(errors, f"{family_name} {prediction} effect", stored["effect"], independent["effect"])
            compare(errors, f"{family_name} {prediction} p", stored["p"], independent["p"])
            compare(errors, f"{family_name} {prediction} p_holm", stored["p_holm"], adjusted[family_name][index])
            for layer, value in independent["per_layer_effects"].items():
                compare(errors, f"{family_name} {prediction} L{layer}", stored["per_layer_effects"][layer], value)
            if family_name == "primary":
                compare(errors, f"primary {prediction} shallow", stored["shallow_band_effect"], independent["shallow"])
                compare(errors, f"primary {prediction} deep", stored["deep_band_effect"], independent["deep"])
                compare(errors, f"primary {prediction} terminal", stored["terminal_band_effect"], independent["terminal"])
                compare(errors, f"primary {prediction} threshold", stored["threshold_passed"], thresholds[index])
                decision = bool(thresholds[index] and adjusted[family_name][index] < 0.05)
                compare(errors, f"primary {prediction} decision", stored["prediction_passed"], decision)
                depth_summary[prediction] = decision
            else:
                null_pass = bool(adjusted[family_name][index] >= 0.05)
                compare(errors, f"{family_name} {prediction} null", stored["passed_as_null"], null_pass)
                all_depth_controls &= null_pass
    compare(errors, "depth prediction summary", depth_report["prediction_summary"], depth_summary)
    compare(errors, "depth threshold summary", depth_report["threshold_summary"], {
        depth_specs[index][0]: thresholds[index] for index in range(4)
    })
    compare(errors, "depth controls summary", depth_report["all_true_null_controls_passed"], all_depth_controls)

    verification = {
        "schema_version": 1,
        "kind": "corpus_v2_a6_independent_readout_verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(Path(__file__)),
        "capture_validation_sha256": sha256_file(CAPTURE_VALIDATION),
        "novelty_sha256": sha256_file(NOVELTY),
        "readouts_sha256": sha256_file(READOUTS),
        "depth_readouts_sha256": sha256_file(DEPTH_READOUTS),
        "independent_v2_summary": v20_summary,
        "independent_depth_summary": depth_summary,
        "independent_v2_controls_passed": v20_controls_passed,
        "independent_depth_controls_passed": all_depth_controls,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(VERIFICATION, verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def self_test() -> None:
    ranks = midrank(np.array([3.0, 1.0, 1.0, 4.0, 2.0]), 5)
    if not np.array_equal(ranks, np.array([0.7, 0.2, 0.2, 0.9, 0.5])):
        raise AssertionError("midrank self-test failed")
    if not np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]):
        raise AssertionError("Holm self-test failed")
    print("corrected verifier self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-capture")
    validate.add_argument("--rehash-shards", action="store_true")
    subparsers.add_parser("confirm")
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "validate-capture":
        validate_capture_command(args)
    elif args.command == "confirm":
        confirm_command(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
