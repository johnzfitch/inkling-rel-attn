"""Independent corpus-v2 capture validation and readout confirmation.

This file deliberately does not import corpus_v2_capture or corpus_v2_pipeline.
It provides two ordered commands: validate-capture before any outcome is read,
then confirm after the registered dump-first analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus_v2"
TOKENIZER_PATH = ROOT / "corpus" / "tokenizer.json"
CAPTURE = ROOT / "dumps" / "round5" / "corpus_v2_capture"
CAPTURE_VALIDATION = ROOT / "analysis" / "round5" / "corpus_v2" / "capture_validation.json"
CLASSES = CORPUS / "classes.json"
WEIGHTS = ROOT / "weights"
APERTURE_DUMP = ROOT / "dumps" / "round5" / "corpus_v2_aperture"
NOVELTY_REPORT = ROOT / "analysis" / "round5" / "corpus_v2" / "novelty.json"
READOUT_REPORT = ROOT / "analysis" / "round5" / "corpus_v2" / "readouts.json"
VERIFICATION_REPORT = ROOT / "analysis" / "round5" / "corpus_v2" / "verification.json"
CONFIRMATION_REPORT = ROOT / "analysis" / "round5" / "corpus_v2" / "confirmation.json"
TEXTS = ["07_slack_human", "08_math_llm"]
MID_GLOBALS = [23, 29, 35, 41, 47]
REGISTRATION_COMMIT = "7fb84ab4e5d164bc759c32e38b0e58803abd5299"
PUBLIC_BOUNDARY_COMMIT = "65b220c2d185829dfc4c8e617a67e673d2fa9cd2"
PERMUTATIONS = 10000

PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
}
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "as",
    "at", "by", "for", "from", "in", "into", "of", "on", "onto", "to",
    "with", "without", "is", "am", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "have", "has", "had", "can", "could",
    "may", "might", "must", "shall", "should", "will", "would", "not",
}


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


def seed_for(label: str) -> int:
    digest = hashlib.sha256(
        f"{REGISTRATION_COMMIT}|corpus-v2|{label}".encode()
    ).hexdigest()
    return int(digest[:16], 16)


def validate_capture_command(_args: argparse.Namespace) -> None:
    errors: list[str] = []
    manifest_path = CAPTURE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    private_manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "corpus_v2_registered_capture":
        errors.append("wrong capture kind")
    if manifest.get("complete") is not True or manifest.get("production_capture") is not True:
        errors.append("capture is not complete production output")
    if manifest.get("registration_commit") != REGISTRATION_COMMIT:
        errors.append("wrong registration commit")
    if manifest.get("public_boundary_commit") != PUBLIC_BOUNDARY_COMMIT:
        errors.append("wrong public boundary commit")
    if manifest.get("texts") != TEXTS or manifest.get("layers") != list(range(66)):
        errors.append("wrong registered text/layer set")
    if manifest.get("seq") != 8192 or manifest.get("normalized_inputs_captured") is not False:
        errors.append("wrong sequence or normalized-input mode")
    if manifest.get("attention_meter_enabled") is not False:
        errors.append("measurement-only attention meter was unexpectedly enabled")
    if manifest.get("private_manifest_sha256") != sha256_file(CORPUS / "manifest.json"):
        errors.append("private manifest hash mismatch")
    source_hash = manifest.get("source_sha256", {}).get("corpus_v2_capture.py")
    if source_hash != sha256_file(ROOT / "scripts" / "corpus_v2_capture.py"):
        errors.append("capture source hash is stale")

    artifacts = manifest.get("artifacts", [])
    if len(artifacts) != 134 or manifest.get("artifact_count") != 134:
        errors.append("wrong artifact count")
    paths = [record.get("path") for record in artifacts]
    if len(paths) != len(set(paths)):
        errors.append("duplicate artifact paths")
    kind_counts = Counter(record.get("kind") for record in artifacts)
    if kind_counts != Counter({"rvec": 132, "next_token_nll": 2}):
        errors.append(f"wrong artifact kinds: {dict(kind_counts)}")

    rvec_checked = 0
    nll_checked = 0
    values_checked = 0
    pattern = re.compile(r"rvec/rvec_L(\d{2})_(07_slack_human|08_math_llm)\.npy")
    for record in artifacts:
        relative = record.get("path")
        path = CAPTURE / str(relative)
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        if sha256_file(path) != record.get("sha256"):
            errors.append(f"artifact hash mismatch: {relative}")
            continue
        if path.stat().st_size != record.get("bytes"):
            errors.append(f"artifact byte-count mismatch: {relative}")
        if record.get("kind") == "rvec":
            match = pattern.fullmatch(str(relative))
            if not match:
                errors.append(f"bad r-vector path: {relative}")
                continue
            layer = int(match.group(1))
            text = match.group(2)
            if layer != record.get("layer") or text != record.get("text"):
                errors.append(f"r-vector metadata mismatch: {relative}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (8192, 64, 16) or array.dtype != np.float16:
                errors.append(f"r-vector shape/dtype mismatch: {relative}")
            elif not np.isfinite(array).all():
                errors.append(f"non-finite r-vector: {relative}")
            else:
                values_checked += int(array.size)
            rvec_checked += 1
        elif record.get("kind") == "next_token_nll":
            text = record.get("text")
            if text not in TEXTS:
                errors.append(f"unknown NLL text: {text}")
                continue
            with np.load(path) as data:
                target_position = data["target_position"]
                target_id = data["target_id"]
                nll = data["nll"]
            ids = np.load(CORPUS / f"{text}.ids.npy", allow_pickle=False)
            expected_hash = private_manifest["texts"][text]["ids_sha256"]
            if sha256_file(CORPUS / f"{text}.ids.npy") != expected_hash:
                errors.append(f"private ID hash mismatch: {text}")
            checks = [
                np.array_equal(target_position, np.arange(1, 8192, dtype=np.int32)),
                target_id.dtype == np.int32 and np.array_equal(target_id, ids[1:]),
                nll.shape == (8191,) and nll.dtype == np.float32,
                bool(np.isfinite(nll).all()),
                bool(np.min(nll) >= -1e-4),
            ]
            if not all(checks):
                errors.append(f"invalid NLL artifact: {relative}")
            else:
                values_checked += int(nll.size)
            nll_checked += 1

    expected_paths = {
        f"rvec/rvec_L{layer:02d}_{text}.npy"
        for layer in range(66)
        for text in TEXTS
    } | {f"nll/nll_{text}.npz" for text in TEXTS}
    if set(paths) != expected_paths:
        errors.append("registered artifact path set mismatch")

    report = {
        "schema_version": 1,
        "kind": "corpus_v2_capture_independent_validation",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        "source_sha256": sha256_file(Path(__file__)),
        "capture_source_sha256": source_hash,
        "capture_git_head": manifest.get("git_head"),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "artifacts_hashed": len(artifacts),
        "rvec_checked": rvec_checked,
        "nll_checked": nll_checked,
        "values_checked": values_checked,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(CAPTURE_VALIDATION, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def independent_classes() -> dict[str, list[int]]:
    private_manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    ids_path = CORPUS / "07_slack_human.ids.npy"
    if sha256_file(ids_path) != private_manifest["texts"]["07_slack_human"]["ids_sha256"]:
        raise RuntimeError("Slack ID hash mismatch")
    ids = np.load(ids_path, allow_pickle=False)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    fragments = tokenizer.decode_batch([[int(value)] for value in ids], skip_special_tokens=False)
    pronouns = []
    function_words = []
    for position, fragment in enumerate(fragments):
        word = fragment.strip().casefold()
        normalized = word if re.fullmatch(r"[a-z]+", word) else None
        if normalized in PRONOUNS:
            pronouns.append(position)
        if normalized in FUNCTION_WORDS:
            function_words.append(position)
    sidecar = json.loads((CORPUS / "07_slack_human.sidecar.json").read_text(encoding="utf-8"))
    return {
        "message_starts": sorted(int(value) for value in sidecar["unit_start_tokens"]),
        "pronouns": pronouns,
        "function_words": function_words,
    }


def independent_rank_percentiles(values: np.ndarray, bin_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    for left in range(0, len(values), bin_size):
        right = min(left + bin_size, len(values))
        block = values[left:right]
        _unique, inverse, counts = np.unique(
            block, return_inverse=True, return_counts=True
        )
        cumulative = np.cumsum(counts)
        first = cumulative - counts + 1
        last = cumulative
        average_rank = (first + last) / 2.0
        output[left:right] = (average_rank[inverse] - 0.5) / len(block)
    return output


def independent_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = independent_rank_percentiles(x, len(x))
    y_rank = independent_rank_percentiles(y, len(y))
    x_centered = x_rank - x_rank.mean(dtype=np.float64)
    y_centered = y_rank - y_rank.mean(dtype=np.float64)
    scale = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if scale == 0:
        raise RuntimeError("constant Spearman input")
    return float(np.dot(x_centered, y_centered) / scale)


def independent_aperture(rvec: np.ndarray, projection: np.ndarray) -> np.ndarray:
    result = np.empty(8192, dtype=np.float64)
    projection64 = np.asarray(projection, dtype=np.float64)
    for left in range(0, 8192, 17):
        right = min(left + 17, 8192)
        coefficients = np.asarray(rvec[left:right], dtype=np.float64)
        curves = np.einsum("thc,cd->thd", coefficients, projection64, optimize=False)
        magnitude = np.abs(curves)
        numerator = magnitude[:, :, 129:].sum(axis=(1, 2), dtype=np.float64)
        denominator = magnitude.sum(axis=(1, 2), dtype=np.float64)
        result[left:right] = numerator / denominator
    return result


def independent_binned_spearman(scores: np.ndarray, nll: np.ndarray) -> dict[str, Any]:
    correlations = []
    for bin_index in range(16):
        left = bin_index * 512
        right = (bin_index + 1) * 512
        positions = np.arange(max(1, left), right, dtype=np.int64)
        correlations.append(
            independent_spearman(scores[positions], nll[positions - 1])
        )
    median = float(np.median(np.asarray(correlations, dtype=np.float64)))
    positive = sum(value > 0 for value in correlations)
    return {
        "bin_correlations": correlations,
        "median_spearman": median,
        "positive_bins": int(positive),
        "passed": bool(median > 0 and positive >= 12),
    }


def independent_permutation(
    scores: np.ndarray,
    positions: list[int],
    *,
    direction: str,
    seed: int,
) -> dict[str, Any]:
    class_positions = np.asarray(sorted(set(positions)), dtype=np.int64)
    observed = float(np.median(scores[class_positions]) - 0.5)
    rng = np.random.Generator(np.random.PCG64(seed))
    bin_specs = []
    for bin_index in range(32):
        left = bin_index * 256
        right = left + 256
        count = int(np.sum((class_positions >= left) & (class_positions < right)))
        if count:
            bin_specs.append((np.arange(left, right, dtype=np.int64), count))
    null = np.empty(PERMUTATIONS, dtype=np.float64)
    for iteration in range(PERMUTATIONS):
        chosen: list[int] = []
        for candidates, count in bin_specs:
            chosen.extend(int(value) for value in rng.choice(candidates, count, replace=False))
        null[iteration] = float(np.median(scores[np.asarray(chosen, dtype=np.int64)]) - 0.5)
    if direction == "positive":
        exceed = int(np.sum(null >= observed))
    else:
        exceed = int(np.sum(null <= observed))
    return {
        "effect": observed,
        "p": float((1 + exceed) / (PERMUTATIONS + 1)),
    }


def independent_holm(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    previous = 0.0
    for rank, (index, value) in enumerate(indexed):
        previous = max(previous, (len(values) - rank) * value)
        output[index] = min(1.0, previous)
    return output


def load_nll(text: str) -> np.ndarray:
    with np.load(CAPTURE / "nll" / f"nll_{text}.npz") as data:
        return np.asarray(data["nll"], dtype=np.float32)


def compare_p2(
    independent: dict[str, Any], reference: dict[str, Any]
) -> bool:
    return bool(
        np.allclose(
            independent["bin_correlations"],
            reference["bin_correlations"],
            atol=1e-12,
            rtol=0,
        )
        and abs(independent["median_spearman"] - reference["median_spearman"]) <= 1e-12
        and independent["positive_bins"] == reference["positive_bins"]
        and independent["passed"] == reference["passed"]
    )


def confirm_command(_args: argparse.Namespace) -> None:
    validation = json.loads(CAPTURE_VALIDATION.read_text(encoding="utf-8"))
    novelty = json.loads(NOVELTY_REPORT.read_text(encoding="utf-8"))
    main = json.loads(READOUT_REPORT.read_text(encoding="utf-8"))
    aperture_manifest = json.loads(
        (APERTURE_DUMP / "manifest.json").read_text(encoding="utf-8")
    )
    frozen_classes = json.loads(CLASSES.read_text(encoding="utf-8"))
    rebuilt_classes = independent_classes()
    class_agreement = rebuilt_classes == {
        name: [int(value) for value in positions]
        for name, positions in frozen_classes["classes"].items()
    }

    recomputed: dict[str, dict[int, np.ndarray]] = {text: {} for text in TEXTS}
    max_aperture_delta = 0.0
    for layer in MID_GLOBALS:
        projection = np.load(
            WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy", allow_pickle=False
        )
        for text in TEXTS:
            rvec = np.load(
                CAPTURE / "rvec" / f"rvec_L{layer:02d}_{text}.npy",
                mmap_mode="r",
            )
            independent = independent_aperture(rvec, projection)
            with np.load(
                APERTURE_DUMP / f"aperture_L{layer:02d}_{text}.npz"
            ) as data:
                blocked = data["aperture_full"]
            max_aperture_delta = max(
                max_aperture_delta,
                float(np.max(np.abs(independent - blocked))),
            )
            recomputed[text][layer] = independent

    novelty_means = {
        text: float(np.mean(load_nll(text), dtype=np.float64)) for text in TEXTS
    }
    novelty_agreement = all(
        abs(novelty_means[text] - float(novelty["mean_nll"][text])) <= 1e-12
        for text in TEXTS
    )

    def independent_scores(text: str, bin_size: int) -> np.ndarray:
        layers = [
            independent_rank_percentiles(recomputed[text][layer], bin_size)
            for layer in MID_GLOBALS
        ]
        return np.mean(layers, axis=0, dtype=np.float64)

    score512 = {text: independent_scores(text, 512) for text in TEXTS}
    nll = {text: load_nll(text) for text in TEXTS}
    p2_primary = {
        text: independent_binned_spearman(score512[text], nll[text]) for text in TEXTS
    }
    p2_cross = {
        "slack_aperture_math_nll": independent_binned_spearman(
            score512["07_slack_human"], nll["08_math_llm"]
        ),
        "math_aperture_slack_nll": independent_binned_spearman(
            score512["08_math_llm"], nll["07_slack_human"]
        ),
    }
    p2_agreement = all(
        compare_p2(p2_primary[text], main["p_v2_2"]["primary"][text])
        for text in TEXTS
    ) and all(
        compare_p2(p2_cross[name], main["p_v2_2"]["cross_arm_controls"][name])
        for name in p2_cross
    )

    score256 = {text: independent_scores(text, 256) for text in TEXTS}
    directions = {
        "message_starts": "positive",
        "pronouns": "negative",
        "function_words": "negative",
    }
    primary = []
    controls = []
    for name, direction in directions.items():
        positions = rebuilt_classes[name]
        primary.append(
            independent_permutation(
                score256["07_slack_human"],
                positions,
                direction=direction,
                seed=seed_for(f"primary|{name}"),
            )
        )
        controls.append(
            independent_permutation(
                score256["08_math_llm"],
                positions,
                direction=direction,
                seed=seed_for(f"cross-control|{name}"),
            )
        )
    primary_adjusted = independent_holm([item["p"] for item in primary])
    control_adjusted = independent_holm([item["p"] for item in controls])
    main_primary = {item["name"]: item for item in main["class_primary"]}
    main_controls = {item["name"]: item for item in main["class_cross_arm_controls"]}
    class_agreements = {}
    for index, name in enumerate(directions):
        primary_ok = bool(
            abs(primary[index]["effect"] - main_primary[name]["effect"]) <= 1e-12
            and primary[index]["p"] == main_primary[name]["p"]
            and primary_adjusted[index] == main_primary[name]["p_holm"]
        )
        control_ok = bool(
            abs(controls[index]["effect"] - main_controls[name]["effect"]) <= 1e-12
            and controls[index]["p"] == main_controls[name]["p"]
            and control_adjusted[index] == main_controls[name]["p_holm"]
        )
        class_agreements[name] = {
            "primary": primary_ok,
            "cross_arm_control": control_ok,
        }
    class_readout_agreement = all(
        all(item.values()) for item in class_agreements.values()
    )

    independent_p2_control = not any(item["passed"] for item in p2_cross.values())
    independent_class_control = all(value >= 0.05 for value in control_adjusted)
    true_null_control_gate = bool(
        independent_p2_control
        and independent_class_control
        and main.get("all_true_null_controls_passed") is True
    )
    prediction_agreement = bool(
        main["prediction_summary"]["P-v2-1"] == novelty["prediction_passed"]
        and main["prediction_summary"]["P-v2-2"]
        == all(item["passed"] for item in p2_primary.values())
        and main["prediction_summary"]["P-v2-3"]
        == main_primary["message_starts"]["prediction_passed"]
        and main["prediction_summary"]["P-v2-4"]
        == bool(
            main_primary["pronouns"]["prediction_passed"]
            and main_primary["function_words"]["prediction_passed"]
        )
    )

    gates = {
        "capture_validation_gate": bool(
            validation.get("passed") is True
            and validation.get("errors") == []
            and validation.get("capture_manifest_sha256")
            == sha256_file(CAPTURE / "manifest.json")
        ),
        "class_freeze_agreement_gate": class_agreement,
        "novelty_rederivation_gate": novelty_agreement,
        "raw_aperture_rederivation_gate": max_aperture_delta <= 1e-12,
        "p_v2_2_rederivation_gate": p2_agreement,
        "class_readout_rederivation_gate": class_readout_agreement,
        "prediction_field_agreement_gate": prediction_agreement,
        "true_null_control_gate": true_null_control_gate,
    }
    verification = {
        "schema_version": 1,
        "kind": "corpus_v2_independent_verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        "source_sha256": sha256_file(Path(__file__)),
        "max_abs_aperture_delta": max_aperture_delta,
        "novelty_means": novelty_means,
        "class_position_counts": {
            name: len(values) for name, values in rebuilt_classes.items()
        },
        "p_v2_2_primary": p2_primary,
        "p_v2_2_cross_arm_controls": p2_cross,
        "class_agreements": class_agreements,
        "independent_primary_effects": {
            name: {
                "effect": primary[index]["effect"],
                "p": primary[index]["p"],
                "p_holm": primary_adjusted[index],
            }
            for index, name in enumerate(directions)
        },
        "independent_control_effects": {
            name: {
                "effect": controls[index]["effect"],
                "p": controls[index]["p"],
                "p_holm": control_adjusted[index],
            }
            for index, name in enumerate(directions)
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    atomic_json(VERIFICATION_REPORT, verification)
    confirmation = {
        "schema_version": 1,
        "kind": "corpus_v2_confirmation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        **gates,
        "methodology_passed": all(gates.values()),
        "prediction_summary": main["prediction_summary"],
        "source_hashes": {
            "capture_validation": sha256_file(CAPTURE_VALIDATION),
            "private_classes": sha256_file(CLASSES),
            "novelty": sha256_file(NOVELTY_REPORT),
            "aperture_manifest": sha256_file(APERTURE_DUMP / "manifest.json"),
            "readouts": sha256_file(READOUT_REPORT),
            "verification": sha256_file(VERIFICATION_REPORT),
        },
    }
    atomic_json(CONFIRMATION_REPORT, confirmation)
    print(json.dumps(confirmation, indent=2, sort_keys=True))
    if not confirmation["methodology_passed"]:
        raise SystemExit(1)


def self_test() -> None:
    values = np.array([3.0, 1.0, 1.0, 4.0, 2.0], dtype=np.float64)
    expected = np.array([0.7, 0.2, 0.2, 0.9, 0.5])
    actual = independent_rank_percentiles(values, len(values))
    if not np.array_equal(actual, expected):
        raise AssertionError((actual, expected))
    if abs(independent_spearman(np.arange(9), np.arange(8, -1, -1)) + 1.0) > 1e-15:
        raise AssertionError("Spearman reversal failed")
    if not np.allclose(independent_holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]):
        raise AssertionError("Holm self-test failed")
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-capture")
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
