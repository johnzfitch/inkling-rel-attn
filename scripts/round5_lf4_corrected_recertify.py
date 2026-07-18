"""Mechanical LF4 re-certification on the A6-corrected widened capture.

All numerical estimators are imported unchanged from round5_lf4_zoom_lens.py;
this wrapper replaces only the capture/provenance gates and output paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import round5_lf4_zoom_lens as frozen
from round5_science_common import (
    CAPTURE,
    MANIFEST_SHA256,
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


DEFAULT_DUMP = ROOT / "dumps" / "round5" / "lf4_a6_corrected"
DEFAULT_OUT = ROOT / "analysis" / "round5" / "lf4_corrected"
LOCI = ROOT / "analysis" / "round5" / "loci.json"
PARITY = ROOT / "analysis" / "round5" / "widened_capture" / "lf5_replay_parity.json"
OLD_REPORT = ROOT / "analysis" / "round5" / "lf4" / "zoom_lens.json"


def require_lf5_parity() -> dict[str, Any]:
    report = json.loads(PARITY.read_text(encoding="utf-8"))
    if (
        report.get("kind") != "round5_lf5_parity"
        or report.get("backend") != "replay"
        or not report.get("passed")
        or report.get("input_manifest_sha256") != MANIFEST_SHA256
        or report.get("layers") != list(range(66))
        or len(report.get("results", [])) != 66
        or not all(row.get("bitwise_equal") and row.get("passed") for row in report["results"])
    ):
        raise RuntimeError("LF5 widened-capture replay parity is missing, failed, or stale")
    return report


def compute(dump: Path, block_tokens: int) -> None:
    _, capture_manifest = require_certified_capture()
    require_lf5_parity()
    records = artifact_index(capture_manifest)
    manifest_path = dump / "manifest.json"
    if dump.exists() and not manifest_path.exists():
        raise RuntimeError(f"existing output directory has no manifest: {dump}")
    dump.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("capture_manifest_sha256") != MANIFEST_SHA256
            or manifest.get("frozen_estimator_sha256") != sha256_file(Path(frozen.__file__))
        ):
            raise RuntimeError("existing corrected LF4 manifest has different inputs or estimator")
    else:
        manifest = {
            "schema_version": 1,
            "kind": "round5_lf4_a6_corrected_aperture_dump",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False,
            "capture_manifest_sha256": MANIFEST_SHA256,
            "capture_validation_sha256": provenance(Path(__file__))["capture_validation_sha256"],
            "lf5_replay_parity_sha256": sha256_file(PARITY),
            "loci_sha256": sha256_file(LOCI),
            "source_sha256": sha256_file(Path(__file__)),
            "frozen_estimator_sha256": sha256_file(Path(frozen.__file__)),
            "block_tokens": block_tokens,
            "layers": list(range(66)),
            "texts": TEXTS,
            "files": {},
        }
        atomic_json(manifest_path, manifest)

    for layer in range(66):
        extent = 1024 if layer in frozen.GLOBAL_LAYERS else 512
        projection_path = ROOT / "weights" / f"layer{layer:02d}_rel_logits_proj.npy"
        projection = np.load(projection_path, mmap_mode="r")
        if projection.shape != (16, extent) or projection.dtype != np.float32:
            raise RuntimeError(f"invalid projection: {projection_path}")
        projection_hash = sha256_file(projection_path)
        for text in TEXTS:
            key = f"L{layer:02d}_{text}"
            output_path = dump / f"aperture_{key}.npz"
            if output_path.exists():
                row = manifest["files"].get(key)
                if row is None or sha256_file(output_path) != row.get("sha256"):
                    raise RuntimeError(f"existing corrected LF4 output is unmanifested/stale: {output_path}")
                continue
            rvec_path = CAPTURE / "replay" / f"rvec_L{layer:02d}_{text}.npy"
            relative = rvec_path.relative_to(CAPTURE).as_posix()
            record = records.get(relative)
            if record is None or record.get("kind") != "rvec":
                raise RuntimeError(f"r-vector is not bound by manifest: {relative}")
            rvec = np.load(rvec_path, mmap_mode="r")
            started = time.time()
            arrays = frozen.aperture_blocked(rvec, projection, block_tokens=block_tokens)
            atomic_npz(output_path, **arrays)
            manifest["files"][key] = {
                "path": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "rvec_sha256": record["sha256"],
                "projection_sha256": projection_hash,
                "extent": extent,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            manifest["last_completed"] = key
            atomic_json(manifest_path, manifest)
            print(f"corrected LF4 {key}: {manifest['files'][key]['elapsed_seconds']:.2f}s", flush=True)
    if len(manifest["files"]) != 396:
        raise RuntimeError(f"corrected LF4 output count={len(manifest['files'])}, expected 396")
    manifest["complete"] = True
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(f"sealed {manifest_path}")


def analyze(dump: Path, out: Path, permutations: int) -> None:
    report_path = out / "zoom_lens_corrected.json"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, results_path)
    require_certified_capture()
    parity = require_lf5_parity()
    manifest = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    if (
        not manifest.get("complete")
        or len(manifest.get("files", {})) != 396
        or manifest.get("capture_manifest_sha256") != MANIFEST_SHA256
    ):
        raise RuntimeError("corrected LF4 aperture dump is incomplete or stale")
    loci = json.loads(LOCI.read_text(encoding="utf-8"))
    primary_specs = [
        ("closers", "02_code", "positive"),
        ("pronouns", "01_prose_en", "positive"),
        ("function_words", "01_prose_en", "negative"),
    ]
    primary = []
    for name, text, direction in primary_specs:
        positions = frozen.class_positions(loci, text, name)
        result = frozen.contrast(
            name=name,
            text=text,
            positions=positions,
            direction=direction,
            dump=dump,
            bin_size=256,
            permutations=permutations,
        )
        result["sensitivity"] = {}
        for bin_size in (128, 512):
            sensitivity = frozen.contrast(
                name=name,
                text=text,
                positions=positions,
                direction=direction,
                dump=dump,
                bin_size=bin_size,
                permutations=permutations,
            )
            sensitivity.pop("head_effects")
            sensitivity.pop("per_layer_effects")
            sensitivity.pop("bootstrap_256")
            result["sensitivity"][str(bin_size)] = sensitivity
        random_scores, _ = frozen.averaged_scores(dump, "06_random", 256)
        random_seed = int(
            hashlib.sha256(
                f"{frozen.REGISTRATION_COMMIT}|LF4|null|{name}".encode()
            ).hexdigest()[:16],
            16,
        )
        result["random_position_mask_control"] = frozen.permutation_test(
            random_scores,
            positions,
            bin_size=256,
            direction=direction,
            permutations=permutations,
            seed=random_seed,
        )
        primary.append(result)
    adjusted = frozen.holm_adjust([row["p"] for row in primary])
    null_adjusted = frozen.holm_adjust(
        [row["random_position_mask_control"]["p"] for row in primary]
    )
    for row, p_adjusted, null_p_adjusted in zip(primary, adjusted, null_adjusted):
        row["p_holm"] = p_adjusted
        control = row["random_position_mask_control"]
        control["p_holm"] = null_p_adjusted
        control["passed_as_null"] = null_p_adjusted >= 0.05
        sign_ok = row["effect"] > 0 if row["direction"] == "positive" else row["effect"] < 0
        row["prediction_passed"] = bool(sign_ok and p_adjusted < 0.05)

    secondary_specs = [
        (
            "sentence_starts",
            "01_prose_en",
            frozen.class_positions(loci, "01_prose_en", "sentence_starts"),
        ),
        *[
            (
                f"rare_bpe_{text}",
                text,
                frozen.class_positions(loci, text, "rare_bpe_primary"),
            )
            for text in TEXTS[:-1]
            if loci["texts"][text]["counts"]["rare_bpe_primary"] > 0
        ],
    ]
    secondary = [
        frozen.contrast(
            name=name,
            text=text,
            positions=positions,
            direction="two-sided",
            dump=dump,
            bin_size=256,
            permutations=permutations,
        )
        for name, text, positions in secondary_specs
    ]
    secondary_adjusted = frozen.bh_adjust([row["p"] for row in secondary])
    for row, p_adjusted in zip(secondary, secondary_adjusted):
        row["p_bh"] = p_adjusted

    old = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    old_by_name = {row["name"]: row for row in old["primary"]}
    no_flips = all(
        row["prediction_passed"] == old_by_name[row["name"]]["prediction_passed"]
        for row in primary
    )
    report = {
        "schema_version": 1,
        "kind": "round5_lf4_a6_corrected_recertification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "registration_commit": frozen.REGISTRATION_COMMIT,
        "plan_commit": frozen.PLAN_COMMIT,
        "capture_manifest_sha256": MANIFEST_SHA256,
        "lf5_replay_parity_sha256": sha256_file(PARITY),
        "lf5_replay_values_compared": sum(row["elements"] for row in parity["results"]),
        "frozen_estimator_sha256": sha256_file(Path(frozen.__file__)),
        "dump_manifest_sha256": sha256_file(dump / "manifest.json"),
        "loci_sha256": sha256_file(LOCI),
        "metric": "sum_h,sum_d>128 |b| / sum_h,sum_all_d |b|",
        "primary_depth_layers": frozen.MID_GLOBALS,
        "primary_position_bin": 256,
        "permutations": permutations,
        "primary": primary,
        "secondary": secondary,
        "negative_controls_passed": all(
            row["random_position_mask_control"]["passed_as_null"] for row in primary
        ),
        "all_three_primary_predictions_passed": all(
            row["prediction_passed"] for row in primary
        ),
        "a6_no_verdict_flip_expectation_confirmed": no_flips,
        "old_report_sha256": sha256_file(OLD_REPORT),
        "old_to_corrected": {
            row["name"]: {
                "old_effect": old_by_name[row["name"]]["effect"],
                "corrected_effect": row["effect"],
                "delta": row["effect"] - old_by_name[row["name"]]["effect"],
                "old_prediction_passed": old_by_name[row["name"]]["prediction_passed"],
                "corrected_prediction_passed": row["prediction_passed"],
            }
            for row in primary
        },
        "provenance": provenance(Path(__file__)),
    }
    atomic_json(report_path, report)
    sentence = next(row for row in secondary if row["name"] == "sentence_starts")
    lines = [
        "# LF4 A6-corrected mechanical re-certification",
        "",
        "**Status: answered; independent re-derivation pending.** The frozen estimator and classes are unchanged; only corrected input/provenance paths differ.",
        "",
        f"- A6 no-verdict-flip expectation: **{str(no_flips).lower()}**.",
        *[
            f"- {row['name']}: effect `{row['effect']:+.9f}`, Holm p `{row['p_holm']:.6g}`, prediction **{'pass' if row['prediction_passed'] else 'fail'}** (old effect `{old_by_name[row['name']]['effect']:+.9f}`)."
            for row in primary
        ],
        f"- Sentence-start secondary effect: `{sentence['effect']:+.9f}`, BH p `{sentence['p_bh']:.6g}`.",
        f"- Random-mask controls all null: **{str(report['negative_controls_passed']).lower()}**.",
        "",
    ]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"no_flips": no_flips, "primary": report["old_to_corrected"]}, indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    frozen.self_test()
    print("round5_lf4_corrected_recertify self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    compute_parser = subparsers.add_parser("compute")
    compute_parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    compute_parser.add_argument("--block-tokens", type=int, default=32)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    analyze_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    analyze_parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "compute":
        compute(args.dump, args.block_tokens)
    else:
        analyze(args.dump, args.out, args.permutations)


if __name__ == "__main__":
    main()
