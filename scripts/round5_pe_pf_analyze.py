"""Registered P-e/P-f aperture analyses on the fresh paired corrected capture.

Operational choices not spelled out in the question-level registrations are
frozen by this source commit before paired r-vector outcomes are opened:

* P-e1 is primary on the multi-conversation arm (single-thread is descriptive),
  with dose permuted within the eight frozen segment-open blocks.
* Its interval is a 5,000-resample segment-block bootstrap.
* P-e3's exact dose bins are floor(log2(1+dose)).
* P-f2 exactly preserves the disclosed exploratory reducer: the mean paired
  deep-minus-shallow token score on starts valid for the full -4..+11 profile.
* P-f top-level Holm p-values are the intersection-union P-f1 p, the paired
  deep-vs-shallow sign-flip p for P-f2, and the max of the two band p-values
  for the conjunctive P-f3 test.

No token positions, classes, bands, effects, or thresholds are outcome-derived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import spearmanr

import round5_lf4_zoom_lens as aperture_tools
from round5_science_common import (
    CAPTURE,
    MANIFEST_SHA256,
    ROOT,
    artifact_index,
    atomic_json,
    atomic_npz,
    holm_adjust,
    provenance,
    refuse_existing,
    require_certified_capture,
    self_test_common,
    sha256_file,
)


ARMS = ["09_pe_single_thread", "10_pe_multi_conversation"]
SINGLE, MULTI = ARMS
SHALLOW = [17, 23, 29]
DEEP = [35, 41, 47, 53, 59]
LAYERS = SHALLOW + DEEP
PRIVATE_ROOT = ROOT / "corpus_v2"
PRIVATE_MANIFEST = PRIVATE_ROOT / "pe_manifest.json"
PRIVATE_CLASSES = PRIVATE_ROOT / "pe_classes.json"
PUBLIC_FREEZE = ROOT / "analysis" / "round5" / "pe" / "corpus_freeze.json"
MATH_APERTURE = ROOT / "dumps" / "round5" / "corpus_v2_corrected_aperture"
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "pe_pf_aperture"
DEFAULT_OUT = ROOT / "analysis" / "round5" / "pe_pf"
WEIGHTS = ROOT / "weights"


def seed_for(name: str) -> int:
    freeze_hash = sha256_file(PUBLIC_FREEZE)
    return int.from_bytes(
        hashlib.sha256(f"{freeze_hash}:{name}".encode("utf-8")).digest()[:8],
        "big",
        signed=False,
    )


def require_private_freeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    private_manifest = json.loads(PRIVATE_MANIFEST.read_text(encoding="utf-8"))
    classes = json.loads(PRIVATE_CLASSES.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC_FREEZE.read_text(encoding="utf-8"))
    if (
        private_manifest.get("kind") != "round5_pe_paired_private_corpus"
        or not private_manifest.get("complete")
        or private_manifest.get("classes", {}).get("sha256") != sha256_file(PRIVATE_CLASSES)
        or public.get("kind") != "round5_pe_pf_public_corpus_freeze"
        or public.get("private_manifest_sha256") != sha256_file(PRIVATE_MANIFEST)
        or public.get("private_classes_sha256") != sha256_file(PRIVATE_CLASSES)
        or not public.get("pair_preservation", {}).get("passed")
        or public.get("paired_ordinary_message_count") != 563
    ):
        raise RuntimeError("private/public P-e/P-f freeze is missing, failed, or stale")
    return private_manifest, classes, public


def compute(dump: Path, block_tokens: int) -> None:
    _, capture_manifest = require_certified_capture()
    private_manifest, _, _ = require_private_freeze()
    records = artifact_index(capture_manifest)
    manifest_path = dump / "manifest.json"
    if dump.exists() and not manifest_path.exists():
        raise RuntimeError(f"existing P-e/P-f output has no manifest: {dump}")
    dump.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("capture_manifest_sha256") != MANIFEST_SHA256
            or manifest.get("private_classes_sha256") != sha256_file(PRIVATE_CLASSES)
            or manifest.get("source_sha256") != sha256_file(Path(__file__))
        ):
            raise RuntimeError("existing P-e/P-f aperture manifest is stale")
    else:
        manifest = {
            "schema_version": 1,
            "kind": "round5_pe_pf_corrected_aperture_dump",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False,
            "capture_manifest_sha256": MANIFEST_SHA256,
            "private_manifest_sha256": sha256_file(PRIVATE_MANIFEST),
            "private_classes_sha256": sha256_file(PRIVATE_CLASSES),
            "public_freeze_sha256": sha256_file(PUBLIC_FREEZE),
            "frozen_aperture_estimator_sha256": sha256_file(Path(aperture_tools.__file__)),
            "source_sha256": sha256_file(Path(__file__)),
            "builder_source_sha256": private_manifest["builder_source_sha256"],
            "layers": LAYERS,
            "arms": ARMS,
            "block_tokens": block_tokens,
            "files": {},
        }
        atomic_json(manifest_path, manifest)
    for layer in LAYERS:
        projection_path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        projection = np.load(projection_path, mmap_mode="r")
        if (
            projection.shape != (16, 1024)
            or projection.dtype != np.float32
            or not np.isfinite(projection).all()
        ):
            raise RuntimeError(f"invalid global projection: {projection_path}")
        projection_hash = sha256_file(projection_path)
        for arm in ARMS:
            key = f"L{layer:02d}_{arm}"
            output_path = dump / f"aperture_{key}.npz"
            if output_path.exists():
                row = manifest["files"].get(key)
                if row is None or sha256_file(output_path) != row.get("sha256"):
                    raise RuntimeError(f"unmanifested/stale paired aperture: {output_path}")
                continue
            rvec_path = CAPTURE / "paired" / f"rvec_L{layer:02d}_{arm}.npy"
            relative = rvec_path.relative_to(CAPTURE).as_posix()
            record = records.get(relative)
            if record is None or record.get("kind") != "rvec":
                raise RuntimeError(f"paired r-vector is not bound by manifest: {relative}")
            if sha256_file(rvec_path) != record.get("sha256"):
                raise RuntimeError(f"paired r-vector hash mismatch: {relative}")
            rvec = np.load(rvec_path, mmap_mode="r")
            if rvec.shape != (8192, 64, 16) or rvec.dtype != np.float16:
                raise RuntimeError(f"invalid paired r-vector: {rvec_path}")
            started = time.time()
            arrays = aperture_tools.aperture_blocked(
                rvec, projection, block_tokens=block_tokens
            )
            atomic_npz(
                output_path,
                aperture_full=arrays["aperture_full"],
                full_numerator=arrays["full_numerator"],
                full_denominator=arrays["full_denominator"],
            )
            manifest["files"][key] = {
                "path": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "rvec_sha256": record["sha256"],
                "projection_sha256": projection_hash,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            atomic_json(manifest_path, manifest)
            print(f"P-e/P-f aperture {key}: {manifest['files'][key]['elapsed_seconds']:.2f}s", flush=True)
    if len(manifest["files"]) != 16:
        raise RuntimeError(f"paired aperture count={len(manifest['files'])}, expected 16")
    manifest["complete"] = True
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(f"sealed {manifest_path}")


def load_rank_arrays(dump: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, str]]:
    ranks: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    paired_manifest = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    if (
        paired_manifest.get("kind") != "round5_pe_pf_corrected_aperture_dump"
        or not paired_manifest.get("complete")
    ):
        raise RuntimeError("paired aperture manifest is not sealed")
    for arm in ARMS:
        arm_ranks = []
        for layer in LAYERS:
            path = dump / f"aperture_L{layer:02d}_{arm}.npz"
            record = paired_manifest["files"].get(f"L{layer:02d}_{arm}")
            if record is None or sha256_file(path) != record.get("sha256"):
                raise RuntimeError(f"stale paired aperture: {path}")
            with np.load(path) as data:
                values = np.asarray(data["aperture_full"], dtype=np.float64)
            if values.shape != (8192,) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid paired aperture: {path}")
            arm_ranks.append(aperture_tools.midrank_percentiles(values, 256))
            hashes[path.relative_to(ROOT).as_posix()] = record["sha256"]
        ranks[arm] = np.asarray(arm_ranks)
    math_manifest = json.loads((MATH_APERTURE / "manifest.json").read_text(encoding="utf-8"))
    if not math_manifest.get("complete") or not math_manifest.get("capture_validation_passed"):
        raise RuntimeError("corrected math aperture control is not certified")
    math_ranks = []
    for layer in LAYERS:
        path = MATH_APERTURE / f"aperture_L{layer:02d}_08_math_llm.npz"
        record = math_manifest["files"].get(f"L{layer:02d}_08_math_llm")
        if record is None or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"stale math control aperture: {path}")
        with np.load(path) as data:
            values = np.asarray(data["aperture_full"], dtype=np.float64)
        if values.shape != (8192,) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid math control aperture: {path}")
        math_ranks.append(aperture_tools.midrank_percentiles(values, 256))
        hashes[path.relative_to(ROOT).as_posix()] = record["sha256"]
    return ranks, np.asarray(math_ranks), hashes


def band_effect(ranks: np.ndarray, positions: list[int] | np.ndarray) -> float:
    index = np.asarray(positions, dtype=np.int64)
    if index.size == 0:
        raise ValueError("empty registered class")
    return float(np.median(np.median(ranks[:, index], axis=1)) - 0.5)


def token_band_score(ranks: np.ndarray) -> np.ndarray:
    return np.mean(ranks, axis=0)


def stratified_null_positions(
    positions: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    selected = []
    for start in range(0, 8192, 256):
        stop = start + 256
        count = int(np.sum((positions >= start) & (positions < stop)))
        if count:
            selected.append(rng.choice(np.arange(start, stop), size=count, replace=False))
    return np.concatenate(selected).astype(np.int64)


def class_permutation_test(
    ranks: np.ndarray,
    positions: list[int] | np.ndarray,
    *,
    direction: Literal["positive", "negative", "two-sided"],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    index = np.asarray(sorted(set(map(int, positions))), dtype=np.int64)
    observed = band_effect(ranks, index)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        null[iteration] = band_effect(ranks, stratified_null_positions(index, rng))
    if direction == "positive":
        exceed = np.sum(null >= observed)
    elif direction == "negative":
        exceed = np.sum(null <= observed)
    else:
        exceed = np.sum(np.abs(null) >= abs(observed))
    return {
        "count": int(index.size),
        "effect": observed,
        "direction": direction,
        "p": float((1 + exceed) / (permutations + 1)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q50": float(np.quantile(null, 0.5)),
        "null_q975": float(np.quantile(null, 0.975)),
    }


def spearman_statistic(scores: np.ndarray, doses: np.ndarray) -> float:
    statistic = float(spearmanr(np.log2(1.0 + doses), scores).statistic)
    if not np.isfinite(statistic):
        raise RuntimeError("non-finite Spearman statistic")
    return statistic


def pe1_test(
    scores: np.ndarray,
    boundary_rows: list[dict[str, Any]],
    *,
    permutations: int = 10000,
    bootstrap: int = 5000,
) -> dict[str, Any]:
    positions = np.asarray([row["token"] for row in boundary_rows], dtype=np.int64)
    doses = np.asarray([row["retired_context_dose"] for row in boundary_rows], dtype=np.float64)
    block_labels = np.asarray([row["segment_open_token"] for row in boundary_rows], dtype=np.int64)
    groups = [np.flatnonzero(block_labels == label) for label in sorted(set(block_labels.tolist()))]
    observed_scores = scores[positions]
    observed = spearman_statistic(observed_scores, doses)
    rng = np.random.default_rng(seed_for("P-e1-permutation"))
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        permuted = doses.copy()
        for group in groups:
            permuted[group] = rng.permutation(permuted[group])
        null[iteration] = spearman_statistic(observed_scores, permuted)
    pvalue = float((1 + np.sum(null >= observed)) / (permutations + 1))
    boot_rng = np.random.default_rng(seed_for("P-e1-block-bootstrap"))
    draws = np.empty(bootstrap, dtype=np.float64)
    for iteration in range(bootstrap):
        chosen = boot_rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[group_index] for group_index in chosen])
        draws[iteration] = spearman_statistic(observed_scores[index], doses[index])
    return {
        "n": int(len(positions)),
        "structure_blocks": len(groups),
        "spearman_rho": observed,
        "one_sided_structure_block_permutation_p": pvalue,
        "bootstrap_q025": float(np.quantile(draws, 0.025)),
        "bootstrap_q50": float(np.quantile(draws, 0.5)),
        "bootstrap_q975": float(np.quantile(draws, 0.975)),
        "passed": bool(observed > 0 and pvalue < 0.05),
    }


def pe2_test(
    single_ranks: np.ndarray,
    multi_ranks: np.ndarray,
    pairs: list[dict[str, Any]],
    permutations: int = 10000,
) -> dict[str, Any]:
    single_positions = np.asarray([row["single_token"] for row in pairs], dtype=np.int64)
    multi_positions = np.asarray([row["multi_token"] for row in pairs], dtype=np.int64)
    single_values = single_ranks[:, single_positions]
    multi_values = multi_ranks[:, multi_positions]
    single_effect = band_effect(single_ranks, single_positions)
    multi_effect = band_effect(multi_ranks, multi_positions)
    observed = single_effect - multi_effect
    rng = np.random.default_rng(seed_for("P-e2-paired-arm-label"))
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        swap = rng.integers(0, 2, size=len(pairs), dtype=np.int8).astype(bool)
        left = np.where(swap[None, :], multi_values, single_values)
        right = np.where(swap[None, :], single_values, multi_values)
        null[iteration] = (
            float(np.median(np.median(left, axis=1)) - 0.5)
            - float(np.median(np.median(right, axis=1)) - 0.5)
        )
    pvalue = float((1 + np.sum(null >= observed)) / (permutations + 1))
    return {
        "paired_messages": len(pairs),
        "single_thread_effect": single_effect,
        "multi_conversation_effect": multi_effect,
        "difference": observed,
        "one_sided_paired_label_permutation_p": pvalue,
        "direction_passed": bool(observed > 0),
    }


def pe3_test(
    scores: np.ndarray,
    boundary_rows: list[dict[str, Any]],
    permutations: int = 10000,
) -> dict[str, Any]:
    higher = [row for row in boundary_rows if row["boundary_type"] == "higher_scope"]
    ordinary = [row for row in boundary_rows if row["boundary_type"] == "ordinary_message"]
    higher_positions = np.asarray([row["token"] for row in higher], dtype=np.int64)
    ordinary_positions = np.asarray([row["token"] for row in ordinary], dtype=np.int64)
    raw = float(np.median(scores[higher_positions]) - np.median(scores[ordinary_positions]))
    all_positions = np.concatenate([higher_positions, ordinary_positions])
    higher_count = len(higher_positions)
    rng = np.random.default_rng(seed_for("P-e3-scope-label"))
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        selected_higher_indices = []
        for start in range(0, 8192, 256):
            stop = start + 256
            pool = np.flatnonzero((all_positions >= start) & (all_positions < stop))
            count = int(np.sum((higher_positions >= start) & (higher_positions < stop)))
            if count:
                selected_higher_indices.append(rng.choice(pool, size=count, replace=False))
        selected_indices = np.concatenate(selected_higher_indices)
        selected_mask = np.zeros(len(all_positions), dtype=bool)
        selected_mask[selected_indices] = True
        if int(selected_mask.sum()) != higher_count:
            raise RuntimeError("P-e3 stratified permutation changed class size")
        null[iteration] = float(
            np.median(scores[all_positions[selected_mask]])
            - np.median(scores[all_positions[~selected_mask]])
        )
    pvalue = float((1 + np.sum(null >= raw)) / (permutations + 1))

    ordinary_by_bin: dict[int, list[float]] = {}
    for row in ordinary:
        dose_bin = int(np.floor(np.log2(1 + row["retired_context_dose"])))
        ordinary_by_bin.setdefault(dose_bin, []).append(float(scores[row["token"]]))
    matched = []
    unmatched = 0
    for row in higher:
        dose_bin = int(np.floor(np.log2(1 + row["retired_context_dose"])))
        controls = ordinary_by_bin.get(dose_bin, [])
        if not controls:
            unmatched += 1
            continue
        matched.append(float(scores[row["token"]]) - float(np.median(controls)))
    adjusted = float(np.median(matched)) if matched else float("nan")
    return {
        "higher_scope_n": len(higher),
        "ordinary_n": len(ordinary),
        "raw_median_contrast": raw,
        "one_sided_position_bin_scope_permutation_p": pvalue,
        "direction_passed": bool(raw > 0),
        "dose_bin_definition": "floor(log2(1+retired_context_dose))",
        "dose_matched_higher_n": len(matched),
        "dose_unmatched_higher_n": unmatched,
        "dose_adjusted_median_contrast": adjusted,
        "attenuated_after_dose_matching": bool(np.isfinite(adjusted) and abs(adjusted) < abs(raw)),
    }


def offset_profile(deep: np.ndarray, shallow: np.ndarray, arm: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the disclosed v2.1 -4..+11 paired-message profile exactly."""

    difference = token_band_score(deep) - token_band_score(shallow)
    ordinary = np.asarray(sorted(map(int, arm["classes"]["ordinary_message_starts"])), dtype=np.int64)
    ordinary = ordinary[(ordinary >= 4) & (ordinary + 11 < 8192)]
    values: dict[int, np.ndarray] = {}
    for offset in range(-4, 12):
        values[offset] = difference[ordinary + offset]
    return {
        "difference": difference,
        "ordinary": ordinary.tolist(),
        "offset_values": values,
        "offset_means": {str(offset): float(np.mean(value)) for offset, value in values.items()},
        "offset_counts": {str(offset): int(len(value)) for offset, value in values.items()},
    }


def pf2_test(
    deep: np.ndarray,
    shallow: np.ndarray,
    arm: dict[str, Any],
    *,
    bootstrap: int = 5000,
    permutations: int = 10000,
) -> dict[str, Any]:
    profile = offset_profile(deep, shallow, arm)
    offset0 = profile["offset_values"][0]
    observed = float(np.mean(offset0))
    rng = np.random.default_rng(seed_for("P-f2-message-bootstrap"))
    draws = np.empty(bootstrap, dtype=np.float64)
    for iteration in range(bootstrap):
        index = rng.integers(0, len(offset0), size=len(offset0))
        draws[iteration] = np.mean(offset0[index])
    sign_rng = np.random.default_rng(seed_for("P-f2-depth-sign-flip"))
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        signs = sign_rng.choice(np.asarray([-1.0, 1.0]), size=len(offset0))
        null[iteration] = np.mean(offset0 * signs)
    pvalue = float((1 + np.sum(null >= observed)) / (permutations + 1))
    body_median = float(np.median([profile["offset_means"][str(offset)] for offset in range(6, 12)]))
    q025, q50, q975 = np.quantile(draws, [0.025, 0.5, 0.975])
    return {
        "ordinary_message_n": len(offset0),
        "offset0_deep_minus_shallow": observed,
        "bootstrap_q025": float(q025),
        "bootstrap_q50": float(q50),
        "bootstrap_q975": float(q975),
        "one_sided_paired_depth_sign_flip_p": pvalue,
        "profile_reducer": "mean paired deep-minus-shallow token score at each offset",
        "valid_start_rule": "ordinary start >= 4 and start + 11 < 8192",
        "offset_means": profile["offset_means"],
        "offset_counts": profile["offset_counts"],
        "body_offset_6_11_median": body_median,
        "substantive_thresholds_passed": bool(q025 > 0.10 and body_median <= 0),
    }


def analyze(dump: Path, out: Path) -> None:
    report_path = out / "pe_pf.json"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, results_path)
    require_certified_capture()
    private_manifest, classes, public = require_private_freeze()
    aperture_manifest = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    if (
        not aperture_manifest.get("complete")
        or len(aperture_manifest.get("files", {})) != 16
        or aperture_manifest.get("capture_manifest_sha256") != MANIFEST_SHA256
        or aperture_manifest.get("private_classes_sha256") != sha256_file(PRIVATE_CLASSES)
    ):
        raise RuntimeError("paired aperture dump is incomplete or stale")
    ranks, math_ranks, aperture_hashes = load_rank_arrays(dump)
    shallow = {arm: ranks[arm][:3] for arm in ARMS}
    deep = {arm: ranks[arm][3:] for arm in ARMS}
    math_shallow, math_deep = math_ranks[:3], math_ranks[3:]
    multi_arm = classes["arms"][MULTI]

    pe1 = pe1_test(token_band_score(deep[MULTI]), multi_arm["boundaries"])
    pe1_shallow = pe1_test(token_band_score(shallow[MULTI]), multi_arm["boundaries"])
    pe1_math = pe1_test(token_band_score(math_deep), multi_arm["boundaries"])
    ordinary_rows = [row for row in multi_arm["boundaries"] if row["boundary_type"] == "ordinary_message"]
    higher_rows = [row for row in multi_arm["boundaries"] if row["boundary_type"] == "higher_scope"]
    random_rows = []
    for rows, mask_name in [
        (ordinary_rows, "ordinary_message_starts"),
        (higher_rows, "higher_scope_starts"),
    ]:
        masks = sorted(map(int, multi_arm["random_masks"][mask_name]))
        if len(rows) != len(masks):
            raise RuntimeError(f"random mask count mismatch: {mask_name}")
        for row, token in zip(sorted(rows, key=lambda item: item["token"]), masks):
            random_rows.append({**row, "token": token})
    random_rows.sort(key=lambda row: row["token"])
    pe1_random = pe1_test(token_band_score(deep[MULTI]), random_rows)

    pe2 = pe2_test(deep[SINGLE], deep[MULTI], classes["paired_ordinary_messages"])
    pe3 = pe3_test(token_band_score(deep[MULTI]), multi_arm["boundaries"])
    pe2_shallow = pe2_test(shallow[SINGLE], shallow[MULTI], classes["paired_ordinary_messages"])
    pe3_shallow = pe3_test(token_band_score(shallow[MULTI]), multi_arm["boundaries"])
    pe3_math = pe3_test(token_band_score(math_deep), multi_arm["boundaries"])
    pe3_random = pe3_test(token_band_score(deep[MULTI]), random_rows)
    pe_secondary_holm = holm_adjust(
        np.asarray(
            [
                pe2["one_sided_paired_label_permutation_p"],
                pe3["one_sided_position_bin_scope_permutation_p"],
            ]
        )
    )
    pe2["p_holm_secondary"] = float(pe_secondary_holm[0])
    pe3["p_holm_secondary"] = float(pe_secondary_holm[1])
    pe2["passed"] = bool(pe2["direction_passed"] and pe_secondary_holm[0] < 0.05)
    pe3["passed"] = bool(pe3["direction_passed"] and pe_secondary_holm[1] < 0.05)

    anchor_names = ["url_tokens", "proper_noun_proxy", "gratitude_tokens"]
    anchors: dict[str, Any] = {}
    main_p = []
    math_p = []
    random_p = []
    for name in anchor_names:
        positions = multi_arm["classes"][name]
        random_positions = multi_arm["random_masks"][name]
        main = class_permutation_test(
            deep[MULTI], positions, direction="positive", permutations=10000, seed=seed_for(f"P-f1-main:{name}")
        )
        main["shallow_effect"] = band_effect(shallow[MULTI], positions)
        math = class_permutation_test(
            math_deep, positions, direction="two-sided", permutations=10000, seed=seed_for(f"P-f1-math:{name}")
        )
        random_control = class_permutation_test(
            deep[MULTI], random_positions, direction="two-sided", permutations=10000, seed=seed_for(f"P-f1-random:{name}")
        )
        anchors[name] = {"main": main, "crossed_math": math, "random_mask": random_control}
        main_p.append(main["p"])
        math_p.append(math["p"])
        random_p.append(random_control["p"])
    main_holm = holm_adjust(np.asarray(main_p))
    math_holm = holm_adjust(np.asarray(math_p))
    random_holm = holm_adjust(np.asarray(random_p))
    anchor_subpasses = []
    for index, name in enumerate(anchor_names):
        row = anchors[name]
        row["main"]["p_holm_internal"] = float(main_holm[index])
        row["crossed_math"]["p_holm_control"] = float(math_holm[index])
        row["random_mask"]["p_holm_control"] = float(random_holm[index])
        row["passed"] = bool(
            row["main"]["effect"] >= 0.05
            and main_holm[index] < 0.05
            and math_holm[index] >= 0.05
            and random_holm[index] >= 0.05
        )
        anchor_subpasses.append(row["passed"])
    pf1_internal_pass = bool(all(anchor_subpasses))
    pf1_top_p = float(np.max(main_p))

    pf2 = pf2_test(deep[MULTI], shallow[MULTI], multi_arm)
    pf2_single_descriptive = pf2_test(deep[SINGLE], shallow[SINGLE], classes["arms"][SINGLE])

    colon_positions = multi_arm["classes"]["label_colons"]
    pf3_shallow = class_permutation_test(
        shallow[MULTI], colon_positions, direction="negative", permutations=10000, seed=seed_for("P-f3-shallow")
    )
    pf3_deep = class_permutation_test(
        deep[MULTI], colon_positions, direction="negative", permutations=10000, seed=seed_for("P-f3-deep")
    )
    pf3_top_p = max(pf3_shallow["p"], pf3_deep["p"])
    pf3_substantive = bool(pf3_shallow["effect"] <= -0.05 and pf3_deep["effect"] <= -0.05)
    pf3 = {
        "shallow": pf3_shallow,
        "deep": pf3_deep,
        "joint_intersection_union_p": pf3_top_p,
        "substantive_thresholds_passed": pf3_substantive,
    }

    top_raw = np.asarray(
        [pf1_top_p, pf2["one_sided_paired_depth_sign_flip_p"], pf3_top_p]
    )
    top_holm = holm_adjust(top_raw)
    pf1_pass = bool(pf1_internal_pass and top_holm[0] < 0.05)
    pf2_pass = bool(pf2["substantive_thresholds_passed"] and top_holm[1] < 0.05)
    pf3_pass = bool(pf3_substantive and top_holm[2] < 0.05)
    pf1 = {
        "anchors": anchors,
        "internal_joint_passed": pf1_internal_pass,
        "top_level_raw_p": pf1_top_p,
        "top_level_holm_p": float(top_holm[0]),
        "passed": pf1_pass,
    }
    pf2["top_level_holm_p"] = float(top_holm[1])
    pf2["passed"] = pf2_pass
    pf3["top_level_holm_p"] = float(top_holm[2])
    pf3["passed"] = pf3_pass

    single_descriptive = {
        name: {
            "deep_effect": band_effect(deep[SINGLE], classes["arms"][SINGLE]["classes"][name]),
            "shallow_effect": band_effect(shallow[SINGLE], classes["arms"][SINGLE]["classes"][name]),
        }
        for name in anchor_names + ["label_colons"]
    }
    report = {
        "schema_version": 1,
        "kind": "round5_pe_pf_registered_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "P_e": {
            "P_e1_primary": pe1,
            "P_e1_shallow_depth_control": pe1_shallow,
            "P_e1_crossed_math_control": pe1_math,
            "P_e1_random_mask_control": pe1_random,
            "P_e2": pe2,
            "P_e2_shallow_depth_control": pe2_shallow,
            "P_e3": pe3,
            "P_e3_shallow_depth_control": pe3_shallow,
            "P_e3_crossed_math_control": pe3_math,
            "P_e3_random_mask_control": pe3_random,
            "registered_verdicts": {
                "P_e1": pe1["passed"],
                "P_e2": pe2["passed"],
                "P_e3": pe3["passed"],
            },
        },
        "P_f": {
            "P_f1": pf1,
            "P_f2": pf2,
            "P_f3": pf3,
            "single_thread_descriptive_replication": single_descriptive,
            "single_thread_boundary_profile_descriptive": pf2_single_descriptive,
            "top_level_raw_p": top_raw.tolist(),
            "top_level_holm_p": top_holm.tolist(),
            "registered_verdicts": {"P_f1": pf1_pass, "P_f2": pf2_pass, "P_f3": pf3_pass},
        },
        "bands": {"shallow": SHALLOW, "deep": DEEP},
        "position_bin": 256,
        "permutations": 10000,
        "public_freeze_sha256": sha256_file(PUBLIC_FREEZE),
        "private_manifest_sha256": sha256_file(PRIVATE_MANIFEST),
        "private_classes_sha256": sha256_file(PRIVATE_CLASSES),
        "aperture_manifest_sha256": sha256_file(dump / "manifest.json"),
        "math_control_manifest_sha256": sha256_file(MATH_APERTURE / "manifest.json"),
        "aperture_input_sha256": aperture_hashes,
        "builder_source_sha256": private_manifest["builder_source_sha256"],
        "provenance": provenance(Path(__file__)),
    }
    atomic_json(report_path, report)
    lines = [
        "# P-e/P-f fresh paired-corpus results",
        "",
        "**Status: answered from the certified corrected capture; independent re-derivation pending.**",
        "",
        "## P-e",
        "",
        f"- P-e1 dose slope: **{'pass' if pe1['passed'] else 'fail'}**; rho `{pe1['spearman_rho']:+.6g}`, one-sided p `{pe1['one_sided_structure_block_permutation_p']:.6g}`, block-bootstrap 95% [`{pe1['bootstrap_q025']:+.6g}`, `{pe1['bootstrap_q975']:+.6g}`].",
        f"- P-e2 paired arm: **{'pass' if pe2['passed'] else 'fail'}**; single-minus-multi effect `{pe2['difference']:+.6g}`, Holm p `{pe2['p_holm_secondary']:.6g}`.",
        f"- P-e3 boundary scope: **{'pass' if pe3['passed'] else 'fail'}**; raw contrast `{pe3['raw_median_contrast']:+.6g}`, Holm p `{pe3['p_holm_secondary']:.6g}`, dose-adjusted `{pe3['dose_adjusted_median_contrast']:+.6g}`.",
        "",
        "## P-f",
        "",
        f"- P-f1 referential anchors: **{'pass' if pf1_pass else 'fail'}**; "
        + ", ".join(f"{name} `{anchors[name]['main']['effect']:+.6g}`" for name in anchor_names)
        + f"; top Holm p `{top_holm[0]:.6g}`.",
        f"- P-f2 boundary transient: **{'pass' if pf2_pass else 'fail'}**; mean offset-0 `{pf2['offset0_deep_minus_shallow']:+.6g}`, bootstrap lower `{pf2['bootstrap_q025']:+.6g}`, median of mean offsets 6-11 `{pf2['body_offset_6_11_median']:+.6g}`, top Holm p `{top_holm[1]:.6g}`.",
        f"- P-f3 label colon: **{'pass' if pf3_pass else 'fail'}**; shallow `{pf3_shallow['effect']:+.6g}`, deep `{pf3_deep['effect']:+.6g}`, top Holm p `{top_holm[2]:.6g}`.",
        "",
    ]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"P_e": report["P_e"]["registered_verdicts"], "P_f": report["P_f"]["registered_verdicts"]}, indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    ranks = np.tile(np.linspace(0, 1, 8192), (5, 1))
    positions = [10, 300, 700]
    if not np.isclose(band_effect(ranks, positions), np.median(ranks[0, positions]) - 0.5):
        raise AssertionError("band effect failed")
    rng = np.random.default_rng(3)
    selected = stratified_null_positions(np.asarray(positions), rng)
    if len(selected) != len(positions):
        raise AssertionError(selected)
    rows = [
        {"token": i, "retired_context_dose": i + 1, "segment_open_token": 0}
        for i in range(1, 101)
    ]
    test = pe1_test(np.arange(8192, dtype=np.float64), rows, permutations=100, bootstrap=100)
    if test["spearman_rho"] < 0.99:
        raise AssertionError(test)
    duplicate_rows = [
        {"token": 10, "boundary_type": "higher_scope", "retired_context_dose": 10},
        {"token": 10, "boundary_type": "ordinary_message", "retired_context_dose": 20},
        {"token": 300, "boundary_type": "ordinary_message", "retired_context_dose": 30},
    ]
    duplicate_test = pe3_test(np.arange(8192, dtype=np.float64), duplicate_rows, permutations=10)
    if duplicate_test["higher_scope_n"] != 1 or duplicate_test["ordinary_n"] != 2:
        raise AssertionError(duplicate_test)
    synthetic_arm = {"classes": {"ordinary_message_starts": [4, 100, 8180, 8181]}}
    profile = offset_profile(ranks[:5], ranks[:3], synthetic_arm)
    if profile["ordinary"] != [4, 100, 8180]:
        raise AssertionError(profile["ordinary"])
    print("round5_pe_pf_analyze self-test passed")


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
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "compute":
        compute(args.dump, args.block_tokens)
    else:
        analyze(args.dump, args.out)


if __name__ == "__main__":
    main()
