"""Analyze all seven registered follow-ups from complete sealed dumps.

This producer owns the frozen verdict logic. It refuses partial campaigns and
refuses to overwrite either the JSON result or the human-readable report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import round5_followup7_runner as F
import round5_r5d_runner as R


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "followup7"
DEFAULT_OUT = ROOT / "analysis" / "round5" / "followup7" / "results.json"
DEFAULT_REPORT = ROOT / "analysis" / "round5" / "followup7" / "RESULTS.md"
PARENT_ARMS = ROOT / "dumps" / "round5" / "r5d" / "arms"
PARENT_RESULTS = ROOT / "analysis" / "round5" / "r5d" / "r5d_results.json"
TEXTS = tuple(R.TEXTS)
N_BOOT = 5000
BLOCK = 256


def seed(label: str) -> int:
    registration_sha = R.sha256_file(F.REGISTRATION)
    digest = hashlib.sha256(f"{registration_sha}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def validate_manifest(root: Path, manifest_path: Path, expected_kind: str) -> dict[str, Any]:
    manifest = R.load_json(manifest_path)
    if manifest.get("kind") != expected_kind or manifest.get("complete") is not True:
        raise RuntimeError(f"incomplete or wrong manifest: {manifest_path}")
    if manifest.get("artifact_count") != len(manifest.get("artifacts", [])):
        raise RuntimeError(f"artifact-count mismatch: {manifest_path}")
    for record in manifest["artifacts"]:
        path = root / record["path"]
        if not path.is_file() or R.sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {path}")
    return manifest


def validate_campaign(dump: Path) -> dict[str, Any]:
    baseline = validate_manifest(
        dump / "baseline_fullvocab",
        dump / "baseline_fullvocab" / "manifest.json",
        "round5_followup7_fullvocab_baseline",
    )
    arm_manifests: dict[str, str] = {}
    for arm in F.ARMS:
        root = dump / "arms" / arm.arm_id
        manifest = validate_manifest(root, root / "manifest.json", "round5_followup7_arm")
        if manifest["arm"]["arm_id"] != arm.arm_id:
            raise RuntimeError(f"arm identity drift: {arm.arm_id}")
        arm_manifests[arm.arm_id] = R.sha256_file(root / "manifest.json")
    fresh = validate_manifest(dump / "fresh", dump / "fresh" / "manifest.json", "round5_followup7_fresh_class_job")
    return {
        "baseline_manifest_sha256": R.sha256_file(dump / "baseline_fullvocab" / "manifest.json"),
        "arm_manifest_sha256": arm_manifests,
        "fresh_manifest_sha256": R.sha256_file(dump / "fresh" / "manifest.json"),
        "artifact_count": int(baseline["artifact_count"] + fresh["artifact_count"] + sum(
            R.load_json(dump / "arms" / arm.arm_id / "manifest.json")["artifact_count"] for arm in F.ARMS
        )),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.array(values[name], copy=True) for name in values.files}


def arm_tokens(dump: Path, arm: str, text: str) -> dict[str, np.ndarray]:
    return load_npz(dump / "arms" / arm / "tokens" / f"{text}.npz")


def parent_tokens(arm: str, text: str) -> dict[str, np.ndarray]:
    return load_npz(PARENT_ARMS / arm / "tokens" / f"{text}.npz")


def baseline_tokens(dump: Path, text: str) -> dict[str, np.ndarray]:
    return load_npz(dump / "baseline_fullvocab" / f"{text}.npz")


def block_effects(arrays: Iterable[np.ndarray]) -> np.ndarray:
    blocks: list[float] = []
    for values in arrays:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (R.SEQ - 1,):
            raise RuntimeError(f"invalid token effect shape: {values.shape}")
        for start in range(0, values.size, BLOCK):
            blocks.append(float(values[start : start + BLOCK].mean()))
    return np.asarray(blocks, dtype=np.float64)


def inference(blocks: np.ndarray, label: str) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed(label)))
    n = blocks.size
    indices = rng.integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    draws = blocks[indices].mean(axis=1)
    signs = rng.integers(0, 2, size=(N_BOOT, n), dtype=np.int8) * 2 - 1
    null = (blocks[None, :] * signs).mean(axis=1)
    observed = float(blocks.mean())
    p_positive = float((1 + np.count_nonzero(null >= observed)) / (N_BOOT + 1))
    p_negative = float((1 + np.count_nonzero(null <= observed)) / (N_BOOT + 1))
    return {
        "effect": observed,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "p_signflip_positive": p_positive,
        "p_signflip_negative": p_negative,
        "n_blocks": int(n),
        "draws": draws,
    }


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * float(pvalues[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def strip_draws(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "draws"}


def pooled_cost(dump: Path, arm: str) -> tuple[float, list[np.ndarray]]:
    arrays = [arm_tokens(dump, arm, text)["delta_nll"].astype(np.float64) for text in TEXTS]
    return float(np.concatenate(arrays).mean()), arrays


def family_f7_1(dump: Path) -> dict[str, Any]:
    single_names = [f"d{d}_off_L29" for d in range(4)]
    singles: dict[str, Any] = {}
    inferential: list[dict[str, Any]] = []
    costs: dict[str, float] = {}
    for arm in [a.arm_id for a in F.ARMS if a.family == "F7-1"]:
        cost, arrays = pooled_cost(dump, arm)
        costs[arm] = cost
        if arm in single_names:
            row = inference(block_effects(arrays), f"F7-1:{arm}")
            inferential.append(row)
            singles[arm] = strip_draws(row)
    adjusted = holm([row["p_signflip_positive"] for row in inferential])
    for arm, value in zip(single_names, adjusted):
        singles[arm]["p_holm_positive"] = value
        singles[arm]["positive_confirmed"] = bool(
            singles[arm]["effect"] > 0 and singles[arm]["ci95"][0] > 0 and value < 0.05
        )
    parent_cost = float(
        np.concatenate([parent_tokens("bias_off_L29", t)["delta_nll"] for t in TEXTS])
        .astype(np.float64)
        .mean()
    )
    rescue = 1.0 - costs["stencil_only_d0_3_L29"] / parent_cost
    return {
        "costs": costs,
        "certified_bias_off_cost": parent_cost,
        "singleton_inference": singles,
        "stencil_rescue_fraction": rescue,
        "prediction_1_passed": any(row["positive_confirmed"] for row in singles.values()),
        "prediction_2_passed": bool(rescue >= 0.50),
        "passed": bool(any(row["positive_confirmed"] for row in singles.values()) and rescue >= 0.50),
    }


def family_f7_2(dump: Path) -> dict[str, Any]:
    specifications = {
        "adjacent_23_29": ("bias_off_L23_L29", (23, 29)),
        "adjacent_29_35": ("bias_off_L29_L35", (29, 35)),
        "triple": ("bias_off_L23_L29_L35", (23, 29, 35)),
        "control_23_35": ("bias_off_L23_L35", (23, 35)),
    }
    rows: dict[str, Any] = {}
    primary_records: list[dict[str, Any]] = []
    for label, (joint, layers) in specifications.items():
        per_text = []
        for text in TEXTS:
            effect = arm_tokens(dump, joint, text)["delta_nll"].astype(np.float64)
            for layer in layers:
                effect -= parent_tokens(f"bias_off_L{layer:02d}", text)["delta_nll"].astype(np.float64)
            per_text.append(effect)
        record = inference(block_effects(per_text), f"F7-2:{label}")
        rows[label] = strip_draws(record)
        if label != "control_23_35":
            primary_records.append(record)
    adjusted = holm([row["p_signflip_positive"] for row in primary_records])
    for label, value in zip(("adjacent_23_29", "adjacent_29_35", "triple"), adjusted):
        rows[label]["p_holm_positive"] = value
        rows[label]["positive_confirmed"] = bool(
            rows[label]["effect"] > 0 and rows[label]["ci95"][0] > 0 and value < 0.05
        )
    return {"interactions": rows, "passed": all(rows[label]["positive_confirmed"] for label in adjusted_labels())}


def adjusted_labels() -> tuple[str, str, str]:
    return ("adjacent_23_29", "adjacent_29_35", "triple")


def family_f7_3(dump: Path) -> dict[str, Any]:
    names = [arm.arm_id for arm in F.ARMS if arm.family == "F7-3"]
    costs = {name: pooled_cost(dump, name)[0] for name in names}
    bias = float(
        np.concatenate([parent_tokens("bias_off_L29", t)["delta_nll"] for t in TEXTS])
        .astype(np.float64)
        .mean()
    )
    ratios = {name: costs[name] / bias for name in names}
    clauses = {
        "remove_mean_at_least_half": ratios["r_remove_mean_L29"] >= 0.5,
        "remove_noncarrier_mean_at_least_half": ratios["r_remove_noncarrier_mean_L29"] >= 0.5,
        "remove_centered_at_most_quarter": ratios["r_remove_centered_L29"] <= 0.25,
        "remove_carrier_mean_null": abs(costs["r_remove_carrier_mean_L29"]) < 0.005,
    }
    return {"costs": costs, "ratios_to_bias_off": ratios, "clauses": clauses, "passed": all(clauses.values())}


def family_f7_4(dump: Path) -> dict[str, Any]:
    quartiles = [f"head_q{q}_off_L29" for q in range(1, 5)]
    costs = {name: pooled_cost(dump, name)[0] for name in [a.arm_id for a in F.ARMS if a.family == "F7-4"]}
    contrasts: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for other in quartiles[1:]:
        arrays = [
            arm_tokens(dump, quartiles[0], text)["delta_nll"].astype(np.float64)
            - arm_tokens(dump, other, text)["delta_nll"].astype(np.float64)
            for text in TEXTS
        ]
        row = inference(block_effects(arrays), f"F7-4:q1-minus:{other}")
        contrasts[other] = strip_draws(row)
        records.append(row)
    adjusted = holm([row["p_signflip_positive"] for row in records])
    for other, value in zip(quartiles[1:], adjusted):
        contrasts[other]["p_holm_positive"] = value
        contrasts[other]["positive_confirmed"] = bool(
            contrasts[other]["effect"] > 0 and contrasts[other]["ci95"][0] > 0 and value < 0.05
        )
    bias = float(
        np.concatenate([parent_tokens("bias_off_L29", t)["delta_nll"] for t in TEXTS])
        .astype(np.float64)
        .mean()
    )
    rescue = 1 - costs["head_top16_stencil_only_L29"] / bias
    return {
        "costs": costs,
        "q1_contrasts": contrasts,
        "top16_stencil_rescue_fraction": rescue,
        "q1_localized": all(row["positive_confirmed"] for row in contrasts.values()),
        "top16_rescue_passed": bool(rescue >= 0.5),
        "passed": bool(all(row["positive_confirmed"] for row in contrasts.values()) and rescue >= 0.5),
    }


def query_bootstrap(parent: np.ndarray, patched: np.ndarray, label: str) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed(label)))
    n = parent.size
    indices = rng.integers(0, n, size=(N_BOOT, n), dtype=np.int16)
    parent_draw = parent[indices].mean(axis=1)
    patched_draw = patched[indices].mean(axis=1)
    absolute = parent_draw - patched_draw
    fraction = absolute / parent_draw
    return {
        "parent_mean": float(parent.mean()),
        "patched_mean": float(patched.mean()),
        "absolute_rescue": float(parent.mean() - patched.mean()),
        "absolute_rescue_ci95": [float(np.quantile(absolute, 0.025)), float(np.quantile(absolute, 0.975))],
        "rescue_fraction": float((parent.mean() - patched.mean()) / parent.mean()),
        "rescue_fraction_ci95": [float(np.quantile(fraction, 0.025)), float(np.quantile(fraction, 0.975))],
    }


def family_f7_5(dump: Path) -> dict[str, Any]:
    queries = F.frozen()["patch_query_positions"].astype(np.int64)
    parent = parent_tokens("bias_off_L29", "05_needles")["delta_nll"].astype(np.float64)[queries]
    query = arm_tokens(dump, "bias_off_L29_patch_query", "05_needles")["delta_nll"].astype(np.float64)[queries]
    sham = arm_tokens(dump, "bias_off_L29_patch_sham", "05_needles")["delta_nll"].astype(np.float64)[queries]
    query_row = query_bootstrap(parent, query, "F7-5:query")
    sham_row = query_bootstrap(parent, sham, "F7-5:sham")
    query_pass = query_row["rescue_fraction"] >= 0.5 and query_row["absolute_rescue_ci95"][0] > 0
    sham_pass = sham_row["rescue_fraction"] < 0.1 and sham_row["absolute_rescue_ci95"][0] <= 0 <= sham_row["absolute_rescue_ci95"][1]
    return {"query_patch": query_row, "sham_patch": sham_row, "query_passed": query_pass,
            "sham_passed": sham_pass, "passed": bool(query_pass and sham_pass)}


def clock_basis(kind: str, layer: int, text: str) -> np.ndarray:
    values = F.frozen()
    if kind == "clock_union":
        basis = values[f"clock_union_L{layer:02d}"]
    elif kind == "clock_pertext":
        basis = values[f"clock_g_L{layer:02d}_{text}"][:, None]
    elif kind == "clock_loto":
        basis = values[f"clock_loto_L{layer:02d}_{text}"]
    elif kind == "clock_sham6":
        basis = values[f"clock_sham6_L{layer:02d}"]
    else:
        raise ValueError(kind)
    return basis.astype(np.float64)


def clock_gain_stat(dump: Path, arm_id: str, kind: str, layer: int, text: str) -> float:
    r = np.load(
        dump / "arms" / arm_id / "clock" / f"rvec_pre_L{layer:02d}_{text}.npy",
        allow_pickle=False,
    ).astype(np.float32).reshape(R.SEQ, R.RFLAT).astype(np.float64)
    mu = F.frozen()[f"mu_L{layer:02d}_{text}"].astype(np.float64)
    u = clock_basis(kind, layer, text)
    transformed = r - ((r - mu) @ u) @ u.T
    blocks = transformed[64:].reshape(127, 64, R.HEADS, R.RPERHEAD).mean(axis=1)
    proj = np.load(ROOT / "weights" / f"layer{layer:02d}_rel_logits_proj.npy", allow_pickle=False).astype(np.float64)
    kernels = blocks @ proj
    mean_kernel = kernels.mean(axis=0)
    denominator = np.sum(mean_kernel * mean_kernel, axis=1)
    gains = np.sum(kernels * mean_kernel[None, :, :], axis=2) / np.maximum(denominator[None, :], 1e-30)
    x = np.log1p(np.arange(64, R.SEQ, 64, dtype=np.float64) + 31.5)
    xc = x - x.mean()
    gc = gains - gains.mean(axis=0)
    corr = (xc[:, None] * gc).sum(axis=0) / np.maximum(
        np.linalg.norm(xc) * np.linalg.norm(gc, axis=0), 1e-30
    )
    return float(np.median(np.abs(corr)))


def family_f7_6(dump: Path) -> dict[str, Any]:
    arm_kind = {arm.arm_id: arm.kind for arm in F.ARMS if arm.family == "F7-6"}
    stats: dict[str, Any] = {}
    for arm_id, kind in arm_kind.items():
        arm = F.ARM_BY_ID[arm_id]
        cells: dict[str, float] = {}
        for layer in arm.layers:
            for text in TEXTS:
                cells[f"L{layer:02d}:{text}"] = clock_gain_stat(dump, arm_id, kind, layer, text)
        stats[arm_id] = {"cells": cells, "median": float(np.median(list(cells.values()))),
                         "maximum": float(np.max(list(cells.values())))}
    loto = stats["clock_loto_L53_L59"]
    sham = stats["clock_sham6_L53_L59"]
    behavior_arms = ("clock_union_L53_L59", "clock_pertext_L53_L59", "clock_loto_L53_L59")
    costs = {arm: pooled_cost(dump, arm)[0] for arm in behavior_arms}
    transfer_pass = loto["maximum"] < 0.20 and sham["median"] >= 0.50
    behavior_pass = all(abs(value) < 0.005 for value in costs.values())
    return {"kernel_gain_correlation": stats, "behavior_costs": costs,
            "loto_transfer_passed": transfer_pass, "behavior_null_passed": behavior_pass,
            "passed": bool(transfer_pass and behavior_pass)}


def ece(probability: np.ndarray, correct: np.ndarray, bins: int) -> float:
    index = np.minimum((probability * bins).astype(np.int64), bins - 1)
    result = 0.0
    n = probability.size
    for b in range(bins):
        mask = index == b
        if mask.any():
            result += mask.sum() / n * abs(float(correct[mask].mean()) - float(probability[mask].mean()))
    return float(result)


def ece_block_stats(probability: list[np.ndarray], correct: list[np.ndarray], bins: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for p, c in zip(probability, correct):
        for start in range(0, p.size, BLOCK):
            pp = p[start : start + BLOCK]
            cc = c[start : start + BLOCK]
            idx = np.minimum((pp * bins).astype(np.int64), bins - 1)
            row = np.zeros((bins, 3), dtype=np.float64)
            for b in range(bins):
                mask = idx == b
                row[b] = (mask.sum(), cc[mask].sum(), pp[mask].sum())
            rows.append(row)
    return np.stack(rows)


def ece_from_stats(stats: np.ndarray) -> float:
    total = stats[:, 0].sum()
    valid = stats[:, 0] > 0
    return float(np.sum(stats[valid, 0] / total * np.abs(
        stats[valid, 1] / stats[valid, 0] - stats[valid, 2] / stats[valid, 0]
    )))


def ece_inference(base_p: list[np.ndarray], base_c: list[np.ndarray], arm_p: list[np.ndarray], arm_c: list[np.ndarray]) -> dict[str, Any]:
    base_stats = ece_block_stats(base_p, base_c, 20)
    arm_stats = ece_block_stats(arm_p, arm_c, 20)
    observed = ece_from_stats(arm_stats.sum(axis=0)) - ece_from_stats(base_stats.sum(axis=0))
    rng = np.random.Generator(np.random.PCG64(seed("F7-7:ece20")))
    draws = np.empty(N_BOOT, dtype=np.float64)
    for draw in range(N_BOOT):
        idx = rng.integers(0, base_stats.shape[0], size=base_stats.shape[0])
        draws[draw] = ece_from_stats(arm_stats[idx].sum(axis=0)) - ece_from_stats(base_stats[idx].sum(axis=0))
    return {"effect": observed, "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def matched_class(
    delta: np.ndarray,
    baseline_nll: np.ndarray,
    positions: np.ndarray,
    excluded: set[int],
    label: str,
) -> dict[str, Any]:
    positions = np.asarray(sorted({int(p) for p in positions if 0 <= int(p) < delta.size}), dtype=np.int64)
    boundaries = np.quantile(baseline_nll, np.linspace(0.1, 0.9, 9))
    decile = np.digitize(baseline_nll, boundaries, right=True)
    block = np.arange(delta.size) // 512
    rng = np.random.Generator(np.random.PCG64(seed(label)))
    control_sum = np.zeros(10000, dtype=np.float64)
    for position in positions:
        candidates = np.flatnonzero(
            (block == block[position]) & (decile == decile[position])
            & np.asarray([index not in excluded for index in range(delta.size)], dtype=bool)
        )
        if candidates.size == 0:
            raise RuntimeError(f"no matched controls for {label} at {position}")
        control_sum += delta[rng.choice(candidates, size=10000, replace=True)]
    class_mean = float(delta[positions].mean())
    contrasts = class_mean - control_sum / positions.size
    return {
        "n": int(positions.size),
        "class_mean": class_mean,
        "matched_contrast": float(contrasts.mean()),
        "ci95": [float(np.quantile(contrasts, .025)), float(np.quantile(contrasts, .975))],
        "p_positive": float((1 + np.count_nonzero(contrasts <= 0)) / 10001),
    }


def family_f7_7(dump: Path) -> dict[str, Any]:
    arm_name = "bias_off_L29_fullvocab"
    baseline = [baseline_tokens(dump, text) for text in TEXTS]
    arm = [arm_tokens(dump, arm_name, text) for text in TEXTS]
    rank_blocks = block_effects([values["delta_log1p_target_rank"] for values in arm])
    accuracy_blocks = block_effects([values["delta_top1_correct"] for values in arm])
    rank = inference(rank_blocks, "F7-7:rank")
    accuracy = inference(accuracy_blocks, "F7-7:accuracy")
    ranking_pass = rank["effect"] > 0 and rank["ci95"][0] > 0 and accuracy["effect"] < 0 and accuracy["ci95"][1] < 0
    base_p = [values["top1_probability"].astype(np.float64) for values in baseline]
    base_c = [values["top1_correct"].astype(np.float64) for values in baseline]
    arm_p = [values["top1_probability"].astype(np.float64) for values in arm]
    arm_c = [values["top1_correct"].astype(np.float64) for values in arm]
    calibration = ece_inference(base_p, base_c, arm_p, arm_c)
    calibration["baseline_ece20"] = ece(np.concatenate(base_p), np.concatenate(base_c), 20)
    calibration["arm_ece20"] = ece(np.concatenate(arm_p), np.concatenate(arm_c), 20)
    calibration["baseline_ece10"] = ece(np.concatenate(base_p), np.concatenate(base_c), 10)
    calibration["arm_ece10"] = ece(np.concatenate(arm_p), np.concatenate(arm_c), 10)
    calibration["evidence_threshold_met"] = bool(
        abs(calibration["effect"]) >= .01
        and not (calibration["ci95"][0] <= 0 <= calibration["ci95"][1])
    )
    descriptive = {
        field: float(np.mean(np.concatenate([values[f"delta_{field}"] for values in arm])))
        for field in ("entropy", "brier", "target_margin", "target_rank")
    }

    fresh_rows: dict[str, Any] = {}
    frozen = F.frozen()
    specifications = (
        ("07b_slack_multi", "first_content", frozen["class_07b_first_content"]),
        ("07b_slack_multi", "pronouns", frozen["class_07b_pronouns"]),
        ("08_math_llm", "unit_starts", frozen["class_08_unit_starts"]),
        ("08_math_llm", "pronouns", frozen["class_08_pronouns"]),
    )
    excluded_by_text: dict[str, set[int]] = {}
    for text, _name, positions in specifications:
        excluded_by_text.setdefault(text, set()).update(int(value) for value in positions)
    for text, name, positions in specifications:
        values = load_npz(dump / "fresh" / "bias_off_L29" / f"{text}.npz")
        baseline_values = load_npz(dump / "fresh" / "baseline" / f"{text}.npz")
        row = matched_class(
            values["delta_nll"].astype(np.float64),
            baseline_values["nll"].astype(np.float64),
            positions,
            excluded_by_text[text],
            f"F7-7:class:{text}:{name}:query",
        )
        target_positions = np.asarray([int(p) - 1 for p in positions if int(p) > 0], dtype=np.int64)
        row["target_aligned_secondary"] = matched_class(
            values["delta_nll"].astype(np.float64),
            baseline_values["nll"].astype(np.float64),
            target_positions,
            excluded_by_text[text],
            f"F7-7:class:{text}:{name}:target",
        )
        fresh_rows[f"{text}:{name}"] = row
    adjusted = holm([row["p_positive"] for row in fresh_rows.values()])
    for row, value in zip(fresh_rows.values(), adjusted):
        row["p_holm_positive"] = value
    slack_pass = all(
        fresh_rows[key]["matched_contrast"] > 0 and fresh_rows[key]["p_holm_positive"] < .05
        for key in ("07b_slack_multi:first_content", "07b_slack_multi:pronouns")
    )
    return {
        "ranking": {**strip_draws(rank), "accuracy": strip_draws(accuracy), "passed": ranking_pass},
        "calibration": calibration,
        "descriptive_fullvocab_deltas": descriptive,
        "fresh_classes": fresh_rows,
        "fresh_slack_replication_passed": slack_pass,
        "passed": bool(ranking_pass and slack_pass),
    }


def build_report(results: dict[str, Any]) -> str:
    f1, f2, f3, f4, f5, f6, f7 = (results["families"][f"F7-{i}"] for i in range(1, 8))
    lines = [
        "# Round 5 seven-experiment follow-up results",
        "",
        f"Status: **ANSWERED_PENDING_INDEPENDENT_VERIFICATION**",
        "",
        f"- F7-1 signed stencil: **{'PASS' if f1['passed'] else 'FAIL'}**; all-head d0..3-only rescue `{f1['stencil_rescue_fraction']:.3f}`.",
        f"- F7-2 shoulder backups: **{'PASS' if f2['passed'] else 'FAIL'}**; interactions "
        + ", ".join(f"{name} `{f2['interactions'][name]['effect']:+.6f}`" for name in adjusted_labels()) + ".",
        f"- F7-3 r decomposition: **{'PASS' if f3['passed'] else 'FAIL'}**; mean-removal ratio `{f3['ratios_to_bias_off']['r_remove_mean_L29']:.3f}`, non-carrier-mean ratio `{f3['ratios_to_bias_off']['r_remove_noncarrier_mean_L29']:.3f}`.",
        f"- F7-4 head localization: **{'PASS' if f4['passed'] else 'FAIL'}**; top-16 stencil rescue `{f4['top16_stencil_rescue_fraction']:.3f}`.",
        f"- F7-5 query patch: **{'PASS' if f5['passed'] else 'FAIL'}**; query rescue `{f5['query_patch']['rescue_fraction']:.3f}`, sham `{f5['sham_patch']['rescue_fraction']:.3f}`.",
        f"- F7-6 clock subspaces: **{'PASS' if f6['passed'] else 'FAIL'}**; LOTO maximum kernel correlation `{f6['kernel_gain_correlation']['clock_loto_L53_L59']['maximum']:.3f}`.",
        f"- F7-7 full vocabulary + fresh classes: **{'PASS' if f7['passed'] else 'FAIL'}**; rank delta `{f7['ranking']['effect']:+.6f}`, accuracy delta `{f7['ranking']['accuracy']['effect']:+.6f}`, ECE20 delta `{f7['calibration']['effect']:+.6f}`.",
        "",
        "Every number above is post-preregistration but remains uncertified until the independent verifier agrees from raw dumps.",
    ]
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> None:
    if args.out.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite follow-up results")
    provenance = validate_campaign(args.dump.resolve())
    families = {
        "F7-1": family_f7_1(args.dump.resolve()),
        "F7-2": family_f7_2(args.dump.resolve()),
        "F7-3": family_f7_3(args.dump.resolve()),
        "F7-4": family_f7_4(args.dump.resolve()),
        "F7-5": family_f7_5(args.dump.resolve()),
        "F7-6": family_f7_6(args.dump.resolve()),
        "F7-7": family_f7_7(args.dump.resolve()),
    }
    results = {
        "schema_version": 1,
        "kind": "round5_followup7_results",
        "created_at_utc": R.utc_now(),
        "registration_commit": F.REGISTERED_COMMIT,
        "registration_sha256": R.sha256_file(F.REGISTRATION),
        "runner_sha256": R.sha256_file(ROOT / "scripts" / "round5_followup7_runner.py"),
        "analyzer_sha256": R.sha256_file(Path(__file__)),
        "frozen_inputs_sha256": R.sha256_file(F.FROZEN_NPZ),
        "dump_provenance": provenance,
        "families": families,
        "all_rows_answered": True,
        "certified": False,
    }
    atomic_json(args.out, results)
    atomic_text(args.report, build_report(results))
    print(json.dumps({name: row["passed"] for name, row in families.items()}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
