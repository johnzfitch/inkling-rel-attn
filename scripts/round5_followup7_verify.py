"""Independent raw-dump verifier for the seven Round-5 follow-ups.

This file deliberately does not import the producer analyzer. It authenticates
every manifest/artifact, reconstructs all registered verdict quantities, and
compares them to ``results.json``. The clock check uses the 16x16 projection
Gram identity rather than materializing the producer's full kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import round5_followup7_runner as F
import round5_r5d_runner as R


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "followup7"
DEFAULT_RESULTS = ROOT / "analysis" / "round5" / "followup7" / "results.json"
DEFAULT_OUT = ROOT / "analysis" / "round5" / "followup7" / "verification.json"
PARENT = ROOT / "dumps" / "round5" / "r5d" / "arms"
TEXTS = tuple(R.TEXTS)
N_BOOT = 5000
BLOCK = 256


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.array(values[name], copy=True) for name in values.files}


def arm(dump: Path, arm_id: str, text: str) -> dict[str, np.ndarray]:
    return load(dump / "arms" / arm_id / "tokens" / f"{text}.npz")


def parent(arm_id: str, text: str) -> dict[str, np.ndarray]:
    return load(PARENT / arm_id / "tokens" / f"{text}.npz")


def baseline(dump: Path, text: str) -> dict[str, np.ndarray]:
    return load(dump / "baseline_fullvocab" / f"{text}.npz")


def seed(label: str) -> int:
    registration_sha = R.sha256_file(F.REGISTRATION)
    return int.from_bytes(hashlib.sha256(f"{registration_sha}:{label}".encode()).digest()[:8], "big")


def blocks(arrays: list[np.ndarray]) -> np.ndarray:
    output: list[float] = []
    for values in arrays:
        x = np.asarray(values, dtype=np.float64)
        for start in range(0, x.size, BLOCK):
            output.append(float(np.add.reduce(x[start : start + BLOCK]) / x[start : start + BLOCK].size))
    return np.asarray(output, dtype=np.float64)


def resample(values: np.ndarray, label: str) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed(label)))
    count = values.size
    chosen = rng.integers(0, count, size=(N_BOOT, count), dtype=np.int32)
    bootstrap = np.add.reduce(values[chosen], axis=1) / count
    signs = 2 * rng.integers(0, 2, size=(N_BOOT, count), dtype=np.int8) - 1
    null = np.add.reduce(values[None, :] * signs, axis=1) / count
    observed = float(np.add.reduce(values) / count)
    return {
        "effect": observed,
        "ci95": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
        "p_positive": float((1 + np.count_nonzero(null >= observed)) / (N_BOOT + 1)),
        "p_negative": float((1 + np.count_nonzero(null <= observed)) / (N_BOOT + 1)),
    }


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, (len(values) - rank) * values[index])
        result[index] = min(maximum, 1.0)
    return result


def close(errors: list[str], label: str, observed: float, reported: float, tolerance: float = 2e-10) -> None:
    if not np.isclose(observed, reported, rtol=tolerance, atol=tolerance):
        errors.append(f"{label}: {observed} != {reported}")


def equal(errors: list[str], label: str, observed: Any, reported: Any) -> None:
    if observed != reported:
        errors.append(f"{label}: {observed!r} != {reported!r}")


def authenticate(dump: Path, errors: list[str]) -> int:
    roots = [(dump / "baseline_fullvocab", "round5_followup7_fullvocab_baseline")]
    roots.extend((dump / "arms" / item.arm_id, "round5_followup7_arm") for item in F.ARMS)
    roots.append((dump / "fresh", "round5_followup7_fresh_class_job"))
    total = 0
    for root, kind in roots:
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"manifest read {manifest_path}: {exc}")
            continue
        if manifest.get("kind") != kind or manifest.get("complete") is not True:
            errors.append(f"manifest incomplete/wrong kind: {manifest_path}")
        if manifest.get("artifact_count") != len(manifest.get("artifacts", [])):
            errors.append(f"manifest count: {manifest_path}")
        for record in manifest.get("artifacts", []):
            path = root / record["path"]
            if not path.is_file() or R.sha256_file(path) != record.get("sha256"):
                errors.append(f"artifact hash: {path}")
            total += 1
    return total


def pooled(dump: Path, arm_id: str) -> tuple[float, list[np.ndarray]]:
    arrays = [arm(dump, arm_id, text)["delta_nll"].astype(np.float64) for text in TEXTS]
    total = sum(float(np.add.reduce(values)) for values in arrays)
    count = sum(values.size for values in arrays)
    return total / count, arrays


def verify_f7_1(dump: Path, reported: dict[str, Any], errors: list[str]) -> bool:
    names = [f"d{index}_off_L29" for index in range(4)]
    rows = []
    for name in names:
        cost, arrays = pooled(dump, name)
        close(errors, f"F7-1 cost {name}", cost, reported["costs"][name])
        row = resample(blocks(arrays), f"F7-1:{name}")
        stored = reported["singleton_inference"][name]
        close(errors, f"F7-1 effect {name}", row["effect"], stored["effect"])
        close(errors, f"F7-1 lower {name}", row["ci95"][0], stored["ci95"][0])
        close(errors, f"F7-1 upper {name}", row["ci95"][1], stored["ci95"][1])
        rows.append(row)
    adjusted = holm([row["p_positive"] for row in rows])
    confirmed = []
    for name, row, p in zip(names, rows, adjusted):
        stored = reported["singleton_inference"][name]
        close(errors, f"F7-1 Holm {name}", p, stored["p_holm_positive"])
        confirmed.append(row["effect"] > 0 and row["ci95"][0] > 0 and p < .05)
    bias = pooled_parent("bias_off_L29")
    stencil = pooled(dump, "stencil_only_d0_3_L29")[0]
    rescue = 1 - stencil / bias
    close(errors, "F7-1 rescue", rescue, reported["stencil_rescue_fraction"])
    verdict = any(confirmed) and rescue >= .5
    equal(errors, "F7-1 verdict", verdict, reported["passed"])
    return verdict


def pooled_parent(arm_id: str) -> float:
    arrays = [parent(arm_id, text)["delta_nll"].astype(np.float64) for text in TEXTS]
    return sum(float(np.add.reduce(x)) for x in arrays) / sum(x.size for x in arrays)


def verify_f7_2(dump: Path, reported: dict[str, Any], errors: list[str]) -> bool:
    specs = [
        ("adjacent_23_29", "bias_off_L23_L29", (23, 29)),
        ("adjacent_29_35", "bias_off_L29_L35", (29, 35)),
        ("triple", "bias_off_L23_L29_L35", (23, 29, 35)),
    ]
    rows = []
    for label, joint, layers in specs:
        arrays = []
        for text in TEXTS:
            x = arm(dump, joint, text)["delta_nll"].astype(np.float64)
            for layer in layers:
                x = x - parent(f"bias_off_L{layer:02d}", text)["delta_nll"].astype(np.float64)
            arrays.append(x)
        row = resample(blocks(arrays), f"F7-2:{label}")
        close(errors, f"F7-2 {label}", row["effect"], reported["interactions"][label]["effect"])
        rows.append(row)
    adjusted = holm([row["p_positive"] for row in rows])
    verdicts = []
    for (label, _joint, _layers), row, p in zip(specs, rows, adjusted):
        close(errors, f"F7-2 Holm {label}", p, reported["interactions"][label]["p_holm_positive"])
        verdicts.append(row["effect"] > 0 and row["ci95"][0] > 0 and p < .05)
    verdict = all(verdicts)
    equal(errors, "F7-2 verdict", verdict, reported["passed"])
    return verdict


def verify_f7_3(dump: Path, reported: dict[str, Any], errors: list[str]) -> bool:
    names = [item.arm_id for item in F.ARMS if item.family == "F7-3"]
    bias = pooled_parent("bias_off_L29")
    costs = {name: pooled(dump, name)[0] for name in names}
    ratios = {name: costs[name] / bias for name in names}
    for name in names:
        close(errors, f"F7-3 cost {name}", costs[name], reported["costs"][name])
        close(errors, f"F7-3 ratio {name}", ratios[name], reported["ratios_to_bias_off"][name])
    clauses = {
        "remove_mean_at_least_half": ratios["r_remove_mean_L29"] >= .5,
        "remove_noncarrier_mean_at_least_half": ratios["r_remove_noncarrier_mean_L29"] >= .5,
        "remove_centered_at_most_quarter": ratios["r_remove_centered_L29"] <= .25,
        "remove_carrier_mean_null": abs(costs["r_remove_carrier_mean_L29"]) < .005,
    }
    verdict = all(clauses.values())
    equal(errors, "F7-3 clauses", clauses, reported["clauses"])
    equal(errors, "F7-3 verdict", verdict, reported["passed"])
    return verdict


def verify_f7_4(dump: Path, reported: dict[str, Any], errors: list[str]) -> bool:
    quartiles = [f"head_q{q}_off_L29" for q in range(1, 5)]
    rows = []
    for other in quartiles[1:]:
        arrays = [
            arm(dump, quartiles[0], text)["delta_nll"].astype(np.float64)
            - arm(dump, other, text)["delta_nll"].astype(np.float64)
            for text in TEXTS
        ]
        row = resample(blocks(arrays), f"F7-4:q1-minus:{other}")
        close(errors, f"F7-4 contrast {other}", row["effect"], reported["q1_contrasts"][other]["effect"])
        rows.append(row)
    adjusted = holm([row["p_positive"] for row in rows])
    localized = all(
        row["effect"] > 0 and row["ci95"][0] > 0 and p < .05
        for row, p in zip(rows, adjusted)
    )
    rescue = 1 - pooled(dump, "head_top16_stencil_only_L29")[0] / pooled_parent("bias_off_L29")
    close(errors, "F7-4 rescue", rescue, reported["top16_stencil_rescue_fraction"])
    verdict = localized and rescue >= .5
    equal(errors, "F7-4 verdict", verdict, reported["passed"])
    return verdict


def patch_summary(reference: np.ndarray, patched: np.ndarray, label: str) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed(label)))
    index = rng.integers(0, reference.size, size=(N_BOOT, reference.size), dtype=np.int16)
    ref = reference[index].mean(axis=1)
    got = patched[index].mean(axis=1)
    absolute = ref - got
    return {
        "fraction": float((reference.mean() - patched.mean()) / reference.mean()),
        "absolute_ci": [float(np.quantile(absolute, .025)), float(np.quantile(absolute, .975))],
    }


def verify_f7_5(dump: Path, reported: dict[str, Any], errors: list[str]) -> bool:
    q = F.frozen()["patch_query_positions"].astype(np.int64)
    reference = parent("bias_off_L29", "05_needles")["delta_nll"].astype(np.float64)[q]
    query = arm(dump, "bias_off_L29_patch_query", "05_needles")["delta_nll"].astype(np.float64)[q]
    sham = arm(dump, "bias_off_L29_patch_sham", "05_needles")["delta_nll"].astype(np.float64)[q]
    qr = patch_summary(reference, query, "F7-5:query")
    sr = patch_summary(reference, sham, "F7-5:sham")
    close(errors, "F7-5 query rescue", qr["fraction"], reported["query_patch"]["rescue_fraction"])
    close(errors, "F7-5 sham rescue", sr["fraction"], reported["sham_patch"]["rescue_fraction"])
    verdict = qr["fraction"] >= .5 and qr["absolute_ci"][0] > 0 and sr["fraction"] < .1 and sr["absolute_ci"][0] <= 0 <= sr["absolute_ci"][1]
    equal(errors, "F7-5 verdict", verdict, reported["passed"])
    return verdict


def basis(kind: str, layer: int, text: str) -> np.ndarray:
    frozen = F.frozen()
    if kind == "clock_union":
        value = frozen[f"clock_union_L{layer:02d}"]
    elif kind == "clock_pertext":
        value = frozen[f"clock_g_L{layer:02d}_{text}"][:, None]
    elif kind == "clock_loto":
        value = frozen[f"clock_loto_L{layer:02d}_{text}"]
    else:
        value = frozen[f"clock_sham6_L{layer:02d}"]
    return value.astype(np.float64)


def gram_clock_stat(dump: Path, arm_id: str, kind: str, layer: int, text: str) -> float:
    values = np.load(
        dump / "arms" / arm_id / "clock" / f"rvec_pre_L{layer:02d}_{text}.npy",
        allow_pickle=False,
    ).astype(np.float32).reshape(R.SEQ, R.RFLAT).astype(np.float64)
    mu = F.frozen()[f"mu_L{layer:02d}_{text}"].astype(np.float64)
    u = basis(kind, layer, text)
    values -= ((values - mu) @ u) @ u.T
    block_mean = values[64:].reshape(127, 64, R.HEADS, R.RPERHEAD).mean(axis=1)
    mean = block_mean.mean(axis=0)
    projection = np.load(ROOT / "weights" / f"layer{layer:02d}_rel_logits_proj.npy", allow_pickle=False).astype(np.float64)
    gram = projection @ projection.T
    numerator = np.einsum("bhi,ij,hj->bh", block_mean, gram, mean)
    denominator = np.einsum("hi,ij,hj->h", mean, gram, mean)
    gain = numerator / np.maximum(denominator[None, :], 1e-30)
    x = np.log1p(np.arange(64, R.SEQ, 64, dtype=np.float64) + 31.5)
    xc = x - x.mean()
    gain -= gain.mean(axis=0)
    corr = np.add.reduce(xc[:, None] * gain, axis=0) / np.maximum(
        np.linalg.norm(xc) * np.linalg.norm(gain, axis=0), 1e-30
    )
    return float(np.median(np.abs(corr)))


def verify_f7_6(dump: Path, reported: dict[str, Any], errors: list[str]) -> bool:
    loto_values = []
    sham_values = []
    for arm_id, destination in (("clock_loto_L53_L59", loto_values), ("clock_sham6_L53_L59", sham_values)):
        kind = F.ARM_BY_ID[arm_id].kind
        for layer in (53, 59):
            for text in TEXTS:
                value = gram_clock_stat(dump, arm_id, kind, layer, text)
                destination.append(value)
                close(errors, f"F7-6 {arm_id} L{layer} {text}", value,
                      reported["kernel_gain_correlation"][arm_id]["cells"][f"L{layer:02d}:{text}"], 2e-8)
    costs = {
        arm_id: pooled(dump, arm_id)[0]
        for arm_id in ("clock_union_L53_L59", "clock_pertext_L53_L59", "clock_loto_L53_L59")
    }
    for name, value in costs.items():
        close(errors, f"F7-6 cost {name}", value, reported["behavior_costs"][name])
    verdict = max(loto_values) < .2 and float(np.median(sham_values)) >= .5 and all(abs(x) < .005 for x in costs.values())
    equal(errors, "F7-6 verdict", verdict, reported["passed"])
    return verdict


def ece(probability: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    assignment = np.clip(np.floor(probability * n_bins).astype(np.int64), 0, n_bins - 1)
    total = probability.size
    value = 0.0
    for index in range(n_bins):
        locations = np.flatnonzero(assignment == index)
        if locations.size:
            confidence = float(np.add.reduce(probability[locations]) / locations.size)
            accuracy = float(np.add.reduce(correct[locations]) / locations.size)
            value += locations.size / total * abs(accuracy - confidence)
    return value


def matched(
    delta: np.ndarray, nll: np.ndarray, positions: np.ndarray, excluded: set[int], label: str
) -> dict[str, float]:
    positions = np.asarray(sorted({int(x) for x in positions if 0 <= int(x) < delta.size}), dtype=np.int64)
    cuts = np.quantile(nll, np.arange(1, 10) / 10)
    decile = np.searchsorted(cuts, nll, side="left")
    token_block = np.arange(delta.size) // 512
    allowed = np.ones(delta.size, dtype=bool)
    if excluded:
        allowed[np.asarray(sorted(x for x in excluded if 0 <= x < delta.size), dtype=np.int64)] = False
    rng = np.random.Generator(np.random.PCG64(seed(label)))
    sum_control = np.zeros(10000, dtype=np.float64)
    for position in positions:
        options = np.flatnonzero(
            allowed & (token_block == token_block[position]) & (decile == decile[position])
        )
        sum_control += delta[rng.choice(options, size=10000, replace=True)]
    differences = float(delta[positions].mean()) - sum_control / positions.size
    return {
        "effect": float(differences.mean()),
        "p": float((1 + np.count_nonzero(differences <= 0)) / 10001),
    }


def verify_f7_7(dump: Path, reported: dict[str, Any], errors: list[str]) -> bool:
    rank_arrays = [arm(dump, "bias_off_L29_fullvocab", text)["delta_log1p_target_rank"] for text in TEXTS]
    accuracy_arrays = [arm(dump, "bias_off_L29_fullvocab", text)["delta_top1_correct"] for text in TEXTS]
    rank = resample(blocks(rank_arrays), "F7-7:rank")
    accuracy = resample(blocks(accuracy_arrays), "F7-7:accuracy")
    close(errors, "F7-7 rank", rank["effect"], reported["ranking"]["effect"])
    close(errors, "F7-7 accuracy", accuracy["effect"], reported["ranking"]["accuracy"]["effect"])
    ranking_pass = rank["effect"] > 0 and rank["ci95"][0] > 0 and accuracy["effect"] < 0 and accuracy["ci95"][1] < 0

    base_probability = np.concatenate([baseline(dump, text)["top1_probability"].astype(np.float64) for text in TEXTS])
    base_correct = np.concatenate([baseline(dump, text)["top1_correct"].astype(np.float64) for text in TEXTS])
    arm_probability = np.concatenate([arm(dump, "bias_off_L29_fullvocab", text)["top1_probability"].astype(np.float64) for text in TEXTS])
    arm_correct = np.concatenate([arm(dump, "bias_off_L29_fullvocab", text)["top1_correct"].astype(np.float64) for text in TEXTS])
    close(errors, "F7-7 baseline ece20", ece(base_probability, base_correct, 20), reported["calibration"]["baseline_ece20"])
    close(errors, "F7-7 arm ece20", ece(arm_probability, arm_correct, 20), reported["calibration"]["arm_ece20"])

    frozen = F.frozen()
    specs = [
        ("07b_slack_multi", "first_content", frozen["class_07b_first_content"]),
        ("07b_slack_multi", "pronouns", frozen["class_07b_pronouns"]),
        ("08_math_llm", "unit_starts", frozen["class_08_unit_starts"]),
        ("08_math_llm", "pronouns", frozen["class_08_pronouns"]),
    ]
    excluded: dict[str, set[int]] = {}
    for text, _name, positions in specs:
        excluded.setdefault(text, set()).update(int(x) for x in positions)
    rows = []
    keys = []
    for text, name, positions in specs:
        fresh_arm = load(dump / "fresh" / "bias_off_L29" / f"{text}.npz")
        fresh_base = load(dump / "fresh" / "baseline" / f"{text}.npz")
        row = matched(
            fresh_arm["delta_nll"].astype(np.float64), fresh_base["nll"].astype(np.float64),
            positions, excluded[text], f"F7-7:class:{text}:{name}:query",
        )
        key = f"{text}:{name}"
        close(errors, f"F7-7 class {key}", row["effect"], reported["fresh_classes"][key]["matched_contrast"])
        rows.append(row)
        keys.append(key)
    adjusted = holm([row["p"] for row in rows])
    for key, value in zip(keys, adjusted):
        close(errors, f"F7-7 class Holm {key}", value, reported["fresh_classes"][key]["p_holm_positive"])
    slack = all(
        rows[index]["effect"] > 0 and adjusted[index] < .05 for index in (0, 1)
    )
    verdict = ranking_pass and slack
    equal(errors, "F7-7 ranking verdict", ranking_pass, reported["ranking"]["passed"])
    equal(errors, "F7-7 fresh verdict", slack, reported["fresh_slack_replication_passed"])
    equal(errors, "F7-7 verdict", verdict, reported["passed"])
    return verdict


def verify(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite verification: {args.out}")
    reported = json.loads(args.results.read_text(encoding="utf-8"))
    errors: list[str] = []
    artifact_count = authenticate(args.dump.resolve(), errors)
    functions = (verify_f7_1, verify_f7_2, verify_f7_3, verify_f7_4, verify_f7_5, verify_f7_6, verify_f7_7)
    verdicts: dict[str, bool] = {}
    for index, function in enumerate(functions, 1):
        try:
            verdicts[f"F7-{index}"] = function(
                args.dump.resolve(), reported["families"][f"F7-{index}"], errors
            )
        except Exception as exc:
            errors.append(f"F7-{index} verifier exception: {type(exc).__name__}: {exc}")
    output = {
        "schema_version": 1,
        "kind": "round5_followup7_independent_verification",
        "created_at_utc": R.utc_now(),
        "passed": not errors,
        "errors": errors,
        "artifact_count_rehashed": artifact_count,
        "results_sha256": R.sha256_file(args.results),
        "verifier_sha256": R.sha256_file(Path(__file__)),
        "producer_imported": False,
        "clock_method": "16x16 projection-Gram identity; no full producer kernels",
        "verdicts": verdicts,
    }
    atomic_json(args.out, output)
    print(json.dumps({"passed": output["passed"], "errors": len(errors), "verdicts": verdicts}, indent=2))
    if errors:
        raise RuntimeError(f"independent verification failed with {len(errors)} errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
