"""Frozen, outcome-blind analyzer for the complete 72-arm R5-D campaign.

This source is committed with the GPU runner, before any intervention arm is
opened.  It refuses to compute or print a partial verdict: every arm manifest
and every artifact hash must validate first.  Results remain
``answered, pending independent raw-dump re-derivation`` until a second
analyst commits a verifier artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr

import round5_r5d_runner as R


ROOT = R.ROOT
DEFAULT_DUMP = R.DEFAULT_DUMP
DEFAULT_JSON = ROOT / "analysis" / "round5" / "r5d" / "r5d_results.json"
DEFAULT_MARKDOWN = ROOT / "analysis" / "round5" / "r5d" / "RESULTS.md"
DEFAULT_ANALYSIS_DUMP = ROOT / "dumps" / "round5" / "r5d" / "analysis" / "resamples.npz"
BOOTSTRAP_DRAWS = 5000
BLOCK = 256


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def validate_complete_batch(dump_root: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    expected_head: str | None = None
    for arm in R.ARMS:
        root = dump_root / "arms" / arm.arm_id
        path = root / "manifest.json"
        if not path.exists():
            raise RuntimeError(f"RUNNING: missing arm {arm.arm_id}")
        manifest = R.load_json(path)
        if manifest.get("kind") != "round5_r5d_intervention_arm" or manifest.get("complete") is not True:
            raise RuntimeError(f"RUNNING: incomplete arm {arm.arm_id}")
        expected_arm = asdict(arm)
        stored_arm = manifest.get("arm")
        # JSON represents tuples as lists.
        normalized = {
            "arm_id": expected_arm["arm_id"],
            "kind": expected_arm["kind"],
            "layers": list(expected_arm["layers"]),
            "start_layer": expected_arm["start_layer"],
            "meter_layers": list(expected_arm["meter_layers"]),
        }
        if stored_arm != normalized:
            raise RuntimeError(f"arm specification drift: {arm.arm_id}")
        if expected_head is None:
            expected_head = manifest["source_git_head"]
        if manifest.get("source_git_head") != expected_head:
            raise RuntimeError("arms were produced from different committed runner revisions")
        records = manifest.get("artifacts", [])
        if manifest.get("artifact_count") != len(records):
            raise RuntimeError(f"artifact-count mismatch: {arm.arm_id}")
        if manifest.get("expected_artifact_count") != len(records):
            raise RuntimeError(f"expected-artifact-count mismatch: {arm.arm_id}")
        paths = [record["path"] for record in records]
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"duplicate artifact path: {arm.arm_id}")
        expected_count = 6 + 1 + 6 * len(arm.meter_layers)
        if arm.kind in {"clock_freeze", "clock_sham"}:
            expected_count += 6 * len(arm.layers)
        if len(records) != expected_count:
            raise RuntimeError(
                f"unexpected artifact count {arm.arm_id}: {len(records)} != {expected_count}"
            )
        for record in records:
            artifact = root / record["path"]
            if not artifact.is_file() or R.sha256_file(artifact) != record["sha256"]:
                raise RuntimeError(f"artifact hash mismatch: {arm.arm_id}/{record['path']}")
        token_texts = {
            record.get("text") for record in records if record.get("kind") == "r5d_token_readout"
        }
        if token_texts != set(R.TEXTS):
            raise RuntimeError(f"token readout inventory mismatch: {arm.arm_id}")
        manifests[arm.arm_id] = manifest
    if len(manifests) != 72:
        raise AssertionError("complete-batch validator did not see 72 arms")
    return manifests


def load_delta(dump_root: Path, arm_id: str, text: str) -> np.ndarray:
    with np.load(dump_root / "arms" / arm_id / "tokens" / f"{text}.npz", allow_pickle=False) as dump:
        positions = dump["target_position"]
        values = np.array(dump["delta_nll"], dtype=np.float64, copy=True)
        if positions.shape != (R.SEQ - 1,) or not np.array_equal(
            positions, np.arange(1, R.SEQ, dtype=np.int32)
        ):
            raise RuntimeError(f"target position drift: {arm_id}/{text}")
        if values.shape != (R.SEQ - 1,) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid delta NLL: {arm_id}/{text}")
        return values


def bootstrap_seed(arm_id: str) -> int:
    parent_hash = R.sha256_file(R.PARENT_AMENDMENT)
    payload = f"{parent_hash}:r5d_256_token_bootstrap:{arm_id}"
    return R.seed_from(payload)


def block_bootstrap(deltas: list[np.ndarray], arm_id: str) -> np.ndarray:
    """Paired 256-token block bootstrap, independently within each text.

    Each 8,191-target text has 32 consecutive blocks (31 of 256 and a final
    block of 255).  A draw samples 32 blocks with replacement separately for
    each of the six equal-weight texts, then takes the token-weighted pooled
    mean.  This method is frozen here before outcomes.
    """

    sums = np.empty((len(deltas), 32), dtype=np.float64)
    counts = np.empty((len(deltas), 32), dtype=np.int64)
    for text_index, values in enumerate(deltas):
        for block_index, start in enumerate(range(0, R.SEQ - 1, BLOCK)):
            stop = min(start + BLOCK, R.SEQ - 1)
            sums[text_index, block_index] = values[start:stop].sum(dtype=np.float64)
            counts[text_index, block_index] = stop - start
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed(arm_id)))
    sample = rng.integers(0, 32, size=(BOOTSTRAP_DRAWS, len(deltas), 32))
    draw_sums = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    draw_counts = np.zeros(BOOTSTRAP_DRAWS, dtype=np.int64)
    for text_index in range(len(deltas)):
        draw_sums += sums[text_index][sample[:, text_index]].sum(axis=1)
        draw_counts += counts[text_index][sample[:, text_index]].sum(axis=1)
    return draw_sums / draw_counts


def summarize_arm(dump_root: Path, arm_id: str) -> tuple[dict[str, Any], np.ndarray]:
    deltas = [load_delta(dump_root, arm_id, text) for text in R.TEXTS]
    per_text = {
        text: float(np.mean(values, dtype=np.float64)) for text, values in zip(R.TEXTS, deltas)
    }
    pooled = float(np.mean(np.concatenate(deltas), dtype=np.float64))
    natural_five = float(np.mean(np.concatenate(deltas[:5]), dtype=np.float64))
    resamples = block_bootstrap(deltas, arm_id)
    interval = np.percentile(resamples, [2.5, 97.5])
    return (
        {
            "pooled_mean_delta_nll": pooled,
            "natural_five_mean_delta_nll": natural_five,
            "per_text_mean_delta_nll": per_text,
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "block_tokens": BLOCK,
                "seed_unsigned_be": bootstrap_seed(arm_id),
                "ci95": [float(interval[0]), float(interval[1])],
                "median": float(np.median(resamples)),
            },
        },
        resamples.astype(np.float32),
    )


def summarize_needles(dump_root: Path, arm_id: str) -> dict[str, Any]:
    path = dump_root / "arms" / arm_id / "needle" / "05_needles.npz"
    with np.load(path, allow_pickle=False) as dump:
        side = dump["side_of_seam"].astype(str)
        output: dict[str, Any] = {"count": int(side.size), "splits": {}}
        for label, mask in (
            ("all", np.ones(side.size, dtype=bool)),
            ("below", side == "below"),
            ("above", side == "above"),
        ):
            output["splits"][label] = {
                "count": int(mask.sum()),
                "mean_delta_target_logit": float(np.mean(dump["delta_target_logit"][mask], dtype=np.float64)),
                "mean_delta_probability": float(np.mean(dump["delta_probability"][mask], dtype=np.float64)),
                "mean_delta_log_probability": float(
                    np.mean(dump["delta_log_probability"][mask], dtype=np.float64)
                ),
                "mean_delta_nll": float(np.mean(dump["delta_nll"][mask], dtype=np.float64)),
            }
        return output


def meter_file_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as dump:
        mass = np.asarray(dump["mass_with"], dtype=np.float64)
        count = np.asarray(dump["count"], dtype=np.float64)
        effective = (
            np.asarray(dump["mean_effective_count"], dtype=np.float64)
            if "mean_effective_count" in dump.files
            else None
        )
        distances = np.arange(mass.shape[1])
        total = mass.sum(axis=1)
        bands = {
            "d_lt_4": distances < 4,
            "d_4_to_128": (distances >= 4) & (distances <= 128),
            "d_gt_128": distances > 128,
            "d_ge_1024": distances >= 1024,
        }
        fractions = {
            label: (mass[:, mask].sum(axis=1) / total).tolist()
            for label, mask in bands.items()
        }
        return {
            "per_head_mass_fraction": fractions,
            "per_head_mean_effective_count": None if effective is None else effective.tolist(),
            "distance_count": int(count.size),
        }


def aggregate_meter_summaries(dump_root: Path, arm: R.Arm) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer in arm.meter_layers:
        arm_files = [
            meter_file_summary(dump_root / "arms" / arm.arm_id / "meters" / f"L{layer:02d}_{text}.npz")
            for text in R.TEXTS
        ]
        baseline_files = [
            meter_file_summary(R.CAPTURE / "meters" / f"layer{layer:02d}_{text}_s{R.SEQ}.npz")
            for text in R.TEXTS
        ]
        bands: dict[str, Any] = {}
        for band in ("d_lt_4", "d_4_to_128", "d_gt_128", "d_ge_1024"):
            arm_head = np.mean(
                [np.asarray(item["per_head_mass_fraction"][band]) for item in arm_files], axis=0
            )
            baseline_head = np.mean(
                [np.asarray(item["per_head_mass_fraction"][band]) for item in baseline_files], axis=0
            )
            bands[band] = {
                "arm_per_head": arm_head.tolist(),
                "baseline_per_head": baseline_head.tolist(),
                "delta_per_head": (arm_head - baseline_head).tolist(),
                "arm_mean_heads": float(arm_head.mean()),
                "baseline_mean_heads": float(baseline_head.mean()),
                "delta_mean_heads": float((arm_head - baseline_head).mean()),
            }
        arm_effective = np.mean(
            [np.asarray(item["per_head_mean_effective_count"]) for item in arm_files], axis=0
        )
        baseline_effective_values = [item["per_head_mean_effective_count"] for item in baseline_files]
        baseline_effective = (
            np.mean([np.asarray(value) for value in baseline_effective_values], axis=0)
            if all(value is not None for value in baseline_effective_values)
            else None
        )
        output[f"L{layer:02d}"] = {
            "bands": bands,
            "effective_count": {
                "arm_per_head": arm_effective.tolist(),
                "baseline_per_head": None if baseline_effective is None else baseline_effective.tolist(),
                "delta_per_head": None if baseline_effective is None else (arm_effective - baseline_effective).tolist(),
                "arm_mean_heads": float(arm_effective.mean()),
                "baseline_mean_heads": None if baseline_effective is None else float(baseline_effective.mean()),
                "baseline_note": (
                    "Certified capture predates the registered effective-count accumulator; arm value is reported without a fabricated baseline."
                    if baseline_effective is None
                    else None
                ),
            },
        }
    return output


def reconstruct_clock_stat(dump_root: Path, arm_id: str, layer: int) -> float:
    path = dump_root / "arms" / arm_id / "clock" / f"rvec_pre_L{layer:02d}_06_random.npy"
    rvec = np.load(path, allow_pickle=False).astype(np.float32).reshape(R.SEQ, R.RFLAT)
    with np.load(R.CLOCK_PATH, allow_pickle=False) as freeze:
        direction_key = "sham_L59" if arm_id == "clock_sham_L59" else f"G_L{layer}"
        direction = freeze[direction_key].astype(np.float32)
        anchor = freeze[f"rbar_L{layer}"].astype(np.float32)
    coefficient = ((rvec - anchor) * direction).sum(axis=1, keepdims=True, dtype=np.float32)
    frozen = rvec - coefficient * direction
    blocks = frozen[64:].reshape(127, 64, R.HEADS, R.RPERHEAD).mean(axis=1, dtype=np.float64)
    projection = np.load(ROOT / "weights" / f"layer{layer:02d}_rel_logits_proj.npy", allow_pickle=False).astype(np.float64)
    kernels = blocks @ projection
    mean_kernel = kernels.mean(axis=0)
    denominator = np.sum(mean_kernel * mean_kernel, axis=1)
    if np.any(denominator <= 0):
        raise RuntimeError(f"zero clock kernel-gain denominator: {arm_id}")
    gain = np.sum(kernels * mean_kernel[None], axis=2) / denominator[None]
    starts = np.arange(64, R.SEQ, 64)
    x = np.log1p(starts + 31.5)
    correlations = np.asarray([np.corrcoef(gain[:, head], x)[0, 1] for head in range(R.HEADS)])
    if not np.isfinite(correlations).all():
        raise RuntimeError(f"nonfinite clock correlations: {arm_id}")
    return float(np.median(np.abs(correlations)))


def ck2_bootstrap(pooled_blocks: np.ndarray) -> tuple[float, np.ndarray, int]:
    starts = np.arange(64, R.SEQ, 64)
    x = np.log1p(starts + 31.5)
    regressor = np.abs(x - x.mean())
    observed_raw = float(spearmanr(pooled_blocks, regressor).statistic)
    # A constant effect has undefined Spearman rho and cannot satisfy a
    # strictly-positive registered clause; freeze its neutral disposition.
    observed = observed_raw if np.isfinite(observed_raw) else 0.0
    amendment_hash = R.sha256_file(R.CLOCK_AMENDMENT)
    seed = R.seed_from(f"{amendment_hash}:ck2_bootstrap")
    rng = np.random.Generator(np.random.PCG64(seed))
    groups = np.asarray([start // 256 for start in starts], dtype=np.int64)
    indices_by_group = [np.flatnonzero(groups == group) for group in range(32)]
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        selected_groups = rng.integers(0, 32, size=32)
        selected = np.concatenate([indices_by_group[group] for group in selected_groups])
        # Rank explicitly so repeated superblocks receive tie-aware average ranks.
        yrank = rankdata(pooled_blocks[selected], method="average")
        xrank = rankdata(regressor[selected], method="average")
        value = float(np.corrcoef(yrank, xrank)[0, 1])
        draws[draw] = value if np.isfinite(value) else 0.0
    return observed, draws, seed


def clock_verdicts(
    dump_root: Path, summaries: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], np.ndarray]:
    l53 = reconstruct_clock_stat(dump_root, "clock_freeze_L53", 53)
    l59 = reconstruct_clock_stat(dump_root, "clock_freeze_L59", 59)
    sham = reconstruct_clock_stat(dump_root, "clock_sham_L59", 59)
    # The amendment says ties fail; equality at either boundary therefore fails.
    ck1_pass = bool(l53 < 0.20 and l59 < 0.20 and sham > 0.50)

    block_by_text = []
    for text in R.TEXTS:
        values = load_delta(dump_root, "clock_freeze_L53_L59", text)
        # target positions 64..8191 correspond to zero-based readout indices 63..8190.
        block_by_text.append(values[63:].reshape(127, 64).mean(axis=1, dtype=np.float64))
    pooled_blocks = np.mean(block_by_text, axis=0)
    observed, draws, seed = ck2_bootstrap(pooled_blocks)
    lower = float(np.percentile(draws, 2.5))
    ck2_pass = bool(observed > 0.0 and lower > 0.0)

    l65_cost = abs(summaries["clock_freeze_L65"]["pooled_mean_delta_nll"])
    ck3_pass = bool(l65_cost < 0.005)
    return (
        {
            "CK1_kernel_gain_flattening": {
                "passed": ck1_pass,
                "tie_policy": "ties fail",
                "clock_freeze_L53_median_abs_corr": l53,
                "clock_freeze_L59_median_abs_corr": l59,
                "clock_sham_L59_median_abs_corr": sham,
                "required": {"real_each_strictly_below": 0.20, "sham_strictly_above": 0.50},
            },
            "CK2_log_extremes_cost": {
                "passed": ck2_pass,
                "spearman_rho": observed,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_unit": "32 aligned 256-token superblocks",
                "bootstrap_seed_unsigned_be": seed,
                "bootstrap_ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                "bootstrap_lower_95": lower,
                "pooled_block_delta_nll": pooled_blocks.tolist(),
            },
            "CK3_L65_exemption": {
                "passed": ck3_pass,
                "absolute_pooled_delta_nll": l65_cost,
                "required_strictly_below": 0.005,
            },
        },
        draws.astype(np.float32),
    )


def parent_verdicts(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bias_values = {
        layer: summaries[f"bias_off_L{layer:02d}"]["pooled_mean_delta_nll"]
        for layer in R.SINGLE_LAYERS
    }
    required_large = {0, 1, 2, 3, 4, 5, 65}
    bias_checks = {
        layer: (value > 0.05 if layer in required_large else abs(value) <= 0.05)
        for layer, value in bias_values.items()
    }

    carrier_rows: dict[int, Any] = {}
    for layer in R.SINGLE_LAYERS:
        bias = bias_values[layer]
        carrier = summaries[f"carrier_out_L{layer:02d}"]["pooled_mean_delta_nll"]
        difference = abs(carrier - bias)
        bound = 0.20 * abs(bias)
        carrier_rows[layer] = {
            "bias_delta_nll": bias,
            "carrier_delta_nll": carrier,
            "absolute_difference": difference,
            "bound": bound,
            "passed": bool(difference <= bound),
        }

    near_values = [
        summaries[f"near_off_L{layer:02d}"]["pooled_mean_delta_nll"] for layer in R.SINGLE_LAYERS
    ]
    far_values = [
        summaries[f"far_off_L{layer:02d}"]["pooled_mean_delta_nll"] for layer in R.SINGLE_LAYERS
    ]
    near_mean = float(np.mean(near_values))
    far_mean = float(np.mean(far_values))
    ratio = float(near_mean / far_mean) if far_mean != 0 else None
    near_far_pass = bool(far_mean > 0.0 and ratio is not None and ratio >= 5.0)

    wall = summaries["wall_heal_global"]["pooled_mean_delta_nll"]
    return {
        "bias_off_depth": {
            "passed": bool(all(bias_checks.values())),
            "pooled_delta_nll": {f"L{layer:02d}": value for layer, value in bias_values.items()},
            "layer_pass": {f"L{layer:02d}": passed for layer, passed in bias_checks.items()},
        },
        "carrier_equivalence": {
            "passed": bool(all(row["passed"] for row in carrier_rows.values())),
            "layers": {f"L{layer:02d}": row for layer, row in carrier_rows.items()},
        },
        "near_dominates_far": {
            "passed": near_far_pass,
            "mean_near_delta_nll": near_mean,
            "mean_far_delta_nll": far_mean,
            "ratio": ratio,
            "required_ratio": 5.0,
            "far_denominator_strictly_positive": far_mean > 0.0,
            "layerwise_ratio_descriptive": {
                f"L{layer:02d}": float(near / far) if far != 0 else None
                for layer, near, far in zip(R.SINGLE_LAYERS, near_values, far_values)
            },
        },
        "wall_incidental_at_8k": {
            "passed": bool(abs(wall) < 0.005),
            "pooled_delta_nll": wall,
            "absolute_delta_nll": abs(wall),
            "required_strictly_below": 0.005,
        },
    }


def render_markdown(results: dict[str, Any]) -> str:
    verdicts = results["verdicts"]
    clock = results["clock_verdicts"]
    lines = [
        "# Round 5 R5-D causal ablations — results",
        "",
        f"Status: **{results['status']}**. All 72 arms were sealed before verdict computation.",
        "",
        "| Registered clause | Verdict | Key value |",
        "|---|---:|---:|",
        f"| Bias-off depth | {'PASS' if verdicts['bias_off_depth']['passed'] else 'FAIL'} | see JSON layer table |",
        f"| Carrier equivalence | {'PASS' if verdicts['carrier_equivalence']['passed'] else 'FAIL'} | 16/16 required |",
        f"| Near dominates far | {'PASS' if verdicts['near_dominates_far']['passed'] else 'FAIL'} | ratio {verdicts['near_dominates_far']['ratio'] if verdicts['near_dominates_far']['ratio'] is not None else 'undefined'} |",
        f"| Wall incidental at 8k | {'PASS' if verdicts['wall_incidental_at_8k']['passed'] else 'FAIL'} | |ΔNLL| {verdicts['wall_incidental_at_8k']['absolute_delta_nll']:.6g} |",
        f"| CK1 clock mechanism | {'PASS' if clock['CK1_kernel_gain_flattening']['passed'] else 'FAIL'} | L53 {clock['CK1_kernel_gain_flattening']['clock_freeze_L53_median_abs_corr']:.4f}; L59 {clock['CK1_kernel_gain_flattening']['clock_freeze_L59_median_abs_corr']:.4f}; sham {clock['CK1_kernel_gain_flattening']['clock_sham_L59_median_abs_corr']:.4f} |",
        f"| CK2 log-extremes cost | {'PASS' if clock['CK2_log_extremes_cost']['passed'] else 'FAIL'} | rho {clock['CK2_log_extremes_cost']['spearman_rho']:.4f}; lower {clock['CK2_log_extremes_cost']['bootstrap_lower_95']:.4f} |",
        f"| CK3 L65 exemption | {'PASS' if clock['CK3_L65_exemption']['passed'] else 'FAIL'} | |ΔNLL| {clock['CK3_L65_exemption']['absolute_pooled_delta_nll']:.6g} |",
        "",
        "The two head-class arms, needle splits, all per-text costs, bootstrap intervals, and full locus-meter summaries are in `r5d_results.json`. No threshold was invented for descriptive readouts.",
        "",
        "Promotion state: answered, pending independent raw-dump re-derivation.",
        "",
    ]
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> None:
    dump_root = args.dump.resolve()
    output_json = args.output_json.resolve()
    output_markdown = args.output_markdown.resolve()
    analysis_dump = args.analysis_dump.resolve()
    for path in (output_json, output_markdown, analysis_dump):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite analysis output: {path}")
    manifests = validate_complete_batch(dump_root)
    summaries: dict[str, dict[str, Any]] = {}
    resamples: dict[str, np.ndarray] = {}
    needles: dict[str, Any] = {}
    meters: dict[str, Any] = {}
    for ordinal, arm in enumerate(R.ARMS, start=1):
        summary, draws = summarize_arm(dump_root, arm.arm_id)
        summaries[arm.arm_id] = summary
        resamples[f"arm__{arm.arm_id}"] = draws
        needles[arm.arm_id] = summarize_needles(dump_root, arm.arm_id)
        meters[arm.arm_id] = aggregate_meter_summaries(dump_root, arm)
        print(f"validated/analyzed {ordinal:02d}/72 {arm.arm_id}", flush=True)
    parent = parent_verdicts(summaries)
    clock, ck2_draws = clock_verdicts(dump_root, summaries)
    resamples["clock__ck2"] = ck2_draws
    atomic_npz(analysis_dump, **resamples)
    results = {
        "schema_version": 1,
        "kind": "round5_r5d_causal_ablation_results",
        "created_at_utc": utc_now(),
        "status": "ANSWERED_PENDING_INDEPENDENT_REDERIVATION",
        "registered_arm_count": 72,
        "complete_arm_count": len(manifests),
        "source_git_head": next(iter(manifests.values()))["source_git_head"],
        "runner_sha256": R.sha256_file(R.SCRIPT_DIR / "round5_r5d_runner.py"),
        "analyzer_sha256": R.sha256_file(Path(__file__)),
        "parent_amendment_sha256": R.sha256_file(R.PARENT_AMENDMENT),
        "clock_amendment_sha256": R.sha256_file(R.CLOCK_AMENDMENT),
        "analysis_resamples": {
            "path": analysis_dump.relative_to(ROOT).as_posix(),
            "sha256": R.sha256_file(analysis_dump),
            "keys": sorted(resamples),
        },
        "verdicts": parent,
        "clock_verdicts": clock,
        "arm_summaries": summaries,
        "needle_readouts": needles,
        "locus_meter_summaries": meters,
        "head_class_readouts": {
            arm_id: summaries[arm_id]
            for arm_id in ("rising_heads_off_L00_L04", "negative_seam_heads_off_L11")
        },
        "certification": {
            "all_raw_artifact_hashes_revalidated": True,
            "independent_second_analyst_artifact": False,
            "promotion_allowed": False,
        },
    }
    atomic_json(output_json, results)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_markdown.with_suffix(output_markdown.suffix + ".tmp")
    temporary.write_text(render_markdown(results), encoding="utf-8", newline="\n")
    os.replace(temporary, output_markdown)
    print(
        json.dumps(
            {
                "status": results["status"],
                "results": str(output_json),
                "results_sha256": R.sha256_file(output_json),
                "markdown": str(output_markdown),
                "analysis_dump": str(analysis_dump),
            },
            indent=2,
        )
    )


def self_test() -> None:
    if len(R.ARMS) != 72:
        raise AssertionError("analyzer sees wrong arm count")
    dummy = [np.linspace(-1, 1, R.SEQ - 1, dtype=np.float64) for _ in R.TEXTS]
    draws_a = block_bootstrap(dummy, "self_test")
    draws_b = block_bootstrap(dummy, "self_test")
    if draws_a.shape != (BOOTSTRAP_DRAWS,) or not np.array_equal(draws_a, draws_b):
        raise RuntimeError("parent bootstrap is not deterministic")
    pooled = np.linspace(0, 1, 127)
    rho, ck_a, seed_a = ck2_bootstrap(pooled)
    _, ck_b, seed_b = ck2_bootstrap(pooled)
    if not rho > 0 or seed_a != seed_b or not np.array_equal(ck_a, ck_b):
        raise RuntimeError("CK2 bootstrap is not deterministic")
    print(
        json.dumps(
            {
                "passed": True,
                "arm_count": len(R.ARMS),
                "parent_bootstrap_draws": BOOTSTRAP_DRAWS,
                "ck2_bootstrap_draws": BOOTSTRAP_DRAWS,
                "ck2_test_rho": rho,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["self-test", "analyze"])
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--analysis-dump", type=Path, default=DEFAULT_ANALYSIS_DUMP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-test":
        self_test()
    else:
        analyze(args)


if __name__ == "__main__":
    main()
