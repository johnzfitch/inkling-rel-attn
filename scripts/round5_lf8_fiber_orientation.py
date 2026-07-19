"""Registered LF8 fiber orientation, depth family, and chirality census."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from round5_science_common import (
    CAPTURE,
    GLOBAL_LAYERS,
    ROOT,
    TEXTS,
    artifact_index,
    atomic_json,
    atomic_npz,
    deterministic_seed,
    holm_adjust,
    provenance,
    refuse_existing,
    require_certified_capture,
    self_test_common,
    sha256_file,
)


DEFAULT_OUT = ROOT / "analysis" / "round5" / "lf8"
TEXT_PAIRS = list(combinations(range(6), 2))


def mean_and_skewness(rvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rvec, dtype=np.float32).reshape(8192, 1024)
    mean = x.mean(axis=0, dtype=np.float64)
    centered = x - mean.astype(np.float32)
    m2 = np.mean(centered * centered, axis=0, dtype=np.float64)
    m3 = np.mean(centered * centered * centered, axis=0, dtype=np.float64)
    if np.any(m2 <= 0):
        raise RuntimeError("constant centered r coordinate")
    g1 = m3 / np.power(m2, 1.5)
    n = x.shape[0]
    adjusted = np.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
    return mean, adjusted


def pairwise_cosines(means: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(means, axis=1)
    if np.any(norms <= 0):
        raise RuntimeError("zero mean-r orientation")
    unit = means / norms[:, None]
    return np.asarray([unit[a] @ unit[b] for a, b in TEXT_PAIRS], dtype=np.float64)


def chirality_test(skewness: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # skewness: text, coordinate
    mean = skewness.mean(axis=0)
    sd = skewness.std(axis=0, ddof=1)
    t_statistic = np.zeros_like(mean)
    nonzero_sd = sd > 0
    t_statistic[nonzero_sd] = mean[nonzero_sd] / (sd[nonzero_sd] / np.sqrt(6.0))
    t_statistic[~nonzero_sd & (mean != 0)] = np.sign(mean[~nonzero_sd & (mean != 0)]) * np.inf
    pvalue = 2.0 * student_t.sf(np.abs(t_statistic), df=5)
    adjusted = holm_adjust(pvalue)
    median = np.median(skewness, axis=0)
    candidate = (adjusted < 0.05) & (np.abs(median) > 0.25)
    return adjusted, median, candidate


def family_statistic(cosine: np.ndarray, globals_set: set[int]) -> float:
    global_index = np.asarray(sorted(globals_set), dtype=np.int64)
    local_index = np.asarray([layer for layer in range(66) if layer not in globals_set], dtype=np.int64)
    gg = cosine[np.ix_(global_index, global_index)]
    ll = cosine[np.ix_(local_index, local_index)]
    gl = cosine[np.ix_(global_index, local_index)]
    gg_mean = float(gg[np.triu_indices(len(global_index), 1)].mean())
    ll_mean = float(ll[np.triu_indices(len(local_index), 1)].mean())
    gl_mean = float(gl.mean())
    return 0.5 * (gg_mean + ll_mean) - gl_mean


def family_test(layer_means: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(layer_means, axis=1)
    if np.any(norms <= 0):
        raise RuntimeError("zero layer mean-r vector")
    unit = layer_means / norms[:, None]
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    observed = family_statistic(cosine, GLOBAL_LAYERS)
    rng = np.random.default_rng(deterministic_seed("lf8-depth-family"))
    null = np.empty(10000, dtype=np.float64)
    for replicate in range(10000):
        pseudo = {block * 6 + int(rng.integers(0, 6)) for block in range(11)}
        null[replicate] = family_statistic(cosine, pseudo)
    pvalue = float((1 + np.sum(null >= observed)) / 10001.0)
    return {
        "observed_balanced_family_statistic": observed,
        "one_sided_depth_matched_permutation_p": pvalue,
        "permutations": 10000,
        "passed": bool(observed > 0 and pvalue <= 0.05),
        "cosine": cosine,
        "null": null,
    }


def run(out: Path) -> None:
    report_path = out / "lf8.json"
    dump_path = out / "lf8_arrays.npz"
    results_path = out / "RESULTS.md"
    refuse_existing(report_path, dump_path, results_path)
    _, manifest = require_certified_capture()
    records = artifact_index(manifest)
    mean_vectors = np.empty((66, 6, 1024), dtype=np.float64)
    skewness = np.empty((66, 6, 1024), dtype=np.float64)
    content_cosine = np.empty((66, 15), dtype=np.float64)
    adjusted_p = np.empty((66, 1024), dtype=np.float64)
    median_skewness = np.empty((66, 1024), dtype=np.float64)
    chirality_candidate = np.zeros((66, 1024), dtype=bool)
    input_hashes: dict[str, str] = {}

    layer_order = [53] + [layer for layer in range(66) if layer != 53]
    for layer in layer_order:
        for text_index, text in enumerate(TEXTS):
            path = CAPTURE / "replay" / f"rvec_L{layer:02d}_{text}.npy"
            relative = path.relative_to(CAPTURE).as_posix()
            record = records.get(relative)
            if record is None or record.get("kind") != "rvec":
                raise RuntimeError(f"r-vector is not bound by manifest: {relative}")
            input_hashes[relative] = record["sha256"]
            rvec = np.load(path, mmap_mode="r")
            if rvec.shape != (8192, 64, 16) or rvec.dtype != np.float16:
                raise RuntimeError(f"invalid r-vector: {path}")
            mean_vectors[layer, text_index], skewness[layer, text_index] = mean_and_skewness(rvec)
        content_cosine[layer] = pairwise_cosines(mean_vectors[layer])
        adjusted_p[layer], median_skewness[layer], chirality_candidate[layer] = chirality_test(
            skewness[layer]
        )
        label = " (pre-disclosed anomaly first)" if layer == 53 else ""
        print(
            f"LF8 L{layer:02d}: min cross-text cos={content_cosine[layer].min():.6f}, "
            f"chirality candidates={chirality_candidate[layer].sum()}{label}",
            flush=True,
        )

    minimum_index = np.unravel_index(np.argmin(content_cosine), content_cosine.shape)
    pair = TEXT_PAIRS[minimum_index[1]]
    stability_passed = bool(np.all(content_cosine > 0.9))
    six_text_layer_mean = mean_vectors.mean(axis=1)
    family = family_test(six_text_layer_mean)
    candidate_locations = np.argwhere(chirality_candidate)
    chirality_passed = bool(candidate_locations.size == 0)
    verdicts = {
        "content_stability_cos_gt_0p9_everywhere": stability_passed,
        "global_local_depth_family_split": family["passed"],
        "no_centered_chirality_candidates": chirality_passed,
        "all_registered_predictions": bool(stability_passed and family["passed"] and chirality_passed),
    }
    report = {
        "schema_version": 1,
        "kind": "round5_lf8_fiber_orientation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANSWERED; independent re-derivation pending",
        "registered_verdicts": verdicts,
        "content_stability": {
            "minimum_cosine": float(content_cosine[minimum_index]),
            "minimum_layer": int(minimum_index[0]),
            "minimum_text_pair": [TEXTS[pair[0]], TEXTS[pair[1]]],
            "layer53_minimum_cosine": float(content_cosine[53].min()),
            "per_layer_minimum": np.min(content_cosine, axis=1).tolist(),
        },
        "depth_family": {
            key: value
            for key, value in family.items()
            if key not in {"cosine", "null"}
        },
        "chirality": {
            "candidate_count": int(len(candidate_locations)),
            "candidate_locations": [
                {
                    "layer": int(layer),
                    "coordinate": int(coordinate),
                    "median_skewness": float(median_skewness[layer, coordinate]),
                    "holm_p": float(adjusted_p[layer, coordinate]),
                }
                for layer, coordinate in candidate_locations
            ],
            "holm_family": "1024 coordinates within each layer",
            "effect_threshold": 0.25,
        },
        "input_rvec_sha256": input_hashes,
        "provenance": provenance(Path(__file__)),
    }
    atomic_npz(
        dump_path,
        mean_vectors=mean_vectors,
        skewness=skewness,
        content_pairwise_cosine=content_cosine,
        chirality_holm_p=adjusted_p,
        median_skewness=median_skewness,
        chirality_candidate=chirality_candidate,
        across_layer_cosine=family["cosine"],
        family_permutation_null=family["null"],
    )
    report["dump_sha256"] = sha256_file(dump_path)
    atomic_json(report_path, report)
    results = [
        "# LF8 — orientation of the relative fiber",
        "",
        "**Status: answered from the certified corrected capture; independent re-derivation pending.**",
        "",
        f"- Content stability >0.9 everywhere: **{str(stability_passed).lower()}**; minimum `{content_cosine[minimum_index]:.6g}` at L{minimum_index[0]} ({TEXTS[pair[0]]} vs {TEXTS[pair[1]]}); corrected L53 minimum `{content_cosine[53].min():.6g}`.",
        f"- Global/local depth-family split: **{str(family['passed']).lower()}** (balanced statistic `{family['observed_balanced_family_statistic']:.6g}`, depth-matched p `{family['one_sided_depth_matched_permutation_p']:.6g}`).",
        f"- No centered chirality beyond the mean: **{str(chirality_passed).lower()}** ({len(candidate_locations)} frozen-rule candidates).",
        "",
        "Full mean orientations, odd moments, adjusted p-values, family null, controls, and source hashes are in `lf8.json` and `lf8_arrays.npz`.",
        "",
    ]
    results_path.write_text("\n".join(results), encoding="utf-8", newline="\n")
    print(json.dumps(verdicts, indent=2))
    print(f"wrote {out}")


def self_test() -> None:
    self_test_common()
    rng = np.random.default_rng(2)
    synthetic = rng.normal(size=(8192, 64, 16)).astype(np.float16)
    mean, skew = mean_and_skewness(synthetic)
    if mean.shape != (1024,) or skew.shape != (1024,) or not np.isfinite(skew).all():
        raise AssertionError((mean.shape, skew.shape))
    stable = np.tile(np.arange(1, 1025, dtype=np.float64), (6, 1))
    if not np.allclose(pairwise_cosines(stable), 1.0):
        raise AssertionError("pairwise cosine self-test failed")
    fake_skew = rng.normal(0, 0.01, size=(6, 1024))
    fake_skew[:, 3] += 1.0
    adjusted, median, candidate = chirality_test(fake_skew)
    if not candidate[3] or median[3] < 0.9 or adjusted[3] >= 0.05:
        raise AssertionError((adjusted[3], median[3], candidate[3]))
    layer_means = rng.normal(size=(66, 1024))
    family = family_test(layer_means)
    if family["null"].shape != (10000,):
        raise AssertionError(family["null"].shape)
    print("round5_lf8_fiber_orientation self-test passed")


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
