"""LF4 zoom-lens aperture dump and frozen class tests.

Outcome-bearing commands refuse to run until LF5's confirmation manifest says
its capture, replay, CPU characterization, and independent checks have passed.
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
CAPTURE = ROOT / "dumps" / "tier2" / "capture"
WEIGHTS = ROOT / "weights"
LOCI = ROOT / "analysis" / "round5" / "loci.json"
LF5_CONFIRMATION = ROOT / "analysis" / "round5" / "lf5" / "confirmation.json"
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "lf4"
DEFAULT_REPORT = ROOT / "analysis" / "round5" / "lf4" / "zoom_lens.json"
REGISTRATION_COMMIT = "34278b4"
PLAN_COMMIT = "d4e2579"
TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
GLOBAL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}
MID_GLOBALS = [23, 29, 35, 41, 47]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def require_lf5(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "capture_gate",
        "replay_gate",
        "cpu_gate",
        "row_dump_gate",
        "bracket_reference_agreement",
    ]
    missing = [name for name in required if not report.get(name)]
    if missing or not report.get("methodology_passed"):
        raise RuntimeError(f"LF5 is not confirmed; missing/failed={missing}")
    return report


def aperture_blocked(
    rvec: np.ndarray,
    projection: np.ndarray,
    *,
    block_tokens: int = 32,
) -> dict[str, np.ndarray]:
    if rvec.ndim != 3 or rvec.shape[1:] != (64, 16):
        raise ValueError(rvec.shape)
    if projection.ndim != 2 or projection.shape[0] != 16:
        raise ValueError(projection.shape)
    tokens = rvec.shape[0]
    extent = projection.shape[1]
    if extent not in (512, 1024):
        raise ValueError(extent)

    full_num = np.zeros(tokens, dtype=np.float64)
    full_den = np.zeros(tokens, dtype=np.float64)
    available_num = np.zeros(tokens, dtype=np.float64)
    available_den = np.zeros(tokens, dtype=np.float64)
    head_num = np.zeros((tokens, 64), dtype=np.float64)
    head_den = np.zeros((tokens, 64), dtype=np.float64)

    projection64 = np.asarray(projection, dtype=np.float64)
    for start in range(0, tokens, block_tokens):
        stop = min(start + block_tokens, tokens)
        coeff = np.asarray(rvec[start:stop], dtype=np.float64)
        curves = coeff.reshape(-1, 16) @ projection64
        absolute = np.abs(curves).reshape(stop - start, 64, extent)
        den_h = absolute.sum(axis=2, dtype=np.float64)
        num_h = absolute[:, :, 129:].sum(axis=2, dtype=np.float64)
        head_den[start:stop] = den_h
        head_num[start:stop] = num_h
        full_den[start:stop] = den_h.sum(axis=1, dtype=np.float64)
        full_num[start:stop] = num_h.sum(axis=1, dtype=np.float64)

        cumulative = np.cumsum(absolute, axis=2, dtype=np.float64)
        max_distance = np.minimum(np.arange(start, stop), extent - 1)
        available_h = np.take_along_axis(
            cumulative,
            max_distance[:, None, None],
            axis=2,
        )[:, :, 0]
        far_h = np.zeros_like(available_h)
        has_far = max_distance > 128
        if np.any(has_far):
            far_h[has_far] = available_h[has_far] - cumulative[has_far, :, 128]
        available_den[start:stop] = available_h.sum(axis=1, dtype=np.float64)
        available_num[start:stop] = far_h.sum(axis=1, dtype=np.float64)

    if np.any(full_den <= 0) or np.any(available_den <= 0):
        raise RuntimeError("zero aperture denominator")
    aperture = full_num / full_den
    available = available_num / available_den
    headwise = np.divide(
        head_num,
        head_den,
        out=np.full_like(head_num, np.nan),
        where=head_den > 0,
    )
    for name, values in {"aperture": aperture, "available": available, "headwise": headwise}.items():
        if not np.isfinite(values).all() or np.any(values < -1e-12) or np.any(values > 1 + 1e-12):
            raise RuntimeError(f"invalid {name}")
    return {
        "aperture_full": aperture,
        "aperture_available": available,
        "full_numerator": full_num,
        "full_denominator": full_den,
        "available_numerator": available_num,
        "available_denominator": available_den,
        "headwise_aperture_full": headwise.astype(np.float32),
    }


def compute_command(args: argparse.Namespace) -> None:
    lf5 = require_lf5(args.lf5_confirmation)
    loci_sha = sha256_file(args.loci)
    layers = list(range(66)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("loci_sha256") != loci_sha:
            raise RuntimeError("existing LF4 dump uses different loci")
    else:
        manifest = {
            "schema_version": 1,
            "kind": "round5_lf4_aperture_dump",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False,
            "registration_commit": REGISTRATION_COMMIT,
            "plan_commit": PLAN_COMMIT,
            "source_sha256": sha256_file(Path(__file__)),
            "loci_sha256": loci_sha,
            "lf5_confirmation_sha256": sha256_file(args.lf5_confirmation),
            "lf5_methodology_passed": bool(lf5["methodology_passed"]),
            "layers": layers,
            "texts": TEXTS,
            "files": {},
        }
        atomic_json(manifest_path, manifest)

    for layer in layers:
        projection_path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        projection = np.load(projection_path)
        expected_extent = 1024 if layer in GLOBAL_LAYERS else 512
        if projection.shape != (16, expected_extent):
            raise RuntimeError((layer, projection.shape))
        for text in TEXTS:
            key = f"L{layer:02d}_{text}"
            output_path = args.out / f"aperture_{key}.npz"
            if output_path.exists():
                if key not in manifest["files"]:
                    raise RuntimeError(f"unmanifested existing output: {output_path}")
                continue
            rvec_path = CAPTURE / f"rvec_L{layer:02d}_{text}.npy"
            rvec = np.load(rvec_path, mmap_mode="r")
            started = time.time()
            arrays = aperture_blocked(rvec, projection, block_tokens=args.block_tokens)
            atomic_npz(output_path, **arrays)
            manifest["files"][key] = {
                "path": output_path.relative_to(args.out).as_posix(),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "rvec_sha256": sha256_file(rvec_path),
                "projection_sha256": sha256_file(projection_path),
                "extent": expected_extent,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            manifest["last_completed"] = key
            atomic_json(manifest_path, manifest)
            print(f"{key}: {manifest['files'][key]['elapsed_seconds']:.2f}s", flush=True)
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


def averaged_scores(dump: Path, text: str, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    token_scores = []
    head_scores = []
    for layer in MID_GLOBALS:
        with np.load(dump / f"aperture_L{layer:02d}_{text}.npz") as data:
            aperture = data["aperture_full"]
            headwise = data["headwise_aperture_full"]
        token_scores.append(midrank_percentiles(aperture, bin_size))
        per_head = np.empty_like(headwise, dtype=np.float64)
        for head in range(64):
            per_head[:, head] = midrank_percentiles(headwise[:, head], bin_size)
        head_scores.append(per_head)
    return np.mean(token_scores, axis=0), np.mean(head_scores, axis=0)


def permutation_test(
    scores: np.ndarray,
    positions: list[int],
    *,
    bin_size: int,
    direction: Literal["positive", "negative", "two-sided"],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    positions_array = np.asarray(sorted(set(positions)), dtype=np.int64)
    if len(positions_array) == 0:
        raise ValueError("empty class")
    observed = float(np.median(scores[positions_array]) - 0.5)
    counts = []
    blocks = []
    for start in range(0, len(scores), bin_size):
        stop = min(start + bin_size, len(scores))
        blocks.append(np.arange(start, stop, dtype=np.int64))
        counts.append(int(np.count_nonzero((positions_array >= start) & (positions_array < stop))))
    rng = np.random.Generator(np.random.PCG64(seed))
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        selected = [
            rng.choice(block, size=count, replace=False)
            for block, count in zip(blocks, counts)
            if count
        ]
        null[iteration] = np.median(scores[np.concatenate(selected)]) - 0.5
    if direction == "positive":
        exceed = np.count_nonzero(null >= observed)
    elif direction == "negative":
        exceed = np.count_nonzero(null <= observed)
    else:
        exceed = np.count_nonzero(np.abs(null) >= abs(observed))
    return {
        "count": int(len(positions_array)),
        "effect": observed,
        "direction": direction,
        "permutations": permutations,
        "seed": seed,
        "p": float((1 + exceed) / (permutations + 1)),
        "null_quantiles": {
            "q025": float(np.quantile(null, 0.025)),
            "q50": float(np.quantile(null, 0.5)),
            "q975": float(np.quantile(null, 0.975)),
        },
    }


def block_bootstrap(
    scores: np.ndarray,
    positions: list[int],
    *,
    block_size: int,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    class_set = set(positions)
    blocks = []
    for start in range(0, len(scores), block_size):
        stop = min(start + block_size, len(scores))
        values = [scores[q] for q in range(start, stop) if q in class_set]
        blocks.append(np.asarray(values, dtype=np.float64))
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        values = [blocks[index] for index in chosen if len(blocks[index])]
        draws[iteration] = np.median(np.concatenate(values)) - 0.5
    return {
        "q025": float(np.quantile(draws, 0.025)),
        "q50": float(np.quantile(draws, 0.5)),
        "q975": float(np.quantile(draws, 0.975)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 1.0
    m = len(p_values)
    for reverse_rank in range(m - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def class_positions(loci: dict[str, Any], text: str, name: str) -> list[int]:
    if name == "closers":
        return [int(x) for x in loci["texts"][text]["loci"]["bracket_query_positions"]]
    if name == "sentence_starts":
        return [int(x["token_pos"]) for x in loci["texts"][text]["loci"]["sentence_starts"]]
    return [int(x["token_pos"]) for x in loci["texts"][text]["classes"][name]]


def contrast(
    *,
    name: str,
    text: str,
    positions: list[int],
    direction: Literal["positive", "negative", "two-sided"],
    dump: Path,
    bin_size: int,
    permutations: int,
) -> dict[str, Any]:
    scores, head_scores = averaged_scores(dump, text, bin_size)
    seed = int(hashlib.sha256(f"{REGISTRATION_COMMIT}|LF4|{name}|{bin_size}".encode()).hexdigest()[:16], 16)
    result = permutation_test(
        scores,
        positions,
        bin_size=bin_size,
        direction=direction,
        permutations=permutations,
        seed=seed,
    )
    result.update(name=name, text=text, bin_size=bin_size)
    result["bootstrap_256"] = block_bootstrap(
        scores,
        positions,
        block_size=256,
        iterations=permutations,
        seed=seed ^ 0xB0057,
    )
    result["head_effects"] = [
        float(np.median(head_scores[np.asarray(positions), head]) - 0.5)
        for head in range(64)
    ]
    result["per_layer_effects"] = {}
    for layer in MID_GLOBALS:
        with np.load(dump / f"aperture_L{layer:02d}_{text}.npz") as data:
            ranks = midrank_percentiles(data["aperture_full"], bin_size)
        result["per_layer_effects"][str(layer)] = float(np.median(ranks[positions]) - 0.5)
    return result


def analyze_command(args: argparse.Namespace) -> None:
    lf5 = require_lf5(args.lf5_confirmation)
    manifest = json.loads((args.dump / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or len(manifest.get("files", {})) != 396:
        raise RuntimeError("LF4 aperture dump is incomplete")
    loci = json.loads(args.loci.read_text(encoding="utf-8"))

    primary_specs = [
        ("closers", "02_code", "positive"),
        ("pronouns", "01_prose_en", "positive"),
        ("function_words", "01_prose_en", "negative"),
    ]
    primary = []
    for name, text, direction in primary_specs:
        key = name if name == "closers" else name
        positions = class_positions(loci, text, key)
        result = contrast(
            name=name,
            text=text,
            positions=positions,
            direction=direction,
            dump=args.dump,
            bin_size=256,
            permutations=args.permutations,
        )
        result["sensitivity"] = {}
        for bin_size in (128, 512):
            sensitivity = contrast(
                name=name,
                text=text,
                positions=positions,
                direction=direction,
                dump=args.dump,
                bin_size=bin_size,
                permutations=args.permutations,
            )
            # Avoid duplicating large diagnostics in sensitivity branches.
            sensitivity.pop("head_effects")
            sensitivity.pop("per_layer_effects")
            sensitivity.pop("bootstrap_256")
            result["sensitivity"][str(bin_size)] = sensitivity

        random_scores, _ = averaged_scores(args.dump, "06_random", 256)
        random_seed = int(
            hashlib.sha256(f"{REGISTRATION_COMMIT}|LF4|null|{name}".encode()).hexdigest()[:16],
            16,
        )
        result["random_position_mask_control"] = permutation_test(
            random_scores,
            positions,
            bin_size=256,
            direction=direction,
            permutations=args.permutations,
            seed=random_seed,
        )
        primary.append(result)

    adjusted = holm_adjust([x["p"] for x in primary])
    null_adjusted = holm_adjust(
        [x["random_position_mask_control"]["p"] for x in primary]
    )
    for result, p_adjusted, null_p_adjusted in zip(primary, adjusted, null_adjusted):
        result["p_holm"] = p_adjusted
        result["random_position_mask_control"]["p_holm"] = null_p_adjusted
        result["random_position_mask_control"]["passed_as_null"] = null_p_adjusted >= 0.05
        sign_ok = result["effect"] > 0 if result["direction"] == "positive" else result["effect"] < 0
        result["prediction_passed"] = bool(sign_ok and p_adjusted < 0.05)

    secondary_specs = [
        ("sentence_starts", "01_prose_en", class_positions(loci, "01_prose_en", "sentence_starts")),
        *[
            (
                f"rare_bpe_{text}",
                text,
                class_positions(loci, text, "rare_bpe_primary"),
            )
            for text in TEXTS[:-1]
            if loci["texts"][text]["counts"]["rare_bpe_primary"] > 0
        ],
    ]
    secondary = [
        contrast(
            name=name,
            text=text,
            positions=positions,
            direction="two-sided",
            dump=args.dump,
            bin_size=256,
            permutations=args.permutations,
        )
        for name, text, positions in secondary_specs
    ]
    secondary_adjusted = bh_adjust([x["p"] for x in secondary])
    for result, p_adjusted in zip(secondary, secondary_adjusted):
        result["p_bh"] = p_adjusted

    report = {
        "schema_version": 1,
        "kind": "round5_lf4_zoom_lens",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "source_sha256": sha256_file(Path(__file__)),
        "loci_sha256": sha256_file(args.loci),
        "dump_manifest_sha256": sha256_file(args.dump / "manifest.json"),
        "lf5_confirmation_sha256": sha256_file(args.lf5_confirmation),
        "lf5_methodology_passed": bool(lf5["methodology_passed"]),
        "metric": "sum_h,sum_d>128 |b| / sum_h,sum_all_d |b|",
        "primary_depth_layers": MID_GLOBALS,
        "primary_position_bin": 256,
        "permutations": args.permutations,
        "primary": primary,
        "secondary": secondary,
        "negative_controls_passed": all(
            x["random_position_mask_control"]["passed_as_null"] for x in primary
        ),
        "all_three_primary_predictions_passed": all(x["prediction_passed"] for x in primary),
    }
    atomic_json(args.report, report)
    print(f"wrote {args.report}")


def self_test() -> None:
    rng = np.random.default_rng(0)
    rvec = rng.normal(size=(17, 64, 16)).astype(np.float16)
    projection = rng.normal(size=(16, 512)).astype(np.float32)
    blocked = aperture_blocked(rvec, projection, block_tokens=5)
    direct = np.abs(rvec.astype(np.float64).reshape(-1, 16) @ projection.astype(np.float64))
    direct = direct.reshape(17, 64, 512)
    expected = direct[:, :, 129:].sum((1, 2)) / direct.sum((1, 2))
    if not np.allclose(blocked["aperture_full"], expected, rtol=0, atol=1e-14):
        raise AssertionError("blocked aperture mismatch")
    ranks = midrank_percentiles(np.array([1.0, 1.0, 3.0, 2.0]), 4)
    if not np.allclose(ranks, [0.25, 0.25, 0.875, 0.625]):
        raise AssertionError(ranks)
    if not np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]):
        raise AssertionError("Holm self-test")
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")

    compute = sub.add_parser("compute")
    compute.add_argument("--layers", default="all")
    compute.add_argument("--block-tokens", type=int, default=32)
    compute.add_argument("--loci", type=Path, default=LOCI)
    compute.add_argument("--lf5-confirmation", type=Path, default=LF5_CONFIRMATION)
    compute.add_argument("--out", type=Path, default=DEFAULT_DUMP)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--permutations", type=int, default=10000)
    analyze.add_argument("--loci", type=Path, default=LOCI)
    analyze.add_argument("--lf5-confirmation", type=Path, default=LF5_CONFIRMATION)
    analyze.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    analyze.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    elif args.command == "compute":
        compute_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    else:
        raise SystemExit("choose compute or analyze, or pass --self-test")


if __name__ == "__main__":
    main()
