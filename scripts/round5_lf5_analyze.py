"""LF5 frozen-locus row dump and registered bracket-matching analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from round5_offline_attention import (
    AMENDMENT_A5_COMMIT,
    DEFAULT_INPUTS,
    OfflineAttention,
    PLAN_COMMIT,
    REGISTRATION_COMMIT,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
LOCI = ROOT / "analysis" / "round5" / "loci.json"
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "lf5"
DEFAULT_REPLAY_GATE = ROOT / "analysis" / "round5" / "lf5" / "parity_replay.json"
DEFAULT_CPU_GATE = ROOT / "analysis" / "round5" / "lf5" / "parity_cpu_sentinels.json"
DEFAULT_A5 = ROOT / "ROUND5_AMENDMENT_A5.md"
DEFAULT_BRACKET_REPORT = ROOT / "analysis" / "round5" / "lf5" / "brackets.json"
TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
COMPONENTS = [
    "content_logits",
    "positional_bias",
    "total_logits",
    "attention_with_bias",
    "attention_without_bias",
]
FP16_MIN_POSITIVE = float(np.nextafter(np.float16(0), np.float16(1)))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def parity_report(path: Path, backend: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("kind") != "round5_lf5_parity" or report.get("backend") != backend:
        raise RuntimeError(f"wrong {backend} gate report: {path}")
    return report


def require_gates(
    replay_path: Path,
    cpu_path: Path,
    input_root: Path,
    amendment_path: Path,
) -> dict[str, Any]:
    replay = parity_report(replay_path, "replay")
    cpu = parity_report(cpu_path, "cpu")
    if replay.get("layers") != list(range(66)) or len(replay.get("results", [])) != 66:
        raise RuntimeError(f"incomplete replay gate report: {replay_path}")
    if not replay.get("passed") or not all(x.get("passed") for x in replay["results"]):
        raise RuntimeError(f"failed replay gate report: {replay_path}")
    cpu_results = cpu.get("results", [])
    if (
        cpu.get("passed") is not False
        or not cpu_results
        or cpu_results[0].get("layer") != 0
        or cpu_results[0].get("passed") is not False
    ):
        raise RuntimeError("A5 requires the registered L0 CPU failure to remain preserved")
    input_sha = sha256_file(input_root / "manifest.json")
    if replay.get("input_manifest_sha256") != input_sha or cpu.get("input_manifest_sha256") != input_sha:
        raise RuntimeError("parity reports do not certify the selected input manifest")
    amendment = amendment_path.read_text(encoding="utf-8")
    required_clauses = [
        "the production LF5 instrument is the **GPU replay backend**",
        "No threshold is relaxed",
        "CPU path is demoted to a non-registered convenience",
    ]
    missing = [clause for clause in required_clauses if clause not in amendment]
    if missing:
        raise RuntimeError(f"A5 amendment is missing required clauses: {missing}")
    return {
        "amendment_a5_commit": AMENDMENT_A5_COMMIT,
        "amendment_a5_sha256": sha256_file(amendment_path),
        "production_backend": "replay",
        "replay_report_sha256": sha256_file(replay_path),
        "replay_gate_passed": True,
        "registered_cpu_gate_passed": False,
        "registered_cpu_failure_preserved": True,
        "cpu_failure_report_sha256": sha256_file(cpu_path),
        "input_manifest_sha256": input_sha,
    }


def query_positions(loci: dict[str, Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for text in TEXTS:
        item = loci["texts"][text]["loci"]
        positions = {int(x["token_pos"]) for x in item["random_controls"]}
        if text == "01_prose_en":
            positions.update(int(x["token_pos"]) for x in item["sentence_starts"])
        elif text == "02_code":
            positions.update(int(x) for x in item["bracket_query_positions"])
        elif text == "03_templated":
            positions.update(int(x["token_pos"]) for x in item["heartbeat_line_ends"])
        result[text] = sorted(positions)
    return result


def group_hashes(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and file.name != "group_manifest.json":
            result[file.name] = {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
    return result


def dump_group(
    instrument: OfflineAttention,
    *,
    text: str,
    qpos: list[int],
    root: Path,
    query_batch: int,
) -> dict[str, Any]:
    layer = instrument.layer
    final = root / f"L{layer:02d}_{text}"
    if final.exists():
        group_manifest = final / "group_manifest.json"
        if not group_manifest.exists():
            raise RuntimeError(f"existing incomplete group: {final}")
        return json.loads(group_manifest.read_text(encoding="utf-8"))
    tmp = root / f"L{layer:02d}_{text}.tmp"
    if tmp.exists():
        raise RuntimeError(f"stale temporary group requires inspection: {tmp}")
    tmp.mkdir(parents=True)

    starts = np.array(
        [0 if instrument.is_global else max(0, q - 511) for q in qpos], dtype=np.int32
    )
    stops = np.array([q + 1 for q in qpos], dtype=np.int32)
    lengths = stops.astype(np.int64) - starts
    indptr = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    total_keys = int(indptr[-1])
    arrays: dict[str, np.memmap] = {}
    temp_paths: dict[str, Path] = {}
    for component in COMPONENTS:
        temp_path = tmp / f"{component}.tmp.npy"
        temp_paths[component] = temp_path
        arrays[component] = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=np.float16, shape=(total_keys, 64)
        )

    started = time.time()
    if instrument.backend == "replay":
        by_chunk: dict[int, list[int]] = {}
        for q in qpos:
            by_chunk.setdefault(q // instrument.qchunk, []).append(q)
        batches = [by_chunk[chunk] for chunk in sorted(by_chunk)]
    else:
        batches = [qpos[start : start + query_batch] for start in range(0, len(qpos), query_batch)]
    index_by_query = {q: index for index, q in enumerate(qpos)}
    completed = 0
    for batch_positions in batches:
        rows = instrument.rows(text, batch_positions, compact=True)
        for row in rows:
            index = index_by_query[row.query_position]
            lo, hi = int(indptr[index]), int(indptr[index + 1])
            if not np.array_equal(row.key_positions, np.arange(starts[index], stops[index])):
                raise RuntimeError(f"ragged support mismatch at L{layer}, {text}, q={row.query_position}")
            for component in COMPONENTS:
                value = getattr(row, component).T
                if value.shape != (hi - lo, 64):
                    raise RuntimeError((component, value.shape, hi - lo))
                arrays[component][lo:hi] = value.astype(np.float16)
        completed += len(rows)
        print(
            f"L{layer:02d} {text}: {completed}/{len(qpos)} rows",
            flush=True,
        )

    for component, array in list(arrays.items()):
        array.flush()
        del arrays[component]
        os.replace(temp_paths[component], tmp / f"{component}.npy")
    np.savez(
        tmp / "index.npz",
        qpos=np.asarray(qpos, dtype=np.int32),
        support_start=starts,
        support_stop=stops,
        indptr=indptr,
    )
    group = {
        "schema_version": 1,
        "layer": layer,
        "text": text,
        "is_global": instrument.is_global,
        "backend": instrument.backend,
        "query_batch_strategy": "one original 512-query chunk at a time",
        "queries": len(qpos),
        "total_key_entries": total_keys,
        "components": COMPONENTS,
        "dtype": "float16",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json(tmp / "group_manifest.json", group)
    group["files"] = group_hashes(tmp)
    atomic_json(tmp / "group_manifest.json", group)
    os.replace(tmp, final)
    return group


def dump_command(args: argparse.Namespace) -> None:
    gate_hashes = require_gates(
        args.replay_gate,
        args.cpu_gate,
        args.input_root,
        args.amendment_a5,
    )
    loci = json.loads(args.loci.read_text(encoding="utf-8"))
    if loci.get("registration_commit") != REGISTRATION_COMMIT or loci.get("plan_commit") != PLAN_COMMIT:
        raise RuntimeError("loci manifest registration mismatch")
    queries = query_positions(loci)
    layers = list(range(66)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("loci_sha256") != sha256_file(args.loci):
            raise RuntimeError("existing LF5 dump uses different loci")
    else:
        manifest = {
            "schema_version": 1,
            "kind": "round5_lf5_frozen_locus_rows",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False,
            "registration_commit": REGISTRATION_COMMIT,
            "plan_commit": PLAN_COMMIT,
            "amendment_a5_commit": AMENDMENT_A5_COMMIT,
            "production_backend": "replay",
            "source_sha256": sha256_file(Path(__file__)),
            "offline_source_sha256": sha256_file(ROOT / "scripts" / "round5_offline_attention.py"),
            "loci_sha256": sha256_file(args.loci),
            "gates": gate_hashes,
            "layers": layers,
            "queries": queries,
            "groups": {},
        }
        atomic_json(manifest_path, manifest)

    for layer in layers:
        with OfflineAttention(layer, backend="replay", input_root=args.input_root) as instrument:
            for text in TEXTS:
                key = f"L{layer:02d}_{text}"
                group = dump_group(
                    instrument,
                    text=text,
                    qpos=queries[text],
                    root=args.out,
                    query_batch=args.query_batch,
                )
                manifest["groups"][key] = group
                manifest["last_completed_group"] = key
                atomic_json(manifest_path, manifest)
    manifest["complete"] = True
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(f"DONE {len(manifest['groups'])} groups")


class RowStore:
    def __init__(self, path: Path):
        with np.load(path / "index.npz") as index:
            self.qpos = index["qpos"].astype(np.int64)
            self.starts = index["support_start"].astype(np.int64)
            self.stops = index["support_stop"].astype(np.int64)
            self.indptr = index["indptr"].astype(np.int64)
        self.lookup = {int(q): i for i, q in enumerate(self.qpos)}
        self.attention = np.load(path / "attention_with_bias.npy", mmap_mode="r")

    def head_mean_at(self, q: int, keys: list[int]) -> np.ndarray:
        index = self.lookup[q]
        start = int(self.starts[index])
        stop = int(self.stops[index])
        if any(k < start or k >= stop for k in keys):
            raise IndexError((q, start, stop, keys))
        lo = int(self.indptr[index])
        offsets = np.asarray(keys, dtype=np.int64) - start + lo
        return np.asarray(self.attention[offsets], dtype=np.float64).mean(axis=1)


def candidate_controls(
    q: int,
    matched: int,
    opener: str,
    openers: dict[str, list[int]],
) -> list[int]:
    distance = q - matched
    tolerance = max(8, round(0.1 * distance))
    candidates = [
        k for k in openers[opener]
        if k < q and k != matched and abs((q - k) - distance) <= tolerance
    ]
    if len(candidates) >= 8:
        return candidates
    log_bin = math.floor(math.log2(max(1, distance)))
    candidates = [
        k for k in openers[opener]
        if k < q
        and k != matched
        and math.floor(math.log2(max(1, q - k))) == log_bin
    ]
    return candidates if len(candidates) >= 8 else []


def permutation_layer(
    layer: int,
    store: RowStore,
    pairs: list[dict[str, Any]],
    openers: dict[str, list[int]],
    permutations: int,
) -> dict[str, Any]:
    observed_values: list[float] = []
    null_sets: list[np.ndarray] = []
    pair_records: list[dict[str, Any]] = []
    for pair in pairs:
        q = int(pair["token_pos"])
        matched = int(pair["opener_token_pos"])
        controls = candidate_controls(q, matched, pair["opener"], openers)
        if not controls:
            continue
        keys = [matched, *controls]
        values = store.head_mean_at(q, keys)
        safe = np.maximum(values, FP16_MIN_POSITIVE)
        log_ratio = float(np.log(safe[0]) - np.log(np.median(safe[1:])))
        observed_values.append(log_ratio)
        null_sets.append(safe)
        pair_records.append(
            {
                "q": q,
                "matched": matched,
                "bracket": pair["bracket"],
                "distance": q - matched,
                "controls": len(controls),
                "matched_attention": float(values[0]),
                "control_median_attention": float(np.median(values[1:])),
                "log_ratio": log_ratio,
            }
        )
    if not observed_values:
        raise RuntimeError(f"no eligible bracket pairs at L{layer}")
    observed = float(np.median(observed_values))
    seed_hex = hashlib.sha256(f"{REGISTRATION_COMMIT}|LF5|bracket|{layer}".encode()).hexdigest()
    seed = int(seed_hex[:16], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        ratios = np.empty(len(null_sets), dtype=np.float64)
        for index, values in enumerate(null_sets):
            chosen = int(rng.integers(len(values)))
            baseline = np.median(np.delete(values, chosen))
            ratios[index] = np.log(values[chosen]) - np.log(baseline)
        null[iteration] = np.median(ratios)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
    return {
        "layer": layer,
        "eligible_pairs": len(observed_values),
        "observed_median_log_ratio": observed,
        "observed_ratio": float(np.exp(observed)),
        "permutations": permutations,
        "seed": seed,
        "p_one_sided": p_value,
        "null_quantiles": {
            "q025": float(np.quantile(null, 0.025)),
            "q50": float(np.quantile(null, 0.5)),
            "q975": float(np.quantile(null, 0.975)),
        },
        "pairs": pair_records,
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


def brackets_command(args: argparse.Namespace) -> None:
    require_gates(
        args.replay_gate,
        args.cpu_gate,
        args.input_root,
        args.amendment_a5,
    )
    dump_manifest = json.loads((args.dump / "manifest.json").read_text(encoding="utf-8"))
    if not dump_manifest.get("complete"):
        raise RuntimeError("LF5 row dump is incomplete")
    loci = json.loads(args.loci.read_text(encoding="utf-8"))
    pairs = loci["texts"]["02_code"]["loci"]["bracket_pairs"]
    openers: dict[str, list[int]] = {"(": [], "[": [], "{": []}
    for pair in pairs:
        openers[pair["opener"]].append(int(pair["opener_token_pos"]))
    openers = {kind: sorted(set(values)) for kind, values in openers.items()}

    results = []
    for layer in GLOBAL_LAYERS:
        store = RowStore(args.dump / f"L{layer:02d}_02_code")
        result = permutation_layer(layer, store, pairs, openers, args.permutations)
        results.append(result)
        print(
            f"L{layer:02d}: pairs={result['eligible_pairs']} "
            f"ratio={result['observed_ratio']:.4g} p={result['p_one_sided']:.6g}",
            flush=True,
        )
    adjusted = holm_adjust([x["p_one_sided"] for x in results])
    for result, p_adjusted in zip(results, adjusted):
        result["p_holm"] = p_adjusted
        result["positive_and_significant"] = bool(
            result["observed_median_log_ratio"] > 0 and p_adjusted < 0.05
        )
    significant = sum(x["positive_and_significant"] for x in results)
    report = {
        "schema_version": 1,
        "kind": "round5_lf5_bracket_matching",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "amendment_a5_commit": AMENDMENT_A5_COMMIT,
        "production_backend": "replay",
        "registered_cpu_gate_passed": False,
        "source_sha256": sha256_file(Path(__file__)),
        "loci_sha256": sha256_file(args.loci),
        "dump_manifest_sha256": sha256_file(args.dump / "manifest.json"),
        "zero_probability_convention": {
            "kind": "left-censor at smallest positive FP16 subnormal",
            "epsilon": FP16_MIN_POSITIVE,
        },
        "control_rule": "same opener type; +/-max(8, round(0.1*d)); then same floor(log2(d)) bin; require >=8",
        "decision_rule": "at least 3/11 global layers positive with Holm one-sided p<0.05",
        "results": results,
        "positive_significant_layers": significant,
        "prediction_passed": significant >= 3,
    }
    atomic_json(args.report, report)
    print(f"wrote {args.report}; prediction_passed={report['prediction_passed']}")


def self_test() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    if not np.allclose(adjusted, [0.03, 0.06, 0.06]):
        raise AssertionError(adjusted)
    openers = {"(": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "[": [], "{": []}
    controls = candidate_controls(120, 20, "(", openers)
    if 20 in controls:
        raise AssertionError("matched opener leaked into controls")
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")

    dump = sub.add_parser("dump")
    dump.add_argument("--layers", default="all")
    dump.add_argument("--query-batch", type=int, default=8)
    dump.add_argument("--loci", type=Path, default=LOCI)
    dump.add_argument("--input-root", type=Path, default=DEFAULT_INPUTS)
    dump.add_argument("--replay-gate", type=Path, default=DEFAULT_REPLAY_GATE)
    dump.add_argument("--cpu-gate", type=Path, default=DEFAULT_CPU_GATE)
    dump.add_argument("--amendment-a5", type=Path, default=DEFAULT_A5)
    dump.add_argument("--out", type=Path, default=DEFAULT_DUMP)

    brackets = sub.add_parser("brackets")
    brackets.add_argument("--permutations", type=int, default=10000)
    brackets.add_argument("--loci", type=Path, default=LOCI)
    brackets.add_argument("--input-root", type=Path, default=DEFAULT_INPUTS)
    brackets.add_argument("--replay-gate", type=Path, default=DEFAULT_REPLAY_GATE)
    brackets.add_argument("--cpu-gate", type=Path, default=DEFAULT_CPU_GATE)
    brackets.add_argument("--amendment-a5", type=Path, default=DEFAULT_A5)
    brackets.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    brackets.add_argument("--report", type=Path, default=DEFAULT_BRACKET_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    elif args.command == "dump":
        dump_command(args)
    elif args.command == "brackets":
        brackets_command(args)
    else:
        raise SystemExit("choose dump or brackets, or pass --self-test")


if __name__ == "__main__":
    main()
