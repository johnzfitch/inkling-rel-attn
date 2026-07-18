"""Build and verify the registered R5-D wall-healing table continuation.

The method is frozen in ROUND5_R5D_EXECUTION_AMENDMENT.md.  This script is
weight-only: it never loads model activations or any causal-ablation output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
REGISTRATION = ROOT / "registrations" / "ROUND5_R5D_EXECUTION_AMENDMENT.md"
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "r5d_wall_tail"
DEFAULT_REPORT = ROOT / "analysis" / "round5" / "r5d" / "tail_fit.json"
GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
FIT_START = 512
ENDPOINT = 1023
MAX_DISTANCE = 8191
RATE_MIN = 1.0 / (2.0 * 1024.0)
RATE_MAX = 0.1
AMPLITUDE_MULTIPLIER = 10.0
REGISTRATION_COMMIT = "2a48f5b3ce476f9b30be68712be04f6fa63bd8a0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def require_committed_source() -> dict[str, str]:
    head = git_output("rev-parse", "HEAD")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", REGISTRATION_COMMIT, head], cwd=ROOT
    ).returncode:
        raise RuntimeError("R5-D registration commit is not an ancestor of HEAD")
    records = {}
    for path in (Path(__file__), REGISTRATION):
        relative = path.relative_to(ROOT).as_posix()
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT).returncode:
            raise RuntimeError(f"uncommitted registered source: {relative}")
        payload = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        blob_hash = hashlib.sha256(payload).hexdigest()
        if blob_hash != sha256_file(path):
            raise RuntimeError(f"working file differs from HEAD blob: {relative}")
        records[relative] = blob_hash
    records["git_head"] = head
    return records


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def rates(parameters: np.ndarray) -> tuple[float, float]:
    slow = float(parameters[1])
    fast = slow + float(parameters[2]) * (RATE_MAX - slow)
    return slow, fast


def curve(
    delta: np.ndarray, parameters: np.ndarray, endpoint_value: float
) -> np.ndarray:
    amplitude, _, _ = map(float, parameters)
    slow, fast = rates(parameters)
    exponent_slow = np.clip(-slow * delta, -700.0, 700.0)
    exponent_fast = np.clip(-fast * delta, -700.0, 700.0)
    return amplitude * np.exp(exponent_slow) + (endpoint_value - amplitude) * np.exp(
        exponent_fast
    )


def fit_row(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (1024,) or not np.isfinite(values).all():
        raise ValueError("wall-tail row must be finite with length 1024")
    observed = values[FIT_START : ENDPOINT + 1]
    delta = np.arange(FIT_START, ENDPOINT + 1, dtype=np.float64) - ENDPOINT
    endpoint_value = float(values[ENDPOINT])
    row_max = float(np.max(np.abs(values)))
    amplitude_bound = AMPLITUDE_MULTIPLIER * row_max + 1e-8
    scale = max(float(np.sqrt(np.mean(observed * observed))), 1e-8)

    amplitude_seeds = [0.0, endpoint_value, 0.5 * endpoint_value]
    if row_max:
        amplitude_seeds.extend([row_max, -row_max])
    slow_seeds = [0.001, 0.005, 0.02]
    fraction_seeds = [0.2, 0.8]
    best: tuple[float, np.ndarray] | None = None
    attempts = 0
    for amplitude in amplitude_seeds:
        for slow in slow_seeds:
            for fraction in fraction_seeds:
                initial = np.asarray(
                    [
                        np.clip(amplitude, -0.95 * amplitude_bound, 0.95 * amplitude_bound),
                        np.clip(slow, RATE_MIN * 1.01, RATE_MAX * 0.99),
                        fraction,
                    ],
                    dtype=np.float64,
                )
                result = least_squares(
                    lambda parameters: (curve(delta, parameters, endpoint_value) - observed)
                    / scale,
                    initial,
                    bounds=(
                        np.asarray([-amplitude_bound, RATE_MIN, 0.0]),
                        np.asarray([amplitude_bound, RATE_MAX, 1.0]),
                    ),
                    max_nfev=4000,
                    ftol=1e-11,
                    xtol=1e-11,
                    gtol=1e-11,
                )
                attempts += 1
                if not result.success or not np.isfinite(result.x).all():
                    continue
                residual = curve(delta, result.x, endpoint_value) - observed
                sse = float(np.dot(residual, residual))
                if best is None or sse < best[0]:
                    best = (sse, result.x.copy())
    if best is None:
        raise RuntimeError("all deterministic wall-tail fit starts failed")

    sse, parameters = best
    slow, fast = rates(parameters)
    total = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1.0 - sse / total) if total > 1e-20 else float(sse <= 1e-20)
    forward_delta = np.arange(0, MAX_DISTANCE - ENDPOINT + 1, dtype=np.float64)
    continuation = curve(forward_delta, parameters, endpoint_value)
    endpoint_error = float(abs(continuation[0] - endpoint_value))
    tail = continuation[1:]
    if tail.shape != (MAX_DISTANCE - ENDPOINT,):
        raise AssertionError(tail.shape)
    record = {
        "endpoint_value": endpoint_value,
        "amplitude_slow": float(parameters[0]),
        "amplitude_fast": float(endpoint_value - parameters[0]),
        "rate_slow": slow,
        "rate_fast": fast,
        "rate_fraction": float(parameters[2]),
        "fit_sse": sse,
        "fit_r2": r2,
        "fit_attempts": attempts,
        "observed_abs_max": row_max,
        "tail_abs_max": float(np.max(np.abs(tail))),
        "endpoint_error_u0": endpoint_error,
        "first_new_value_d1024": float(tail[0]),
        "last_value_d8191": float(tail[-1]),
        "amplitude_bound": amplitude_bound,
        "slow_rate_at_lower_bound": bool(slow <= RATE_MIN * 1.0001),
        "slow_rate_at_upper_bound": bool(slow >= RATE_MAX * 0.9999),
        "fast_rate_at_upper_bound": bool(fast >= RATE_MAX * 0.9999),
    }
    return tail.astype(np.float32), record


def build(dump: Path, report_path: Path) -> None:
    if dump.exists() and any(dump.iterdir()):
        raise FileExistsError(f"refusing to overwrite wall-tail dump: {dump}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite wall-tail report: {report_path}")
    source = require_committed_source()
    arrays: dict[str, np.ndarray] = {}
    fits: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    failures: list[str] = []
    for layer in GLOBALS:
        path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        table = np.load(path, allow_pickle=False)
        if table.shape != (16, 1024) or table.dtype != np.float32 or not np.isfinite(table).all():
            raise RuntimeError(f"invalid global projection table: {path}")
        input_hashes[path.name] = sha256_file(path)
        rows = []
        tail_rows = []
        for row_index, values in enumerate(table):
            tail, record = fit_row(values)
            record["row"] = row_index
            rows.append(record)
            tail_rows.append(tail)
            if record["endpoint_error_u0"] > 1e-6:
                failures.append(f"L{layer:02d}/row{row_index}: endpoint continuity")
            limit = AMPLITUDE_MULTIPLIER * record["observed_abs_max"] + 1e-6
            if record["tail_abs_max"] > limit:
                failures.append(f"L{layer:02d}/row{row_index}: tail overshoot")
        layer_tail = np.stack(tail_rows).astype(np.float32)
        if layer_tail.shape != (16, MAX_DISTANCE - ENDPOINT) or not np.isfinite(layer_tail).all():
            failures.append(f"L{layer:02d}: invalid tail array")
        arrays[f"L{layer:02d}"] = layer_tail
        fits[f"L{layer:02d}"] = rows
        print(
            f"L{layer:02d}: median R2={np.median([row['fit_r2'] for row in rows]):.4f} "
            f"max|tail|/max|row|={max(row['tail_abs_max'] / (row['observed_abs_max'] + 1e-30) for row in rows):.3g}",
            flush=True,
        )
    if failures:
        raise RuntimeError(f"wall-tail registered gate failed: {failures}")
    dump.mkdir(parents=True, exist_ok=True)
    tail_path = dump / "tail_tables.npz"
    atomic_npz(tail_path, **arrays)
    report = {
        "schema_version": 1,
        "kind": "round5_r5d_registered_wall_tail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "registration_commit": REGISTRATION_COMMIT,
        "registration_sha256": sha256_file(REGISTRATION),
        "source": source,
        "input_sha256": input_hashes,
        "tail_dump": {
            "path": tail_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(tail_path),
            "keys": sorted(arrays),
            "shape_per_layer": [16, MAX_DISTANCE - ENDPOINT],
            "dtype": "float32",
        },
        "method": {
            "fit_window": [FIT_START, ENDPOINT],
            "first_new_distance": ENDPOINT + 1,
            "last_distance": MAX_DISTANCE,
            "rate_min": RATE_MIN,
            "rate_max": RATE_MAX,
            "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
            "model": "continuity-constrained signed two-exponential",
        },
        "fit_count": sum(len(rows) for rows in fits.values()),
        "fits": fits,
        "gate": {
            "all_finite": True,
            "max_endpoint_error_u0": max(
                row["endpoint_error_u0"] for rows in fits.values() for row in rows
            ),
            "max_tail_to_observed_abs_ratio": max(
                row["tail_abs_max"] / (row["observed_abs_max"] + 1e-30)
                for rows in fits.values()
                for row in rows
            ),
        },
    }
    atomic_json(report_path, report)
    print(f"sealed {tail_path}")
    print(f"wrote {report_path}")


def verify(dump: Path, report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tail_path = dump / "tail_tables.npz"
    errors = []
    if (
        report.get("kind") != "round5_r5d_registered_wall_tail"
        or not report.get("passed")
        or report.get("fit_count") != 176
        or report.get("registration_sha256") != sha256_file(REGISTRATION)
        or report.get("tail_dump", {}).get("sha256") != sha256_file(tail_path)
    ):
        errors.append("report contract or hash mismatch")
    with np.load(tail_path, allow_pickle=False) as payload:
        if sorted(payload.files) != [f"L{layer:02d}" for layer in GLOBALS]:
            errors.append("tail key inventory mismatch")
        for key in payload.files:
            values = payload[key]
            if values.shape != (16, 7168) or values.dtype != np.float32 or not np.isfinite(values).all():
                errors.append(f"invalid tail payload: {key}")
    for name, expected in report.get("input_sha256", {}).items():
        if sha256_file(WEIGHTS / name) != expected:
            errors.append(f"input hash mismatch: {name}")
    if errors:
        raise RuntimeError(errors)
    print("round5_r5d_tail verification passed")


def self_test() -> None:
    d = np.arange(1024, dtype=np.float64)
    endpoint = 0.4
    truth = 0.9 * np.exp(-0.002 * (d - ENDPOINT)) + (endpoint - 0.9) * np.exp(
        -0.011 * (d - ENDPOINT)
    )
    tail, record = fit_row(truth.astype(np.float32))
    if record["fit_r2"] < 0.999999 or record["endpoint_error_u0"] > 1e-8:
        raise AssertionError(record)
    if tail.shape != (7168,) or not np.isfinite(tail).all():
        raise AssertionError(tail.shape)
    zero_tail, zero_record = fit_row(np.zeros(1024, dtype=np.float32))
    if np.max(np.abs(zero_tail)) > 1e-7 or zero_record["endpoint_error_u0"] > 1e-8:
        raise AssertionError(zero_record)
    print("round5_r5d_tail self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    for command in ("build", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
        child.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "build":
        build(args.dump, args.report)
    else:
        verify(args.dump, args.report)


if __name__ == "__main__":
    main()
