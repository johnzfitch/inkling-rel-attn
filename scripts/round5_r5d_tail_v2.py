"""Build Amendment-A's R5-D wall tail from the frozen LF6 mode-0 envelope."""

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


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
LF6_FITS = ROOT / "dumps" / "round5" / "lf6" / "fits_manifest.json"
LF6_PUBLIC = ROOT / "analysis" / "round5" / "lf6" / "lf6_mi_mimicry.json"
AMENDMENT = ROOT / "registrations" / "ROUND5_R5D_TAIL_AMENDMENT_A.md"
DEFAULT_DUMP = ROOT / "dumps" / "round5" / "r5d_wall_tail_v2"
DEFAULT_REPORT = ROOT / "analysis" / "round5" / "r5d" / "tail_fit_v2.json"
GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
ENDPOINT = 1023
MAX_DISTANCE = 8191
TAIL_LENGTH = MAX_DISTANCE - ENDPOINT
AMENDMENT_COMMIT = "c1554f6463a0d36dde6abaee033006ba2fe9767c"


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
        ["git", "merge-base", "--is-ancestor", AMENDMENT_COMMIT, head], cwd=ROOT
    ).returncode:
        raise RuntimeError("tail Amendment A is not an ancestor of HEAD")
    records = {"git_head": head}
    for path in (Path(__file__), AMENDMENT):
        relative = path.relative_to(ROOT).as_posix()
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT).returncode:
            raise RuntimeError(f"uncommitted source: {relative}")
        payload = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        expected = hashlib.sha256(payload).hexdigest()
        if expected != sha256_file(path):
            raise RuntimeError(f"source differs from HEAD: {relative}")
        records[relative] = expected
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


def normalized_envelope(parameters: list[float]) -> np.ndarray:
    if len(parameters) != 4:
        raise ValueError(f"expected LF6 exp2 [a1,r1,a2,r2], got {parameters}")
    a1, r1, a2, r2 = map(float, parameters)
    if not all(np.isfinite([a1, r1, a2, r2])) or min(a1, r1, a2, r2) <= 0:
        raise ValueError(f"non-positive/non-finite LF6 envelope: {parameters}")
    distance = ENDPOINT + np.arange(TAIL_LENGTH + 1, dtype=np.float64)
    raw = a1 * np.exp(-r1 * distance) + a2 * np.exp(-r2 * distance)
    envelope = raw / raw[0]
    if (
        not np.isfinite(envelope).all()
        or np.any(envelope <= 0)
        or abs(float(envelope[0]) - 1.0) > 1e-12
        or np.any(np.diff(envelope) > 1e-14)
        or not envelope[-1] < envelope[0]
    ):
        raise RuntimeError("LF6 normalized envelope gate failed")
    return envelope


def load_bound_fits() -> tuple[dict[str, Any], dict[str, Any]]:
    public = json.loads(LF6_PUBLIC.read_text(encoding="utf-8"))
    fits = json.loads(LF6_FITS.read_text(encoding="utf-8"))
    if (
        public.get("kind") != "round5_lf6_mi_mimicry"
        or public.get("fits_manifest_sha256") != sha256_file(LF6_FITS)
        or fits.get("kind") != "round5_lf6_fit_dump"
        or fits.get("fit_lo") != 32
    ):
        raise RuntimeError("LF6 public/private fit binding failed")
    return public, fits


def build(dump: Path, report_path: Path) -> None:
    if dump.exists() and any(dump.iterdir()):
        raise FileExistsError(f"refusing to overwrite amended tail dump: {dump}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite amended tail report: {report_path}")
    source = require_committed_source()
    public, fits = load_bound_fits()
    arrays: dict[str, np.ndarray] = {}
    layers: dict[str, Any] = {}
    for layer in GLOBALS:
        key = f"L{layer:02d}"
        weight_path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        weight_hash = sha256_file(weight_path)
        if fits["input_sha256"].get(weight_path.name) != weight_hash:
            raise RuntimeError(f"LF6 input hash mismatch: {weight_path.name}")
        table = np.load(weight_path, allow_pickle=False)
        if table.shape != (16, 1024) or table.dtype != np.float32 or not np.isfinite(table).all():
            raise RuntimeError(f"invalid projection table: {weight_path}")
        parameters = fits["fits"][key]["exp2"]["params"]
        envelope = normalized_envelope(parameters)
        tail = table[:, ENDPOINT, None].astype(np.float64) * envelope[None, 1:]
        if tail.shape != (16, TAIL_LENGTH) or not np.isfinite(tail).all():
            raise RuntimeError(f"invalid amended tail: {key}")
        endpoint_abs = np.abs(table[:, ENDPOINT]).astype(np.float64)
        tail_abs_max = np.max(np.abs(tail), axis=1)
        if np.any(tail_abs_max > endpoint_abs + 1e-12):
            raise RuntimeError(f"amended tail exceeds learned endpoint: {key}")
        arrays[key] = tail.astype(np.float32)
        layers[key] = {
            "exp2_parameters": list(map(float, parameters)),
            "projection_sha256": weight_hash,
            "envelope_u0": float(envelope[0]),
            "envelope_u1": float(envelope[1]),
            "envelope_u7168": float(envelope[-1]),
            "monotonic_nonincreasing": bool(np.all(np.diff(envelope) <= 1e-14)),
            "max_endpoint_abs": float(endpoint_abs.max()),
            "max_tail_abs": float(np.max(np.abs(tail))),
        }
        print(
            f"{key}: g(1)={envelope[1]:.9f} g(7168)={envelope[-1]:.6g}",
            flush=True,
        )
    dump.mkdir(parents=True, exist_ok=True)
    tail_path = dump / "tail_tables.npz"
    atomic_npz(tail_path, **arrays)
    report = {
        "schema_version": 1,
        "kind": "round5_r5d_amendment_a_wall_tail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "amendment_commit": AMENDMENT_COMMIT,
        "amendment_sha256": sha256_file(AMENDMENT),
        "source": source,
        "lf6_public_sha256": sha256_file(LF6_PUBLIC),
        "lf6_fits_manifest_sha256": sha256_file(LF6_FITS),
        "lf6_public_binding_verified": public["fits_manifest_sha256"],
        "tail_dump": {
            "path": tail_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(tail_path),
            "keys": sorted(arrays),
            "shape_per_layer": [16, TAIL_LENGTH],
            "dtype": "float32",
        },
        "layers": layers,
        "gate": {
            "fit_count_reused": 11,
            "all_parameters_positive_finite": True,
            "all_envelopes_positive_monotonic": True,
            "all_tails_bounded_by_endpoint": True,
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
        report.get("kind") != "round5_r5d_amendment_a_wall_tail"
        or not report.get("passed")
        or report.get("amendment_sha256") != sha256_file(AMENDMENT)
        or report.get("tail_dump", {}).get("sha256") != sha256_file(tail_path)
    ):
        errors.append("report contract/hash mismatch")
    _, fits = load_bound_fits()
    with np.load(tail_path, allow_pickle=False) as payload:
        if sorted(payload.files) != [f"L{layer:02d}" for layer in GLOBALS]:
            errors.append("tail key mismatch")
        for layer in GLOBALS:
            key = f"L{layer:02d}"
            values = payload[key]
            if values.shape != (16, TAIL_LENGTH) or values.dtype != np.float32 or not np.isfinite(values).all():
                errors.append(f"invalid tail: {key}")
            envelope = normalized_envelope(fits["fits"][key]["exp2"]["params"])
            table = np.load(WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy")
            expected = table[:, ENDPOINT, None].astype(np.float64) * envelope[None, 1:]
            if not np.array_equal(values, expected.astype(np.float32)):
                errors.append(f"tail does not reproduce: {key}")
    if errors:
        raise RuntimeError(errors)
    print("round5_r5d_tail_v2 verification passed")


def self_test() -> None:
    envelope = normalized_envelope([0.3, 0.001, 0.7, 0.01])
    if envelope.shape != (TAIL_LENGTH + 1,) or envelope[0] != 1.0:
        raise AssertionError(envelope[:3])
    if not np.all(np.diff(envelope) < 0):
        raise AssertionError("synthetic envelope is not decreasing")
    print("round5_r5d_tail_v2 self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    for name in ("build", "verify"):
        child = commands.add_parser(name)
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
