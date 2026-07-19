"""LF1 — architectural numerology: single-distance pips at powers of two.

Registered prediction (ROUND5_LEFTFIELD_SPEC.md, blind): no pips survive
multiple-comparison control — 128 flagged as the most likely surprise (tau
constant 128,000 = 128 * 1000). Peek disclosure in the spec: 16-token band
STEPS at d in {256..768} were examined earlier (null except the d=512 echo);
single-distance pips were not.

Method frozen before outcomes:
  - Curve object: mode-0 of each layer's proj bank (first right singular
    vector of proj [16, extent], sign-fixed so the near-field mean is
    positive) — the same object Round 4's kernel fits used. Secondary,
    reported not decided: max pip across the 16 raw bank rows.
  - pip(d) = |c(d) - median(window)| / MAD(window), window = c[d-8 .. d+8]
    excluding d itself, for d in [16, extent-17].
  - Tested distances: powers of two within [16, extent-17]: globals
    {16, 32, 64, 128, 256, 512}, locals {16, 32, 64, 128, 256}.
  - Empirical p per (layer, power) = fraction of ALL eligible d with
    pip >= pip(power). Holm across all (layer, power) tests within the
    global family (primary) and the local family (secondary).
  - d=512 on globals is reported but carries the peek flag (the echo step).

Dump-first: `dump` writes every pip series to dumps/round5/lf1/ before
`analyze` reads only the dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DUMP_DIR = ROOT / "dumps" / "round5" / "lf1"
OUT_DIR = ROOT / "analysis" / "round5" / "lf1"
GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
HALF_WINDOW = 8
MARGIN = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mode0(proj: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(proj.astype(np.float64), full_matrices=False)
    curve = vt[0]
    if np.mean(curve[:32]) < 0:
        curve = -curve
    return curve


def pip_series(curve: np.ndarray) -> np.ndarray:
    n = len(curve)
    out = np.full(n, np.nan)
    for d in range(MARGIN, n - MARGIN):
        window = np.concatenate([curve[d - HALF_WINDOW:d], curve[d + 1:d + HALF_WINDOW + 1]])
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        out[d] = abs(curve[d] - med) / max(mad, 1e-12)
    return out


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def dump_command() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    series = {}
    row_series = {}
    sources = {}
    for layer in range(66):
        path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        proj = np.load(path)
        sources[path.name] = sha256_file(path)
        series[f"L{layer:02d}"] = pip_series(mode0(proj))
        row_series[f"L{layer:02d}"] = np.stack(
            [pip_series(proj[k].astype(np.float64)) for k in range(16)])
        print(f"L{layer:02d} done", flush=True)
    np.savez(DUMP_DIR / "lf1_pips.npz",
             **{f"mode0_{k}": v for k, v in series.items()},
             **{f"rows_{k}": v for k, v in row_series.items()})
    manifest = {"kind": "round5_lf1_pip_dump", "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "method": {"half_window": HALF_WINDOW, "margin": MARGIN,
                           "curve": "mode0 of proj; rows secondary"},
                "source_sha256": sha256_file(Path(__file__)),
                "input_sha256": sources,
                "dump_sha256": sha256_file(DUMP_DIR / "lf1_pips.npz")}
    (DUMP_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("dump complete")


def analyze_command() -> None:
    manifest = json.loads((DUMP_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dump_sha256"] != sha256_file(DUMP_DIR / "lf1_pips.npz"):
        raise RuntimeError("dump hash mismatch")
    z = np.load(DUMP_DIR / "lf1_pips.npz", allow_pickle=False)

    def family(layers: list[int]) -> list[dict]:
        tests = []
        for layer in layers:
            pips = z[f"mode0_L{layer:02d}"]
            valid = ~np.isnan(pips)
            extent = len(pips)
            powers = [p for p in (16, 32, 64, 128, 256, 512, 1024)
                      if MARGIN <= p < extent - MARGIN]
            for p in powers:
                stat = float(pips[p])
                pval = float(np.mean(pips[valid] >= stat))
                tests.append({"layer": layer, "d": p, "pip": stat, "p": pval,
                              "peeked_band_step": bool(p == 512 and layer in GLOBALS)})
        for test, adj in zip(tests, holm([t["p"] for t in tests])):
            test["p_holm"] = adj
            test["significant"] = bool(adj < 0.05)
        return tests

    global_tests = family(GLOBALS)
    local_tests = family([l for l in range(66) if l not in GLOBALS])
    survivors = [t for t in global_tests if t["significant"]]
    survivors_unpeeked = [t for t in survivors if not t["peeked_band_step"]]
    report = {
        "kind": "round5_lf1_pips", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dump_sha256": manifest["dump_sha256"],
        "source_sha256": sha256_file(Path(__file__)),
        "global_tests": global_tests,
        "local_tests": local_tests,
        "n_local_tests": len(local_tests),
        "notes": ["d=16 was not among the spec's example powers {32,64,128,"
                  "256,...}; it is an unregistered addition to the family "
                  "(disclosed; its inclusion only makes Holm stricter and no "
                  "verdict depends on it)"],
        "global_survivors": survivors,
        "prediction": {
            "registered": "no pips survive control; 128 most likely surprise",
            "unpeeked_survivors": survivors_unpeeked,
            "passed": bool(len(survivors_unpeeked) == 0),
            "d128_result": [t for t in global_tests if t["d"] == 128],
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lf1_pips.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("global survivors:", [(t["layer"], t["d"], round(t["p_holm"], 4),
                                 "PEEKED-512" if t["peeked_band_step"] else "")
                                for t in survivors])
    print("local survivors:", [(t["layer"], t["d"]) for t in local_tests if t["significant"]])
    print("prediction passed:", report["prediction"]["passed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["dump", "analyze"])
    args = parser.parse_args()
    dump_command() if args.stage == "dump" else analyze_command()


if __name__ == "__main__":
    main()
