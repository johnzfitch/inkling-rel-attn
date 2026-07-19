"""LF2 sensitivity artifact (audit follow-up): block-residual surrogates.

The registered LF2 null resamples log-residuals IID; the audit measured
residual lag-1 autocorrelation up to 0.915, which makes the IID null
optimistic. This artifact repeats the surrogate test with circular
block-resampled residuals (block sizes 16 and 32), 2,000 surrogates each,
same detector, Holm across the 11 global layers per block size. The
registered IID verdict stands as registered; this file reports robustness.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import round5_lf2_knees as lf2

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "dumps" / "round5" / "lf2"
OUT = ROOT / "analysis" / "round5" / "lf2" / "lf2_block_sensitivity.json"
SURROGATES = 2000
SEED = 0x1F2B


def block_resample(residuals: np.ndarray, block: int, rng) -> np.ndarray:
    n = len(residuals)
    parts = []
    while sum(len(p) for p in parts) < n:
        start = int(rng.integers(0, n))
        parts.append(np.take(residuals, np.arange(start, start + block), mode="wrap"))
    return np.concatenate(parts)[:n]


def main() -> None:
    manifest = json.loads((DUMP / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(DUMP / "lf2_knees.npz", allow_pickle=False)
    d = z["d"]
    results = {}
    for block in (16, 32):
        rng = np.random.default_rng(SEED + block)
        rows = []
        for layer in lf2.GLOBALS:
            y = z[f"y_L{layer:02d}"]
            smooth = z[f"smooth_L{layer:02d}"]
            residuals = y - smooth
            observed = manifest["observed"][f"L{layer:02d}"]
            null = np.empty(SURROGATES)
            for i in range(SURROGATES):
                surrogate = smooth + block_resample(residuals, block, rng)
                null[i], _ = lf2.detect(surrogate, d)
            p = float((1 + np.sum(null >= observed["improvement"])) / (SURROGATES + 1))
            rows.append({"layer": layer, "p": p,
                         "improvement": observed["improvement"],
                         "breakpoint": observed["breakpoint"]})
        for row, adj in zip(rows, lf2.holm([r["p"] for r in rows])):
            row["p_holm"] = adj
            row["significant"] = bool(adj < 0.05)
        results[f"block_{block}"] = rows
        lag1 = {f"L{l:02d}": float(np.corrcoef(
            (z[f"y_L{l:02d}"] - z[f"smooth_L{l:02d}"])[:-1],
            (z[f"y_L{l:02d}"] - z[f"smooth_L{l:02d}"])[1:])[0, 1])
            for l in lf2.GLOBALS}
    report = {"kind": "round5_lf2_block_sensitivity", "schema_version": 1,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dump_sha256": manifest["dump_sha256"],
              "source_sha256": hashlib.sha256(
                  Path(__file__).read_bytes()).hexdigest(),
              "residual_lag1_autocorr": lag1,
              "surrogates": SURROGATES, "results": results}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for block, rows in results.items():
        sig = [r["layer"] for r in rows if r["significant"]]
        print(block, "significant:", sig)


if __name__ == "__main__":
    main()
