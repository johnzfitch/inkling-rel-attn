"""R5-C (rvec side) — activation-space geometry of the r-fiber, corrected capture.

The R5-C registered predictions (ROUND5_LEFTFIELD_SPEC.md) are hidden-state
statements — carrier share of HIDDEN variance < 1%, intrinsic-dim/rotation
discontinuities of the HIDDEN stream — and remain OPEN until the widened pass
(D1) captures corrected hidden-state inputs. This script runs the rvec-side
geometry the ledger unblocked now, as DESCRIPTIVE output (no registered
thresholds evaluated):

  per (layer, arm) on the corrected rvec [8192, 64, 16] flattened to 1024-dim:
  - carrier share: ||mean_r||^2 / mean(||r_t||^2)  (the communal component's
    share of the fiber's second moment);
  - participation ratio of the centered covariance: (sum lambda)^2 / sum
    lambda^2 (intrinsic dimension of fluctuations);
  - cross-arm stability: cos(mean_r) between arms per layer;
  - adjacent-layer rotation: cos(mean_r) between consecutive layers per arm.

Note: LF8's registered fiber-orientation predictions are defined on the six
v1 texts and stay gated on the widened pass; nothing here evaluates them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "dumps" / "round5" / "corpus_v2_corrected_capture" / "rvec"
OUT = ROOT / "analysis" / "round5" / "r5c_rvec"
TEXTS = ["07_slack_human", "08_math_llm", "07b_slack_multi", "01_prose_en"]
GLOBALS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    means: dict[tuple[int, str], np.ndarray] = {}
    rows = {}
    for layer in range(66):
        for text in TEXTS:
            r = np.load(CAP / f"rvec_L{layer:02d}_{text}.npy").astype(np.float32)
            flat = r.reshape(8192, -1).astype(np.float64)
            mean = flat.mean(0)
            second = float(np.mean(np.sum(flat * flat, 1)))
            carrier = float(np.sum(mean * mean) / max(second, 1e-300))
            centered = flat - mean
            s = np.linalg.svd(centered, compute_uv=False)
            lam = s * s
            pr = float(lam.sum() ** 2 / np.sum(lam * lam))
            means[(layer, text)] = mean
            rows[f"L{layer:02d}_{text}"] = {
                "carrier_share_of_fiber": carrier,
                "participation_ratio": pr,
                "mean_norm": float(np.linalg.norm(mean)),
            }
        print(f"L{layer:02d} done", flush=True)

    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    cross_arm = {f"L{layer:02d}": {
        f"{a}|{b}": cos(means[(layer, a)], means[(layer, b)])
        for a, b in combinations(TEXTS, 2)} for layer in range(66)}
    adjacent = {text: {f"L{l:02d}->L{l+1:02d}": cos(means[(l, text)], means[(l + 1, text)])
                       for l in range(65)} for text in TEXTS}

    report = {
        "kind": "round5_r5c_rvec_side", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("DESCRIPTIVE - the registered R5-C predictions are "
                   "hidden-state statements, still gated on the widened pass"),
        "source_sha256": sha256_file(Path(__file__)),
        "capture_manifest_sha256": sha256_file(CAP.parent / "manifest.json"),
        "per_layer_arm": rows,
        "cross_arm_mean_cos": cross_arm,
        "adjacent_layer_mean_cos": adjacent,
    }
    (OUT / "r5c_rvec_side.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    carrier = np.array([[rows[f"L{l:02d}_{t}"]["carrier_share_of_fiber"]
                         for t in TEXTS] for l in range(66)])
    pr = np.array([[rows[f"L{l:02d}_{t}"]["participation_ratio"]
                    for t in TEXTS] for l in range(66)])
    stab = np.array([min(cross_arm[f"L{l:02d}"].values()) for l in range(66)])
    print("carrier share: median %.3f  range [%.3f, %.3f]"
          % (np.median(carrier), carrier.min(), carrier.max()))
    print("participation ratio: median %.1f  range [%.1f, %.1f]"
          % (np.median(pr), pr.min(), pr.max()))
    print("cross-arm min cos(mean_r): median %.4f  min %.4f (layer L%02d)"
          % (np.median(stab), stab.min(), int(np.argmin(stab))))


if __name__ == "__main__":
    main()
