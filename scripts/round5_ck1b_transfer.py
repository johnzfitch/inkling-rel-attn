"""CK1b clock-transfer test, exactly as registered in
ROUND5_R5D_CLOCK_TRANSFER_PREREG.md (commit 3f5a467).

Reads only the sealed clock-arm pre-intervention r-vectors and the sealed
clock_freeze artifact. Verdict cells: {03_templated, 05_needles} x {L53, L59};
prose/code/multilingual reported descriptively.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "dumps" / "round5" / "r5d" / "arms"
CLOCK = ROOT / "analysis" / "round5" / "r5d_clock" / "clock_freeze.npz"
OUT = ROOT / "analysis" / "round5" / "r5d" / "ck1b_transfer.json"

SEQ, HEADS, RPERHEAD = 8192, 64, 16
RFLAT = HEADS * RPERHEAD
VERDICT_TEXTS = ["03_templated", "05_needles"]
DESCRIPTIVE_TEXTS = ["01_prose_en", "02_code", "04_multilingual"]
LAYERS = [53, 59]

starts = np.arange(64, SEQ, 64)
x = np.log1p(starts + 31.5)
xc = x - x.mean()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stat(blocks: np.ndarray, layer: int) -> float:
    proj = np.load(ROOT / "weights" / f"layer{layer:02d}_rel_logits_proj.npy").astype(np.float64)
    kernels = blocks.reshape(127, HEADS, RPERHEAD) @ proj
    meanc = kernels.mean(0)
    gain = (kernels * meanc).sum(2) / (meanc * meanc).sum(1)
    gc = gain - gain.mean(0)
    with np.errstate(invalid="ignore"):
        corr = (xc @ gc) / (np.linalg.norm(xc) * np.linalg.norm(gc, axis=0))
    return float(np.median(np.abs(np.nan_to_num(corr, nan=0.0))))


def cell(layer: int, text: str) -> dict[str, float]:
    arm = f"clock_freeze_L{layer}"
    pre_path = DUMP / arm / "clock" / f"rvec_pre_L{layer:02d}_{text}.npy"
    pre = np.load(pre_path, allow_pickle=False).astype(np.float64).reshape(SEQ, RFLAT)
    with np.load(CLOCK, allow_pickle=False) as freeze:
        direction = freeze[f"G_L{layer}"].astype(np.float64)
        anchor = freeze[f"rbar_L{layer}"].astype(np.float64)
    frozen = pre - ((pre - anchor) @ direction)[:, None] * direction
    baseline_blocks = pre[64:].reshape(127, 64, RFLAT).mean(1)
    frozen_blocks = frozen[64:].reshape(127, 64, RFLAT).mean(1)
    return {
        "baseline": stat(baseline_blocks, layer),
        "transferred_freeze": stat(frozen_blocks, layer),
        "input_sha256": sha256_file(pre_path),
    }


cells = {f"L{layer}_{text}": cell(layer, text) for layer in LAYERS for text in VERDICT_TEXTS}
descriptive = {f"L{layer}_{text}": cell(layer, text) for layer in LAYERS for text in DESCRIPTIVE_TEXTS}
passes = {
    key: bool(values["baseline"] > 0.50 and values["transferred_freeze"] < 0.20)
    for key, values in cells.items()
}
report = {
    "kind": "round5_r5d_ck1b_clock_transfer",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "registration": "ROUND5_R5D_CLOCK_TRANSFER_PREREG.md @ 3f5a467",
    "source_sha256": sha256_file(Path(__file__)),
    "clock_freeze_sha256": sha256_file(CLOCK),
    "verdict_cells": cells,
    "cell_passes": passes,
    "passed": bool(all(passes.values())),
    "descriptive_cells": descriptive,
    "tie_policy": "ties fail; baseline <= 0.50 fails the cell",
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: report[k] for k in ("verdict_cells", "cell_passes", "passed", "descriptive_cells")}, indent=1))
