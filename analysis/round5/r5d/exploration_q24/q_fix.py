"""Fix batch: Q19 on normalized scale, Q21 chirality overlap, Q23 class overlays."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"R:\inkling")
ARMS = ROOT / "dumps" / "round5" / "r5d" / "arms"
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
out = {}

# ---------- Q19 corrected: r_proj reads the NORMALIZED attention input
wr29 = np.load(ROOT / "weights" / "layer29_wr_du.npy").astype(np.float64)
bits = np.load(CAP / "normalized" / "attn_in_L29_05_needles.npy", mmap_mode="r")
h_norm = (np.asarray(bits[:, 4786]).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
col = wr29[:, 4786]
r29 = np.load(CAP / "replay" / "rvec_L29_05_needles.npy").astype(np.float32).reshape(8192, 1024).astype(np.float64)
rbar = r29.mean(0)
contrib_mean = np.abs(h_norm).mean() * np.linalg.norm(col)
# token-level: how much of r variance does 4786 alone explain?
pred = h_norm[:, None] * col[None, :]
resid_var = float(((r29 - r29.mean(0)) - (pred - pred.mean(0))).var(axis=0).sum())
total_var = float(r29.var(axis=0).sum())
out["q19_normalized"] = {
    "mean_abs_h4786_normalized": float(np.abs(h_norm).mean()),
    "contribution_norm_vs_rbar_norm": float(contrib_mean / np.linalg.norm(rbar)),
    "r_variance_explained_by_4786_alone": float(1 - resid_var / total_var),
    "corr_h4786_with_r_along_col": float(np.corrcoef(h_norm, r29 @ (col / np.linalg.norm(col)))[0, 1]),
}

# ---------- Q21 chirality overlap with clock directions
lf8 = json.loads((ROOT / "analysis" / "round5" / "lf8" / "lf8.json").read_text(encoding="utf-8"))
cand = lf8["chirality"]["candidate_locations"]
clock = np.load(ROOT / "analysis" / "round5" / "r5d_clock" / "clock_freeze.npz")
rows = {}
for L in (53, 59, 65):
    coords = [c["coordinate"] for c in cand if c["layer"] == L]
    G = clock[f"G_L{L}"].astype(np.float64)
    rows[f"L{L}"] = {
        "n_chirality_coords": len(coords),
        "G_energy_on_them": float((G[coords] ** 2).sum()) if coords else 0.0,
        "expected_if_uniform": len(coords) / 1024,
    }
layer_hist = {}
for c in cand:
    layer_hist[c["layer"]] = layer_hist.get(c["layer"], 0) + 1
out["q21_chirality"] = {"per_layer_counts": dict(sorted(layer_hist.items())), "clock_overlap": rows}

# ---------- Q23: class overlays, prose + needles at L29 (+ L65, clock joint for contrast)
loci = json.loads((ROOT / "analysis" / "round5" / "loci.json").read_text(encoding="utf-8"))
texts_loci = loci["texts"]


def class_table(arm, text):
    with np.load(ARMS / arm / "tokens" / f"{text}.npz") as z:
        d = np.asarray(z["delta_nll"], dtype=np.float64)
    rows = {"__all__": {"n": len(d), "mean_dnll": float(d.mean())}}
    entry = texts_loci.get(text, {})
    positions_by_class = entry.get("positions", entry.get("classes", entry))
    if isinstance(positions_by_class, dict):
        for name, val in positions_by_class.items():
            if isinstance(val, list) and val and isinstance(val[0], int):
                # delta index i predicts target i+1 from query token i; class token as QUERY
                idx = [p for p in val if 0 <= p < len(d)]
                if len(idx) >= 8:
                    rows[name] = {"n": len(idx), "mean_dnll_as_query": float(d[idx].mean())}
    return rows


for arm, text in (("bias_off_L29", "01_prose_en"), ("bias_off_L29", "02_code"),
                  ("clock_freeze_L53_L59", "01_prose_en"), ("bias_off_L65", "01_prose_en")):
    out[f"q23_{arm}_{text}"] = class_table(arm, text)

print(json.dumps(out, indent=1, default=str))
Path(__file__).with_suffix(".out.json").write_text(json.dumps(out, indent=1, default=str))
