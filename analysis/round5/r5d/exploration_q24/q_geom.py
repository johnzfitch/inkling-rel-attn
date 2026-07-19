"""Geometry/meter batch: Q2, Q4-predictor, Q8, Q17, Q18, Q19, Q20, Q21, Q23."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"R:\inkling")
ARMS = ROOT / "dumps" / "round5" / "r5d" / "arms"
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
out = {}

def bands(mass):  # mass [64, dmax] -> shares
    p = mass / mass.sum(1, keepdims=True)
    return {
        "d_lt4": float(p[:, :4].sum(1).mean()),
        "d_4_128": float(p[:, 4:129].sum(1).mean()),
        "d_129_1023": float(p[:, 129:1024].sum(1).mean()),
        "d_ge_1024": float(p[:, 1024:].sum(1).mean()),
    }

# ---------- Q8 + Q4 predictor: L29 attention redistribution per text
per_text_cost = {"01_prose_en": 0.17647, "02_code": 0.22439, "03_templated": 0.02025,
                 "04_multilingual": 0.09264, "05_needles": 0.00117, "06_random": 0.20983}
redis = {}
mid_loss = []
costs = []
for t in TEXTS:
    with np.load(CAP / "meters" / f"layer29_{t}_s8192.npz") as z:
        base = bands(z["mass_with"])
    with np.load(ARMS / "bias_off_L29" / "meters" / f"L29_{t}.npz") as z:
        ablated = bands(z["mass_with"])
    redis[t] = {"baseline": base, "bias_off": ablated,
                "delta_mid_4_128": ablated["d_4_128"] - base["d_4_128"]}
    mid_loss.append(base["d_4_128"] - ablated["d_4_128"])
    costs.append(per_text_cost[t])
out["q8_redistribution"] = redis
out["q4_mid_band_loss_vs_cost_corr"] = float(np.corrcoef(mid_loss, costs)[0, 1])
out["q4_baseline_mid_share_vs_cost_corr"] = float(
    np.corrcoef([redis[t]["baseline"]["d_4_128"] for t in TEXTS], costs)[0, 1])

# ---------- Q17: dispersion vs causality depths (r5b profile)
r5b = np.load(ROOT / "analysis" / "round5" / "r5b" / "r5b_arrays.npz", allow_pickle=False)
keys = r5b.files
prof = None
for k in keys:
    if "dispersion" in k or "layer" in k:
        arr = r5b[k]
        if arr.ndim == 1 and len(arr) == 66:
            prof = arr; break
if prof is not None:
    out["q17"] = {"disp_L29": float(prof[29]), "disp_L41": float(prof[41]),
                  "disp_argmax": int(np.argmax(prof))}
else:
    out["q17"] = {"r5b_keys": keys}

# ---------- Q18: L53 vs L29 orientation stability
def min_cos(L):
    M = np.stack([np.load(CAP / "replay" / f"rvec_L{L:02d}_{t}.npy").astype(np.float32)
                  .reshape(8192, 1024).mean(0).astype(float) for t in TEXTS])
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = Mn @ Mn.T
    return float(C[np.triu_indices(6, 1)].min()), M

c53, _ = min_cos(53)
c29, M29 = min_cos(29)
out["q18"] = {"min_cos_L53": c53, "min_cos_L29": c29,
              "bias_off_cost_L53": 0.00226, "bias_off_cost_L29": 0.12079}

# ---------- Q19 / Q2 / Q20 / Q21: r-space anatomy at L29 (and 53/59)
basis = np.load(ROOT / "analysis" / "subspace_anatomy" / "common_bases_top4.npz")["basis"]
wr29 = np.load(ROOT / "weights" / "layer29_wr_du.npy").astype(np.float64)  # [1024, 6144]
P29 = np.load(ROOT / "weights" / "layer29_rel_logits_proj.npy").astype(np.float64)  # [16,1024ext]

# carrier image in r-space at L29
carrier29 = basis[29, 0].astype(np.float64)
img29 = wr29 @ carrier29                                   # 1024
rbar29 = M29.mean(0)                                       # grand mean r
img_unit = img29 / np.linalg.norm(img29)
coef = float(rbar29 @ img_unit)
rbar_perp = rbar29 - coef * img_unit
k_full = rbar29.reshape(64, 16) @ P29                      # mean kernel per head
k_perp = rbar_perp.reshape(64, 16) @ P29
k_img = (coef * img_unit).reshape(64, 16) @ P29
def flatness(k):  # per-head std over d / |mean over d|, median heads
    return float(np.median(np.std(k, axis=1) / np.maximum(np.abs(k.mean(axis=1)), 1e-9)))
out["q2_q20_L29"] = {
    "rbar_energy_share_along_carrier_image": float(coef**2 / (rbar29 @ rbar29)),
    "kernel_change_if_carrier_component_removed": float(
        np.linalg.norm(k_full - k_perp) / np.linalg.norm(k_full)),
    "carrier_image_kernel_flatness_med": flatness(k_img),
    "mean_kernel_flatness_med": flatness(k_full),
    "carrier_kernel_level_vs_full": float(np.abs(k_img).mean() / np.abs(k_full).mean()),
}
# token-level: variance of r explained by carrier image (needles at L29)
r29n = np.load(CAP / "replay" / "rvec_L29_05_needles.npy").astype(np.float32).reshape(8192, 1024).astype(np.float64)
rc = r29n - r29n.mean(0)
proj_var = float(((rc @ img_unit) ** 2).mean())
out["q2_q20_L29"]["token_r_variance_share_carrier_image"] = float(proj_var / (rc * rc).sum(1).mean())

# 4786 channel at L29
col = wr29[:, 4786]
col_norms = np.linalg.norm(wr29, axis=0)
h28 = np.load(CAP / "states" / "hidden_L28_05_needles.npy", mmap_mode="r")[:, 4786]
h28 = (np.asarray(h28).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
contrib = np.abs(h28).mean() * np.linalg.norm(col)
out["q19"] = {
    "wr_col4786_norm": float(np.linalg.norm(col)),
    "col_norm_percentile": float((col_norms <= np.linalg.norm(col)).mean()),
    "mean_abs_h4786_L28_needles": float(np.abs(h28).mean()),
    "contribution_norm_vs_rbar_norm": float(contrib / np.linalg.norm(rbar29)),
    "carrier29_loading_on_4786": float(basis[29, 0][4786]),
    "cos_col4786_vs_carrier_image": float((col @ img29) / (np.linalg.norm(col) * np.linalg.norm(img29))),
}

# Q21 overlaps at L53/L59: clock G vs chirality coords, carrier image, P modes
clock = np.load(ROOT / "analysis" / "round5" / "r5d_clock" / "clock_freeze.npz")
lf8 = json.loads((ROOT / "analysis" / "round5" / "lf8" / "lf8.json").read_text(encoding="utf-8"))
chir = lf8.get("chirality", {})
cand = chir.get("candidates", chir if isinstance(chir, list) else [])
overlap = {}
for L in (53, 59):
    G = clock[f"G_L{L}"].astype(np.float64)
    wr = np.load(ROOT / "weights" / f"layer{L:02d}_wr_du.npy").astype(np.float64)
    img = wr @ basis[L, 0].astype(np.float64)
    P = np.load(ROOT / "weights" / f"layer{L:02d}_rel_logits_proj.npy").astype(np.float64)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)      # U: 16 x 16 modes
    w = np.linalg.norm(G.reshape(64, 16), axis=0)          # dim marginal
    Gm = G.reshape(64, 16)
    mode_energy = [float((np.linalg.norm(Gm @ U[:, m]) ** 2)) for m in range(4)]
    coords = [c.get("coordinate", c.get("coord")) for c in cand
              if isinstance(c, dict) and c.get("layer") == L]
    chir_energy = float((G[coords] ** 2).sum()) if coords else None
    overlap[f"L{L}"] = {
        "cos_G_carrier_image": float((G @ img) / np.linalg.norm(img)),
        "G_energy_on_P_modes_0_3": [round(m, 4) for m in mode_energy],
        "n_chirality_coords_here": len(coords),
        "G_energy_on_chirality_coords": chir_energy,
    }
out["q21"] = overlap
out["q21_chirality_total_candidates"] = len(cand) if isinstance(cand, list) else "see lf8"

# ---------- Q23: class overlays on prose at L29 (loci.json)
try:
    loci = json.loads((ROOT / "analysis" / "round5" / "loci.json").read_text(encoding="utf-8"))
    with np.load(ARMS / "bias_off_L29" / "tokens" / "01_prose_en.npz") as z:
        d = np.asarray(z["delta_nll"], dtype=np.float64)
    classes = None
    for key in ("classes", "loci", "01_prose_en"):
        if key in loci:
            classes = loci[key]; break
    rows = {}
    if isinstance(classes, dict):
        source = classes.get("01_prose_en", classes)
        for name, positions in source.items():
            if isinstance(positions, list) and positions and isinstance(positions[0], int):
                idx = [p for p in positions if 0 < p < 8191]
                rows[name] = {"n": len(idx), "mean_dnll": float(d[[p for p in idx]].mean())}
    rows["__all__"] = {"n": 8191, "mean_dnll": float(d.mean())}
    out["q23_prose_L29"] = rows
except Exception as exc:
    out["q23_prose_L29"] = {"error": repr(exc)}

print(json.dumps(out, indent=1, default=str))
Path(__file__).with_suffix(".out.json").write_text(json.dumps(out, indent=1, default=str))
