"""Second-analyst re-derivation: R5-C carrier + lifecycle, R5-B depth profile.

Independent code paths: lifecycle coverage comes from the massive-census
artifacts (a different artifact family than the states the producer used),
with a direct-state cross-check; the carrier share is recomputed from the
full BF16 state and the frozen communal basis. "Component 1" follows the
basis artifact's own one-indexed naming (array index 0) — confirmed by the
frozen live_share field, whose column-0 median is the registered 0.641155.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
GLOBALS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}
SEQ = 8192
out = {}

def bf16(path):
    bits = np.load(path)
    return (bits.astype(np.uint32) << 16).view(np.float32)

# ---- R5-C carrier share at claimed max cell L20 / 05_needles
bases = np.load(ROOT / "analysis" / "subspace_anatomy" / "common_bases_top4.npz")
def carrier_vec(layer):
    v = bases["basis"][layer, 0]
    return (v / np.linalg.norm(v)).astype(np.float64)

def carrier_share(layer, text):
    h = bf16(CAP / "states" / f"hidden_L{layer:02d}_{text}.npy").astype(np.float64)
    v = carrier_vec(layer)
    s = h @ v
    total = h.var(axis=0, ddof=0).sum()
    return float(s.var(ddof=0) / total)

try:
    out["carrier_L20_needles"] = carrier_share(20, "05_needles")
    out["carrier_spots"] = {f"L{L}_{t}": carrier_share(L, t)
                            for L, t in [(20, "01_prose_en"), (35, "05_needles"), (5, "02_code")]}
except Exception as exc:
    out["carrier_error"] = repr(exc)

# live_share median (frozen basis artifact; component 1 = column 0, see docstring)
ls = np.asarray(bases["live_share"], dtype=np.float64)
out["live_share_median_comp1"] = float(np.median(ls[:, 0]))

# ---- R5-C lifecycle: coverage via massive census (independent artifact path)
def coverage_census(layer, text, channel):
    z = np.load(CAP / "massive" / f"massive_L{layer:02d}_{text}.npz")
    return float((z["channel"] == channel).sum() / SEQ)

cov = {ch: np.zeros((9, 6)) for ch in (4786, 3290)}
for li, L in enumerate(range(22, 31)):
    for ti, t in enumerate(TEXTS):
        for ch in cov:
            cov[ch][li, ti] = coverage_census(L, t, ch)

def onset(c):  # c: 9 x 6 coverage for layers 22..30
    for l in range(23, 29):
        i = l - 22
        sustained = ((c[i] > c[i - 1]) & (c[i + 1] > c[i]) & (c[i + 2] > c[i + 1])).sum()
        if sustained >= 4:
            return l
    return None

out["lifecycle"] = {
    "onset_4786": onset(cov[4786]),
    "onset_3290": onset(cov[3290]),
    "cov4786_L25_28_prose": cov[4786][3:7, 0].tolist(),
    "cov3290_max": float(cov[3290].max()),
}
# cross-check census vs raw state for one cell
h = bf16(CAP / "states" / "hidden_L26_01_prose_en.npy")
direct = float((np.abs(h[:, 4786]) > 30000).mean())
out["lifecycle_crosscheck_L26_prose_4786"] = {
    "census": coverage_census(26, "01_prose_en", 4786), "direct_state": direct}
del h

# ---- R5-B depth profile
WEIGHTS = ROOT / "weights"
disp = np.zeros(66)
cent = {t: [] for t in TEXTS}
for L in range(66):
    proj = np.asarray(np.load(WEIGHTS / f"layer{L:02d}_rel_logits_proj.npy"), dtype=np.float64)
    mr = np.stack([
        np.load(CAP / "replay" / f"rvec_L{L:02d}_{t}.npy").astype(np.float32)
        .mean(0).astype(np.float64) for t in TEXTS])            # 6 x 64 x 16
    curves = mr @ proj                                          # 6 x 64 x extent
    dists = []
    for i in range(6):
        for j in range(i + 1, 6):
            a, b = curves[i], curves[j]
            na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
            dists.append(np.linalg.norm(a - b, axis=1) / (0.5 * (na + nb)))
    disp[L] = float(np.median(np.concatenate(dists)))
    meanc = curves.mean(0)
    nm = np.linalg.norm(meanc, axis=1)
    for i, t in enumerate(TEXTS):
        d = np.linalg.norm(curves[i] - meanc, axis=1) / (0.5 * (np.linalg.norm(curves[i], axis=1) + nm))
        cent[t].append(d)

gl = sorted(GLOBALS)
out["r5b"] = {
    "argmax_layer": int(disp.argmax()),
    "unique_max": bool((disp == disp.max()).sum() == 1),
    "median_L0_5": float(np.median(disp[0:6])),
    "median_L23_47": float(np.median(disp[23:48])),
    "L65": float(disp[65]),
    "L65_below_all_mid_globals": bool(all(disp[65] < disp[L] for L in gl if 23 <= L <= 47)),
    "centrality": {t: float(np.median(np.concatenate(v))) for t, v in cent.items()},
}
# LF9 paired-per-head aggregation (reconciles the registered magnitudes exactly)
lf9p = {}
for L in [5, 23, 29, 35, 41, 47, 65]:
    effs = []
    for t in TEXTS:
        z = np.load(CAP / "meters" / f"layer{L:02d}_{t}_s{SEQ}.npz")
        pw = z["mass_with"] / z["mass_with"].sum(1, keepdims=True)
        po = z["mass_without"] / z["mass_without"].sum(1, keepdims=True)
        effs.append(float(np.median(pw[:, 257:].sum(1) - po[:, 257:].sum(1))))
    lf9p[f"L{L}"] = float(np.median(effs))
out["lf9_paired_effects"] = lf9p

print(json.dumps(out, indent=1, default=str))
dest = ROOT / "analysis" / "round5" / "dump_science_batch" / "verification_states_r5b.json"
dest.write_text(json.dumps(out, indent=1, default=str))
