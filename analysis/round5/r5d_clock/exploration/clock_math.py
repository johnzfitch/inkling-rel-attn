"""Mathematical structure of the LF3 clock (exploratory, unregistered).

Q1 functional form: log fit quality vs linear/sqrt/power on block means.
Q2 rank: PCA of tail block-mean matrix — is the clock one direction?
Q3 separability: clock direction reshaped 64x16 — outer-product share.
Q4 kernel action: per-block realized kernel gain vs log p (entropy-compensation sign test).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"R:\inkling")
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
SEQ = 8192
starts = np.arange(64, SEQ, 64)
mid = (starts + 31.5).astype(float)
lg = np.log1p(mid)

def blockmeans(L, t="06_random"):
    r = np.load(CAP / "replay" / f"rvec_L{L:02d}_{t}.npy").astype(np.float32).reshape(SEQ, 1024).astype(float)
    return r[64:].reshape(127, 64, 1024).mean(1)

def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - resid.var() / y.var()

out = {}

# ---------- Q1: functional form on the strongest coordinate (L59 c650) and
# on the clock PC1 scores (below), random arm
B = blockmeans(59)
y = B[:, 650]
forms = {
    "linear": mid,
    "log1p": lg,
    "sqrt": np.sqrt(mid),
    "p^0.25": mid ** 0.25,
    "1/p": 1.0 / mid,
}
out["q1_c650_r2"] = {k: round(r2(y, v), 4) for k, v in forms.items()}
# two-parameter log(p + p0) scan
best = max(((p0, r2(y, np.log(mid + p0))) for p0 in [1, 8, 32, 64, 128, 256, 512, 1024]),
           key=lambda kv: kv[1])
out["q1_c650_log_p0_scan"] = {"best_p0": best[0], "r2": round(best[1], 4)}

# ---------- Q2: rank of the drift. Column-center B, PCA via SVD.
Bc = B - B.mean(0)
U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
share = S ** 2 / (S ** 2).sum()
pc1 = U[:, 0] * S[0]
out["q2_L59"] = {
    "pc1_var_share": round(float(share[0]), 4),
    "pc2_var_share": round(float(share[1]), 4),
    "corr_pc1_logp": round(float(np.corrcoef(pc1, lg)[0, 1]), 4),
    "corr_pc2_logp": round(float(np.corrcoef(U[:, 1] * S[1], lg)[0, 1]), 4),
}

# clock direction: regression beta of each coordinate on log p (unit-normed)
lgc = lg - lg.mean()
beta = (lgc @ Bc) / (lgc @ lgc)             # 1024 slope vector
v_clock = beta / np.linalg.norm(beta)
out["q2_cos_clockdir_vs_pc1"] = round(float(abs(v_clock @ Vt[0])), 4)

# ---------- Q3: separability of the clock direction across heads x dims
M = v_clock.reshape(64, 16)
Um, Sm, Vm = np.linalg.svd(M, full_matrices=False)
out["q3_L59"] = {
    "outer_product_share": round(float(Sm[0] ** 2 / (Sm ** 2).sum()), 4),
    "dim_loadings_rank1": [round(float(x), 3) for x in Vm[0]],
    "head_gain_stats": {
        "frac_same_sign": round(float(max((Um[:, 0] > 0).mean(), (Um[:, 0] < 0).mean())), 3),
        "top5_heads": [int(i) for i in np.argsort(-np.abs(Um[:, 0]))[:5]],
        "participation_ratio": round(float((Um[:, 0] ** 2).sum() ** 2 / (Um[:, 0] ** 4).sum()), 1),
    },
}

# ---------- Q2/Q3 across all global layers (random arm)
per_layer = {}
clock_dirs = {}
for L in GLOBALS:
    Bl = blockmeans(L)
    Blc = Bl - Bl.mean(0)
    Ul, Sl, Vtl = np.linalg.svd(Blc, full_matrices=False)
    sh = float(Sl[0] ** 2 / (Sl ** 2).sum())
    c1 = float(np.corrcoef(Ul[:, 0] * Sl[0], lg)[0, 1])
    b = (lgc @ Blc) / (lgc @ lgc)
    vb = b / np.linalg.norm(b)
    clock_dirs[L] = vb
    Msep = np.linalg.svd(vb.reshape(64, 16), compute_uv=True)
    per_layer[f"L{L}"] = {
        "pc1_share": round(sh, 3),
        "corr_pc1_logp": round(abs(c1), 3),
        "sep_share": round(float(Msep[1][0] ** 2 / (Msep[1] ** 2).sum()), 3),
        "dim_peak": int(np.abs(Msep[2][0]).argmax()),
    }
out["per_global_layer"] = per_layer

# dim-marginal loading agreement across layers: |cos| of 16-dim marginals
dims = {L: np.linalg.norm(clock_dirs[L].reshape(64, 16), axis=0) for L in GLOBALS}
ref = dims[59] / np.linalg.norm(dims[59])
out["dim_marginal_cos_vs_L59"] = {
    f"L{L}": round(float(ref @ (dims[L] / np.linalg.norm(dims[L]))), 3) for L in GLOBALS}

# ---------- Q4: kernel action at L59. Realized kernel per block per head:
# curve_b = B_b (64x16) @ proj (16x1024). Gain against the mean kernel.
proj = np.asarray(np.load(ROOT / "weights" / "layer59_rel_logits_proj.npy"), dtype=np.float64)
curves = B.reshape(127, 64, 16) @ proj                 # 127 x 64 x 1024
meanc = curves.mean(0)                                  # 64 x 1024
gain = (curves * meanc).sum(2) / (meanc * meanc).sum(1) # 127 x 64
gcorr = np.array([np.corrcoef(gain[:, h], lg)[0, 1] for h in range(64)])
out["q4_L59"] = {
    "heads_gain_UP_with_logp": int((gcorr > 0.5).sum()),
    "heads_gain_DOWN": int((gcorr < -0.5).sum()),
    "median_gain_corr": round(float(np.median(gcorr)), 3),
    "gain_range_median_head": None,
}
h_med = int(np.argsort(gcorr)[len(gcorr) // 2])
out["q4_L59"]["gain_range_median_head"] = [round(float(gain[:, h_med].min()), 3),
                                           round(float(gain[:, h_med].max()), 3)]
# near-field bias drift: change of realized kernel at d in [1,64] between
# first and last decile of blocks, per head, sign census
near = curves[:, :, 1:65].mean(2)                       # 127 x 64
early = near[:12].mean(0); late = near[-12:].mean(0)
out["q4_nearfield"] = {
    "heads_nearfield_bias_UP": int((late > early).sum()),
    "median_delta": round(float(np.median(late - early)), 4),
    "mean_kernel_near_positive_heads": int((meanc[:, 1:65].mean(1) > 0).sum()),
}
# whole-kernel drift decomposition: is the drift parallel to the mean kernel
# (pure gain) or a different shape? For each head: cos(drift_dir, mean kernel)
drift = np.stack([(lgc @ (curves[:, h, :] - curves[:, h, :].mean(0))) / (lgc @ lgc)
                  for h in range(64)])                  # 64 x 1024
cosdrift = (drift * meanc).sum(1) / (np.linalg.norm(drift, axis=1) * np.linalg.norm(meanc, axis=1))
out["q4_drift_vs_meankernel_cos"] = {
    "median": round(float(np.median(cosdrift)), 3),
    "frac_abs_gt_0.8": round(float((np.abs(cosdrift) > 0.8).mean()), 3),
}

print(json.dumps(out, indent=1))
Path(__file__).with_suffix(".out.json").write_text(json.dumps(out, indent=1))
