"""Token-level anatomy for registered questions 4-16, 22-24 (exploratory)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"R:\inkling")
ARMS = ROOT / "dumps" / "round5" / "r5d" / "arms"
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
out = {}

def tok(arm, text):
    with np.load(ARMS / arm / "tokens" / f"{text}.npz", allow_pickle=False) as z:
        return {k: np.asarray(z[k], dtype=np.float64) for k in
                ("delta_nll", "delta_target_logit", "delta_log_normalizer", "nll")}

def alltok(arm, key="delta_nll"):
    return {t: tok(arm, t)[key] for t in TEXTS}

# ---------- Q5 sparsity of bias_off_L29
d29 = alltok("bias_off_L29")
pooled = np.concatenate([d29[t] for t in TEXTS])
total = pooled.sum()
srt = np.sort(pooled)[::-1]
n = len(srt)
out["q5"] = {
    "total_delta_nll_sum": total,
    "share_top_0.1pct": float(srt[: max(1, n // 1000)].sum() / total),
    "share_top_1pct": float(srt[: n // 100].sum() / total),
    "share_top_10pct": float(srt[: n // 10].sum() / total),
    "share_positive_tokens": float((pooled > 0).mean()),
    "negative_mass_share": float(pooled[pooled < 0].sum() / total),
}

# ---------- Q6 / Q24 decomposition at L29 (and L65 for contrast)
for arm in ("bias_off_L29", "bias_off_L65", "far_off_L29"):
    rows = {}
    for t in TEXTS:
        v = tok(arm, t)
        rows[t] = {
            "mean_dnll": float(v["delta_nll"].mean()),
            "mean_d_target_logit": float(v["delta_target_logit"].mean()),
            "mean_d_log_normalizer": float(v["delta_log_normalizer"].mean()),
        }
    pooled_t = np.concatenate([tok(arm, t)["delta_target_logit"] for t in TEXTS])
    pooled_z = np.concatenate([tok(arm, t)["delta_log_normalizer"] for t in TEXTS])
    out[f"q6_{arm}"] = {
        "per_text": rows,
        "pooled_mean_d_target": float(pooled_t.mean()),
        "pooled_mean_d_logZ": float(pooled_z.mean()),
    }

# ---------- Q7 near vs bias tokenwise at L29
near = np.concatenate([alltok("near_off_L29")[t] for t in TEXTS])
corr = float(np.corrcoef(near, pooled)[0, 1])
k = n // 100
top_b = set(np.argsort(pooled)[::-1][:k].tolist())
top_n = set(np.argsort(near)[::-1][:k].tolist())
out["q7"] = {
    "tokenwise_corr": corr,
    "top1pct_jaccard": float(len(top_b & top_n) / len(top_b | top_n)),
    "near_over_bias_slope": float((near @ pooled) / (pooled @ pooled)),
    "pooled_ratio": float(near.mean() / pooled.mean()),
}

# ---------- Q9 beneficiaries (negative dnll) in far_off/L65 arms
bene = {}
for arm in ("far_off_L29", "far_off_L65", "bias_off_L65", "far_off_L11"):
    rows = {}
    for t in TEXTS:
        v = tok(arm, t)
        d = v["delta_nll"]
        neg = d < -0.05
        rows[t] = {
            "n_neg_gt_0.05": int(neg.sum()),
            "neg_mass": float(d[neg].sum()),
            "pos_mass": float(d[d > 0.05].sum()),
            "mean_baseline_nll_of_beneficiaries": float(v["nll"][neg].mean()) if neg.any() else None,
            "mean_baseline_nll_all": float(v["nll"].mean()),
        }
    bene[arm] = rows
out["q9"] = bene

# ---------- Q12 wall-heal cancellation check
rows = {}
for t in TEXTS:
    d = tok("wall_heal_global", t)["delta_nll"]
    rows[t] = {
        "mean": float(d.mean()),
        "mean_abs": float(np.abs(d).mean()),
        "n_abs_gt_0.1": int((np.abs(d) > 0.1).sum()),
        "max": float(d.max()),
        "min": float(d.min()),
    }
out["q12"] = rows

# ---------- Q13 clock redundancy tokenwise
c53 = np.concatenate([alltok("clock_freeze_L53")[t] for t in TEXTS])
c59 = np.concatenate([alltok("clock_freeze_L59")[t] for t in TEXTS])
cj = np.concatenate([alltok("clock_freeze_L53_L59")[t] for t in TEXTS])
out["q13"] = {
    "corr_L53_L59": float(np.corrcoef(c53, c59)[0, 1]),
    "corr_joint_vs_sum": float(np.corrcoef(cj, c53 + c59)[0, 1]),
    "regression_joint_on_sum": float((cj @ (c53 + c59)) / ((c53 + c59) @ (c53 + c59))),
    "mean_abs_joint": float(np.abs(cj).mean()),
    "mean_abs_sum": float(np.abs(c53 + c59).mean()),
}

# ---------- Q14 where does clock freezing matter (joint arm)
rows = {}
for t in TEXTS:
    d = alltok("clock_freeze_L53_L59")[t]
    b = tok("clock_freeze_L53_L59", t)["nll"]
    rows[t] = {
        "mean": float(d.mean()),
        "mean_abs": float(np.abs(d).mean()),
        "corr_with_baseline_nll": float(np.corrcoef(d, b)[0, 1]),
        "top_position": int(np.argmax(np.abs(d))) + 1,
        "top_value": float(d[np.argmax(np.abs(d))]),
        "first256_mean": float(d[:256].mean()),
        "last256_mean": float(d[-256:].mean()),
    }
out["q14"] = rows

# ---------- needle table: Q10, Q11, Q15, Q16 + clock/wall
needle_rows = {}
sidecar = json.loads((ROOT / "corpus" / "05_needles.sidecar.json").read_text(encoding="utf-8"))
dist = np.asarray([e["distance"] for e in sidecar["entities"]], dtype=np.float64)
arm_list = [f"bias_off_L{L:02d}" for L in (5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65)] + [
    "near_off_L29", "far_off_L29", "rising_heads_off_L00_L04",
    "negative_seam_heads_off_L11", "wall_heal_global",
    "clock_freeze_L53", "clock_freeze_L59", "clock_freeze_L53_L59",
]
for arm in arm_list:
    with np.load(ARMS / arm / "needle" / "05_needles.npz", allow_pickle=True) as z:
        d = np.asarray(z["delta_nll"], dtype=np.float64)
        sides = np.asarray([str(s) for s in z["side_of_seam"]])
    below = sides == "below"
    needle_rows[arm] = {
        "mean_all": float(d.mean()),
        "mean_below_seam": float(d[below].mean()),
        "mean_above_seam": float(d[~below].mean()),
        "n_below": int(below.sum()),
        "corr_with_distance": float(np.corrcoef(d, dist)[0, 1]) if d.std() > 0 else 0.0,
        "max": float(d.max()),
    }
out["needle_table"] = needle_rows

# ordinary-needle-token contrast for Q10
d = alltok("bias_off_L29")["05_needles"]
q = np.asarray([int(e["token_positions"][1]) for e in sidecar["entities"]])
mask = np.zeros(8191, dtype=bool)
mask[q - 1 + 1 - 1] = True  # delta index for target position q+1 is q-... see runner: index q
mask = np.zeros(8191, dtype=bool)
mask[q] = False
mask[q] = True
out["q10"] = {
    "needle_query_mean": float(d[q].mean()),
    "ordinary_mean": float(d[~mask].mean()),
    "needle_over_ordinary": float(d[q].mean() / max(abs(d[~mask].mean()), 1e-9)),
}

# ---------- Q22 surprisal/aperture -> vulnerability at L29
rows = {}
for t in TEXTS:
    v = tok("bias_off_L29", t)
    ap = np.load(ROOT / "dumps" / "round5" / "lf4_a6_corrected" / f"aperture_L29_{t}.npz")["aperture_full"]
    # delta index i corresponds to target position i+1 predicted FROM token i -> aperture of query token i
    a = ap[:-1]
    d = v["delta_nll"]
    b = v["nll"]
    hi = a >= np.quantile(a, 0.9)
    lo = a <= np.quantile(a, 0.1)
    rows[t] = {
        "corr_dnll_baseline_nll": float(np.corrcoef(d, b)[0, 1]),
        "corr_dnll_aperture": float(np.corrcoef(d, a)[0, 1]),
        "mean_dnll_top_decile_aperture": float(d[hi].mean()),
        "mean_dnll_bottom_decile_aperture": float(d[lo].mean()),
    }
out["q22_bias_L29"] = rows

print(json.dumps(out, indent=1, default=str))
Path(__file__).with_suffix(".out.json").write_text(json.dumps(out, indent=1, default=str))
