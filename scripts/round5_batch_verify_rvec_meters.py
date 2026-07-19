"""Second-analyst independent re-derivation of LF3, LF8, LF9 headline numbers.

Reads ONLY the certified widened-capture dumps. No producer imports.
Algorithms deliberately re-implemented: correlations via explicit centered
dot products, skewness via two-pass central moments, Holm by hand.

Verified against the first analyst's registered outcomes (d3cca8f):
LF3 tail stat/search p exact; LF8 min cosine, family stat, and 47-candidate
chirality census exact; LF9 signs at all seven registered layers (paired
per-head magnitudes reconciled exactly in round5_batch_verify_states_r5b.py's
companion run; the marginal aggregation used here flips no sign).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
GLOBALS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}
SEQ, NC = 8192, 1024

# ---- LF3 setup: 127 complete 64-token blocks after tokens 0..63
starts = np.arange(64, SEQ, 64)            # 127 starts
mid = starts + 31.5                        # block midpoints
REGS = {"mid": mid.astype(np.float64), "log1p": np.log1p(mid.astype(np.float64))}

def corr_all_shifts(B, reg):
    """Pearson r of each column of B (127 x NC) against every circular shift of reg.
    Returns (n_shifts x NC)."""
    Bc = B - B.mean(0)
    Bn = np.sqrt((Bc * Bc).sum(0))
    Bn[Bn == 0] = np.inf
    n = len(reg)
    shifts = np.stack([np.roll(reg, s) for s in range(n)])   # n x 127
    Sc = shifts - shifts.mean(1, keepdims=True)
    Sn = np.sqrt((Sc * Sc).sum(1))
    return (Sc @ Bc) / (Sn[:, None] * Bn[None, :])

# accumulators
lf3 = {t: {"obs": np.zeros((66, 2)), "null": np.zeros((66, 2, 127))} for t in ("01_prose_en", "06_random")}
mean_r = np.zeros((66, 6, NC))
skew = np.zeros((66, 6, NC))

for L in range(66):
    for ti, t in enumerate(TEXTS):
        r = np.load(CAP / "replay" / f"rvec_L{L:02d}_{t}.npy").astype(np.float32).reshape(SEQ, NC)
        x = r.astype(np.float64)
        mu = x.mean(0)
        mean_r[L, ti] = mu
        xc = x - mu
        m2 = (xc * xc).mean(0)
        m3 = (xc * xc * xc).mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            skew[L, ti] = np.where(m2 > 0, m3 / np.power(m2, 1.5, where=m2 > 0), 0.0)
        if t in lf3:
            B = x[64:].reshape(127, 64, NC).mean(1)
            for ri, reg in enumerate(REGS.values()):
                C = np.abs(corr_all_shifts(B, reg))       # 127 x NC
                per_shift = C.max(1)                       # max over coords
                lf3[t]["obs"][L, ri] = per_shift[0]
                lf3[t]["null"][L, ri] = per_shift
    print(f"L{L:02d} done", flush=True)

out = {}
# ---- LF3 verdicts
for t, d in lf3.items():
    search = d["null"].max(axis=(0, 1))                    # max over layers+regressors, per shift
    obs = search[0]
    p = (1 + int((search[1:] >= obs).sum())) / 127
    Lbest, ribest = np.unravel_index(d["obs"].argmax(), d["obs"].shape)
    out[f"lf3_{t}"] = {"max_abs_r": float(obs), "argmax_layer": int(Lbest),
                       "regressor": list(REGS)[ribest], "search_p": p}

# ---- LF8: stability cosines, family split, chirality
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

min_cos, min_where = 2.0, None
for L in range(66):
    for i in range(6):
        for j in range(i + 1, 6):
            c = cos(mean_r[L, i], mean_r[L, j])
            if c < min_cos:
                min_cos, min_where = c, (L, TEXTS[i], TEXTS[j])
out["lf8_min_cosine"] = {"value": min_cos, "where": min_where}

six_mean = mean_r.mean(1)                                  # 66 x NC
six_mean = six_mean / np.linalg.norm(six_mean, axis=1, keepdims=True)
C66 = six_mean @ six_mean.T
gl = np.array(sorted(GLOBALS)); lo = np.array([l for l in range(66) if l not in GLOBALS])
def fam_stat(gidx):
    lidx = np.array([l for l in range(66) if l not in set(gidx.tolist())])
    gg = C66[np.ix_(gidx, gidx)][np.triu_indices(len(gidx), 1)].mean()
    ll = C66[np.ix_(lidx, lidx)][np.triu_indices(len(lidx), 1)].mean()
    glm = C66[np.ix_(gidx, lidx)].mean()
    return 0.5 * (gg + ll) - glm
obs_fam = fam_stat(gl)
rng = np.random.Generator(np.random.PCG64(20260718))
null_fam = np.array([fam_stat(np.array([6 * b + rng.integers(6) for b in range(11)])) for _ in range(10000)])
out["lf8_family"] = {"stat": float(obs_fam),
                     "p_one_sided": float((1 + (null_fam >= obs_fam).sum()) / 10001)}

cand = 0
cand_list = []
for L in range(66):
    sk = skew[L]                                           # 6 x NC
    tstat, praw = stats.ttest_1samp(sk, 0.0, axis=0)
    order = np.argsort(praw)
    holm = np.minimum.accumulate((praw[order] * (NC - np.arange(NC)))[::-1])[::-1]
    adj = np.empty(NC); adj[order] = np.minimum(holm, 1.0)
    med = np.abs(np.median(sk, axis=0))
    hits = np.where((adj < 0.05) & (med > 0.25))[0]
    cand += len(hits)
    for h in hits[:3]:
        cand_list.append((L, int(h), float(np.median(sk[:, h]))))
out["lf8_chirality_candidates"] = cand
out["lf8_chirality_sample"] = cand_list[:8]

# ---- LF9 far-share effects at registered layers
def far_share(mass):                                       # mass: 64 x dmax
    p = mass / mass.sum(1, keepdims=True)
    return p[:, 257:].sum(1)                               # d > 256

lf9 = {}
for L in [5, 23, 29, 35, 41, 47, 65]:
    per_cond = {}
    for cond in ("with", "without"):
        vals = []
        for t in TEXTS:
            z = np.load(CAP / "meters" / f"layer{L:02d}_{t}_s{SEQ}.npz")
            vals.append(np.median(far_share(z[f"mass_{cond}"])))
        per_cond[cond] = float(np.median(vals))
    lf9[f"L{L}"] = {"with": per_cond["with"], "without": per_cond["without"],
                    "effect": per_cond["with"] - per_cond["without"]}
out["lf9"] = lf9

print(json.dumps(out, indent=1, default=str))
dest = ROOT / "analysis" / "round5" / "dump_science_batch" / "verification_rvec_meters.json"
dest.write_text(json.dumps(out, indent=1, default=str))
