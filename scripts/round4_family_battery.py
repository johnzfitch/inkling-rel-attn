"""
Round 4 family battery (F1-F10) -- which classical positional encoding did Inkling
learn? Pre-registered spec in ROUND4_SPEC.md, run with the review-#6 fitting fixes.

Dump-first: reads dumps/round3/{mode_curves (D1), perhead_svd (D0)} + weights/_meta.json.
No GPU, no network. Writes analysis/round4/family_battery.json + fingerprints.json,
and rewrites C4 truncated_fraction with the constant removed from extrapolation.

Corrections applied vs the literal spec table (all from code-review #6, documented
so the amendment is explicit):
  - all decay rates bounded > 0 (F3 c2>0, F4 p in (0,2], F9 rates>0, F10 delta>0);
  - F9 two rates made identifiable by the ordered (rate + positive gap) form;
  - all frequencies bounded to [0, pi] (F6/F7/F10) so rho and its 2pi/-rho aliases
    are not observationally equivalent;
  - F10 uses the phase-IDENTIFIABLE (a_cos, a_sin) form (5 params) rather than the
    spec's phase-anchored pure-cosine (4 params): anchoring phase at d=0 misspecifies
    genuine phase-shifted oscillations. Param count updated for BIC accordingly.
  - C4 extrapolation drops the fitted constant (gauge): past the extent the physical
    bias is pinned to 0, not to c, so only the decaying kernel is extrapolated.

Selection protocol (verbatim from spec): fit on EVEN distances in [8,extent), score
held-out R^2 on ODD distances, BIC on the full far-field. Winner = lowest BIC AND
held-out R^2 within 0.02 of the best held-out; else "contested" (report both).
Ties are ΔBIC<10.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit, nnls

D0 = r"R:\inkling\dumps\round3\perhead_svd"
D1 = r"R:\inkling\dumps\round3\mode_curves"
W = r"R:\inkling\weights"
OUT = r"R:\inkling\analysis\round4"
CUR = os.path.join(OUT, "curiosity")
GLOBAL = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
NEAR = 8


def safe_exp(x):
    return np.exp(np.clip(x, -700.0, 700.0))


# ---- family forms (nonlinear ones only; linear families fit in closed form) ----
def f_log(d, r, c2, c):      return -r * np.log1p(c2 * d) + c
def f_pow(d, r, p, c):       return -r * np.power(d, p) + c
def f_fourier(d, a1, a2, a3, w1, w2, w3, c):
    return a1 * np.cos(w1 * d) + a2 * np.cos(w2 * d) + a3 * np.cos(w3 * d) + c
def f_rope(d, a1, a2, a3, w0, g):
    return a1 * np.cos(w0 * d) + a2 * np.cos(w0 / g * d) + a3 * np.cos(w0 / g**2 * d)
def f_exp2(d, a_slow, r_slow, a_fast, r_gap, c):
    return a_slow * safe_exp(-r_slow * d) + a_fast * safe_exp(-(r_slow + r_gap) * d) + c
def f_dsin(d, a_cos, a_sin, delta, rho, c):
    return safe_exp(-delta * d) * (a_cos * np.cos(rho * d) + a_sin * np.sin(rho * d)) + c


PI = np.pi


def fit_family(fam, x, y, extent):
    """Return (predict_fn, k_params) fitting on (x,y). predict_fn(d)->b."""
    d, yv = x.astype(np.float64), y.astype(np.float64)
    if fam == "F1_const":
        c = yv.mean();  return (lambda dd: np.full_like(np.asarray(dd, float), c)), 1
    if fam == "F2_linear":
        A = np.column_stack([d, np.ones_like(d)]); m, c = np.linalg.lstsq(A, yv, rcond=None)[0]
        return (lambda dd: m * np.asarray(dd, float) + c), 2
    if fam == "F8_gaussian":
        A = np.column_stack([d**2, np.ones_like(d)]); a, c = np.linalg.lstsq(A, yv, rcond=None)[0]
        return (lambda dd: a * np.asarray(dd, float)**2 + c), 2
    if fam == "F5_bucket":
        knots = np.unique(np.round(np.logspace(np.log10(NEAR), np.log10(extent), 9)).astype(int))
        edges = knots
        vals = []
        for i in range(len(edges) - 1):
            mask = (d >= edges[i]) & (d < edges[i + 1])
            vals.append(yv[mask].mean() if mask.any() else 0.0)
        vals = np.array(vals)
        def pred(dd):
            dd = np.asarray(dd, float); out = np.zeros_like(dd)
            for i in range(len(edges) - 1):
                out[(dd >= edges[i]) & (dd < edges[i + 1])] = vals[i]
            out[dd >= edges[-1]] = vals[-1]; out[dd < edges[0]] = vals[0]
            return out
        return pred, len(vals)
    # nonlinear families
    span = yv.max() - yv.min() + 1e-9
    specs = {
        "F3_log":     (f_log,  [span, 0.01, yv.max()], ([0, 1e-4, -np.inf], [np.inf, 10, np.inf])),
        "F4_power":   (f_pow,  [span, 1.0, yv.max()],  ([0, 1e-3, -np.inf], [np.inf, 2.0, np.inf])),
        "F6_fourier": (f_fourier, [span, span/2, span/4, 0.05, 0.2, 1.0, yv.mean()],
                        ([-np.inf]*3 + [0, 0, 0] + [-np.inf], [np.inf]*3 + [PI, PI, PI] + [np.inf])),
        "F7_rope":    (f_rope, [span, span/2, span/4, 0.1, 2.0],
                        ([-np.inf]*3 + [1e-3, 1.01], [np.inf]*3 + [PI, 50.0])),
        "F9_exp2":    (f_exp2, [span, 0.01, span/2, 0.05, yv.min()],
                        ([-np.inf, 1e-5, -np.inf, 0.0, -np.inf], [np.inf, 1.0, np.inf, 1.0, np.inf])),
        "F10_dsin":   (f_dsin, [span, 0.0, 0.01, 0.1, yv.mean()],
                        ([-np.inf, -np.inf, 1e-5, 0.0, -np.inf], [np.inf, np.inf, 1.0, PI, np.inf])),
    }
    func, p0, bounds = specs[fam]
    best = None
    # modest multi-start on the frequency/rate seeds (review: multi-start as in R2 E1)
    seeds = [p0]
    if fam in ("F6_fourier", "F7_rope", "F10_dsin"):
        for w in (0.02, 0.1, 0.5, 1.5):
            q = list(p0)
            if fam == "F10_dsin": q[3] = w
            elif fam == "F7_rope": q[3] = w
            else: q[3], q[4], q[5] = w, min(w*3, PI), min(w*8, PI)
            seeds.append(q)
    for s in seeds:
        try:
            popt, _ = curve_fit(func, d, yv, p0=s, bounds=bounds, maxfev=8000)
            sse = np.sum((func(d, *popt) - yv) ** 2)
            if best is None or sse < best[1]:
                best = (popt, sse)
        except Exception:
            continue
    if best is None:
        return (lambda dd: np.full_like(np.asarray(dd, float), yv.mean())), len(p0)
    popt = best[0]
    return (lambda dd: func(np.asarray(dd, float), *popt)), len(p0), popt, func


FAMILIES = ["F1_const", "F2_linear", "F3_log", "F4_power", "F5_bucket",
            "F6_fourier", "F7_rope", "F8_gaussian", "F9_exp2", "F10_dsin"]


def evaluate_curve(curve, extent):
    """Run the full battery on one far-field curve. Returns per-family scores + winner."""
    d = np.arange(extent)
    ff = d >= NEAR
    d_ff, y_ff = d[ff].astype(float), curve[ff].astype(float)
    even = (d_ff.astype(int) % 2 == 0)
    xe, ye = d_ff[even], y_ff[even]
    xo, yo = d_ff[~even], y_ff[~even]
    if len(xe) < 10 or len(xo) < 5:
        return None
    sst_o = np.sum((yo - yo.mean()) ** 2) + 1e-12
    sst_full = np.sum((y_ff - y_ff.mean()) ** 2) + 1e-12
    n = len(y_ff)
    scores = {}
    for fam in FAMILIES:
        res = fit_family(fam, xe, ye, extent)
        pred, k = res[0], res[1]
        r2_odd = 1 - np.sum((pred(xo) - yo) ** 2) / sst_o
        sse_full = np.sum((pred(d_ff) - y_ff) ** 2)
        bic = n * np.log(sse_full / n + 1e-30) + k * np.log(n)
        scores[fam] = dict(held_out_r2=float(r2_odd), bic=float(bic), k=k,
                           r2_full=float(1 - sse_full / sst_full))
    best_bic = min(scores, key=lambda f: scores[f]["bic"])
    best_r2 = max(scores, key=lambda f: scores[f]["held_out_r2"])
    bic_ok = scores[best_bic]["held_out_r2"] >= scores[best_r2]["held_out_r2"] - 0.02
    ties = [f for f in scores if scores[f]["bic"] - scores[best_bic]["bic"] < 10]
    verdict = ("agreed" if best_bic == best_r2 else "consistent" if bic_ok else "contested")
    return dict(winner_bic=best_bic, winner_r2=best_r2, verdict=verdict,
                ties=ties, scores=scores)


def named_scheme(fam):
    return {"F1_const": "NoPE", "F2_linear": "ALiBi", "F3_log": "KERPLE-log",
            "F4_power": "KERPLE-power", "F5_bucket": "T5", "F6_fourier": "sinusoidal-APE",
            "F7_rope": "RoPE", "F8_gaussian": "Gaussian/window", "F9_exp2": "RetNet/decay",
            "F10_dsin": "damped-rotation"}[fam]


def run_battery():
    meta = json.load(open(os.path.join(W, "_meta.json")))
    per_mode, per_head = {}, {}
    winners_mode0 = []
    for Ls in sorted(meta, key=int):
        L = int(Ls); ext = meta[Ls]["extent"]
        d1 = np.load(os.path.join(D1, f"layer{L:02d}.npz"))
        modes = {}
        for k in range(16):
            curve = d1["S"][k] * d1["Vt"][k, :]
            r = evaluate_curve(curve, ext)
            if r:
                modes[k] = dict(winner_bic=r["winner_bic"], verdict=r["verdict"],
                                held_out_r2=r["scores"][r["winner_bic"]]["held_out_r2"],
                                ties=r["ties"])
                if k == 0:
                    winners_mode0.append((L, r["winner_bic"], r["verdict"]))
        per_mode[Ls] = modes
        # per-head dominant mode
        d0 = np.load(os.path.join(D0, f"layer{L:02d}.npz"))
        heads = {}
        for h in range(64):
            curve = d0["S"][h, 0] * d0["U"][h, :, 0]
            r = evaluate_curve(curve, ext)
            if r:
                heads[h] = dict(winner_bic=r["winner_bic"], verdict=r["verdict"])
        per_head[Ls] = heads
        wc = {}
        for h in heads.values():
            wc[h["winner_bic"]] = wc.get(h["winner_bic"], 0) + 1
        print(f"L{L:2d} {'G' if L in GLOBAL else ' '} mode0={modes.get(0,{}).get('winner_bic','-'):11} "
              f"head-winners: {dict(sorted(wc.items(), key=lambda x:-x[1]))}", flush=True)

    # overall verdict block
    from collections import Counter
    m0 = Counter(w for _, w, _ in winners_mode0)
    allheads = Counter()
    for hs in per_head.values():
        for h in hs.values():
            allheads[h["winner_bic"]] += 1
    verdict = dict(
        mode0_winner_counts={k: v for k, v in m0.most_common()},
        head_winner_counts={k: v for k, v in allheads.most_common()},
        dominant_family=m0.most_common(1)[0][0],
        dominant_named_scheme=named_scheme(m0.most_common(1)[0][0]),
        note="'shape' verdict only; named-scheme requires the fingerprint ladder (see fingerprints.json).")
    return dict(verdict=verdict, per_mode=per_mode, per_head=per_head)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== family battery (per layer: mode-0 winner + per-head winner counts) ===")
    res = run_battery()
    json.dump(res, open(os.path.join(OUT, "family_battery.json"), "w"), indent=2)
    print("\nVERDICT:", json.dumps(res["verdict"], indent=2))
    print(f"\nwrote {OUT}\\family_battery.json")


if __name__ == "__main__":
    main()
