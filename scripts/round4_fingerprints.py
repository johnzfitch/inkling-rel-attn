"""
Round 4 fingerprint tests + corrected C4 truncation.

The family battery says which SHAPE fits; the fingerprints say whether a NAMED
scheme is realized -- named schemes fix cross-head parameter LADDERS, not just a
curve shape. Matching the shape but not the ladder = "same family, different
organization". Run after round4_family_battery.py (reads family_battery.json).

Also rewrites C4 truncated_fraction with the constant removed from the
extrapolation (per today's correction): past the extent the physical bias is
pinned to 0, not to the fitted DC constant c, so only the decaying kernel is
extrapolated and the realized integral is taken of (b - c).

Dump-first: D0/D1 + weights/_meta.json + family_battery.json. No GPU/network.
Writes analysis/round4/fingerprints.json and rewrites curiosity/C4_truncated_fraction.json.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

D0 = r"R:\inkling\dumps\round3\perhead_svd"
D1 = r"R:\inkling\dumps\round3\mode_curves"
W = r"R:\inkling\weights"
OUT = r"R:\inkling\analysis\round4"
CUR = os.path.join(OUT, "curiosity")
GLOBAL = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
NEAR = 8
RNG = np.random.default_rng(0)


def safe_exp(x):
    return np.exp(np.clip(x, -700.0, 700.0))


def geometric_ladder_test(values, n_perm=1000):
    """Is the ENGINEERED ladder present, i.e. are sorted values a geometric
    sequence with REGULAR log-spacing -- distinct from a generic scale-free
    spread of learned rates?

    Fix (audit): the old test compared log(sorted) vs rank R^2 to a UNIFORM null,
    but sorted log-uniform order statistics are also ~linear in log-vs-rank, so it
    only separated 'geometric-ish' from 'uniform-ish', not 'named ladder' from
    'scale-free spread'. Here we test the REGULARITY of successive log-spacings
    (a RetNet/ALiBi ladder has near-constant spacing; a scale-free draw does not)
    against a LOG-UNIFORM null over the observed log-range. Pinned values must be
    filtered out by the caller before calling this.
    """
    v = np.sort(np.asarray(values, float))
    v = v[v > 0]
    if len(v) < 8:
        return dict(cv_logspacing=None, p_value=None, n=int(len(v)), geometric=None)
    lv = np.log(v)
    def cv_spacing(logvals):
        sp = np.diff(np.sort(logvals))
        sp = sp[sp > 0]
        if len(sp) < 3:
            return np.inf
        return float(np.std(sp) / (np.mean(sp) + 1e-12))   # low CV => regular ladder
    obs = cv_spacing(lv)
    lo, hi = lv.min(), lv.max()
    null = np.array([cv_spacing(RNG.uniform(lo, hi, len(v))) for _ in range(n_perm)])
    p = float((null <= obs).mean())          # fraction of scale-free draws AT LEAST as regular
    return dict(cv_logspacing=float(obs), p_value=p, n=int(len(v)),
                geometric=bool(obs < 0.5 and p < 0.05))


def f_exp2(d, a_slow, r_slow, a_fast, r_gap, c):
    return a_slow * safe_exp(-r_slow * d) + a_fast * safe_exp(-(r_slow + r_gap) * d) + c


def per_head_slopes_rates(L, ext):
    """Fit F2 slope m_h and F9 slow-rate r_h per head. The slow-rate lower bound
    is 1/(2*extent) -- the slowest rate the window can distinguish from constant;
    rates that pin there are unresolvable and returned masked so the ladder test
    filters them (a run of identical boundary rates would corrupt the regression)."""
    d0 = np.load(os.path.join(D0, f"layer{L:02d}.npz"))
    d = np.arange(ext); ff = d >= NEAR
    x = d[ff].astype(float)
    rmin = 1.0 / (2.0 * ext)
    slopes, rates, pinned = [], [], []
    for h in range(64):
        y = (d0["S"][h, 0] * d0["U"][h, :, 0])[ff].astype(float)
        m = np.polyfit(x, y, 1)[0]
        slopes.append(abs(m))
        try:
            span = y.max() - y.min() + 1e-9
            popt, _ = curve_fit(f_exp2, x, y, p0=[span, max(0.01, rmin*2), span/2, 0.05, y.min()],
                                bounds=([-np.inf, rmin, -np.inf, 0, -np.inf], [np.inf, 1, np.inf, 1, np.inf]),
                                maxfev=6000)
            r = float(popt[1])
            rates.append(r); pinned.append(r <= rmin * 1.05)
        except Exception:
            rates.append(np.nan); pinned.append(True)
    return np.array(slopes), np.array(rates), np.array(pinned), rmin


def fingerprints():
    meta = json.load(open(os.path.join(W, "_meta.json")))
    fb = json.load(open(os.path.join(OUT, "family_battery.json")))
    # F2 competitiveness per layer (ALiBi slope ladder only interpretable where F2 wins heads)
    f2_wins = {Ls: sum(1 for h in hs.values() if h["winner_bic"] == "F2_linear")
               for Ls, hs in fb["per_head"].items()}
    out = {"_desc": "named-scheme ladder tests: shape match is necessary but not sufficient; "
                    "a named scheme also fixes the cross-head parameter LADDER (regular log-spacing). "
                    "ALiBi slope ladder only interpretable where F2 is competitive (>=8 head wins)."}
    per_layer = {}
    for Ls in sorted(meta, key=int):
        L = int(Ls); ext = meta[Ls]["extent"]
        slopes, rates, pinned, rmin = per_head_slopes_rates(L, ext)
        rr = rates[np.isfinite(rates) & ~pinned]          # drop pinned/unresolved rates
        n_pinned = int(pinned.sum())
        alibi = geometric_ladder_test(slopes)
        alibi["interpretable"] = bool(f2_wins.get(Ls, 0) >= 8)
        per_layer[Ls] = dict(
            FP_RetNet_rate_ladder=geometric_ladder_test(rr),
            n_rates_pinned=n_pinned, resolvable_rate_min=float(rmin),
            FP_ALiBi_slope_ladder=alibi, f2_head_wins=f2_wins.get(Ls, 0),
        )
    out["per_layer"] = per_layer
    retnet = [v["FP_RetNet_rate_ladder"].get("geometric") for v in per_layer.values()]
    alibi_ok = [v["FP_ALiBi_slope_ladder"].get("geometric") and v["FP_ALiBi_slope_ladder"]["interpretable"]
                for v in per_layer.values()]
    out["summary"] = dict(
        layers_with_retnet_ladder=int(sum(bool(x) for x in retnet)),
        layers_with_alibi_ladder_where_interpretable=int(sum(bool(x) for x in alibi_ok)),
        dominant_shape=fb["verdict"]["dominant_family"],
        null_model="log-spacing regularity vs a log-uniform (scale-free) null; distinguishes an "
                   "engineered ladder from a generic scale-free spread of learned rates.",
        interpretation="retnet_ladder>0 => Inkling re-derived a RetNet retention ladder by SGD; "
                       "0 => exp-decay SHAPE without the named ladder.")
    return out


def fix_c4():
    """C4 with (1) constant dropped from extrapolation AND (2) slow-rate lower
    bound = 1/(2*extent). Where the slow rate still pins at that bound the decay
    length exceeds the window -- the kernel is a flat tail the data cannot
    distinguish from constant, and integral(decay)[ext,4ext]/integral(decay)[8,ext]
    -> ~3 for ANY pinned fit regardless of data. Those layers are reported
    UNIDENTIFIED, not a number (audit fix: dropping c alone re-imports the
    degeneracy through a_slow*exp(-rmin*d))."""
    meta = json.load(open(os.path.join(W, "_meta.json")))
    out = {"_label": "F9 fit, decaying kernel ONLY (constant dropped); slow-rate bound "
                     "1/(2*extent). Extrapolation, not data.",
           "_correction": "pinned-rate fits reported 'unidentified' -- the window cannot "
                          "resolve a decay length > extent from a constant."}
    for L in GLOBAL:
        ext = meta[str(L)]["extent"]
        rmin = 1.0 / (2.0 * ext)
        d1 = np.load(os.path.join(D1, f"layer{L:02d}.npz"))
        curve = d1["S"][0] * d1["Vt"][0, :]
        d = np.arange(ext); ff = d >= NEAR
        x, y = d[ff].astype(float), curve[ff].astype(float)
        span = y.max() - y.min() + 1e-9
        sst = np.sum((y - y.mean()) ** 2) + 1e-12
        # multistart over the slow-rate seed so pinning reflects the DATA, not the seed
        best = None
        for rs0 in (rmin * 2, 0.005, 0.02, 0.1):
            try:
                popt, _ = curve_fit(f_exp2, x, y, p0=[span, rs0, span/2, 0.05, y.min()],
                                    bounds=([-np.inf, rmin, -np.inf, 0, -np.inf], [np.inf, 1, np.inf, 1, np.inf]),
                                    maxfev=8000)
                sse = np.sum((f_exp2(x, *popt) - y) ** 2)
                if best is None or sse < best[1]:
                    best = (popt, sse)
            except Exception:
                continue
        if best is None:
            out[str(L)] = dict(status="error"); continue
        a_s, r_s, a_f, r_g, c = best[0]
        r2 = 1 - best[1] / sst
        r_f = r_s + r_g
        decay = lambda dd: a_s*safe_exp(-r_s*dd) + a_f*safe_exp(-(r_f)*dd)  # NO c
        xe = np.arange(ext, 4*ext, dtype=float)
        # F9 misfit -> a truncation from the wrong family is meaningless.
        if r2 < 0.80:
            out[str(L)] = dict(status="unidentified", reason=f"F9 misfit (r2={r2:.2f}); "
                               "not the winning family here", fit_r2=float(r2), slow_rate=float(r_s)); continue
        # A pinned slow rate is only fatal if the EXTRAPOLATED TAIL is dominated by
        # that unresolvable component. If a_slow is negligible the kernel is
        # effectively single-exponential (fast, resolvable) -> report the number.
        tail_slow = np.trapezoid(np.abs(a_s*safe_exp(-r_s*xe)), xe)
        tail_fast = np.trapezoid(np.abs(a_f*safe_exp(-r_f*xe)), xe)
        pinned = r_s <= rmin * 1.05
        if pinned and tail_slow > 0.5 * (tail_slow + tail_fast):
            out[str(L)] = dict(status="unidentified",
                               reason="extrapolated tail dominated by a slow component pinned at "
                                      "1/(2*extent) (decay length > window)",
                               fit_r2=float(r2), slow_rate=float(r_s),
                               slow_tail_share=float(tail_slow/(tail_slow+tail_fast+1e-30))); continue
        realized = np.trapezoid(np.abs(decay(x)), x)
        truncated = np.trapezoid(np.abs(decay(xe)), xe)
        out[str(L)] = dict(status="resolved", fit_r2=float(r2),
                           truncated_fraction=float(truncated/(realized+1e-12)),
                           slow_rate=float(r_s), fast_rate=float(r_f),
                           slow_rate_pinned=bool(pinned),
                           slow_tail_share=float(tail_slow/(tail_slow+tail_fast+1e-30)))
    vals = [v["truncated_fraction"] for v in out.values()
            if isinstance(v, dict) and v.get("status") == "resolved"]
    unident = [k for k, v in out.items() if isinstance(v, dict) and v.get("status") == "unidentified"]
    out["_median_truncated_fraction_resolved"] = float(np.median(vals)) if vals else None
    out["_n_resolved"] = len(vals); out["_unidentified_layers"] = unident
    return out


def c5_crosstab():
    """Cross-tabulate F10_dsin (damped-rotation) wins against the C5 flip band
    (L13-28). If oscillation concentrates in the regime-switch band, C5 and the
    battery are two views of one object: decay everywhere except a rotation-
    flavored crossover where rho transiently leaves the rho=0 plane."""
    fb = json.load(open(os.path.join(OUT, "family_battery.json")))
    GLOBAL_S = set(GLOBAL)
    rows = []
    for Ls in sorted(fb["per_head"], key=int):
        L = int(Ls)
        dsin = sum(1 for h in fb["per_head"][Ls].values() if h["winner_bic"] == "F10_dsin")
        mode0 = fb["per_mode"][Ls].get("0", {}).get("winner_bic", "-")
        rows.append(dict(layer=L, in_c5_band=bool(13 <= L <= 28), is_global=L in GLOBAL_S,
                         f10_dsin_head_wins=dsin, mode0_winner=mode0))
    band = [r["f10_dsin_head_wins"] for r in rows if r["in_c5_band"] and not r["is_global"]]
    out_band = [r["f10_dsin_head_wins"] for r in rows if not r["in_c5_band"] and not r["is_global"]]
    return dict(
        rows=rows,
        mean_f10_in_c5_band=float(np.mean(band)) if band else None,
        mean_f10_outside_band=float(np.mean(out_band)) if out_band else None,
        enrichment_ratio=float(np.mean(band) / (np.mean(out_band) + 1e-9)) if band and out_band else None,
        interpretation="F10_dsin concentrated in L13-28 => the rising->decay phase transition (C5) "
                       "passes through a damped-oscillation shape: rho!=0 exactly at the crossover. "
                       "C5 and the battery are two views of one depth-resolved object.")


def main():
    print("fingerprints ..."); fp = fingerprints()
    json.dump(fp, open(os.path.join(OUT, "fingerprints.json"), "w"), indent=2)
    print("  ", json.dumps(fp["summary"], indent=2))
    print("C5 x battery cross-tab ..."); ct = c5_crosstab()
    json.dump(ct, open(os.path.join(OUT, "c5_battery_crosstab.json"), "w"), indent=2)
    print(f"   F10_dsin head-wins: in C5 band={ct['mean_f10_in_c5_band']:.1f} vs "
          f"outside={ct['mean_f10_outside_band']:.1f}  (enrichment {ct['enrichment_ratio']:.1f}x)")
    print("C4 corrected ..."); c4 = fix_c4()
    json.dump(c4, open(os.path.join(CUR, "C4_truncated_fraction.json"), "w"), indent=2)
    print(f"   median truncated_fraction (resolved only) = {c4['_median_truncated_fraction_resolved']} "
          f"| unidentified layers: {c4['_unidentified_layers']}  "
          f"(was ~2.0 with the constant)")
    print("done")


if __name__ == "__main__":
    main()
