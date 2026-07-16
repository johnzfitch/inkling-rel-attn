"""P4 v2: continuous per-head rate estimator, decay heads only, log-uniform null.
Plus F7_rope frequency audit (are the 216 rope-winning heads truly oscillatory?)."""
import json, os
import numpy as np
from scipy.optimize import curve_fit

D0 = r"R:\inkling\dumps\round3\perhead_svd"
W = r"R:\inkling\weights"
meta = json.load(open(os.path.join(W, "_meta.json")))
RNG = np.random.default_rng(0)
NEAR = 8

def cont_rate(curve, ext):
    """Continuous rate: OLS slope of log|smoothed| on the decaying stretch.
    Returns nan for rising/non-decay or unresolvable profiles."""
    y = np.abs(curve).astype(float)
    ys = np.convolve(y, np.ones(9)/9, mode="same")[NEAR:ext-4]
    if len(ys) < 64 or ys[0] <= 0:
        return np.nan
    peak = ys[:32].max()
    if ys[:64].argmax() > 48:           # rising profile -> not a decay head
        return np.nan
    lo = np.where(ys <= 0.05 * peak)[0]
    end = lo[0] if len(lo) else len(ys)
    if end < 48:                         # too fast to resolve a rate cleanly
        end = 48
    seg = ys[:end]
    seg = np.clip(seg, peak * 1e-4, None)
    x = np.arange(len(seg), dtype=float)
    slope = np.polyfit(x, np.log(seg), 1)[0]
    return -slope if slope < -1e-5 else np.nan

def spacing_cv(rates):
    lv = np.log(np.sort(rates))
    sp = np.diff(lv)
    return sp.std() / (sp.mean() + 1e-12) if len(sp) >= 8 else np.nan

print("=== P4 v2: decay-head rate ladder, continuous estimator ===")
n_reg = 0; rows = []
for Ls in sorted(meta, key=int):
    L = int(Ls); ext = meta[Ls]["extent"]
    d0 = np.load(os.path.join(D0, f"layer{L:02d}.npz"))
    rates = []
    for h in range(64):
        r = cont_rate(d0["S"][h, 0] * d0["U"][h, :, 0], ext)
        if np.isfinite(r) and r > 0:
            rates.append(r)
    rates = np.array(rates)
    if len(rates) < 16:
        rows.append((L, np.nan, np.nan, len(rates))); continue
    obs = spacing_cv(rates)
    lo, hi = np.log(rates.min()), np.log(rates.max())
    null = np.array([spacing_cv(np.exp(RNG.uniform(lo, hi, len(rates)))) for _ in range(500)])
    p = float((null <= obs).mean())
    rows.append((L, obs, p, len(rates)))
    if p < 0.05: n_reg += 1
ok = [r for r in rows if np.isfinite(r[1])]
print(f"testable {len(ok)}/66, regular (p<0.05): {n_reg}")
print("five most regular:", [(l, round(o,2), p, n) for l, o, p, n in sorted(ok, key=lambda t: t[2])[:5]])
print("rate dynamic range (max/min) per layer, median:",
      "n/a" if not ok else "")
# dynamic ranges
drs = []
for Ls in sorted(meta, key=int):
    L = int(Ls); ext = meta[Ls]["extent"]
    d0 = np.load(os.path.join(D0, f"layer{L:02d}.npz"))
    rates = [cont_rate(d0["S"][h,0]*d0["U"][h,:,0], ext) for h in range(64)]
    rates = np.array([r for r in rates if np.isfinite(r) and r > 0])
    if len(rates) >= 16:
        drs.append(rates.max()/rates.min())
print(f"median cross-head rate dynamic range: {np.median(drs):.1f}x  (RetNet's is ~128x over heads)")

print("\n=== F7_rope audit: fitted frequencies of rope-winning layers ===")
def f_rope(d, a1, a2, a3, w0, g):
    return a1*np.cos(w0*d) + a2*np.cos(w0/g*d) + a3*np.cos(w0/g**2*d)
D1 = r"R:\inkling\dumps\round3\mode_curves"
for L in [27, 28, 39, 50]:   # rope mode-0 winners / heavy rope-head layers
    ext = meta[str(L)]["extent"]
    d1 = np.load(os.path.join(D1, f"layer{L:02d}.npz"))
    curve = d1["S"][0] * d1["Vt"][0, :]
    d = np.arange(ext); x = d[d>=NEAR].astype(float); y = curve[d>=NEAR].astype(float)
    span = y.max()-y.min()+1e-9
    best = None
    for w in (0.005, 0.02, 0.1, 0.5):
        try:
            popt, _ = curve_fit(f_rope, x, y, p0=[span, span/2, span/4, w, 2.0],
                                bounds=([-np.inf]*3+[1e-3,1.01], [np.inf]*3+[np.pi,50.0]), maxfev=8000)
            sse = ((f_rope(x,*popt)-y)**2).sum()
            if best is None or sse < best[1]: best = (popt, sse)
        except Exception: pass
    if best:
        a1,a2,a3,w0,g = best[0]
        r2 = 1 - best[1]/(((y-y.mean())**2).sum()+1e-12)
        pers = [2*np.pi/w for w in (w0, w0/g, w0/g**2)]
        print(f"L{L:02d}: r2={r2:.3f} w0={w0:.4f} g={g:.2f} periods={[int(p) for p in pers]} tokens "
          f"(window {ext}); cycles in window: {[round(ext/p,2) for p in pers]}")
