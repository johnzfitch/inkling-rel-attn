"""
E1 -- Model-selection refit of the relative-attention mode curves.

Replaces Round 1's single damped-sinusoid fit (see analysis/VERIFICATION.md,
issues 1-3): nonlinear amplitude/phase wrap symmetries are replaced by linear
cosine/sine coefficients, the near field (d < 8) is reported raw instead of
forced through a smooth decay, and the far field is fit with a 5-model ladder
selected by BIC.

Models (far field, d in [8, extent)):
  const: a
  exp:   a*exp(-delta*d) + c
  exp2:  a1*exp(-d1*d) + a2*exp(-d2*d) + c
  log:   a + b*log1p(d)
  dsin:  exp(-delta*d)*(a_cos*cos(rho*d)+a_sin*sin(rho*d)) + c

Exponential decay rates are constrained strictly positive, and the two-rate
model is parameterized during optimization as delta_fast = delta_slow + gap
with a positive gap.  The damped sinusoid uses linear cosine/sine coefficients
instead of amplitude+phase, has nonnegative damping, and constrains rho to the
observable interval (rho_min, pi). rho_min requires at least half a cycle over
the fitted window; slower bends belong in the non-oscillatory families.

Output: analysis/round2/transport_fits.json  (plain floats/ints/strings only)
"""
import json
import os

import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.signal import hilbert

WEIGHTS_DIR = r"R:\inkling\weights"
OUT_DIR = r"R:\inkling\analysis\round2"
NEAR = 8            # near-field cutoff: d in [0, NEAR) reported raw
MODE_THRESH = 0.05  # fit modes with S[k] >= MODE_THRESH * S[0]


def safe_exp(x):
    return np.exp(np.clip(x, -700.0, 700.0))


def f_const(d, a):
    return np.full_like(np.asarray(d, dtype=np.float64), a)


def f_exp(d, a, delta, c):
    return a * safe_exp(-delta * d) + c


def f_exp2(d, a1, d1, a2, d2, c):
    return a1 * safe_exp(-d1 * d) + a2 * safe_exp(-d2 * d) + c


def f_exp2_ordered(d, a_slow, delta_slow, a_fast, delta_gap, c):
    """Optimization form that makes the two decay rates identifiable by order."""
    return f_exp2(d, a_slow, delta_slow, a_fast, delta_slow + delta_gap, c)


def f_log(d, a, b):
    return a + b * np.log1p(d)


def f_dsin(d, a_cos, a_sin, delta, rho, c):
    envelope = safe_exp(-delta * d)
    return envelope * (a_cos * np.cos(rho * d) + a_sin * np.sin(rho * d)) + c


MODELS = {
    "const": (f_const, 1),
    "exp": (f_exp, 3),
    "exp2": (f_exp2, 5),
    "log": (f_log, 2),
    "dsin": (f_dsin, 5),
}
PARAM_NAMES = {
    "const": ("a",),
    "exp": ("a", "delta", "c"),
    "exp2": ("a1", "delta1", "a2", "delta2", "c"),
    "log": ("a", "b"),
    "dsin": ("a_cos", "a_sin", "delta", "rho", "c"),
}
CONST_PARAM_INDEX = {"exp": 2, "exp2": 4, "dsin": 4}  # index of c; const/log: none


def envelope_delta_seed(d, y):
    """Round 1's decay seed: log-slope of the analytic-signal envelope."""
    env = np.abs(hilbert(y))
    env = np.clip(env, 1e-8 * (env.max() + 1e-12), None)
    if env.max() > 0:
        delta0 = -np.polyfit(d, np.log(env), 1)[0]
    else:
        delta0 = 0.0
    return max(delta0, 1e-6)


def fft_rho_seed(y):
    """Round 1's rho seed: dominant non-DC FFT bin."""
    spec = np.fft.rfft(y - y.mean())
    freqs = np.fft.rfftfreq(len(y), d=1.0) * 2 * np.pi
    k = np.argmax(np.abs(spec[1:])) + 1
    return float(freqs[k])


def _dsin_seed(d, y, delta, rho):
    """Least-squares seed for the linear coefficients at fixed delta/rho."""
    envelope = safe_exp(-delta * d)
    design = np.column_stack(
        [envelope * np.cos(rho * d), envelope * np.sin(rho * d), np.ones_like(d)]
    )
    a_cos, a_sin, c = np.linalg.lstsq(design, y, rcond=None)[0]
    return [float(a_cos), float(a_sin), float(delta), float(rho), float(c)]


def _canonicalize_params(fam, params):
    """Remove output-only label symmetries after fitting."""
    params = np.asarray(params, dtype=np.float64)
    if fam == "exp2" and params[1] > params[3]:
        params = params[[2, 3, 0, 1, 4]]
    return params


def _amp_seed(y, c0, delta, d0):
    a0 = float(y[0] - c0) * float(safe_exp(delta * d0))
    if not np.isfinite(a0) or abs(a0) > 1e12:
        a0 = float(y[0] - c0)
    return a0


def fit_far_field(d, y):
    """Fit all models; return {family: {params, rss, bic, aicc, r2, ok, error}}.

    Multi-start: the Round 1 seeds (envelope delta, FFT rho) are always tried,
    plus a few log-spaced restarts, so no family loses model selection merely
    by landing in a bad local minimum. Best rss per family wins.
    """
    n = len(d)
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12

    # shared seeds -- Round 1 envelope/FFT seeds plus log-spaced restarts
    delta0 = envelope_delta_seed(d, y)
    delta_seeds = sorted({delta0, 0.5 / n, 2.0 / n, 8.0 / n, 32.0 / n})
    c0 = float(y[-max(8, n // 8):].mean())
    rho0 = float(np.clip(fft_rho_seed(y), 0.0, np.pi))
    fitted_span = max(float(d[-1] - d[0]), 1.0)
    rho_min = float(np.pi / fitted_span)
    rate_floor = max(1e-12, 1e-6 / fitted_span)
    rho_eps = min(1e-8, 0.01 * rho_min)
    rho_low = rho_min + rho_eps
    rho_high = np.pi - rho_eps
    rho_seeds = sorted(
        {
            float(np.clip(rho0, rho_low, rho_high)),
            float(np.clip(2 * rho_min, rho_low, rho_high)),
            float(np.clip(4 * rho_min, rho_low, rho_high)),
        }
    )

    seed_sets = {
        "const": [[float(y.mean())]],
        "exp": [[_amp_seed(y, c0, dl, d[0]), dl, c0] for dl in delta_seeds],
        "log": [[float(y[0]),
                 float((y[-1] - y[0]) / (np.log1p(d[-1]) - np.log1p(d[0]) + 1e-12))]],
        "dsin": [_dsin_seed(d, y, dl, rh) for dl in delta_seeds for rh in rho_seeds],
    }

    bounds = {
        "exp": ([-np.inf, rate_floor, -np.inf], [np.inf, np.inf, np.inf]),
        "exp2": (
            [-np.inf, rate_floor, -np.inf, rate_floor, -np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
        "dsin": (
            [-np.inf, -np.inf, 0.0, rho_low, -np.inf],
            [np.inf, np.inf, np.inf, rho_high, np.inf],
        ),
    }

    results = {}
    for fam in ["const", "exp", "exp2", "log", "dsin"]:
        func, k = MODELS[fam]
        if fam == "exp2":
            # Prony-style seed: d1, d2 at 1x and 5x the exp-fit delta
            de = results["exp"]["params"][1] if results.get("exp", {}).get("ok") else delta0
            ce = results["exp"]["params"][2] if results.get("exp", {}).get("ok") else c0
            ae = results["exp"]["params"][0] if results.get("exp", {}).get("ok") else \
                _amp_seed(y, c0, delta0, d[0])
            de = max(de, 1e-6)
            p0s = [[ae / 2, de, ae / 2, 4 * de, ce]]
            p0s += [[_amp_seed(y, c0, dl, d[0]) / 2, dl,
                     _amp_seed(y, c0, dl, d[0]) / 2, 4 * dl, c0]
                    for dl in delta_seeds]
        else:
            p0s = seed_sets[fam]
        try:
            best = None
            last_err = None
            for p0 in p0s:
                try:
                    if fam == "const":
                        cand = np.array([float(y.mean())])  # exact LS solution
                    else:
                        kwargs = {"bounds": bounds[fam]} if fam in bounds else {}
                        fit_func = f_exp2_ordered if fam == "exp2" else func
                        cand, _ = curve_fit(
                            fit_func, d, y, p0=p0, maxfev=20000, **kwargs
                        )
                        if fam == "exp2":
                            cand = np.array(
                                [cand[0], cand[1], cand[2], cand[1] + cand[3], cand[4]]
                            )
                    cand = _canonicalize_params(fam, cand)
                    r = float(np.sum((y - func(d, *cand)) ** 2))
                    if best is None or r < best[1]:
                        best = (cand, r)
                except Exception as e:
                    last_err = e
            if best is None:
                raise last_err if last_err else RuntimeError("no seed converged")
            popt, rss = best
            bic = n * np.log(rss / n + 1e-300) + k * np.log(n)
            if n - k - 1 > 0:
                aicc = n * np.log(rss / n + 1e-300) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
            else:
                aicc = float("inf")
            results[fam] = {
                "ok": True, "params": [float(p) for p in popt],
                "param_names": list(PARAM_NAMES[fam]),
                "rss": rss, "bic": float(bic), "aicc": float(aicc),
                "r2": float(1 - rss / ss_tot), "error": None,
            }
        except Exception as e:  # record failures, keep going
            results[fam] = {"ok": False, "params": None,
                            "param_names": list(PARAM_NAMES[fam]), "rss": None,
                            "bic": None, "aicc": None, "r2": None,
                            "error": f"{type(e).__name__}: {e}"}
    return results


def derived_scales(fam, params, extent):
    """Return fitted value at d=NEAR and a decay-scale diagnostic.

    For exp and dsin, d_half is the absolute distance at which the analytic
    envelope is half its value at the start of the fitted region (d=NEAR).
    For exp2 it remains the first magnitude half-crossing of the combined
    non-constant component and is therefore a descriptive crossing, not a
    single exponential half-life. Const/log have no decay half-distance.
    """
    func, _ = MODELS[fam]
    amp0 = float(func(np.array([float(NEAR)]), *params)[0])
    if fam in ("const", "log"):
        return amp0, None
    if fam == "exp":
        delta = float(params[1])
        return amp0, (None if delta <= 0.0 else float(NEAR + np.log(2.0) / delta))
    if fam == "dsin":
        delta = float(params[2])
        return amp0, (None if delta <= 0.0 else float(NEAR + np.log(2.0) / delta))

    c = params[CONST_PARAM_INDEX[fam]]

    def g(x):
        return float(func(np.array([float(x)]), *params)[0]) - c

    g8 = g(NEAR)
    if abs(g8) < 1e-12:
        return amp0, None
    target = abs(g8) / 2.0
    # exp2 only: first magnitude half-crossing of the combined decaying part.
    grid = np.linspace(NEAR, 20.0 * extent, 20000)
    vals = np.abs(np.array([g(x) for x in grid])) - target
    sign_change = np.nonzero((vals[:-1] > 0.0) & (vals[1:] <= 0.0))[0]
    if len(sign_change) == 0:
        return amp0, None
    i = sign_change[0]
    try:
        root = brentq(lambda x: abs(g(x)) - target, grid[i], grid[i + 1])
    except ValueError:
        root = float(grid[i])
    return amp0, float(root)


def main():
    meta = json.load(open(os.path.join(WEIGHTS_DIR, "_meta.json")))
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    mode0_winners = {}
    for i_str in sorted(meta, key=int):
        i = int(i_str)
        m = meta[i_str]
        proj = np.load(os.path.join(WEIGHTS_DIR, f"layer{i:02d}_rel_logits_proj.npy"))
        extent = proj.shape[1]
        U, S, Vt = np.linalg.svd(proj, full_matrices=False)

        layer_modes = {}
        for k in range(len(S)):
            if S[k] < MODE_THRESH * S[0]:
                continue
            y = (S[k] * Vt[k]).astype(np.float64)
            if y[:NEAR].mean() < 0:  # sign canonicalization
                y = -y
            near_field = [float(v) for v in y[:NEAR]]
            d = np.arange(NEAR, extent, dtype=np.float64)
            fits = fit_far_field(d, y[NEAR:])

            fitted = {f: r for f, r in fits.items() if r["ok"]}
            if not fitted:
                raise RuntimeError(f"layer {i} mode {k}: all model families failed")
            winner = min(fitted, key=lambda f: fitted[f]["bic"])
            amp0, d_half = derived_scales(winner, fitted[winner]["params"], extent)

            layer_modes[str(k)] = {
                "singular_value": float(S[k]),
                "near_field": near_field,
                "winner": winner,
                "winner_params": fitted[winner]["params"],
                "winner_param_names": fitted[winner]["param_names"],
                "winner_r2": fitted[winner]["r2"],
                "bic": {f: (r["bic"] if r["ok"] else None) for f, r in fits.items()},
                "aicc": {f: (r["aicc"] if r["ok"] else None) for f, r in fits.items()},
                "fit_errors": {f: r["error"] for f, r in fits.items() if not r["ok"]},
                "amp0": amp0,
                "d_half": d_half,
            }

        results[str(i)] = {
            "is_local": bool(m["is_local"]),
            "extent": int(extent),
            "singular_values": [float(s) for s in S],
            "modes": layer_modes,
        }
        m0 = layer_modes["0"]
        mode0_winners[i] = (m0["winner"], m0["winner_params"], extent)
        dh = f"{m0['d_half']:.1f}" if m0["d_half"] is not None else "None"
        print(f"layer {i:02d} ({'local' if m['is_local'] else 'global'}, extent={extent}): "
              f"{len(layer_modes)} modes fit | mode0 winner={m0['winner']} "
              f"r2={m0['winner_r2']:.3f} amp0={m0['amp0']:.3f} d_half={dh}")

    # --- claim check: dominant modes should prefer an exponential family. ---
    n_expish = sum(1 for w, _, _ in mode0_winners.values() if w in ("exp", "exp2"))
    n_layers = len(mode0_winners)
    print(f"\nmode-0 winners: {n_expish}/{n_layers} exp/exp2")
    if n_expish < n_layers / 2:
        from collections import Counter
        cnt = Counter(w for w, _, _ in mode0_winners.values())
        print(f"[CONTRADICTION] dominant modes do NOT prefer exp/exp2 everywhere: "
              f"mode-0 winner counts = {dict(cnt)}")
    n_dsin = sum(1 for w, _, _ in mode0_winners.values() if w == "dsin")
    if n_dsin:
        print(
            f"dsin wins mode 0 in {n_dsin} layers; this family is defined with "
            "rho above the half-cycle observability floor"
        )

    out_path = os.path.join(OUT_DIR, "transport_fits.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print("Written", out_path)


if __name__ == "__main__":
    main()
