"""
DEPRECATED legacy Round 1 fitter.

Weight-level spectral decomposition of Inkling's learned relative-position
transport.

Per layer, `rel_logits_proj.proj` has shape [d_rel=16, extent] and is SHARED
across all attention heads in that layer (confirmed via safetensors header
inspection: no head axis in the tensor name or shape). The per-head,
per-token feature R = wr_du(hidden_state) (shape [.., num_heads, d_rel]) is
data-dependent, so a *pure weight-level* per-head response curve does not
exist independent of activations -- what IS purely weight-level is `proj`
itself: a bank of 16 basis curves b_k(d), d in [0, extent), shared per layer.

We decompose that bank two ways:
  1. SVD of proj [16, extent] -> singular values = "the spectrum" the task
     asked to dump, right-singular vectors = the 16 orthogonal basis curves
     over distance.
  2. For each of the 16 (left-singular-weighted) basis curves, fit a damped
     sinusoid model to extract (rho, delta, sigma):
         b_k(d) ~= sigma_k * exp(-delta_k * d) * cos(rho_k * d + phi_k)
     via nonlinear least squares (scipy), seeded from the curve's dominant
     FFT frequency and the log-decay slope of its envelope (Hilbert
     transform).  This is a standard damped-sinusoid / Prony-style fit for
     any convolution-kernel-shaped tensor and is the natural way to answer
     "does the model use a rotation, a decay, or a scale to represent
     distance" -- rho = rotation frequency, delta = decay rate, sigma =
     amplitude/scale.  NOTE: this parametric form is our analysis choice,
     not something documented by Thinking Machines -- the checkpoint gives
     us the raw b_k(d) curves; whether they are well-described by a damped
     sinusoid is an empirical question this script also reports (fit R^2).

The fit below is retained only to reproduce the historical artifact.  Its
unconstrained decay and amplitude/phase/frequency symmetries make the fitted
parameters non-identifiable, and it must not be used for current conclusions.
Use ``fit_transport_models.py`` for the corrected constrained model ladder.
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert

WEIGHTS_DIR = r"R:\inkling\weights"
OUT_DIR = r"R:\inkling\analysis"


def damped_sinusoid(d, sigma, delta, rho, phi):
    return sigma * np.exp(-delta * d) * np.cos(rho * d + phi)


def fit_component(curve):
    d = np.arange(len(curve), dtype=np.float64)
    y = curve.astype(np.float64)

    # seed rho from dominant FFT frequency (excluding DC)
    spec = np.fft.rfft(y - y.mean())
    freqs = np.fft.rfftfreq(len(y), d=1.0) * 2 * np.pi
    k = np.argmax(np.abs(spec[1:])) + 1
    rho0 = freqs[k]

    # seed delta from decay of the analytic-signal envelope (avoid log(0))
    env = np.abs(hilbert(y))
    env = np.clip(env, 1e-8 * (env.max() + 1e-12), None)
    if env.max() > 0:
        logenv = np.log(env)
        delta0 = -np.polyfit(d, logenv, 1)[0]
    else:
        delta0 = 0.0
    delta0 = max(delta0, 1e-6)

    sigma0 = float(np.abs(y).max() + 1e-8)
    p0 = [sigma0, delta0, rho0, 0.0]
    try:
        popt, _ = curve_fit(damped_sinusoid, d, y, p0=p0, maxfev=20000)
    except RuntimeError:
        popt = p0

    yhat = damped_sinusoid(d, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1 - ss_res / ss_tot

    sigma, delta, rho, phi = popt
    # canonicalize: sigma >= 0, rho in [0, pi]
    if sigma < 0:
        sigma, phi = -sigma, phi + np.pi
    rho = float(np.mod(rho, 2 * np.pi))
    if rho > np.pi:
        rho, phi = 2 * np.pi - rho, -phi
    return dict(sigma=float(sigma), delta=float(delta), rho=float(rho),
                phi=float(np.mod(phi, 2 * np.pi)), r2=float(r2))


def legacy_main():
    meta = json.load(open(os.path.join(WEIGHTS_DIR, "_meta.json")))
    results = {}
    for i_str, m in meta.items():
        i = int(i_str)
        proj = np.load(os.path.join(WEIGHTS_DIR, f"layer{i:02d}_rel_logits_proj.npy"))
        U, S, Vt = np.linalg.svd(proj, full_matrices=False)  # proj = U diag(S) Vt

        components = []
        for k in range(len(S)):
            curve = S[k] * Vt[k]  # this component's contribution scaled into the curve
            fit = fit_component(curve)
            components.append({"index": k, "singular_value": float(S[k]), **fit})

        results[i] = {
            "is_local": m["is_local"],
            "extent": m["extent"],
            "singular_values": S.tolist(),
            "components": components,
        }
        print(f"layer {i:02d} ({'local' if m['is_local'] else 'global'}, extent={m['extent']}): "
              f"top singular value={S[0]:.4f}, "
              f"top component rho={components[0]['rho']:.4f} "
              f"delta={components[0]['delta']:.5f} sigma={components[0]['sigma']:.4f} "
              f"r2={components[0]['r2']:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "rel_attn_spectrum.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print("\nFull spectrum dump written to", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the deprecated, non-identifiable Round 1 fit."
    )
    parser.add_argument(
        "--acknowledge-invalid-legacy-fit",
        action="store_true",
        help="explicitly allow regeneration of the historical invalid fit",
    )
    args = parser.parse_args()
    if not args.acknowledge_invalid_legacy_fit:
        parser.error(
            "this fitter is deprecated and its parameters are non-identifiable; "
            "use scripts/fit_transport_models.py instead. To reproduce the old "
            "artifact only, pass --acknowledge-invalid-legacy-fit."
        )
    print(
        "[WARNING] reproducing the deprecated non-identifiable Round 1 fit; "
        "do not use these parameters for current conclusions."
    )
    legacy_main()


if __name__ == "__main__":
    main()
