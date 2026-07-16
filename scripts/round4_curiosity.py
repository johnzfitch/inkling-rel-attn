"""
Round 4 curiosity register (C1-C7) -- weight-level tests on existing dumps.

Dump-first, per ROUND4_SPEC ground rules: reads ONLY
  dumps/round3/perhead_svd/*.npz   (D0: per head S[64,16] U[64,extent,16] V[64,16,6144])
  dumps/round3/mode_curves/*.npz   (D1: proj SVD  S[16] U[16,16] Vt[16,extent])
  weights/layerNN_{wr_du,rel_logits_proj}.npy   (raw, for C2 energy + C1 sign)
  analysis/round3/head_taxonomy_v2.json          (A3 classes)
No GPU, no network, no new dumps. One JSON per item -> analysis/round4/curiosity/.

Sign note (C1): D0/D1 apply a sign canonicalization (flip so U[:8].sum()>=0). The
OPERATOR C_h = U diag(S) V is sign-invariant, but any single mode curve's sign is
convention-dependent. C1 therefore reports the weight-space value under the D0
convention AND flags that the physical sign needs activations (the Tier-2
mass_with seam and the captured r-vectors settle it -- see round4_curiosity_act.py).
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

D0 = r"R:\inkling\dumps\round3\perhead_svd"
D1 = r"R:\inkling\dumps\round3\mode_curves"
W = r"R:\inkling\weights"
TAX = r"R:\inkling\analysis\round3\head_taxonomy_v2.json"
OUT = r"R:\inkling\analysis\round4\curiosity"
GLOBAL = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
NEAR = 8


def meta():
    return json.load(open(os.path.join(W, "_meta.json")))


def d0(layer):
    return np.load(os.path.join(D0, f"layer{layer:02d}.npz"))


def d1(layer):
    return np.load(os.path.join(D1, f"layer{layer:02d}.npz"))


def taxonomy():
    return json.load(open(TAX))["trunk_layers"]


def safe_exp(x):
    return np.exp(np.clip(x, -700.0, 700.0))


# ----------------------------------------------------------------- C1

def c1_seam_direction():
    """Weight-space prediction of the seam step sign, per global layer/head.
    Uses the dominant per-head distance profile S[h,0]*U[h,:,0] near the boundary.
    Sign is under the D0 canonicalization -> flagged convention-dependent."""
    out = {"_caveat": "sign is under D0 canonicalization (U[:8].sum()>=0); physical "
                      "sign requires activations. Authoritative answer: Tier-2 mass_with "
                      "seam + captured r-vectors. This is the registered blind prediction.",
           "layers": {}}
    for L in GLOBAL:
        d = d0(L)
        S, U = d["S"], d["U"]                       # [64,16], [64,extent,16]
        prof = S[:, 0:1] * U[:, :, 0]               # [64, extent] dominant profile per head
        edge = prof[:, 1008:1024].mean(1)           # [64] mean over the just-inside band
        pos = float((edge > 0).mean())
        out["layers"][str(L)] = dict(
            edge_mean=float(edge.mean()), heads_positive_frac=pos,
            predicted_seam_direction=("positive (drop)" if edge.mean() > 0 else "negative (rise)"))
    # single registered scalar for the Tier-2 hook (majority across global layers)
    majority = np.mean([v["heads_positive_frac"] for v in out["layers"].values()])
    out["predicted_seam_direction"] = "positive (drop)" if majority > 0.5 else "negative (rise)"
    out["confidence"] = "LOW (registered); convention-dependent, see caveat"
    return out


# ----------------------------------------------------------------- C2

def c2_interface_utilization():
    """Does wr_du feed proj's dead modes? Per layer: E_k = energy each head injects
    into proj-mode k, vs proj singular values S_k. Cosine of the two profiles and
    the vestigial fraction (energy in modes beyond proj's effective rank)."""
    m = meta()
    out = {}
    for Ls in sorted(m, key=int):
        L = int(Ls)
        proj = np.load(os.path.join(W, f"layer{L:02d}_rel_logits_proj.npy")).astype(np.float64)  # [16, extent]
        wr = np.load(os.path.join(W, f"layer{L:02d}_wr_du.npy")).astype(np.float64)               # [1024, 6144]
        wr = wr.reshape(64, 16, 6144)                          # head-major
        Up, Sp, _ = np.linalg.svd(proj, full_matrices=False)  # Up [16,16] d_rel-space left vecs
        # E_k per head = ||Up[:,k]^T @ Wr_h||^2 ; average energy profile over heads
        proj_modes = Up.T @ wr.transpose(1, 0, 2).reshape(16, -1)   # [16, 64*6144]
        # do it per head to keep [16] profile
        Ek = np.zeros(16)
        for h in range(64):
            m_hk = Up.T @ wr[h]                # [16, 6144]
            Ek += (m_hk ** 2).sum(1)
        Ek /= 64.0
        Sp2 = Sp ** 2
        cos = float(np.dot(Ek, Sp2) / (np.linalg.norm(Ek) * np.linalg.norm(Sp2) + 1e-12))
        # effective rank of proj (participation ratio) and vestigial energy fraction
        pr = float((Sp2.sum() ** 2) / (Sp2 ** 2).sum())
        eff = int(np.ceil(pr))
        vestigial_frac = float(Ek[eff:].sum() / (Ek.sum() + 1e-12))
        # baseline: if heads injected energy UNIFORMLY across d_rel modes, this
        # fraction would land in the (16-eff) modes proj barely uses. Compare.
        uniform_baseline = float((16 - eff) / 16)
        out[str(L)] = dict(proj_S2=Sp2.tolist(), head_energy_Ek=Ek.tolist(),
                           cosine_energy_vs_S=cos, proj_participation_rank=pr,
                           vestigial_energy_fraction=vestigial_frac,
                           uniform_baseline_vestigial=uniform_baseline,
                           feeds_live_modes_better_than_uniform=bool(vestigial_frac < uniform_baseline))
    vest = [v["vestigial_energy_fraction"] for v in out.values()]
    null = [v["uniform_baseline_vestigial"] for v in out.values()]
    out["_audit"] = dict(
        status="NULL: indistinguishable from uniform injection",
        median_vestigial=float(np.median(vest)),
        median_uniform_baseline=float(np.median(null)),
        note="This weight-level test carries no information; use C2_interface_utilization_act.json for live co-adaptation.")
    return out


# ----------------------------------------------------------------- C3

def principal_angle_cos(A, B):
    """cos of principal angles between row-spaces of A[k,n], B[k,n]."""
    Qa, _ = np.linalg.qr(A.T)     # [n,k]
    Qb, _ = np.linalg.qr(B.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0, 1)


def c3_shared_circuit():
    """Do heads read positional recipe from a shared hidden subspace? Per layer:
    similarity of heads' top-2 right-singular subspaces (V), clustered; within- vs
    across-taxonomy-class similarity; effective # distinct read-subspaces."""
    tax = taxonomy()
    out = {}
    for Ls in sorted(tax, key=int):
        L = int(Ls)
        V = d0(L)["V"][:, :2, :]                 # [64,2,6144] top-2 read directions
        classes = tax[Ls]["head_class"]
        sim = np.zeros((64, 64))
        for i in range(64):
            for j in range(i, 64):
                c = principal_angle_cos(V[i], V[j]).mean()
                sim[i, j] = sim[j, i] = c
        off = sim[~np.eye(64, dtype=bool)]
        # within vs across taxonomy class
        same = np.array([[classes[i] == classes[j] for j in range(64)] for i in range(64)])
        np.fill_diagonal(same, False)
        across_mask = (~same) & ~np.eye(64, dtype=bool)
        within = float(sim[same].mean()) if same.any() else None
        across = float(sim[across_mask].mean()) if across_mask.any() else None  # None: single-class layer
        # effective # subspaces: participation ratio over eigenvalues of the sim matrix
        ev = np.linalg.eigvalsh(sim).clip(0)
        eff_subspaces = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))
        out[str(L)] = dict(mean_offdiag_sim=float(off.mean()),
                           within_class_sim=within, across_class_sim=across,
                           eff_num_read_subspaces=eff_subspaces)
    # rising-head cross-layer sharing (layers 0-13): one "look far" circuit or 64?
    rising_layers = [L for L in range(14) if "rising" in [c for c in tax.get(str(L), {}).get("head_class", [])]]
    Vr = []
    for L in rising_layers:
        cls = tax[str(L)]["head_class"]
        V = d0(L)["V"]
        for h in range(64):
            if cls[h] == "rising":
                Vr.append(V[h, 0, :])            # top read direction of each rising head
    Vr = np.array(Vr)                             # [n_rising_total, 6144]
    if len(Vr) > 1:
        Vn = Vr / (np.linalg.norm(Vr, axis=1, keepdims=True) + 1e-12)
        gram = np.abs(Vn @ Vn.T)
        ev = np.linalg.eigvalsh(Vn @ Vn.T).clip(0)
        out["_rising_cross_layer"] = dict(
            n_rising_heads=int(len(Vr)),
            mean_abs_alignment=float(gram[~np.eye(len(Vr), dtype=bool)].mean()),
            eff_num_directions=float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12)),
            note="eff_num_directions ~1 => one shared 'look far' direction reused; "
                 "~n => independent implementations")
    return out


# ----------------------------------------------------------------- C4

def f_exp2_ordered(d, a_slow, delta_slow, a_fast, delta_gap, c):
    return a_slow * safe_exp(-delta_slow * d) + a_fast * safe_exp(-(delta_slow + delta_gap) * d) + c


def c4_truncated_fraction():
    """How much kernel got amputated by the extent wall? Fit F9 (bounded positive
    rates, per review #6) to each global layer's mode-0 curve; extrapolate past the
    extent. truncated_fraction = int|b_extrap|[extent,4*extent] / int|b|[8,extent)."""
    m = meta()
    out = {"_label": "DEPRECATED extrapolation diagnostic: the free constant is non-decaying and dominates the out-of-window integral; do not interpret this as amputated kernel mass.",
           "_audit_status": "ARTIFACT. See analysis/revised_mechanisms/revised_mechanism_summary.json for c=0 fits with rate>=1/extent (median 0.321; L5/L65 still pinned)."}
    for L in GLOBAL:
        ext = m[str(L)]["extent"]
        d = d1(L)
        curve = d["S"][0] * d["Vt"][0, :]              # mode-0 curve [extent]
        dd = np.arange(ext, dtype=np.float64)
        band = (dd >= NEAR)
        x, y = dd[band], curve[band]
        # bounded fit: a in R, rates > 0, gap >= 0
        a0 = y[0]
        p0 = [a0, 0.01, a0 * 0.5, 0.05, 0.0]
        bounds = ([-np.inf, 1e-5, -np.inf, 0.0, -np.inf], [np.inf, 1.0, np.inf, 1.0, np.inf])
        try:
            popt, _ = curve_fit(f_exp2_ordered, x, y, p0=p0, bounds=bounds, maxfev=20000)
            yhat = f_exp2_ordered(x, *popt)
            r2 = 1 - np.sum((y - yhat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
            xe = np.arange(ext, 4 * ext, dtype=np.float64)
            be = f_exp2_ordered(xe, *popt)
            realized = np.trapezoid(np.abs(y), x)
            truncated = np.trapezoid(np.abs(be), xe)
            out[str(L)] = dict(fit_r2=float(r2), truncated_fraction=float(truncated / (realized + 1e-12)),
                               rates=[float(popt[1]), float(popt[1] + popt[3])])
        except Exception as e:
            out[str(L)] = dict(error=str(e))
    vals = [v["truncated_fraction"] for v in out.values() if isinstance(v, dict) and "truncated_fraction" in v]
    out["_median_truncated_fraction"] = float(np.median(vals)) if vals else None
    out["_prediction"] = "substantial (>0.3), MEDIUM confidence"
    return out


# ----------------------------------------------------------------- C5

def c5_flip_trajectory():
    """Is rising->decay a smooth rotation or a phase transition? Adjacent-layer
    distance in mode-0 curve space (resampled to 512, L2-normed) and in full
    rank-16 subspace (principal angles)."""
    m = meta()
    layers = sorted((int(k) for k in m), key=int)
    curves = []
    subs = []
    for L in layers:
        d = d1(L)
        c = d["S"][0] * d["Vt"][0, :]
        xp = np.linspace(0, 1, len(c)); xt = np.linspace(0, 1, 512)
        c512 = np.interp(xt, xp, c)
        c512 = c512 / (np.linalg.norm(c512) + 1e-12)
        curves.append(c512)
        subs.append(d["Vt"])                          # [16, extent] rank-16 distance modes
    curves = np.array(curves)
    curve_step = [float(np.linalg.norm(curves[i] - curves[i - 1])) for i in range(1, len(layers))]
    # subspace distance: principal angles between consecutive layers' Vt row-spaces
    # (resample to min extent so shapes match)
    sub_step = []
    for i in range(1, len(layers)):
        A, B = subs[i - 1], subs[i]
        n = min(A.shape[1], B.shape[1])
        cs = principal_angle_cos(A[:, :n], B[:, :n])
        sub_step.append(float(np.sqrt(np.sum(1 - cs ** 2))))   # chordal distance
    out = dict(layers=layers[1:], curve_step_mode0=curve_step, subspace_step_rank16=sub_step,
               global_layers=GLOBAL,
               transition_band_note="taxonomy flips ~L13-28; look for a spike there vs a smooth arc")
    # crude verdict: is the transition-band step an outlier vs the median step?
    cs = np.array(curve_step); med = np.median(cs)
    band = [i for i, L in enumerate(layers[1:]) if 13 <= L <= 28]
    if band:
        out["transition_vs_median_ratio"] = float(np.mean([cs[i] for i in band]) / (med + 1e-12))
        out["verdict"] = ("phase transition (spike)" if out["transition_vs_median_ratio"] > 1.5
                          else "smooth rotation")
        out["audit"] = dict(
            status="VERDICT SURVIVES; MAGNITUDE INFLATED",
            corrected_sign_invariant_common_raw_support_ratio=1.930727429429633,
            note="Original metric mixed near-field sign canonicalization and 512/1024 extent rescaling. See scripts/revised_mechanism_figures.py.")
    return out


# ----------------------------------------------------------------- C6

def c6_within_head_anatomy():
    """mode1 = far-field engine, mode2 = near-field corrector? Per head, near-field
    (d<8) L2-mass fraction of U[:,0] vs U[:,1]."""
    m = meta()
    ratios = []
    per_layer = {}
    for Ls in sorted(m, key=int):
        L = int(Ls)
        U = d0(L)["U"]                                # [64,extent,16]
        def nf_frac(vec):                             # near-field share of L2 mass
            e = vec ** 2
            return e[:, :NEAR].sum(1) / (e.sum(1) + 1e-12)
        nf0 = nf_frac(U[:, :, 0]); nf1 = nf_frac(U[:, :, 1])
        per_layer[Ls] = dict(mode0_nf_mean=float(nf0.mean()), mode1_nf_mean=float(nf1.mean()),
                             mode1_more_nearfield_frac=float((nf1 > nf0).mean()))
        ratios.extend((nf1 / (nf0 + 1e-12)).tolist())
    r = np.array(ratios)
    return dict(per_layer=per_layer,
                ratio_mode1_over_mode0_nf=dict(median=float(np.median(r)), mean=float(r.mean()),
                                               frac_gt_1=float((r > 1).mean())),
                verdict=("mode-2 systematically more near-field (two-part machine)"
                         if (r > 1).mean() > 0.6 else "no systematic near-field split"))


# ----------------------------------------------------------------- C7

def c7_mtp_nearfield():
    """Does draft depth shift the MTP near-field spike? Per MTP layer k, near-field
    argmax of mode/head curves vs draft depth (predicts token n+k+1)."""
    out = {}
    for k in range(8):
        fp = os.path.join(D0, f"mtp{k}.npz")
        if not os.path.exists(fp):
            continue
        U = np.load(fp)["U"]                          # [64,extent,16]
        prof = np.abs(U[:, :, 0])                     # dominant profile magnitude per head
        argmax = prof[:, :32].argmax(1)              # near-field argmax (first 32 distances)
        out[f"mtp{k}"] = dict(draft_depth=k, predicts_token_offset=k + 1,
                              nearfield_argmax_median=int(np.median(argmax)),
                              nearfield_argmax_hist=np.bincount(argmax, minlength=8)[:8].tolist())
    depths = [v["draft_depth"] for v in out.values()]
    peaks = [v["nearfield_argmax_median"] for v in out.values()]
    if len(depths) > 1:
        corr = float(np.corrcoef(depths, peaks)[0, 1])
        out["_verdict"] = dict(depth_peak_corr=corr,
                               tracks_draft_depth=bool(corr > 0.5),
                               note="prediction: NO shift (corr~0). If corr>0.5, spike tracks draft depth.")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    items = {"C1_seam_direction": c1_seam_direction,
             "C2_interface_utilization": c2_interface_utilization,
             "C3_shared_circuit": c3_shared_circuit,
             "C4_truncated_fraction": c4_truncated_fraction,
             "C5_flip_trajectory": c5_flip_trajectory,
             "C6_within_head_anatomy": c6_within_head_anatomy,
             "C7_mtp_nearfield": c7_mtp_nearfield}
    for name, fn in items.items():
        print(f"running {name} ...", flush=True)
        res = fn()
        json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    print(f"\nwrote 7 JSONs -> {OUT}")


if __name__ == "__main__":
    main()
