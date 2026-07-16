"""
Round 4 curiosity register -- ACTIVATION enrichment for C1/C2/C3.

Uses the Tier-2 capture pass (dumps/tier2/capture/: per-layer r-vectors and
hidden states) to answer the activation versions the weight-space tests couldn't:

  C1-act: the PHYSICAL seam-direction sign. b_h(d) = mean_q rvec[q,h] . proj[:,d]
          near d=1024 -- the real signed bias, no canonicalization. Resolves the
          weight-space sign ambiguity and cross-checks Tier-2's mean_bias. This is
          the authoritative answer to "which way does the seam step".
  C2-act: which proj modes do LIVE tokens actually excite? Project r-vectors onto
          proj's d_rel singular directions; compare the live mode-energy profile
          to proj's S^2 (what the table reads) and to weight-space E_k. Settles
          co-adapted vs vestigial with activations, not just weights.
  C3-act: do heads' live positional reads point the same way (shared) or not?

Reads capture/*.npy, weights/, and dumps/tier2/ (Tier-2 mean_bias for cross-check).
No GPU, no network. Writes analysis/round4/curiosity/*_act.json.
"""
import glob
import json
import os

import numpy as np

CAP = r"R:\inkling\dumps\tier2\capture"
W = r"R:\inkling\weights"
T2 = r"R:\inkling\dumps\tier2"
OUT = r"R:\inkling\analysis\round4\curiosity"
GLOBAL = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]


def have_capture():
    return len(glob.glob(os.path.join(CAP, "rvec_L*.npy"))) >= 66 * len(TEXTS)


def proj(L):
    return np.load(os.path.join(W, f"layer{L:02d}_rel_logits_proj.npy")).astype(np.float64)  # [16, extent]


def rvec(L, text):
    return np.load(os.path.join(CAP, f"rvec_L{L:02d}_{text}.npy")).astype(np.float64)  # [S,64,16]


def c1_act():
    """Physical seam sign from live r-vectors, per global layer/head, vs Tier-2."""
    out = {"_desc": "b_h(d)=mean_q rvec.proj near d in [1008,1024); physical sign, no canonicalization."}
    for L in GLOBAL:
        p = proj(L)                                   # [16, extent]
        band = slice(1008, 1024)
        per_text = {}
        for t in TEXTS:
            r = rvec(L, t)                            # [S,64,16]
            # bias each query assigns at each distance: [S,64,extent] = r @ p
            # near-seam band mean over distance, then over queries -> per head
            b_band = np.einsum("shk,kd->shd", r, p[:, band])   # [S,64,16(dist)]
            bh = b_band.mean(axis=(0, 2))            # [64] per-head mean in-window bias
            per_text[t] = dict(edge_bias_mean=float(bh.mean()),
                               heads_positive_frac=float((bh > 0).mean()))
        pos = np.mean([v["heads_positive_frac"] for v in per_text.values()])
        emean = np.mean([v["edge_bias_mean"] for v in per_text.values()])
        # cross-check vs Tier-2 measured mean_bias in the same band
        t2 = np.load(os.path.join(T2, f"layer{L:02d}_06_random_s8192.npz"), allow_pickle=True)
        t2_bias = float(np.nanmean(t2["mean_bias"][:, 1008:1024]))
        out[str(L)] = dict(heads_positive_frac=float(pos), edge_bias_mean=float(emean),
                           physical_seam_direction=("positive (drop)" if emean > 0 else "negative (rise)"),
                           tier2_meanbias_inband=t2_bias, texts=per_text)
    return out


def c2_act():
    """Which proj modes do live r-vectors excite? Live mode-energy vs proj S^2."""
    out = {}
    for L in range(66):
        p = proj(L)
        Up, Sp, _ = np.linalg.svd(p, full_matrices=False)   # Up [16,16] d_rel left dirs
        live = np.zeros(16)
        n = 0
        for t in TEXTS:
            r = rvec(L, t)                            # [S,64,16]
            coeff = np.einsum("shk,km->shm", r, Up)  # project onto proj modes -> [S,64,16]
            live += (coeff ** 2).sum(axis=(0, 1))
            n += r.shape[0] * r.shape[1]
        live /= n
        Sp2 = Sp ** 2
        cos = float(np.dot(live, Sp2) / (np.linalg.norm(live) * np.linalg.norm(Sp2) + 1e-12))
        pr = float((Sp2.sum() ** 2) / (Sp2 ** 2).sum()); eff = int(np.ceil(pr))
        live_vestigial = float(live[eff:].sum() / (live.sum() + 1e-12))
        out[str(L)] = dict(live_mode_energy=live.tolist(), proj_S2=Sp2.tolist(),
                           cosine_live_vs_S=cos, live_vestigial_fraction=live_vestigial,
                           proj_eff_rank=pr)
    coss = [v["cosine_live_vs_S"] for v in out.values()]
    vest = [v["live_vestigial_fraction"] for v in out.values()]
    out["_summary"] = dict(median_cosine=float(np.median(coss)), median_live_vestigial=float(np.median(vest)),
                           verdict=("co-adapted: live tokens excite the modes proj reads"
                                    if np.median(coss) > 0.8 else
                                    "MISALIGNED: live tokens excite modes proj under-reads (or vice versa)"))
    return out


def c3_act():
    """Do heads' live positional reads share a direction in d_rel space?"""
    out = {}
    for L in range(66):
        # per-head dominant live r-direction (SVD over tokens), pooled across texts
        dirs = []
        for h in range(64):
            acc = []
            for t in TEXTS:
                acc.append(rvec(L, t)[:, h, :])       # [S,16]
            R = np.concatenate(acc, 0)                # [S*ntext,16]
            u, s, vt = np.linalg.svd(R - R.mean(0, keepdims=True), full_matrices=False)
            dirs.append(vt[0])                        # dominant read direction (16-dim)
        D = np.array(dirs)                            # [64,16]
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        gram = np.abs(Dn @ Dn.T)
        ev = np.linalg.eigvalsh(Dn @ Dn.T).clip(0)
        out[str(L)] = dict(mean_abs_alignment=float(gram[~np.eye(64, dtype=bool)].mean()),
                           eff_num_read_dirs=float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12)))
    al = [v["mean_abs_alignment"] for v in out.values()]
    out["_summary"] = dict(median_alignment=float(np.median(al)),
                           note="d_rel is only 16-dim shared via proj; high alignment => heads read "
                                "positional info along a common few directions (shared), low => spread")
    return out


def main():
    if not have_capture():
        n = len(glob.glob(os.path.join(CAP, "rvec_L*.npy")))
        print(f"capture incomplete ({n}/{66*len(TEXTS)} rvec files) -- run after the capture pass finishes.")
        return 1
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("C1_seam_direction_act", c1_act), ("C2_interface_utilization_act", c2_act),
                     ("C3_shared_circuit_act", c3_act)]:
        print(f"running {name} ...", flush=True)
        json.dump(fn(), open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    print(f"wrote activation JSONs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
