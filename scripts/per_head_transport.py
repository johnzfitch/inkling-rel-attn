"""
E2 -- Per-head composed transport with source-confirmed wr_du reshape.

Weight-level per-head object: C_h = proj.T @ Wr_h  ([extent, 6144]) where
Wr_h is head h's 16-row block of wr_du [1024, 6144], grouped head-major as
``wr.reshape(64, 16, 6144)`` (rows ``16h : 16h+16``).

The official implementation reshapes the projection output with
``view(..., num_heads, -1)``, which fixes grouping A (head-major).  The old
top-1-energy concentration comparison has been removed: concentration is not
a valid way to infer tensor layout and must not select grouping B or cause
duplicate full runs.

Then, for all 66 layers x 64 heads: batched GPU SVD of C_h, storing top 8
singular values, top-2 left singular vectors mean-pooled to 128 points, the
full top-1 profile for d in [0, 32), and a rule-based head taxonomy.

Outputs: analysis/round2/per_head_summary.json,
         analysis/round2/head_profiles.npz
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

WEIGHTS_DIR = r"R:\inkling\weights"
OUT_DIR = r"R:\inkling\analysis\round2"
NUM_LAYERS = 66
NUM_HEADS = 64
D_REL = 16
HIDDEN = 6144
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OFFICIAL_HYPOTHESIS = "A"
RESHAPE_SOURCE = (
    ".venv-tier2/Lib/site-packages/transformers/models/inkling/"
    "modeling_inkling.py: relative_states.view(*input_shape, self.num_heads, -1)"
)


def load_layer(i):
    wr = np.load(os.path.join(WEIGHTS_DIR, f"layer{i:02d}_wr_du.npy"))
    proj = np.load(os.path.join(WEIGHTS_DIR, f"layer{i:02d}_rel_logits_proj.npy"))
    return wr, proj


def per_head_blocks(wr, hypothesis):
    """[1024, 6144] -> [64, 16, 6144] using the official head-major view."""
    if hypothesis != OFFICIAL_HYPOTHESIS:
        raise ValueError(
            f"unsupported reshape {hypothesis!r}; the official implementation "
            "fixes head-major hypothesis 'A'"
        )
    return wr.reshape(NUM_HEADS, D_REL, HIDDEN)


def composed_C(wr, proj, hypothesis):
    """C for all heads: [64, extent, 6144] on GPU float32."""
    blocks = torch.from_numpy(np.ascontiguousarray(per_head_blocks(wr, hypothesis))).to(DEV)
    projT = torch.from_numpy(proj.T.copy()).to(DEV)          # [extent, 16]
    return torch.matmul(projT.unsqueeze(0), blocks)          # [64, extent, 6144]


def batched_svd(C, chunk=NUM_HEADS):
    """SVD of [H, extent, 6144]; returns U [H, extent, extent], S [H, extent]."""
    Us, Ss = [], []
    h = 0
    while h < C.shape[0]:
        try:
            U, S, _ = torch.linalg.svd(C[h:h + chunk], full_matrices=False)
            Us.append(U)
            Ss.append(S)
            h += chunk
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 4)
            print(f"  [OOM] retrying with chunk={chunk}")
    return torch.cat(Us), torch.cat(Ss)


def mean_pool(v, out_len=128):
    """Mean-pool a length-extent vector to out_len points (extent % out_len == 0)."""
    f = len(v) // out_len
    return v[: f * out_len].reshape(out_len, f).mean(axis=1)


def classify(profile):
    """Rule-based taxonomy on the (full-length) top distance profile."""
    p = np.abs(profile)
    if np.sum(p[:4] ** 2) > 0.5 * np.sum(p ** 2):
        return "prev_token"
    rho = spearmanr(p, np.arange(len(p))).statistic
    if rho < -0.5:
        return "decay"
    if (p.max() - p.min()) < 0.1 * p.max():
        return "flat"
    return "other"


def run_full(hypothesis, meta, suffix=""):
    profiles = np.zeros((NUM_LAYERS, NUM_HEADS, 2, 128), dtype=np.float32)
    near = np.zeros((NUM_LAYERS, NUM_HEADS, 32), dtype=np.float32)
    svals = np.zeros((NUM_LAYERS, NUM_HEADS, 8), dtype=np.float32)
    classes = {}
    counts = {}
    for i in range(NUM_LAYERS):
        wr, proj = load_layer(i)
        C = composed_C(wr, proj, hypothesis)
        U, S = batched_svd(C)
        del C
        torch.cuda.empty_cache()
        U = U.cpu().numpy()
        S = S.cpu().numpy()
        svals[i] = S[:, :8]
        layer_classes = []
        for h in range(NUM_HEADS):
            for r in range(2):
                u = U[h, :, r]
                if u[:8].mean() < 0:      # sign canonicalization (as in E1)
                    u = -u
                profiles[i, h, r] = mean_pool(u)
                if r == 0:
                    near[i, h] = u[:32]
                    layer_classes.append(classify(u))
        classes[str(i)] = layer_classes
        cnt = {c: layer_classes.count(c) for c in ["prev_token", "decay", "flat", "other"]}
        counts[str(i)] = cnt
        print(f"[{hypothesis}] layer {i:02d} "
              f"({'local' if meta[str(i)]['is_local'] else 'global'}): "
              f"prev_token={cnt['prev_token']:2d} decay={cnt['decay']:2d} "
              f"flat={cnt['flat']:2d} other={cnt['other']:2d} "
              f"| mean S0={S[:, 0].mean():.3f}")

    np.savez_compressed(os.path.join(OUT_DIR, f"head_profiles{suffix}.npz"),
                        profiles=profiles, near=near, svals=svals)
    return classes, counts


def main():
    meta = json.load(open(os.path.join(WEIGHTS_DIR, "_meta.json")))
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. source-confirmed reshape ----
    print(
        "layout selection: A (head-major), fixed by official source; "
        "the legacy concentration discriminator is not run"
    )
    reshape_test = {
        "selection_basis": "official implementation tensor view",
        "official_source": RESHAPE_SOURCE,
        "winner": OFFICIAL_HYPOTHESIS,
        "winner_meaning": "head-major reshape(64,16,6144)",
        "legacy_concentration_discriminator": (
            "removed; SVD concentration cannot identify a serialized tensor layout"
        ),
    }

    # ---- 2-4. full per-head pass ----
    runs = [(OFFICIAL_HYPOTHESIS, "")]
    summary = {"reshape_test": reshape_test, "runs": {}}
    for hyp, suffix in runs:
        classes, counts = run_full(hyp, meta, suffix)
        summary["runs"][hyp] = {
            "npz_file": f"head_profiles{suffix}.npz",
            "class_counts_per_layer": counts,
            "head_classes_per_layer": classes,
        }

    out_path = os.path.join(OUT_DIR, "per_head_summary.json")
    json.dump(summary, open(out_path, "w"), indent=2)
    print("Written", out_path)


if __name__ == "__main__":
    main()
