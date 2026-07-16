"""
E3 -- Spectrum-vs-depth + local/global comparison + near-field census.

All inputs are the Round 1 .npy dumps on disk; no network, no GPU.

1. Singular spectrum matrix [66, 16] + participation ratio per layer
   -> analysis/round2/spectrum_depth.json
2. Local-vs-global: for each global layer g, Pearson-correlate its 16 mode
   curves (truncated to d < 512) against all 16 mode curves of the adjacent
   local layers g-1 and g+1 (sign-canonicalized), record full 16x16 matrices
   and per-mode best matches.
3. Near-field census: layer-mean curve mean_k |S_k * Vt_k| over d in [0, 32).
4. Plots (PNG, 150 dpi) -> analysis/round2/figs/
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEIGHTS_DIR = r"R:\inkling\weights"
OUT_DIR = r"R:\inkling\analysis\round2"
FIG_DIR = os.path.join(OUT_DIR, "figs")
NUM_LAYERS = 66
NEAR = 8  # sign-canonicalization window (as in E1)


def canon(curve):
    return -curve if curve[:NEAR].mean() < 0 else curve


def participation_ratio(S):
    """PR of the eigenvalue spectrum lambda_i = S_i^2: (sum l)^2 / sum l^2."""
    lam = S.astype(np.float64) ** 2
    return float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-300))


def load_layer(i):
    proj = np.load(os.path.join(WEIGHTS_DIR, f"layer{i:02d}_rel_logits_proj.npy"))
    U, S, Vt = np.linalg.svd(proj, full_matrices=False)
    return proj, S, Vt


def main():
    meta = json.load(open(os.path.join(WEIGHTS_DIR, "_meta.json")))
    os.makedirs(FIG_DIR, exist_ok=True)

    spectra = np.zeros((NUM_LAYERS, 16))
    prs = np.zeros(NUM_LAYERS)
    near_census = np.zeros((NUM_LAYERS, 32))
    curves = {}   # layer -> [16, extent] sign-canonicalized mode curves S_k*Vt_k
    is_local = {}

    for i in range(NUM_LAYERS):
        m = meta[str(i)]
        is_local[i] = bool(m["is_local"])
        proj, S, Vt = load_layer(i)
        spectra[i] = S
        prs[i] = participation_ratio(S)
        mode_curves = S[:, None] * Vt                       # [16, extent]
        near_census[i] = np.abs(mode_curves[:, :32]).mean(axis=0)
        curves[i] = np.stack([canon(c) for c in mode_curves])
        print(f"layer {i:02d} ({'local' if is_local[i] else 'global'}): "
              f"S0={S[0]:.3f} PR={prs[i]:.2f} near-field peak d="
              f"{int(np.argmax(near_census[i]))} ({near_census[i].max():.4f})")

    # sanity check against VERIFICATION.md quoted PRs
    quoted = {0: 2.6, 5: 3.1, 23: 3.2, 40: 1.7, 65: 1.4}
    for l, q in quoted.items():
        if abs(prs[l] - q) > 0.15:
            print(f"[CONTRADICTION] participation ratio layer {l}: measured "
                  f"{prs[l]:.2f} vs VERIFICATION.md {q}")

    # --- item 2: local-vs-global mode-curve correlation ---
    global_layers = [g for g in range(NUM_LAYERS) if not is_local[g]]
    lg = {}
    for g in global_layers:
        entry = {}
        gc = curves[g][:, :512]  # truncate global curves to d < 512
        for nb in (g - 1, g + 1):
            if nb < 0 or nb >= NUM_LAYERS or not is_local[nb]:
                continue
            nc = curves[nb]  # local extent is 512
            corr = np.zeros((16, 16))
            for a in range(16):
                for b in range(16):
                    corr[a, b] = np.corrcoef(gc[a], nc[b])[0, 1]
            best = [{"local_mode": int(np.argmax(np.abs(corr[a]))),
                     "corr": float(corr[a, np.argmax(np.abs(corr[a]))])}
                    for a in range(16)]
            entry[str(nb)] = {
                "corr_matrix": [[float(v) for v in row] for row in corr],
                "best_match_per_global_mode": best,
            }
        lg[str(g)] = entry
        b0 = [entry[k]["best_match_per_global_mode"][0] for k in entry]
        print(f"global layer {g:02d}: mode-0 best |corr| vs neighbors = "
              + ", ".join(f"{d['corr']:+.3f} (local mode {d['local_mode']})" for d in b0))

    # --- item 3 summary: is the near-field spike universal? ---
    spike = near_census[:, :4].max(axis=1) / (near_census.max(axis=1) + 1e-12)
    n_spiky = int(np.sum(spike > 0.99))  # layers whose global max lies in d<4
    peak_pos = near_census.argmax(axis=1)
    print(f"\nnear-field census: {int(np.sum(peak_pos < 4))}/{NUM_LAYERS} layers "
          f"peak at d<4; peak-position histogram: "
          f"{np.bincount(peak_pos, minlength=32).tolist()}")

    out = {
        "singular_values": [[float(v) for v in row] for row in spectra],
        "participation_ratio": [float(v) for v in prs],
        "is_local": [bool(is_local[i]) for i in range(NUM_LAYERS)],
        "near_field_census": [[float(v) for v in row] for row in near_census],
        "near_field_peak_position": [int(p) for p in peak_pos],
        "local_vs_global": lg,
    }
    out_path = os.path.join(OUT_DIR, "spectrum_depth.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print("Written", out_path)

    # --- item 4: plots ---
    ink, muted = "#1a1a2e", "#6b7280"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.grid": False, "font.size": 9})

    # (a) spectrum-vs-depth heatmap (log10 magnitude, single-hue sequential)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(np.log10(spectra + 1e-12).T, aspect="auto", cmap="Blues",
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("layer"); ax.set_ylabel("mode index")
    ax.set_title("Singular spectrum vs depth (log10 S)")
    for g in global_layers:
        ax.axvline(g, color="#b45309", lw=0.6, alpha=0.5)
    fig.colorbar(im, ax=ax, label="log10 singular value")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "spectrum_depth_heatmap.png"), dpi=150)
    plt.close(fig)

    # (b) participation ratio vs depth, global layers marked
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(range(NUM_LAYERS), prs, color="#2563eb", lw=2, zorder=2)
    gx = global_layers
    ax.scatter(gx, prs[gx], marker="D", s=36, facecolor="#b45309",
               edgecolor="white", linewidth=1, zorder=3, label="global layer")
    ax.set_xlabel("layer"); ax.set_ylabel("participation ratio")
    ax.set_title("Effective rank (participation ratio) vs depth")
    ax.grid(True, color="#e5e7eb", lw=0.5, zorder=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "participation_ratio_depth.png"), dpi=150)
    plt.close(fig)

    # (c) near-field mean curves, 11x6 small multiples
    fig, axes = plt.subplots(11, 6, figsize=(12, 16), sharex=True)
    ymax = near_census.max()
    for i in range(NUM_LAYERS):
        ax = axes[i // 6, i % 6]
        col = "#2563eb" if is_local[i] else "#b45309"
        ax.plot(range(32), near_census[i], color=col, lw=1.2)
        ax.set_ylim(0, ymax * 1.05)
        ax.set_title(f"L{i} ({'loc' if is_local[i] else 'GLO'})",
                     fontsize=7, color=ink)
        ax.tick_params(labelsize=6)
    fig.suptitle("Near-field mean |mode curve|, d in [0,32) -- blue=local, orange=global",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(FIG_DIR, "near_field_small_multiples.png"), dpi=150)
    plt.close(fig)
    print("Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
