"""
E4 -- Extend the relative-attention dump to the 8 MTP drafter layers, then run
the E1 model-selection fit and E3 items 1+3 (spectrum/participation-ratio and
near-field census) on them.

Tensor names (confirmed against the safetensors index, all in mtp.safetensors):
  model.mtp.layers.N.transformer_block.attn.wr_du.weight
  model.mtp.layers.N.transformer_block.attn.rel_logits_proj.proj      N in 0..7
mtp_config.local_layer_ids = [0, 2, 4, 5, 6, 7].

Fetch via HTTP range requests (helpers reused from extract_rel_attn.py) into
R:\\inkling\\weights\\mtp\\ (idempotent: skips files already on disk).

Question: does the drafter learn the same transport as the trunk, or a
cheaper one?  Output: analysis/round2/mtp_spectrum.json
"""
import json
import os

import numpy as np

from extract_rel_attn import fetch_tensor, load_weight_map
from fit_transport_models import NEAR, derived_scales, fit_far_field
from depth_and_extent import participation_ratio

MTP_DIR = r"R:\inkling\weights\mtp"
OUT_DIR = r"R:\inkling\analysis\round2"
TRUNK_FITS = r"R:\inkling\analysis\round2\transport_fits.json"
NUM_MTP = 8
LOCAL_IDS = {0, 2, 4, 5, 6, 7}
MODE_THRESH = 0.05


def fetch_all():
    missing = any(
        not (
            os.path.exists(os.path.join(MTP_DIR, f"mtp{i}_wr_du.npy"))
            and os.path.exists(os.path.join(MTP_DIR, f"mtp{i}_rel_logits_proj.npy"))
        )
        for i in range(NUM_MTP)
    )
    weight_map = load_weight_map() if missing else None
    os.makedirs(MTP_DIR, exist_ok=True)
    meta = {}
    for i in range(NUM_MTP):
        wr_key = f"model.mtp.layers.{i}.transformer_block.attn.wr_du.weight"
        proj_key = f"model.mtp.layers.{i}.transformer_block.attn.rel_logits_proj.proj"
        wr_path = os.path.join(MTP_DIR, f"mtp{i}_wr_du.npy")
        proj_path = os.path.join(MTP_DIR, f"mtp{i}_rel_logits_proj.npy")
        if os.path.exists(wr_path) and os.path.exists(proj_path):
            wr = np.load(wr_path)
            proj = np.load(proj_path)
            print(f"mtp layer {i}: already on disk, skipping fetch")
        else:
            assert weight_map is not None
            wr = fetch_tensor(weight_map[wr_key], wr_key)
            proj = fetch_tensor(weight_map[proj_key], proj_key)
            np.save(wr_path, wr)
            np.save(proj_path, proj)
            print(f"mtp layer {i} ({'local' if i in LOCAL_IDS else 'global'}): "
                  f"fetched wr_du {wr.shape}, proj {proj.shape}")
        meta[i] = {"is_local": i in LOCAL_IDS, "extent": int(proj.shape[1]),
                   "wr_du_shape": [int(x) for x in wr.shape],
                   "proj_shape": [int(x) for x in proj.shape]}
    json.dump(meta, open(os.path.join(MTP_DIR, "_meta.json"), "w"), indent=2)
    return meta


def analyze(meta):
    results = {}
    for i in range(NUM_MTP):
        proj = np.load(os.path.join(MTP_DIR, f"mtp{i}_rel_logits_proj.npy"))
        extent = proj.shape[1]
        U, S, Vt = np.linalg.svd(proj, full_matrices=False)
        mode_curves = S[:, None] * Vt
        near_census = np.abs(mode_curves[:, :32]).mean(axis=0)

        modes = {}
        for k in range(len(S)):
            if S[k] < MODE_THRESH * S[0]:
                continue
            y = mode_curves[k].astype(np.float64)
            if y[:NEAR].mean() < 0:
                y = -y
            d = np.arange(NEAR, extent, dtype=np.float64)
            fits = fit_far_field(d, y[NEAR:])
            fitted = {f: r for f, r in fits.items() if r["ok"]}
            if not fitted:
                raise RuntimeError(f"MTP layer {i} mode {k}: all model families failed")
            winner = min(fitted, key=lambda f: fitted[f]["bic"])
            amp0, d_half = derived_scales(winner, fitted[winner]["params"], extent)
            modes[str(k)] = {
                "singular_value": float(S[k]),
                "near_field": [float(v) for v in y[:NEAR]],
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

        pr = participation_ratio(S)
        results[str(i)] = {
            "is_local": bool(meta[i]["is_local"]),
            "extent": int(extent),
            "singular_values": [float(s) for s in S],
            "participation_ratio": float(pr),
            "near_field_census": [float(v) for v in near_census],
            "near_field_peak_position": int(np.argmax(near_census)),
            "modes": modes,
        }
        m0 = modes["0"]
        dh = f"{m0['d_half']:.1f}" if m0["d_half"] is not None else "None"
        print(f"mtp layer {i} ({'local' if meta[i]['is_local'] else 'global'}, "
              f"extent={extent}): S0={S[0]:.3f} PR={pr:.2f} "
              f"near-peak d={int(np.argmax(near_census))} | mode0 winner={m0['winner']} "
              f"r2={m0['winner_r2']:.3f} d_half={dh}")
    return results


def compare_with_trunk(results):
    """Qualitative drafter-vs-trunk comparison, printed for the record."""
    if not os.path.exists(TRUNK_FITS):
        print("trunk transport_fits.json not found; skipping comparison")
        return {}
    trunk = json.load(open(TRUNK_FITS))
    t_winners = [trunk[str(i)]["modes"]["0"]["winner"] for i in range(66)]
    t_expish = sum(1 for w in t_winners if w in ("exp", "exp2"))
    t_pr = [participation_ratio(np.array(trunk[str(i)]["singular_values"]))
            for i in range(66)]
    m_winners = [results[str(i)]["modes"]["0"]["winner"] for i in range(NUM_MTP)]
    m_expish = sum(1 for w in m_winners if w in ("exp", "exp2"))
    m_pr = [results[str(i)]["participation_ratio"] for i in range(NUM_MTP)]
    m_peaks = [results[str(i)]["near_field_peak_position"] for i in range(NUM_MTP)]
    print(f"\ntrunk:   mode-0 exp/exp2 winners {t_expish}/66, "
          f"PR range [{min(t_pr):.2f}, {max(t_pr):.2f}]")
    print(f"drafter: mode-0 exp/exp2 winners {m_expish}/{NUM_MTP}, "
          f"PR range [{min(m_pr):.2f}, {max(m_pr):.2f}], "
          f"near-field peaks at d={m_peaks}")
    return {
        "trunk_mode0_expish_frac": float(t_expish / 66),
        "mtp_mode0_expish_frac": float(m_expish / NUM_MTP),
        "trunk_pr_min_max": [float(min(t_pr)), float(max(t_pr))],
        "mtp_pr_min_max": [float(min(m_pr)), float(max(m_pr))],
        "mtp_mode0_winners": m_winners,
        "mtp_near_field_peaks": [int(p) for p in m_peaks],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = fetch_all()
    results = analyze(meta)
    comparison = compare_with_trunk(results)
    out = {"layers": results, "trunk_comparison": comparison}
    out_path = os.path.join(OUT_DIR, "mtp_spectrum.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print("Written", out_path)


if __name__ == "__main__":
    main()
