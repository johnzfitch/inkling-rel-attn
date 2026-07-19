"""Post-verification raw-dump pattern exploration for the follow-up campaign.

EXPLORATORY — nothing here is a registered verdict. Reads only sealed dumps
and frozen inputs; input files are hash-stamped; output refuses overwrite.
No decoded token text is emitted (corpus privacy).
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DUMP = ROOT / "dumps" / "round5" / "followup7"
PARENT = ROOT / "dumps" / "round5" / "r5d" / "arms"
FROZEN = ROOT / "analysis" / "round5" / "followup7" / "frozen_inputs.npz"
OUT = Path(__file__).with_suffix(".out.json")
if OUT.exists():
    raise FileExistsError(OUT)

TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
out = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
       "note": "exploratory pattern pass; no registered claims", "input_sha256": {}}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tok(arm: str, text: str, fields=("delta_nll",)) -> dict:
    path = DUMP / "arms" / arm / "tokens" / f"{text}.npz"
    key = f"{arm}/{text}"
    if key not in out["input_sha256"]:
        out["input_sha256"][key] = sha(path)
    with np.load(path, allow_pickle=False) as z:
        return {f: np.asarray(z[f], dtype=np.float64) for f in fields}


def ptok(arm: str, text: str) -> np.ndarray:
    path = PARENT / arm / "tokens" / f"{text}.npz"
    key = f"parent:{arm}/{text}"
    if key not in out["input_sha256"]:
        out["input_sha256"][key] = sha(path)
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["delta_nll"], dtype=np.float64)


frozen = {k: np.array(v) for k, v in np.load(FROZEN, allow_pickle=False).items()}
out["input_sha256"]["frozen_inputs"] = sha(FROZEN)

# ---- P1: anatomy of d0 dominance -------------------------------------------
p1 = {}
for t in TEXTS:
    d0 = tok("d0_off_L29", t)["delta_nll"]
    full = ptok("bias_off_L29", t)
    rest = tok("restore_d0_L29", t)["delta_nll"]
    sten = tok("stencil_only_d0_3_L29", t)["delta_nll"]
    p1[t] = {
        "corr_d0off_vs_biasoff": float(np.corrcoef(d0, full)[0, 1]),
        "slope_d0off_on_biasoff": float((d0 * full).sum() / (full * full).sum()),
        "mean_d0off": float(d0.mean()), "mean_biasoff": float(full.mean()),
        "mean_restore_d0": float(rest.mean()), "mean_stencil_only": float(sten.mean()),
        "corr_residual_vs_restore_d0": float(np.corrcoef(full - d0, rest)[0, 1]),
    }
out["p1_d0_anatomy"] = p1

# ---- P2: head stencil-score structure ---------------------------------------
scores = frozen["head_stencil_score_median"].astype(np.float64)
order = frozen["head_order"].astype(int)
p2 = {
    "sorted_scores": [round(float(scores[h]), 4) for h in order],
    "q1_heads": frozen["head_q1"].astype(int).tolist(),
    "top8_heads": frozen["head_top08"].astype(int).tolist(),
    "score_top16_mean": float(scores[order[:16]].mean()),
    "score_bottom48_mean": float(scores[order[16:]].mean()),
    "per_text_score_corr": np.corrcoef(frozen["head_stencil_score_by_text"].astype(np.float64)).round(3).tolist(),
}
out["p2_head_structure"] = p2

# ---- P3: full-vocabulary anatomy of bias-off --------------------------------
fields = ("delta_nll", "delta_target_rank", "delta_log1p_target_rank", "delta_top1_correct",
          "delta_entropy", "delta_target_margin", "target_rank", "top1_correct")
fv = {t: tok("bias_off_L29_fullvocab", t, fields) for t in TEXTS}
base_rank = {t: fv[t]["target_rank"] - fv[t]["delta_target_rank"] for t in TEXTS}
dr = np.concatenate([fv[t]["delta_target_rank"] for t in TEXTS])
p3 = {
    "delta_rank_quantiles": {q: float(np.quantile(dr, float(q))) for q in
                             ("0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "0.999")},
    "frac_rank_worse": float((dr > 0).mean()), "frac_rank_better": float((dr < 0).mean()),
    "mean_delta_rank": float(dr.mean()),
    "share_of_mean_from_top_0.1pct": float(np.sort(dr)[-len(dr) // 1000:].sum() / dr.sum()),
    "top1_flip_rate": float(np.concatenate(
        [np.abs(fv[t]["delta_top1_correct"]) for t in TEXTS]).mean()),
    "per_text": {},
}
for t in TEXTS:
    v = fv[t]
    p3["per_text"][t] = {
        "acc_base": float((v["top1_correct"] - v["delta_top1_correct"]).mean()),
        "acc_arm": float(v["top1_correct"].mean()),
        "mean_delta_rank": float(v["delta_target_rank"].mean()),
        "median_delta_rank": float(np.median(v["delta_target_rank"])),
        "mean_delta_entropy": float(v["delta_entropy"].mean()),
        "mean_delta_margin": float(v["delta_target_margin"].mean()),
    }
# where do catastrophic rank blowups live?
blow_positions = []
for t in TEXTS:
    idx = np.flatnonzero(fv[t]["delta_target_rank"] >= 1000)
    blow_positions.append({"text": t, "n": int(idx.size),
                           "median_baseline_rank_there": float(np.median(base_rank[t][idx])) if idx.size else None,
                           "frac_in_first_512": float((idx < 512).mean()) if idx.size else None})
p3["rank_blowups_ge_1000"] = blow_positions
out["p3_fullvocab_anatomy"] = p3

# ---- P4: the triple knockout ------------------------------------------------
p4 = {}
trip_all = []
for t in TEXTS:
    trip = tok("bias_off_L23_L29_L35", t)["delta_nll"]
    single = ptok("bias_off_L29", t)
    trip_all.append(trip)
    p4[t] = {"mean": float(trip.mean()),
             "frac_gt_1nat": float((trip > 1).mean()),
             "corr_with_L29_single": float(np.corrcoef(trip, single)[0, 1])}
trip_pool = np.concatenate(trip_all)
p4["pooled"] = {"mean": float(trip_pool.mean()),
                "median": float(np.median(trip_pool)),
                "frac_gt_1nat": float((trip_pool > 1).mean()),
                "q95": float(np.quantile(trip_pool, 0.95))}
out["p4_triple_knockout"] = p4

# ---- P5: clock geometry — shared vs private ---------------------------------
p5 = {}
for L in (53, 59):
    G = np.stack([frozen[f"clock_g_L{L:02d}_{t}"].astype(np.float64) for t in TEXTS])
    cosmat = (G @ G.T).round(3)
    proj_onto_loto = {}
    for i, t in enumerate(TEXTS):
        basis = frozen[f"clock_loto_L{L:02d}_{t}"].astype(np.float64)
        coef = basis.T @ G[i]
        proj_onto_loto[t] = float(np.sqrt((coef ** 2).sum()))
    sv = np.linalg.svd(G, compute_uv=False)
    p5[f"L{L}"] = {"pairwise_cos": cosmat.tolist(),
                   "mean_offdiag_cos": float((cosmat.sum() - 6) / 30),
                   "sv_energy_share": (sv ** 2 / (sv ** 2).sum()).round(3).tolist(),
                   "heldout_dir_share_inside_other5_span": proj_onto_loto}
out["p5_clock_geometry"] = p5

# ---- P6: needle recall across the new arms ----------------------------------
p6 = {}
needle_arms = ["d0_off_L29", "d1_3_off_L29", "stencil_only_d0_3_L29", "restore_d0_L29",
               "head_q1_off_L29", "head_top16_stencil_only_L29", "r_remove_mean_L29",
               "r_remove_centered_L29", "bias_off_L23_L29_L35", "clock_loto_L53_L59"]
for arm in needle_arms:
    path = DUMP / "arms" / arm / "needle" / "05_needles.npz"
    out["input_sha256"][f"{arm}/needle"] = sha(path)
    with np.load(path, allow_pickle=False) as z:
        d = np.asarray(z["delta_nll"], dtype=np.float64)
        side = np.asarray(z["side_of_seam"])
    p6[arm] = {"mean_recall_delta": float(d.mean()),
               "n_hurt_gt_0.1": int((d > 0.1).sum()),
               "below_seam_mean": float(d[side == "below"].mean()),
               "above_seam_mean": float(d[side == "above"].mean())}
out["p6_needle_recall"] = p6

# ---- P7: per-query patch anatomy ---------------------------------------------
queries = frozen["patch_query_positions"].astype(np.int64)
parent_needles = ptok("bias_off_L29", "05_needles")[queries]
qp = tok("bias_off_L29_patch_query", "05_needles")["delta_nll"][queries]
sp = tok("bias_off_L29_patch_sham", "05_needles")["delta_nll"][queries]
p7 = {"per_query": [{"parent": round(float(a), 4), "patched": round(float(b), 4),
                     "sham": round(float(c), 4)} for a, b, c in zip(parent_needles, qp, sp)],
      "n_queries_fully_rescued_lt_0.05": int((qp < 0.05).sum()),
      "n_parent_hurt_gt_0.1": int((parent_needles > 0.1).sum()),
      "median_rescue_fraction_hurt_queries": float(np.median(
          1 - qp[parent_needles > 0.1] / parent_needles[parent_needles > 0.1]))}
out["p7_patch_anatomy"] = p7

# ---- P8: fresh-text generalization of the L29 effect --------------------------
p8 = {}
for text in ("07b_slack_multi", "08_math_llm"):
    path = DUMP / "fresh" / "bias_off_L29" / f"{text}.npz"
    out["input_sha256"][f"fresh/{text}"] = sha(path)
    with np.load(path, allow_pickle=False) as z:
        d = np.asarray(z["delta_nll"], dtype=np.float64)
        rank_delta = np.asarray(z["delta_target_rank"], dtype=np.float64)
        acc_delta = np.asarray(z["delta_top1_correct"], dtype=np.float64)
    p8[text] = {"mean_delta_nll": float(d.mean()),
                "median_delta_nll": float(np.median(d)),
                "frac_hurt": float((d > 0).mean()),
                "share_gross_top1pct": float(np.sort(d)[-len(d) // 100:].sum() / d[d > 0].sum()),
                "mean_delta_rank": float(rank_delta.mean()),
                "delta_accuracy": float(acc_delta.mean())}
out["p8_fresh_generalization"] = p8

# ---- P9: is follow-up damage still difficulty-agnostic? -----------------------
p9 = {}
for t in ("01_prose_en", "02_code"):
    v = tok("bias_off_L23_L29_L35", t, ("delta_nll", "nll"))
    base = v["nll"] - v["delta_nll"]
    p9[f"triple:{t}"] = float(np.corrcoef(v["delta_nll"], base)[0, 1])
    w = tok("d0_off_L29", t, ("delta_nll", "nll"))
    p9[f"d0_off:{t}"] = float(np.corrcoef(w["delta_nll"], w["nll"] - w["delta_nll"])[0, 1])
out["p9_difficulty_agnostic"] = p9

OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
compact = {k: out[k] for k in out if k.startswith("p")}
print(json.dumps(compact, indent=1))
