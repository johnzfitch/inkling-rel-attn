"""Corrected exploration batch (v2) after the first analyst's audit.

Fixes relative to the original q_*.py (kept in place for provenance):
- BASELINE NLL is reconstructed as nll - delta_nll (the arm npz `nll` field is
  the INTERVENED readout; the original scripts misused it as baseline).
- Q5 sparsity is reported against POSITIVE GROSS damage, not net.
- Per-text costs are read from r5d_results.json, not hard-coded.
- Class tables are actually emitted, with query-side semantics stated.
- Adds: the auditor's L29 contrastive-stencil check, near-off band masses,
  and a code-level resolution of the rising-arm semantics question.
- Inputs are hash-stamped; output refuses overwrite.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARMS = ROOT / "dumps" / "round5" / "r5d" / "arms"
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
RESULTS = ROOT / "analysis" / "round5" / "r5d" / "r5d_results.json"
OUT = Path(__file__).with_suffix(".out.json")
if OUT.exists():
    raise FileExistsError(OUT)
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
out = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": {}}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tok(arm, text):
    path = ARMS / arm / "tokens" / f"{text}.npz"
    out["input_sha256"][f"{arm}/{text}"] = sha(path)
    with np.load(path, allow_pickle=False) as z:
        d = np.asarray(z["delta_nll"], dtype=np.float64)
        nll_arm = np.asarray(z["nll"], dtype=np.float64)
    return d, nll_arm - d          # (delta, TRUE baseline)


results = json.loads(RESULTS.read_text(encoding="utf-8"))
costs = {t: results["arm_summaries"]["bias_off_L29"]["per_text_mean_delta_nll"][t] for t in TEXTS}

# ---- Q4/Q22 corrected: surprisal gating with true baseline
rows = {}
for t in TEXTS:
    d, base = tok("bias_off_L29", t)
    rows[t] = {"pearson_dnll_vs_true_baseline_nll": float(np.corrcoef(d, base)[0, 1])}
out["q4_q22_corrected_surprisal"] = rows

# ---- Q5 corrected: shares of positive gross damage
pooled = np.concatenate([tok("bias_off_L29", t)[0] for t in TEXTS])
pos = pooled[pooled > 0]
gross = pos.sum()
srt = np.sort(pooled)[::-1]
n = len(pooled)
out["q5_corrected"] = {
    "positive_gross": float(gross),
    "net": float(pooled.sum()),
    "share_of_gross_top_0.1pct": float(srt[: max(1, n // 1000)].sum() / gross),
    "share_of_gross_top_1pct": float(srt[: n // 100].sum() / gross),
    "share_of_gross_top_10pct": float(srt[: n // 10].sum() / gross),
    "frac_tokens_hurt": float((pooled > 0).mean()),
}

# ---- Q9 corrected: beneficiary difficulty with true baseline
rows = {}
for arm in ("far_off_L29", "bias_off_L65"):
    per = {}
    for t in TEXTS:
        d, base = tok(arm, t)
        neg = d < -0.05
        per[t] = {
            "n": int(neg.sum()),
            "mean_true_baseline_nll_beneficiaries": float(base[neg].mean()) if neg.any() else None,
            "mean_true_baseline_nll_all": float(base.mean()),
        }
    rows[arm] = per
out["q9_corrected"] = rows

# ---- Q14 corrected: clock joint vs true baseline difficulty
rows = {}
for t in TEXTS:
    d, base = tok("clock_freeze_L53_L59", t)
    rows[t] = {"corr_dnll_true_baseline": float(np.corrcoef(d, base)[0, 1])}
out["q14_corrected"] = rows

# ---- Auditor's stencil hypothesis: realized L29 kernel at d=0 vs d=1..3
proj = np.load(ROOT / "weights" / "layer29_rel_logits_proj.npy").astype(np.float64)
out["input_sha256"]["layer29_rel_logits_proj"] = sha(ROOT / "weights" / "layer29_rel_logits_proj.npy")
stencil = {}
contrasts, cost_vec = [], []
for t in TEXTS:
    r = np.load(CAP / "replay" / f"rvec_L29_{t}.npy").astype(np.float32).reshape(8192, 64, 16).astype(np.float64)
    k = r.mean(0) @ proj                       # 64 x 1024 mean realized kernel
    d0 = float(k[:, 0].mean())
    d13 = float(k[:, 1:4].mean())
    stencil[t] = {"k_d0_headmean": d0, "k_d1_3_headmean": d13, "contrast": d13 - d0}
    contrasts.append(d13 - d0)
    cost_vec.append(costs[t])
out["stencil_L29"] = {
    "per_text": stencil,
    "corr_contrast_vs_cost_n6": float(np.corrcoef(contrasts, cost_vec)[0, 1]),
}

# ---- Q8 addendum: near-off band masses (the mediation check)
def bands(mass):
    p = mass / mass.sum(1, keepdims=True)
    return {"d_lt4": float(p[:, :4].sum(1).mean()), "d_4_128": float(p[:, 4:129].sum(1).mean())}

rows = {}
for t in TEXTS:
    with np.load(CAP / "meters" / f"layer29_{t}_s8192.npz") as z:
        base = bands(z["mass_with"])
    with np.load(ARMS / "near_off_L29" / "meters" / f"L29_{t}.npz") as z:
        near = bands(z["mass_with"])
    rows[t] = {"baseline_4_128": base["d_4_128"], "near_off_4_128": near["d_4_128"],
               "baseline_lt4": base["d_lt4"], "near_off_lt4": near["d_lt4"]}
out["q8_near_off_bands"] = rows

# ---- Rising-arm semantics: code-level record
out["rising_arm_semantics"] = {
    "runner_intervention": "intervene_bias(kind='rising_heads_off') returns torch.zeros_like(bias) "
                           "— ALL 64 heads zeroed at each of L0..L4 (round5_r5d_runner.py)",
    "registration": "ROUND5_R5D_EXECUTION_AMENDMENT.md: 'all 64 heads at each of L0--L4 are rising'",
    "conclusion": "joint arm is semantically identical to simultaneous all-head bias-off at L0..L4; "
                  "the superadditivity comparison against the five singles is valid",
    "joint_cost": float(results["arm_summaries"]["rising_heads_off_L00_L04"]["pooled_mean_delta_nll"]),
    "sum_of_singles": float(sum(
        results["arm_summaries"][f"bias_off_L{L:02d}"]["pooled_mean_delta_nll"] for L in range(5))),
}

# ---- Q23 corrected: class tables actually emitted, query-side semantics
loci = json.loads((ROOT / "analysis" / "round5" / "loci.json").read_text(encoding="utf-8"))
tables = {}
for text in ("01_prose_en",):
    entry = loci["texts"][text]
    groups = {}
    for name, val in list(entry["loci"].items()) + list(entry["classes"].items()):
        if isinstance(val, list) and val and isinstance(val[0], dict) and "token_pos" in val[0]:
            groups[name] = [p["token_pos"] for p in val]
    d, _ = tok("bias_off_L29", text)
    table = {"__all__": {"n": len(d), "mean_dnll": float(d.mean())}}
    for name, pos in groups.items():
        pos = [p for p in pos if 0 <= p < len(d)]
        if len(pos) >= 8:
            table[name] = {"n": len(pos), "mean_dnll_prediction_FROM_class_token": float(d[pos].mean())}
    tables[text] = table
out["q23_tables"] = tables
out["q23_semantics_note"] = ("delta index p scores the prediction MADE FROM the class token "
                             "(target p+1), per the runner's alignment")

OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: out[k] for k in ("q4_q22_corrected_surprisal", "q5_corrected", "stencil_L29",
                                      "q8_near_off_bands", "rising_arm_semantics")}, indent=1))
