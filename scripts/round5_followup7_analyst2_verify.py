"""Second-analyst independent verification of the seven-experiment follow-up.

Written from the registration (`ROUND5_FOLLOWUP7_PREREG.md`) and the sealed raw
dumps only. Shares no code with `round5_followup7_analyze.py` or
`round5_followup7_verify.py` (neither is imported); every statistic, bootstrap,
kernel correlation, matched-class contrast, and verdict is re-derived here from
first principles. Seeds are regenerated from the registration file's byte hash
per the registered rule so the producer's resampling is reproduced exactly.

Refuses to overwrite its output.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "dumps" / "round5" / "followup7"
PARENT = ROOT / "dumps" / "round5" / "r5d" / "arms"
REG = ROOT / "registrations" / "ROUND5_FOLLOWUP7_PREREG.md"
FROZEN = ROOT / "analysis" / "round5" / "followup7" / "frozen_inputs.npz"
RESULTS = ROOT / "analysis" / "round5" / "followup7" / "results.json"
OUT = ROOT / "analysis" / "round5" / "followup7" / "analyst2_verification.json"
if OUT.exists():
    raise FileExistsError(OUT)

TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
N_BOOT = 5000
BLOCK = 256
errors: list[str] = []
report: dict = {"kind": "round5_followup7_second_analyst_verification",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "errors": errors}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        errors.append(f"{name}: {detail}")


# ---------------------------------------------------------------- seal integrity
REG_SHA = sha256_file(REG)
results = json.loads(RESULTS.read_text(encoding="utf-8"))
check("registration_sha", REG_SHA == results["registration_sha256"],
      f"{REG_SHA} != {results['registration_sha256']}")

rehashed = 0
manifest_roots = [DUMP / "baseline_fullvocab", DUMP / "fresh"] + sorted((DUMP / "arms").iterdir())
for root in manifest_roots:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    check(f"manifest_complete:{root.name}", manifest.get("complete") is True)
    for record in manifest["artifacts"]:
        path = root / record["path"]
        ok = path.is_file() and path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"]
        check(f"artifact:{root.name}/{record['path']}", ok)
        rehashed += 1
report["artifact_count_rehashed"] = rehashed
check("artifact_count", rehashed == results["dump_provenance"]["artifact_count"],
      f"{rehashed} != {results['dump_provenance']['artifact_count']}")


def seed(label: str) -> int:
    digest = hashlib.sha256(f"{REG_SHA}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def tokens(arm: str, text: str) -> dict:
    with np.load(DUMP / "arms" / arm / "tokens" / f"{text}.npz", allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def parent_delta(arm: str, text: str) -> np.ndarray:
    with np.load(PARENT / arm / "tokens" / f"{text}.npz", allow_pickle=False) as z:
        return np.asarray(z["delta_nll"], dtype=np.float64)


def pooled(arm: str) -> tuple[float, list[np.ndarray]]:
    arrays = [tokens(arm, t)["delta_nll"].astype(np.float64) for t in TEXTS]
    return float(np.concatenate(arrays).mean()), arrays


def blocks_of(arrays: list[np.ndarray]) -> np.ndarray:
    rows = []
    for values in arrays:
        values = np.asarray(values, dtype=np.float64)
        assert values.shape == (8191,)
        for start in range(0, 8191, BLOCK):
            rows.append(values[start:start + BLOCK].mean())
    return np.asarray(rows, dtype=np.float64)


def infer(blocks: np.ndarray, label: str) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed(label)))
    n = blocks.size
    idx = rng.integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    draws = blocks[idx].mean(axis=1)
    signs = rng.integers(0, 2, size=(N_BOOT, n), dtype=np.int8) * 2 - 1
    null = (blocks[None, :] * signs).mean(axis=1)
    obs = float(blocks.mean())
    return {
        "effect": obs,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "p_signflip_positive": float((1 + np.count_nonzero(null >= obs)) / (N_BOOT + 1)),
        "p_signflip_negative": float((1 + np.count_nonzero(null <= obs)) / (N_BOOT + 1)),
        "n_blocks": int(n),
    }


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    out = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        out[index] = min(1.0, running)
    return out.tolist()


def close(a: float, b: float, tol: float = 2e-10) -> bool:
    return abs(a - b) <= tol


def match_record(name: str, mine: dict, theirs: dict) -> None:
    for key in ("effect", "p_signflip_positive", "p_signflip_negative"):
        check(f"{name}:{key}", close(mine[key], theirs[key]), f"{mine[key]} != {theirs[key]}")
    for i in (0, 1):
        check(f"{name}:ci{i}", close(mine["ci95"][i], theirs["ci95"][i]),
              f"{mine['ci95'][i]} != {theirs['ci95'][i]}")


frozen = {k: np.array(v) for k, v in np.load(FROZEN, allow_pickle=False).items()}
check("frozen_inputs_sha", sha256_file(FROZEN) == results["frozen_inputs_sha256"])

BIAS_OFF_PARENT = float(np.concatenate([parent_delta("bias_off_L29", t) for t in TEXTS]).mean())
check("f71:parent_cost", close(BIAS_OFF_PARENT, results["families"]["F7-1"]["certified_bias_off_cost"]))

# ------------------------------------------------------------ exact-copy gate
for t in TEXTS:
    mine = tokens("bias_off_L29_fullvocab", t)
    with np.load(PARENT / "bias_off_L29" / "tokens" / f"{t}.npz", allow_pickle=False) as z:
        for field in ("target_logit", "log_normalizer", "nll", "probability", "delta_nll"):
            if field in z.files:
                check(f"fullvocab_exact_copy:{t}:{field}",
                      np.array_equal(mine[field], np.asarray(z[field])))

# ---------------------------------------------------------------------- F7-1
f1 = results["families"]["F7-1"]
my_costs = {}
single_rows = {}
for d in range(4):
    arm = f"d{d}_off_L29"
    cost, arrays = pooled(arm)
    my_costs[arm] = cost
    single_rows[arm] = infer(blocks_of(arrays), f"F7-1:{arm}")
for arm in ("d1_3_off_L29", "restore_d0_L29", "restore_d1_3_L29", "stencil_only_d0_3_L29"):
    my_costs[arm] = pooled(arm)[0]
for arm, cost in my_costs.items():
    check(f"f71:cost:{arm}", close(cost, f1["costs"][arm]), f"{cost} != {f1['costs'][arm]}")
adj = holm([single_rows[f"d{d}_off_L29"]["p_signflip_positive"] for d in range(4)])
confirmed = []
for d, p_h in zip(range(4), adj):
    arm = f"d{d}_off_L29"
    row = single_rows[arm]
    match_record(f"f71:{arm}", row, f1["singleton_inference"][arm])
    check(f"f71:{arm}:holm", close(p_h, f1["singleton_inference"][arm]["p_holm_positive"]))
    confirmed.append(row["effect"] > 0 and row["ci95"][0] > 0 and p_h < 0.05)
rescue1 = 1.0 - my_costs["stencil_only_d0_3_L29"] / BIAS_OFF_PARENT
check("f71:rescue", close(rescue1, f1["stencil_rescue_fraction"]))
my_f1_pass = any(confirmed) and rescue1 >= 0.50
check("f71:verdict", my_f1_pass == f1["passed"])
report["F7-1"] = {"passed": bool(my_f1_pass), "rescue": rescue1, "costs": my_costs}

# ---------------------------------------------------------------------- F7-2
f2 = results["families"]["F7-2"]
spec = {"adjacent_23_29": ("bias_off_L23_L29", (23, 29)),
        "adjacent_29_35": ("bias_off_L29_L35", (29, 35)),
        "triple": ("bias_off_L23_L29_L35", (23, 29, 35)),
        "control_23_35": ("bias_off_L23_L35", (23, 35))}
rows2 = {}
for label, (joint, layers) in spec.items():
    arrays = []
    for t in TEXTS:
        effect = tokens(joint, t)["delta_nll"].astype(np.float64)
        for layer in layers:
            effect = effect - parent_delta(f"bias_off_L{layer:02d}", t)
        arrays.append(effect)
    rows2[label] = infer(blocks_of(arrays), f"F7-2:{label}")
    match_record(f"f72:{label}", rows2[label], f2["interactions"][label])
adj = holm([rows2[k]["p_signflip_positive"] for k in ("adjacent_23_29", "adjacent_29_35", "triple")])
my_f2_pass = all(
    rows2[k]["effect"] > 0 and rows2[k]["ci95"][0] > 0 and p_h < 0.05
    for k, p_h in zip(("adjacent_23_29", "adjacent_29_35", "triple"), adj))
check("f72:verdict", my_f2_pass == f2["passed"])
report["F7-2"] = {"passed": bool(my_f2_pass),
                  "interactions": {k: rows2[k]["effect"] for k in rows2}}

# ---------------------------------------------------------------------- F7-3
f3 = results["families"]["F7-3"]
names3 = ["r_remove_mean_L29", "r_remove_centered_L29", "r_remove_carrier_all_L29",
          "r_remove_noncarrier_all_L29", "r_remove_carrier_mean_L29", "r_remove_noncarrier_mean_L29"]
costs3 = {n: pooled(n)[0] for n in names3}
for n in names3:
    check(f"f73:cost:{n}", close(costs3[n], f3["costs"][n]))
    check(f"f73:ratio:{n}", close(costs3[n] / BIAS_OFF_PARENT, f3["ratios_to_bias_off"][n]))
my_f3_pass = (costs3["r_remove_mean_L29"] / BIAS_OFF_PARENT >= 0.5
              and costs3["r_remove_noncarrier_mean_L29"] / BIAS_OFF_PARENT >= 0.5
              and costs3["r_remove_centered_L29"] / BIAS_OFF_PARENT <= 0.25
              and abs(costs3["r_remove_carrier_mean_L29"]) < 0.005)
check("f73:verdict", my_f3_pass == f3["passed"])
report["F7-3"] = {"passed": bool(my_f3_pass), "ratios": {n: costs3[n] / BIAS_OFF_PARENT for n in names3}}

# ---------------------------------------------------------------------- F7-4
f4 = results["families"]["F7-4"]
names4 = ([f"head_q{q}_off_L29" for q in range(1, 5)]
          + ["head_top16_only_L29", "head_top08_stencil_only_L29",
             "head_top16_stencil_only_L29", "head_top32_stencil_only_L29"])
costs4 = {n: pooled(n)[0] for n in names4}
for n in names4:
    check(f"f74:cost:{n}", close(costs4[n], f4["costs"][n]))
rows4 = {}
for other in ("head_q2_off_L29", "head_q3_off_L29", "head_q4_off_L29"):
    arrays = [tokens("head_q1_off_L29", t)["delta_nll"].astype(np.float64)
              - tokens(other, t)["delta_nll"].astype(np.float64) for t in TEXTS]
    rows4[other] = infer(blocks_of(arrays), f"F7-4:q1-minus:{other}")
    match_record(f"f74:{other}", rows4[other], f4["q1_contrasts"][other])
adj = holm([rows4[k]["p_signflip_positive"] for k in rows4])
localized = all(rows4[k]["effect"] > 0 and rows4[k]["ci95"][0] > 0 and p_h < 0.05
                for k, p_h in zip(rows4, adj))
rescue4 = 1 - costs4["head_top16_stencil_only_L29"] / BIAS_OFF_PARENT
check("f74:rescue", close(rescue4, f4["top16_stencil_rescue_fraction"]))
my_f4_pass = localized and rescue4 >= 0.5
check("f74:verdict", my_f4_pass == f4["passed"])
report["F7-4"] = {"passed": bool(my_f4_pass), "rescue": rescue4, "costs": costs4}

# ---------------------------------------------------------------------- F7-5
f5 = results["families"]["F7-5"]
queries = frozen["patch_query_positions"].astype(np.int64)
parent5 = np.concatenate([parent_delta("bias_off_L29", "05_needles")])[queries]
rows5 = {}
for kind, arm in (("query", "bias_off_L29_patch_query"), ("sham", "bias_off_L29_patch_sham")):
    patched = tokens(arm, "05_needles")["delta_nll"].astype(np.float64)[queries]
    rng = np.random.Generator(np.random.PCG64(seed(f"F7-5:{kind}")))
    idx = rng.integers(0, queries.size, size=(N_BOOT, queries.size), dtype=np.int16)
    pd_, td_ = parent5[idx].mean(axis=1), patched[idx].mean(axis=1)
    absolute, fraction = pd_ - td_, (pd_ - td_) / pd_
    rows5[kind] = {
        "rescue_fraction": float((parent5.mean() - patched.mean()) / parent5.mean()),
        "absolute_rescue": float(parent5.mean() - patched.mean()),
        "absolute_rescue_ci95": [float(np.quantile(absolute, 0.025)), float(np.quantile(absolute, 0.975))],
        "rescue_fraction_ci95": [float(np.quantile(fraction, 0.025)), float(np.quantile(fraction, 0.975))],
    }
for kind, name in (("query", "query_patch"), ("sham", "sham_patch")):
    for key in ("rescue_fraction", "absolute_rescue"):
        check(f"f75:{kind}:{key}", close(rows5[kind][key], f5[name][key]))
    for key in ("absolute_rescue_ci95", "rescue_fraction_ci95"):
        for i in (0, 1):
            check(f"f75:{kind}:{key}:{i}", close(rows5[kind][key][i], f5[name][key][i]))
q_pass = rows5["query"]["rescue_fraction"] >= 0.5 and rows5["query"]["absolute_rescue_ci95"][0] > 0
s_pass = (rows5["sham"]["rescue_fraction"] < 0.1
          and rows5["sham"]["absolute_rescue_ci95"][0] <= 0 <= rows5["sham"]["absolute_rescue_ci95"][1])
my_f5_pass = q_pass and s_pass
check("f75:verdict", my_f5_pass == f5["passed"])
report["F7-5"] = {"passed": bool(my_f5_pass), "query": rows5["query"], "sham": rows5["sham"]}

# ---------------------------------------------------------------------- F7-6
f6 = results["families"]["F7-6"]
POSITIONS = np.arange(64, 8192, 64, dtype=np.float64)
XC = np.log1p(POSITIONS + 31.5)
XC = XC - XC.mean()
PROJ = {L: np.load(ROOT / "weights" / f"layer{L:02d}_rel_logits_proj.npy",
                   allow_pickle=False).astype(np.float64) for L in (53, 59)}


def basis_for(kind: str, layer: int, text: str) -> np.ndarray:
    if kind == "clock_union":
        b = frozen[f"clock_union_L{layer:02d}"]
    elif kind == "clock_pertext":
        b = frozen[f"clock_g_L{layer:02d}_{text}"][:, None]
    elif kind == "clock_loto":
        b = frozen[f"clock_loto_L{layer:02d}_{text}"]
    else:
        b = frozen[f"clock_sham6_L{layer:02d}"]
    return b.astype(np.float64)


def cell_stat(arm: str, kind: str, layer: int, text: str) -> float:
    r = np.load(DUMP / "arms" / arm / "clock" / f"rvec_pre_L{layer:02d}_{text}.npy",
                allow_pickle=False).astype(np.float32).reshape(8192, 1024).astype(np.float64)
    mu = frozen[f"mu_L{layer:02d}_{text}"].astype(np.float64)
    u = basis_for(kind, layer, text)
    r = r - ((r - mu) @ u) @ u.T
    blocks = r[64:].reshape(127, 64, 64, 16).mean(axis=1)
    kernels = np.einsum("bhd,de->bhe", blocks, PROJ[layer])
    mean_kernel = kernels.mean(axis=0)
    denom = np.maximum((mean_kernel * mean_kernel).sum(axis=1), 1e-30)
    gains = (kernels * mean_kernel[None]).sum(axis=2) / denom[None]
    gc = gains - gains.mean(axis=0)
    corr = (XC[:, None] * gc).sum(axis=0) / np.maximum(np.linalg.norm(XC) * np.linalg.norm(gc, axis=0), 1e-30)
    return float(np.median(np.abs(corr)))


arm_specs = {"clock_union_L53": ("clock_union", (53,)), "clock_union_L59": ("clock_union", (59,)),
             "clock_union_L53_L59": ("clock_union", (53, 59)),
             "clock_pertext_L53_L59": ("clock_pertext", (53, 59)),
             "clock_loto_L53_L59": ("clock_loto", (53, 59)),
             "clock_sham6_L53_L59": ("clock_sham6", (53, 59))}
my_cells = {}
for arm, (kind, layers) in arm_specs.items():
    cells = {f"L{L:02d}:{t}": cell_stat(arm, kind, L, t) for L in layers for t in TEXTS}
    my_cells[arm] = cells
    for key, value in cells.items():
        theirs = f6["kernel_gain_correlation"][arm]["cells"][f"L{key.split(':')[0][1:]}:{key.split(':')[1]}"] \
            if False else f6["kernel_gain_correlation"][arm]["cells"][key]
        check(f"f76:{arm}:{key}", abs(value - theirs) <= 1e-6, f"{value} != {theirs}")
costs6 = {arm: pooled(arm)[0] for arm in ("clock_union_L53_L59", "clock_pertext_L53_L59", "clock_loto_L53_L59")}
for arm, cost in costs6.items():
    check(f"f76:cost:{arm}", close(cost, f6["behavior_costs"][arm]))
loto_max = max(my_cells["clock_loto_L53_L59"].values())
sham_med = float(np.median(list(my_cells["clock_sham6_L53_L59"].values())))
my_f6_pass = (loto_max < 0.20 and sham_med >= 0.50) and all(abs(v) < 0.005 for v in costs6.values())
check("f76:verdict", my_f6_pass == f6["passed"])
report["F7-6"] = {"passed": bool(my_f6_pass), "loto_max": loto_max, "sham_median": sham_med,
                  "behavior_costs": costs6}

# ---------------------------------------------------------------------- F7-7
f7 = results["families"]["F7-7"]
arm7 = [tokens("bias_off_L29_fullvocab", t) for t in TEXTS]
base7 = []
for t in TEXTS:
    with np.load(DUMP / "baseline_fullvocab" / f"{t}.npz", allow_pickle=False) as z:
        base7.append({k: np.array(z[k]) for k in z.files})
rank_row = infer(blocks_of([v["delta_log1p_target_rank"].astype(np.float64) for v in arm7]), "F7-7:rank")
acc_row = infer(blocks_of([v["delta_top1_correct"].astype(np.float64) for v in arm7]), "F7-7:accuracy")
match_record("f77:rank", rank_row, f7["ranking"])
match_record("f77:accuracy", acc_row, f7["ranking"]["accuracy"])
ranking_pass = (rank_row["effect"] > 0 and rank_row["ci95"][0] > 0
                and acc_row["effect"] < 0 and acc_row["ci95"][1] < 0)
check("f77:ranking_verdict", ranking_pass == f7["ranking"]["passed"])


def ece_pooled(p: np.ndarray, c: np.ndarray, bins: int) -> float:
    idx = np.minimum((p * bins).astype(np.int64), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(float(c[mask].mean()) - float(p[mask].mean()))
    return total


def ece_stats(plist, clist, bins):
    rows = []
    for p, c in zip(plist, clist):
        for start in range(0, p.size, BLOCK):
            pp, cc = p[start:start + BLOCK], c[start:start + BLOCK]
            idx = np.minimum((pp * bins).astype(np.int64), bins - 1)
            row = np.zeros((bins, 3))
            for b in range(bins):
                mask = idx == b
                row[b] = (mask.sum(), cc[mask].sum(), pp[mask].sum())
            rows.append(row)
    return np.stack(rows)


def ece_of(stats):
    total = stats[:, 0].sum()
    valid = stats[:, 0] > 0
    return float(np.sum(stats[valid, 0] / total
                        * np.abs(stats[valid, 1] / stats[valid, 0] - stats[valid, 2] / stats[valid, 0])))


bp = [v["top1_probability"].astype(np.float64) for v in base7]
bc = [v["top1_correct"].astype(np.float64) for v in base7]
ap = [v["top1_probability"].astype(np.float64) for v in arm7]
ac = [v["top1_correct"].astype(np.float64) for v in arm7]
bs, as_ = ece_stats(bp, bc, 20), ece_stats(ap, ac, 20)
ece_obs = ece_of(as_.sum(axis=0)) - ece_of(bs.sum(axis=0))
rng = np.random.Generator(np.random.PCG64(seed("F7-7:ece20")))
draws = np.empty(N_BOOT)
for i in range(N_BOOT):
    idx = rng.integers(0, bs.shape[0], size=bs.shape[0])
    draws[i] = ece_of(as_[idx].sum(axis=0)) - ece_of(bs[idx].sum(axis=0))
check("f77:ece_effect", close(ece_obs, f7["calibration"]["effect"]))
check("f77:ece_ci0", close(float(np.quantile(draws, 0.025)), f7["calibration"]["ci95"][0]))
check("f77:ece_ci1", close(float(np.quantile(draws, 0.975)), f7["calibration"]["ci95"][1]))
for name, mine in (("baseline_ece20", ece_pooled(np.concatenate(bp), np.concatenate(bc), 20)),
                   ("arm_ece20", ece_pooled(np.concatenate(ap), np.concatenate(ac), 20)),
                   ("baseline_ece10", ece_pooled(np.concatenate(bp), np.concatenate(bc), 10)),
                   ("arm_ece10", ece_pooled(np.concatenate(ap), np.concatenate(ac), 10))):
    check(f"f77:{name}", close(mine, f7["calibration"][name]))
ece_evidence = abs(ece_obs) >= 0.01 and not (np.quantile(draws, 0.025) <= 0 <= np.quantile(draws, 0.975))
check("f77:ece_threshold", ece_evidence == f7["calibration"]["evidence_threshold_met"])
for field in ("entropy", "brier", "target_margin", "target_rank"):
    mine = float(np.mean(np.concatenate([v[f"delta_{field}"].astype(np.float64) for v in arm7])))
    check(f"f77:desc:{field}", close(mine, f7["descriptive_fullvocab_deltas"][field], 5e-5))


def matched(delta, base_nll, positions, excluded, label):
    positions = np.asarray(sorted({int(p) for p in positions if 0 <= int(p) < delta.size}), dtype=np.int64)
    boundaries = np.quantile(base_nll, np.linspace(0.1, 0.9, 9))
    decile = np.digitize(base_nll, boundaries, right=True)
    block = np.arange(delta.size) // 512
    keep = np.ones(delta.size, dtype=bool)
    keep[list(excluded)] = False
    rng = np.random.Generator(np.random.PCG64(seed(label)))
    control = np.zeros(10000)
    for position in positions:
        candidates = np.flatnonzero((block == block[position]) & (decile == decile[position]) & keep)
        if candidates.size == 0:
            raise RuntimeError(f"no controls for {label} at {position}")
        control += delta[rng.choice(candidates, size=10000, replace=True)]
    class_mean = float(delta[positions].mean())
    contrasts = class_mean - control / positions.size
    return {"n": int(positions.size), "class_mean": class_mean,
            "matched_contrast": float(contrasts.mean()),
            "ci95": [float(np.quantile(contrasts, 0.025)), float(np.quantile(contrasts, 0.975))],
            "p_positive": float((1 + np.count_nonzero(contrasts <= 0)) / 10001)}


spec7 = (("07b_slack_multi", "first_content", frozen["class_07b_first_content"]),
         ("07b_slack_multi", "pronouns", frozen["class_07b_pronouns"]),
         ("08_math_llm", "unit_starts", frozen["class_08_unit_starts"]),
         ("08_math_llm", "pronouns", frozen["class_08_pronouns"]))
excluded_by_text: dict[str, set] = {}
for text, _n, positions in spec7:
    excluded_by_text.setdefault(text, set()).update(int(v) for v in positions)
fresh_rows = {}
for text, name, positions in spec7:
    with np.load(DUMP / "fresh" / "bias_off_L29" / f"{text}.npz", allow_pickle=False) as z:
        delta = np.asarray(z["delta_nll"], dtype=np.float64)
    with np.load(DUMP / "fresh" / "baseline" / f"{text}.npz", allow_pickle=False) as z:
        base_nll = np.asarray(z["nll"], dtype=np.float64)
    row = matched(delta, base_nll, positions, excluded_by_text[text], f"F7-7:class:{text}:{name}:query")
    target_positions = np.asarray([int(p) - 1 for p in positions if int(p) > 0], dtype=np.int64)
    row["target_aligned_secondary"] = matched(delta, base_nll, target_positions,
                                              excluded_by_text[text], f"F7-7:class:{text}:{name}:target")
    fresh_rows[f"{text}:{name}"] = row
    theirs = f7["fresh_classes"][f"{text}:{name}"]
    for key in ("n", "class_mean", "matched_contrast", "p_positive"):
        check(f"f77:{text}:{name}:{key}", close(float(row[key]), float(theirs[key])))
    for i in (0, 1):
        check(f"f77:{text}:{name}:ci{i}", close(row["ci95"][i], theirs["ci95"][i]))
    for key in ("class_mean", "matched_contrast", "p_positive"):
        check(f"f77:{text}:{name}:target:{key}",
              close(float(row["target_aligned_secondary"][key]),
                    float(theirs["target_aligned_secondary"][key])))
adj = holm([fresh_rows[f"{t}:{n}"]["p_positive"] for t, n, _p in spec7])
for (t, n, _p), p_h in zip(spec7, adj):
    check(f"f77:{t}:{n}:holm", close(p_h, f7["fresh_classes"][f"{t}:{n}"]["p_holm_positive"]))
slack_pass = all(fresh_rows[k]["matched_contrast"] > 0
                 and f7["fresh_classes"][k]["p_holm_positive"] < 0.05
                 for k in ("07b_slack_multi:first_content", "07b_slack_multi:pronouns"))
my_f7_pass = ranking_pass and slack_pass
check("f77:verdict", my_f7_pass == f7["passed"])
report["F7-7"] = {"passed": bool(my_f7_pass), "ranking_passed": bool(ranking_pass),
                  "slack_replication_passed": bool(slack_pass)}

# -------------------------------------------------------------------- summary
report["verdicts"] = {f"F7-{i}": bool(report[f"F7-{i}"]["passed"]) for i in range(1, 8)}
theirs = {k: bool(v["passed"]) for k, v in results["families"].items()}
check("verdict_agreement", report["verdicts"] == theirs, f"{report['verdicts']} != {theirs}")
report["passed"] = not errors
report["results_sha256"] = sha256_file(RESULTS)
report["producer_imported"] = False
OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
print(json.dumps({"passed": report["passed"], "n_errors": len(errors),
                  "errors_head": errors[:20], "verdicts": report["verdicts"],
                  "artifact_count_rehashed": rehashed}, indent=1))
