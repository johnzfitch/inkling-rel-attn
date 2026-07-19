"""Second-analyst independent re-derivation of the R5-D causal campaign.

Reads only the sealed arm dumps, frozen inputs, and registered amendments.
Imports neither the runner nor the analyzer. Re-hashes every sealed artifact,
recomputes every pooled cost, regenerates every bootstrap from its seed rule,
and reproduces all seven registered verdicts.

Also records the CK1 real-arm structural note: for the single-layer clock
arms the pre-intervention r-vectors are bitwise the certified capture data
from which G was estimated as the OLS slope on the CK1 regressor, so
projecting G out annihilates the regressor correlation in every coordinate
identically. The real-arm clause is therefore an algebraic identity given
the locus-identity gate, not an empirical outcome; the demonstration below
computes the exact residual. The sham clause is unaffected.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "dumps" / "round5" / "r5d"
RESULTS = ROOT / "analysis" / "round5" / "r5d" / "r5d_results.json"
CLOCK_PATH = ROOT / "analysis" / "round5" / "r5d_clock" / "clock_freeze.npz"
PARENT = ROOT / "registrations" / "ROUND5_R5D_EXECUTION_AMENDMENT.md"
CLOCK_AMENDMENT = ROOT / "registrations" / "ROUND5_R5D_CLOCK_AMENDMENT.md"
OUT = ROOT / "analysis" / "round5" / "r5d" / "verification.json"

TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
SINGLE_LAYERS = [0, 1, 2, 3, 4, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
SEQ, HEADS, RPERHEAD = 8192, 64, 16
RFLAT = HEADS * RPERHEAD
DRAWS = 5000
TOL = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_from(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


errors: list[str] = []
report: dict[str, object] = {
    "kind": "round5_r5d_independent_verification",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": sha256_file(Path(__file__)),
    "results_sha256": sha256_file(RESULTS),
}
results = json.loads(RESULTS.read_text(encoding="utf-8"))

# ---- 1. batch integrity: every sealed manifest, every artifact re-hashed
arm_ids = sorted(p.name for p in (DUMP / "arms").iterdir() if p.is_dir())
if len(arm_ids) != 72:
    errors.append(f"expected 72 arm directories, found {len(arm_ids)}")
rehashed = 0
for arm_id in arm_ids:
    root = DUMP / "arms" / arm_id
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        errors.append(f"arm not sealed: {arm_id}")
        continue
    if len(manifest["artifacts"]) != manifest.get("expected_artifact_count"):
        errors.append(f"artifact count mismatch: {arm_id}")
    for record in manifest["artifacts"]:
        path = root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"artifact hash mismatch: {arm_id}/{record['path']}")
        else:
            rehashed += 1
report["artifacts_rehashed"] = rehashed

# baseline binding to certified NLL
CAPTURE = ROOT / "dumps" / "round5" / "widened_corrected_capture"
for text in TEXTS:
    with np.load(DUMP / "baseline" / f"{text}.npz", allow_pickle=False) as base, np.load(
        CAPTURE / "nll" / f"nll_{text}.npz", allow_pickle=False
    ) as certified:
        if not np.array_equal(base["nll"], certified["nll"]):
            errors.append(f"baseline NLL differs from certified: {text}")

# ---- 2. pooled costs + bootstraps for all 72 arms
parent_hash = sha256_file(PARENT)
pooled: dict[str, float] = {}
for arm_id in arm_ids:
    deltas = []
    for text in TEXTS:
        with np.load(DUMP / "arms" / arm_id / "tokens" / f"{text}.npz", allow_pickle=False) as z:
            deltas.append(np.asarray(z["delta_nll"], dtype=np.float64))
    per_text = [float(d.mean()) for d in deltas]
    pooled_value = float(np.mean(per_text))
    pooled[arm_id] = pooled_value
    summary = results["arm_summaries"][arm_id]
    if abs(summary["pooled_mean_delta_nll"] - pooled_value) > TOL:
        errors.append(f"pooled delta mismatch: {arm_id}")
    for text, value in zip(TEXTS, per_text):
        if abs(summary["per_text_mean_delta_nll"][text] - value) > TOL:
            errors.append(f"per-text delta mismatch: {arm_id}/{text}")
    natural = float(np.mean(per_text[:5]))
    if abs(summary["natural_five_mean_delta_nll"] - natural) > TOL:
        errors.append(f"natural-five mismatch: {arm_id}")
    # bootstrap: paired 256-token blocks within each text (positions 1..8191;
    # block index = zero-based target index // 256 -> 32 blocks)
    block_sums = np.stack([
        np.add.reduceat(d, np.arange(0, SEQ - 1, 256)) for d in deltas
    ])                                                    # 6 x 32
    block_counts = np.diff(np.append(np.arange(0, SEQ - 1, 256), SEQ - 1))
    rng = np.random.Generator(np.random.PCG64(seed_from(f"{parent_hash}:r5d_256_token_bootstrap:{arm_id}")))
    sample = rng.integers(0, 32, size=(DRAWS, len(deltas), 32))
    sums = block_sums[np.arange(6)[None, :, None], sample].sum(axis=(1, 2))
    counts = block_counts[sample].sum(axis=(1, 2))
    draws = sums / counts
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    reported = summary["bootstrap"]["ci95"]
    if abs(ci[0] - reported[0]) > 1e-6 or abs(ci[1] - reported[1]) > 1e-6:
        errors.append(f"bootstrap CI mismatch: {arm_id}: {ci} vs {reported}")
report["pooled_costs_verified"] = len(pooled)

# ---- 3. parent verdicts
bias = {layer: pooled[f"bias_off_L{layer:02d}"] for layer in SINGLE_LAYERS}
required_large = {0, 1, 2, 3, 4, 5, 65}
bias_pass = all(
    (value > 0.05) if layer in required_large else (abs(value) <= 0.05)
    for layer, value in bias.items()
)
carrier_pass_layers = [
    layer
    for layer in SINGLE_LAYERS
    if abs(pooled[f"carrier_out_L{layer:02d}"] - bias[layer]) <= 0.20 * abs(bias[layer])
]
near_mean = float(np.mean([pooled[f"near_off_L{layer:02d}"] for layer in SINGLE_LAYERS]))
far_mean = float(np.mean([pooled[f"far_off_L{layer:02d}"] for layer in SINGLE_LAYERS]))
ratio = near_mean / far_mean if far_mean > 0 else float("nan")
my_verdicts = {
    "bias_off_depth_pass": bool(bias_pass),
    "bias_argmax_layer": int(max(bias, key=lambda k: bias[k])),
    "carrier_equivalent_layers": carrier_pass_layers,
    "carrier_pass": bool(len(carrier_pass_layers) == 16),
    "near_over_far_ratio": ratio,
    "near_pass": bool(far_mean > 0 and ratio >= 5),
    "wall_abs_delta": abs(pooled["wall_heal_global"]),
    "wall_pass": bool(abs(pooled["wall_heal_global"]) < 0.005),
}
theirs = results["verdicts"]
checks = [
    ("bias_off_depth", my_verdicts["bias_off_depth_pass"]),
    ("carrier_equivalence", my_verdicts["carrier_pass"]),
    ("near_dominates_far", my_verdicts["near_pass"]),
    ("wall_incidental_at_8k", my_verdicts["wall_pass"]),
]
for key, mine in checks:
    if bool(theirs[key]["passed"]) != mine:
        errors.append(f"parent verdict mismatch: {key}")
report["parent_verdicts"] = my_verdicts

# ---- 4. clock verdicts
starts = np.arange(64, SEQ, 64)
x = np.log1p(starts + 31.5)
xc = x - x.mean()


def clock_stat(arm_id: str, layer: int) -> tuple[float, float]:
    """Return (median |corr|, exact residual ||Ytilde^T xc||_inf) in float64."""
    pre = np.load(
        DUMP / "arms" / arm_id / "clock" / f"rvec_pre_L{layer:02d}_06_random.npy",
        allow_pickle=False,
    ).astype(np.float64).reshape(SEQ, RFLAT)
    with np.load(CLOCK_PATH, allow_pickle=False) as freeze:
        key = "sham_L59" if arm_id == "clock_sham_L59" else f"G_L{layer}"
        direction = freeze[key].astype(np.float64)
        anchor = freeze[f"rbar_L{layer}"].astype(np.float64)
    frozen = pre - ((pre - anchor) @ direction)[:, None] * direction
    blocks = frozen[64:].reshape(127, 64, RFLAT).mean(1)
    centered = blocks - blocks.mean(0)
    residual = float(np.max(np.abs(xc @ centered)))
    proj = np.load(ROOT / "weights" / f"layer{layer:02d}_rel_logits_proj.npy").astype(np.float64)
    kernels = blocks.reshape(127, HEADS, RPERHEAD) @ proj
    meanc = kernels.mean(0)
    gain = (kernels * meanc).sum(2) / (meanc * meanc).sum(1)
    gc = gain - gain.mean(0)
    with np.errstate(invalid="ignore"):
        corr = (xc @ gc) / (np.linalg.norm(xc) * np.linalg.norm(gc, axis=0))
    corr = np.nan_to_num(corr, nan=0.0)
    return float(np.median(np.abs(corr))), residual


ck1_l53, resid_53 = clock_stat("clock_freeze_L53", 53)
ck1_l59, resid_59 = clock_stat("clock_freeze_L59", 59)
ck1_sham, _ = clock_stat("clock_sham_L59", 59)
ck1 = results["clock_verdicts"]["CK1_kernel_gain_flattening"]
for mine, theirs_value, label in (
    (ck1_l53, ck1["clock_freeze_L53_median_abs_corr"], "CK1 L53"),
    (ck1_l59, ck1["clock_freeze_L59_median_abs_corr"], "CK1 L59"),
    (ck1_sham, ck1["clock_sham_L59_median_abs_corr"], "CK1 sham"),
):
    if abs(mine - theirs_value) > 1e-6:
        errors.append(f"{label} mismatch: {mine} vs {theirs_value}")
report["ck1"] = {
    "l53": ck1_l53,
    "l59": ck1_l59,
    "sham": ck1_sham,
    "structural_note": (
        "real-arm pre-r is bitwise the certified data G was OLS-fit on, so "
        "projection annihilates the regressor correlation identically; the "
        "float64 residuals below bound the statistic by construction"
    ),
    "exact_projection_residual_L53": resid_53,
    "exact_projection_residual_L59": resid_59,
}

# CK2
deltas = []
for text in TEXTS:
    with np.load(DUMP / "arms" / "clock_freeze_L53_L59" / "tokens" / f"{text}.npz", allow_pickle=False) as z:
        deltas.append(np.asarray(z["delta_nll"], dtype=np.float64))
pooled_blocks = np.mean([d[63:].reshape(127, 64).mean(1) for d in deltas], axis=0)
regressor = np.abs(xc)
yr = rankdata(pooled_blocks, method="average")
xr = rankdata(regressor, method="average")
rho = float(np.corrcoef(yr, xr)[0, 1])
amendment_hash = sha256_file(CLOCK_AMENDMENT)
rng = np.random.Generator(np.random.PCG64(seed_from(f"{amendment_hash}:ck2_bootstrap")))
groups = np.asarray([s // 256 for s in starts])
by_group = [np.flatnonzero(groups == g) for g in range(32)]
draws = np.empty(DRAWS)
for i in range(DRAWS):
    sel = np.concatenate([by_group[g] for g in rng.integers(0, 32, size=32)])
    a = rankdata(pooled_blocks[sel], method="average")
    b = rankdata(regressor[sel], method="average")
    value = float(np.corrcoef(a, b)[0, 1])
    draws[i] = value if np.isfinite(value) else 0.0
lower = float(np.percentile(draws, 2.5))
ck2 = results["clock_verdicts"]["CK2_log_extremes_cost"]
if abs(rho - ck2["spearman_rho"]) > 1e-9 or abs(lower - ck2["bootstrap_lower_95"]) > 1e-9:
    errors.append(f"CK2 mismatch: rho {rho} vs {ck2['spearman_rho']}, lower {lower} vs {ck2['bootstrap_lower_95']}")
report["ck2"] = {"rho": rho, "lower_95": lower, "pass": bool(rho > 0 and lower > 0)}

# CK3
ck3_value = abs(pooled["clock_freeze_L65"])
if abs(ck3_value - results["clock_verdicts"]["CK3_L65_exemption"]["absolute_pooled_delta_nll"]) > TOL:
    errors.append("CK3 mismatch")
report["ck3"] = {"abs_pooled_delta_nll": ck3_value, "pass": bool(ck3_value < 0.005)}

# ---- 5. spot physical checks on locus meters
with np.load(DUMP / "arms" / "wall_heal_global" / "meters" / "L65_06_random.npz", allow_pickle=False) as z:
    far_mass = float(z["mass_with"][:, 1024:].sum())
if far_mass <= 0:
    errors.append("wall-heal L65 meter shows no mass beyond d=1024")
report["wall_heal_L65_random_mass_beyond_1024"] = far_mass
with np.load(DUMP / "arms" / "near_off_L29" / "meters" / "L29_06_random.npz", allow_pickle=False) as z:
    near_bias = float(np.abs(z["bias_sum"][:, :4]).sum())
if near_bias != 0.0:
    errors.append("near-off L29 meter shows nonzero bias at d<4")
report["near_off_L29_bias_below_d4"] = near_bias

report["errors"] = errors
report["passed"] = not errors
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: report[k] for k in ("passed", "errors", "artifacts_rehashed", "pooled_costs_verified", "ck1", "ck2", "ck3", "parent_verdicts")}, indent=1, default=str))
