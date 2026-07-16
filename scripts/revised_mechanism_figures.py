"""Recompute the corrected Inkling mechanism figures from saved artifacts.

This is a dump-first analysis.  It does not execute the model or use a GPU.  It
audits C2/C4/C5, analyzes the captured needle rows, and exposes the learned
512-token echo, the L65 terminal wall, and content retrieval beyond the bias
horizon.  Outputs go to ``analysis/revised_mechanisms``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "revised_mechanisms"
TIER2 = ROOT / "dumps" / "tier2"
CAPTURE = TIER2 / "capture"
WEIGHTS = ROOT / "weights"
D1 = ROOT / "dumps" / "round3" / "mode_curves"
ROUND4 = ROOT / "analysis" / "round4" / "curiosity"
NEEDLES = ROOT / "analysis" / "needles" / "needle_results.json"
CORPUS = ROOT / "corpus"

GLOBAL = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
ECHO_TEXTS = TEXTS[:3]

BLUE = "#2F6BDE"
ORANGE = "#D97706"
RED = "#D1495B"
TEAL = "#16857A"
PURPLE = "#7C5CFC"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
INK = "#172033"
NOTE_BOX = {
    "boxstyle": "round,pad=0.25",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.90,
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def tier2(layer: int, text: str):
    return np.load(TIER2 / f"layer{layer:02d}_{text}_s8192.npz", allow_pickle=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "axes.edgecolor": "#9CA3AF",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "none",
            "legend.framealpha": 0.90,
            "figure.titlesize": 15,
            "figure.titleweight": "normal",
        }
    )


def safe_exp(x):
    return np.exp(np.clip(x, -700.0, 700.0))


def f_exp2_const(d, a_slow, rate_slow, a_fast, rate_gap, constant):
    return (
        a_slow * safe_exp(-rate_slow * d)
        + a_fast * safe_exp(-(rate_slow + rate_gap) * d)
        + constant
    )


def f_exp2_zero(d, a_slow, rate_slow, a_fast, rate_gap):
    return a_slow * safe_exp(-rate_slow * d) + a_fast * safe_exp(
        -(rate_slow + rate_gap) * d
    )


def c4_refit() -> dict[str, dict[str, float | bool | list[float]]]:
    """Audit the free-constant fit and refit with c=0 and rate >= 1/extent."""
    out: dict[str, dict[str, float | bool | list[float]]] = {}
    for layer in GLOBAL:
        z = np.load(D1 / f"layer{layer:02d}.npz")
        curve = z["S"][0] * z["Vt"][0]
        extent = len(curve)
        x = np.arange(extent, dtype=np.float64)[8:]
        y = curve[8:].astype(np.float64)

        p0_old = [y[0], 0.01, y[0] * 0.5, 0.05, 0.0]
        popt_old, _ = curve_fit(
            f_exp2_const,
            x,
            y,
            p0=p0_old,
            bounds=(
                [-np.inf, 1e-5, -np.inf, 0.0, -np.inf],
                [np.inf, 1.0, np.inf, 1.0, np.inf],
            ),
            maxfev=20_000,
        )
        xe = np.arange(extent, 4 * extent, dtype=np.float64)
        old_extrap = np.trapezoid(np.abs(f_exp2_const(xe, *popt_old)), xe)
        c_integral = abs(float(popt_old[4])) * float(xe[-1] - xe[0])

        min_rate = 1.0 / extent
        p0_new = [y[0], max(0.01, 2 * min_rate), y[0] * 0.5, 0.05]
        popt_new, _ = curve_fit(
            f_exp2_zero,
            x,
            y,
            p0=p0_new,
            bounds=(
                [-np.inf, min_rate, -np.inf, 0.0],
                [np.inf, 1.0, np.inf, 1.0],
            ),
            maxfev=100_000,
        )
        fitted = f_exp2_zero(x, *popt_new)
        r2 = 1 - np.sum((y - fitted) ** 2) / (
            np.sum((y - y.mean()) ** 2) + 1e-12
        )
        realized = np.trapezoid(np.abs(y), x)
        extrap = np.trapezoid(np.abs(f_exp2_zero(xe, *popt_new)), xe)
        slow = float(popt_new[1])
        out[str(layer)] = {
            # Reconstruct the originally reported free-c statistic directly.
            # The registered JSON may already have been replaced by the audited
            # c=0 result, where unresolved layers intentionally omit this field.
            "old_truncated_fraction": float(old_extrap / (realized + 1e-12)),
            "old_constant": float(popt_old[4]),
            "old_slow_rate": float(popt_old[1]),
            "old_constant_over_full_extrapolated_abs_integral": float(
                c_integral / (old_extrap + 1e-12)
            ),
            "corrected_truncated_fraction": float(extrap / (realized + 1e-12)),
            "corrected_fit_r2": float(r2),
            "corrected_rates": [slow, float(slow + popt_new[3])],
            "corrected_slow_rate_pinned": bool(
                np.isclose(slow, min_rate, rtol=1e-3, atol=1e-12)
            ),
        }
    return out


def c5_corrected() -> dict[str, object]:
    """Sign-invariant mode-0 steps on a common raw d=0..511 support."""
    old = read_json(ROUND4 / "C5_flip_trajectory.json")
    layers = list(range(66))
    curves = []
    for layer in layers:
        z = np.load(D1 / f"layer{layer:02d}.npz")
        curve = (z["S"][0] * z["Vt"][0])[:512].astype(np.float64)
        curves.append(curve / (np.linalg.norm(curve) + 1e-12))
    step = np.array(
        [
            min(
                np.linalg.norm(curves[i] - curves[i - 1]),
                np.linalg.norm(curves[i] + curves[i - 1]),
            )
            for i in range(1, len(curves))
        ]
    )
    destination = np.arange(1, 66)
    band = (destination >= 13) & (destination <= 28)
    ratio = float(step[band].mean() / np.median(step))
    return {
        "layers": destination.tolist(),
        "reported_step": old["curve_step_mode0"],
        "corrected_step": step.tolist(),
        "reported_ratio": float(old["transition_vs_median_ratio"]),
        "corrected_ratio": ratio,
    }


def figure_audit(c4: dict, c5: dict, summary: dict) -> None:
    c2w = read_json(ROUND4 / "C2_interface_utilization.json")
    vest = np.array([c2w[str(layer)]["vestigial_energy_fraction"] for layer in range(66)])
    base = np.array(
        [c2w[str(layer)]["uniform_baseline_vestigial"] for layer in range(66)]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), layout="constrained")
    fig.suptitle("Audit corrections: one null result, one marginal fit, one surviving transition")

    ax = axes[0]
    for i in range(66):
        ax.plot([0, 1], [base[i], vest[i]], color=LIGHT_GRAY, lw=0.7, alpha=0.7)
    ax.scatter(np.zeros(66), base, s=12, color=GRAY, alpha=0.55)
    ax.scatter(np.ones(66), vest, s=12, color=BLUE, alpha=0.55)
    med_base, med_vest = float(np.median(base)), float(np.median(vest))
    ax.scatter([0, 1], [med_base, med_vest], s=90, color=[GRAY, BLUE], zorder=4)
    ax.plot([0, 1], [med_base, med_vest], color=INK, lw=2, zorder=3)
    ax.set_xticks([0, 1], ["uniform null", "weight-side C2"])
    ax.set(ylabel="fraction in weak table modes", title="A. Weight-side C2 is indistinguishable from random")
    ax.set_ylim(0.55, 0.95)
    ax.grid(axis="y")
    ax.text(
        0.5,
        0.575,
        f"median {med_vest:.3f} vs {med_base:.3f}",
        ha="center",
        color=INK,
    )

    ax = axes[1]
    layers = np.array(GLOBAL)
    old = np.array([c4[str(layer)]["old_truncated_fraction"] for layer in layers])
    new = np.array([c4[str(layer)]["corrected_truncated_fraction"] for layer in layers])
    pinned = np.array([c4[str(layer)]["corrected_slow_rate_pinned"] for layer in layers])
    for x, a, b in zip(layers, old, new):
        ax.plot([x, x], [a, b], color=LIGHT_GRAY, lw=1)
    ax.scatter(layers, old, color=RED, s=35, label="reported: free c")
    ax.scatter(layers[~pinned], new[~pinned], color=TEAL, s=38, label="c=0, resolvable rate")
    ax.scatter(
        layers[pinned],
        new[pinned],
        facecolors="white",
        edgecolors=ORANGE,
        marker="X",
        s=70,
        label="corrected rate pinned",
        zorder=4,
    )
    ax.axhline(0.3, color=GRAY, ls="--", lw=1, label="registered threshold")
    ax.set_yscale("log")
    ax.set(
        xlabel="global layer",
        ylabel="extrapolated / observed |kernel| integral",
        title="B. C4 falls from strong to marginal",
    )
    ax.grid(axis="y", which="both")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        7,
        0.11,
        f"median {np.median(old):.2f} → {np.median(new):.2f}",
        color=INK,
    )

    ax = axes[2]
    dst = np.array(c5["layers"])
    reported = np.array(c5["reported_step"])
    corrected = np.array(c5["corrected_step"])
    ax.axvspan(13, 28, color=ORANGE, alpha=0.10, label="registered transition band")
    ax.plot(dst, reported, color=LIGHT_GRAY, lw=1.2, label="reported (signed + rescaled)")
    ax.plot(dst, corrected, color=PURPLE, lw=1.8, label="sign-invariant, raw d≤511")
    ax.set(
        xlabel="destination layer",
        ylabel="mode-0 change from previous layer",
        title="C. C5 survives, but the spike was inflated",
    )
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.03,
        0.96,
        f"band / median\n{c5['reported_ratio']:.2f}× → {c5['corrected_ratio']:.2f}×",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PURPLE,
        bbox=NOTE_BOX,
    )

    save(fig, "audit-corrections.png")
    summary["audit"] = {
        "C2_weight_median_vestigial": med_vest,
        "C2_uniform_median_vestigial": med_base,
        "C4_reported_median_truncated": float(np.median(old)),
        "C4_corrected_median_truncated": float(np.median(new)),
        "C4_corrected_pinned_layers": layers[pinned].tolist(),
        "C5_reported_transition_ratio": c5["reported_ratio"],
        "C5_corrected_transition_ratio": c5["corrected_ratio"],
    }


def figure_interface_activation(summary: dict) -> None:
    data = read_json(ROUND4 / "C2_interface_utilization_act.json")
    raw_weak, effective_weak, cosines = [], [], []
    raw_profiles, effective_profiles = [], []
    for layer in range(66):
        entry = data[str(layer)]
        live = np.asarray(entry["live_mode_energy"], dtype=float)
        s2 = np.asarray(entry["proj_S2"], dtype=float)
        cutoff = int(np.ceil(entry["proj_eff_rank"]))
        response = live * s2
        raw_weak.append(live[cutoff:].sum() / live.sum())
        effective_weak.append(response[cutoff:].sum() / response.sum())
        cosines.append(entry["cosine_live_vs_S"])
        raw_profiles.append(live / live.sum())
        effective_profiles.append(response / response.sum())

    raw_weak = np.asarray(raw_weak)
    effective_weak = np.asarray(effective_weak)
    raw_profile = np.median(np.asarray(raw_profiles), axis=0)
    raw_profile /= raw_profile.sum()
    effective_profile = np.median(np.asarray(effective_profiles), axis=0)
    effective_profile /= effective_profile.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), layout="constrained")
    fig.suptitle("Live co-adaptation appears in what the table reads—not where raw energy goes")

    ax = axes[0]
    for i in range(66):
        ax.plot([0, 1], [raw_weak[i], effective_weak[i]], color=LIGHT_GRAY, lw=0.8)
    ax.scatter(np.zeros(66), raw_weak, color=ORANGE, s=15, alpha=0.65)
    ax.scatter(np.ones(66), effective_weak, color=TEAL, s=15, alpha=0.65)
    med_raw = float(np.median(raw_weak))
    med_eff = float(np.median(effective_weak))
    ax.scatter([0, 1], [med_raw, med_eff], color=[ORANGE, TEAL], s=100, zorder=4)
    ax.set_xticks([0, 1], ["raw r-vector energy", "after S² readout weighting"])
    ax.set(
        ylabel="fraction in weak table modes",
        title="A. Most raw energy is ignored by the positional table",
    )
    ax.set_ylim(-0.02, 0.95)
    ax.grid(axis="y")
    ax.annotate(
        f"median {med_raw:.2f}",
        xy=(0, med_raw),
        xytext=(8, 12),
        textcoords="offset points",
        ha="left",
        color=ORANGE,
        bbox=NOTE_BOX,
    )
    ax.annotate(
        f"median {med_eff:.2f}",
        xy=(1, med_eff),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        color=TEAL,
        bbox=NOTE_BOX,
    )

    ax = axes[1]
    modes = np.arange(1, 17)
    ax.plot(modes, raw_profile, marker="o", color=ORANGE, lw=1.8, label="raw live excitation")
    ax.plot(
        modes,
        effective_profile,
        marker="o",
        color=TEAL,
        lw=1.8,
        label="effective response = live × S²",
    )
    ax.set_yscale("log")
    ax.set(
        xlabel="table mode, ordered by singular value",
        ylabel="median normalized share (log scale)",
        title="B. S² weighting concentrates the functional signal",
    )
    ax.set_xticks([1, 2, 4, 8, 12, 16])
    ax.grid(axis="y", which="both")
    ax.legend()
    ax.text(
        9.2,
        0.20,
        f"live-vs-S² cosine\nmedian {np.median(cosines):.2f}",
        color=INK,
        bbox=NOTE_BOX,
    )

    save(fig, "interface-live-coadaptation.png")
    summary["interface_activation"] = {
        "median_raw_live_energy_in_weak_modes": med_raw,
        "median_effective_response_in_weak_modes_after_S2": med_eff,
        "median_live_vs_S2_cosine": float(np.median(cosines)),
    }


def needle_setup():
    sidecar = read_json(CORPUS / "05_needles.sidecar.json")
    entities = [e for e in sidecar["entities"] if len(e["token_positions"]) >= 2]
    tokenizer = Tokenizer.from_file(str(CORPUS / "tokenizer.json"))
    widths = {
        e["codeword"]: len(tokenizer.encode(" " + e["codeword"]).ids) for e in entities
    }
    return entities, widths


def needle_breadth(entities) -> tuple[np.ndarray, list[int], list[int]]:
    """Conservative breadth on a fixed eight-token intro-target span."""
    breadth, top_median, top_frequency = [], [], []
    for layer in GLOBAL:
        z = np.load(CAPTURE / f"needlerows_L{layer:02d}.npz")
        qpos, rows = z["qpos"], z["rows"]
        masses = []
        for entity in entities:
            p0, q = entity["token_positions"][:2]
            i = int(np.flatnonzero(qpos == q)[0])
            masses.append(rows[i, :, p0 : p0 + 8].astype(np.float64).sum(-1))
        mass = np.asarray(masses)
        breadth.append(float(np.median((mass > 0.05).sum(1))))
        top_median.append(int(np.median(mass, axis=0).argmax()))
        top_frequency.append(int(np.bincount(mass.argmax(1), minlength=64).argmax()))
    return np.asarray(breadth), top_median, top_frequency


def needle_reconstruction(layer: int, entities, widths) -> list[dict[str, float | int | str]]:
    """Exact same-head with/without-bias reconstruction for each needle."""
    z = np.load(CAPTURE / f"needlerows_L{layer:02d}.npz")
    qpos, rows = z["qpos"], z["rows"].astype(np.float64)
    projection = np.load(WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy").astype(
        np.float64
    )
    rvec = np.load(CAPTURE / f"rvec_L{layer:02d}_05_needles.npy", mmap_mode="r")
    out = []
    keys = np.arange(rows.shape[-1])
    for entity in entities:
        p0, q = entity["token_positions"][:2]
        i = int(np.flatnonzero(qpos == q)[0])
        row = rows[i]
        distance = q - keys
        in_horizon = (distance >= 0) & (distance < projection.shape[1])
        bias_by_distance = np.asarray(rvec[q], dtype=np.float64) @ projection
        bias = np.zeros_like(row)
        bias[:, in_horizon] = bias_by_distance[:, distance[in_horizon]]
        without = row * np.exp(-bias)
        without[:, keys > q] = 0.0
        without /= without.sum(-1, keepdims=True) + 1e-300
        win = slice(p0, p0 + widths[entity["codeword"]])
        with_target = row[:, win].sum(-1)
        without_target = without[:, win].sum(-1)
        head = int(with_target.argmax())
        causal_far = (distance >= projection.shape[1]) & (distance <= q)
        out.append(
            {
                "side": entity["side_of_seam"],
                "distance": int(q - p0),
                "head": head,
                "with_target": float(with_target[head]),
                "without_target": float(without_target[head]),
                "target_multiplier": float(
                    with_target[head] / (without_target[head] + 1e-300)
                ),
                "target_bias": float(bias[head, win].mean()),
                "with_horizon_mass": float(row[head, in_horizon].sum()),
                "without_horizon_mass": float(without[head, in_horizon].sum()),
                "with_far_mass": float(row[head, causal_far].sum()),
                "without_far_mass": float(without[head, causal_far].sum()),
            }
        )
    return out


def figure_needle_retrieval(summary: dict, entities) -> None:
    data = read_json(NEEDLES)
    layers = np.array(GLOBAL)
    medians, q1, q3, pvals = {}, {}, {}, []
    for side in ["below", "above"]:
        values = [
            np.array([p["with_max"] for p in data[str(layer)] if p["side"] == side])
            for layer in layers
        ]
        medians[side] = np.array([np.median(v) for v in values])
        q1[side] = np.array([np.quantile(v, 0.25) for v in values])
        q3[side] = np.array([np.quantile(v, 0.75) for v in values])
    for layer in layers:
        below = [p["with_max"] for p in data[str(layer)] if p["side"] == "below"]
        above = [p["with_max"] for p in data[str(layer)] if p["side"] == "above"]
        pvals.append(float(mannwhitneyu(below, above, alternative="two-sided").pvalue))

    multipliers = {}
    for side in ["below", "above"]:
        multipliers[side] = np.array(
            [
                np.median(
                    [
                        p["with_max"] / (p["wo_max"] + 1e-300)
                        for p in data[str(layer)]
                        if p["side"] == side
                    ]
                )
                for layer in layers
            ]
        )
    breadth, top_median, top_frequency = needle_breadth(entities)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), layout="constrained")
    fig.suptitle("Needle retrieval crosses the seam through the trunk—and uses many heads")

    ax = axes[0]
    for side, color, label in [
        ("below", BLUE, "below seam: d≈900–1000"),
        ("above", ORANGE, "above seam: d≈1050–1150"),
    ]:
        ax.fill_between(layers, q1[side], q3[side], color=color, alpha=0.12)
        ax.plot(layers, medians[side], marker="o", color=color, lw=1.8, label=label)
    ax.axhline(0.007, color=GRAY, ls="--", lw=1, label="chance ≈ 0.007")
    for layer, pvalue in zip(layers, pvals):
        if pvalue < 0.05:
            y = max(medians["below"][layers == layer][0], medians["above"][layers == layer][0])
            at_left_edge = layer == layers[0]
            ax.annotate(
                f"p={pvalue:.3f}",
                xy=(layer, min(0.96, y + 0.10)),
                xytext=(6 if at_left_edge else -6, 0),
                textcoords="offset points",
                ha="left" if at_left_edge else "right",
                va="center",
                fontsize=8,
                color=RED,
                bbox=NOTE_BOX,
            )
    ax.set(
        xlabel="global layer",
        ylabel="best-head mass on intro target",
        title="A. Only L5 and L65 show a seam deficit",
    )
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.axhline(1, color=GRAY, ls="--", lw=1)
    ax.plot(layers, multipliers["below"], marker="o", color=BLUE, label="below seam")
    ax.plot(layers, multipliers["above"], marker="o", color=ORANGE, label="above seam")
    ax.set(
        xlabel="global layer",
        ylabel="with-bias / without-bias target mass",
        title="B. Bias often helps targets on both sides",
    )
    ax.set_ylim(0.3, 1.75)
    ax.grid(axis="y")
    ax.legend()
    i41 = GLOBAL.index(41)
    ax.text(41, multipliers["above"][i41] + 0.09, f"{multipliers['above'][i41]:.2f}×", ha="center", color=ORANGE)

    ax = axes[2]
    ax.plot(layers, breadth, marker="o", color=TEAL, lw=1.8)
    for layer, value, head, common in zip(layers, breadth, top_median, top_frequency):
        label = f"h{head}" if head == common else f"h{common}/{head}"
        ax.text(layer, value + 1.2, label, ha="center", fontsize=7.5, color=INK)
    ax.set(
        xlabel="global layer",
        ylabel="median heads with target mass > 0.05",
        title="C. Retrieval is broad; the top head rotates",
    )
    ax.set_ylim(0, max(30, breadth.max() + 5))
    ax.grid(axis="y")
    ax.text(
        0.02,
        0.04,
        "labels: most-common / median-top head\n(8-token intro target)",
        transform=ax.transAxes,
        fontsize=8,
        color=GRAY,
        bbox=NOTE_BOX,
    )

    save(fig, "needle-retrieval-across-seam.png")
    summary["needle_retrieval"] = {
        "mann_whitney_two_sided_p_by_layer": dict(zip(map(str, layers), pvals)),
        "median_best_head_mass_below": dict(zip(map(str, layers), medians["below"].tolist())),
        "median_best_head_mass_above": dict(zip(map(str, layers), medians["above"].tolist())),
        "median_with_without_multiplier_below": dict(zip(map(str, layers), multipliers["below"].tolist())),
        "median_with_without_multiplier_above": dict(zip(map(str, layers), multipliers["above"].tolist())),
        "median_head_breadth_fixed_8_token_target": dict(zip(map(str, layers), breadth.tolist())),
        "top_head_by_median": dict(zip(map(str, layers), top_median)),
        "top_head_by_argmax_frequency": dict(zip(map(str, layers), top_frequency)),
    }


def figure_contrast_enhancer(summary: dict, recon41: list[dict]) -> None:
    above = [r for r in recon41 if r["side"] == "above"]
    without_target = np.array([r["without_target"] for r in above])
    with_target = np.array([r["with_target"] for r in above])
    multiplier = np.array([r["target_multiplier"] for r in above])

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8), layout="constrained")
    fig.suptitle("At L41 the bias helps an out-of-horizon target without touching it")

    ax = axes[0]
    order = np.argsort(without_target)
    for i in order:
        ax.plot([0, 1], [without_target[i], with_target[i]], color=LIGHT_GRAY, lw=1)
    ax.scatter(np.zeros(len(above)), without_target, color=GRAY, s=32, label="without bias")
    ax.scatter(np.ones(len(above)), with_target, color=ORANGE, s=32, label="with bias")
    ax.set_xticks([0, 1], ["without bias", "with bias"])
    ax.set(
        ylabel="same-head mass on above-seam target",
        title="A. Every target itself has exactly zero bias",
    )
    ax.grid(axis="y")
    ax.text(
        0.5,
        0.96,
        f"median gain {np.median(multiplier):.2f}×",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=ORANGE,
        bbox=NOTE_BOX,
    )

    ax = axes[1]
    horizon = [
        np.median([r["without_horizon_mass"] for r in above]),
        np.median([r["with_horizon_mass"] for r in above]),
    ]
    far = [
        np.median([r["without_far_mass"] for r in above]),
        np.median([r["with_far_mass"] for r in above]),
    ]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, [horizon[0], far[0]], width, color=GRAY, label="without bias")
    ax.bar(x + width / 2, [horizon[1], far[1]], width, color=TEAL, label="with bias")
    ax.set_xticks(x, ["inside bias horizon\n(competing keys)", "beyond horizon\n(target side)"])
    ax.set(
        ylabel="median attention mass",
        title="B. Suppressed competitors release normalizer mass",
    )
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y")
    ax.legend(fontsize=8)

    ax = axes[2]
    for side, color in [("below", BLUE), ("above", ORANGE)]:
        rows = [r for r in recon41 if r["side"] == side]
        ax.scatter(
            [r["distance"] for r in rows],
            [r["target_multiplier"] for r in rows],
            color=color,
            s=42,
            label=side,
        )
    ax.axvline(1024, color=RED, ls="--", lw=1, label="bias horizon")
    ax.axhline(1, color=GRAY, ls=":", lw=1)
    ax.set(
        xlabel="target distance",
        ylabel="same-head with / without target mass",
        title="C. Above-seam gains are pure renormalization",
    )
    ax.grid(axis="y")
    ax.legend(fontsize=8)

    save(fig, "bias-as-contrast-enhancer.png")
    summary["contrast_enhancer_L41"] = {
        "above_seam_same_head_target_multiplier_median": float(np.median(multiplier)),
        "above_seam_target_bias_max_abs": float(
            max(abs(float(r["target_bias"])) for r in above)
        ),
        "without_bias_horizon_mass_median": float(horizon[0]),
        "with_bias_horizon_mass_median": float(horizon[1]),
        "without_bias_beyond_horizon_mass_median": float(far[0]),
        "with_bias_beyond_horizon_mass_median": float(far[1]),
    }


def echo_metrics():
    curves: dict[tuple[int, str], np.ndarray] = {}
    deltas = np.zeros((len(GLOBAL), len(ECHO_TEXTS)))
    head_fracs = np.zeros_like(deltas)
    for i, layer in enumerate(GLOBAL):
        for j, text in enumerate(ECHO_TEXTS):
            z = tier2(layer, text)
            bias = z["mean_bias"]
            curves[(layer, text)] = bias.mean(0)
            per_head = bias[:, 512:528].mean(1) - bias[:, 496:512].mean(1)
            deltas[i, j] = per_head.mean()
            head_fracs[i, j] = (per_head > 0).mean()
    controls = [128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896]
    consistent_positive = []
    for boundary in controls:
        n_layers = 0
        for layer in GLOBAL:
            vals = []
            for text in ECHO_TEXTS:
                curve = curves[(layer, text)]
                vals.append(
                    curve[boundary : boundary + 16].mean()
                    - curve[boundary - 16 : boundary].mean()
                )
            n_layers += int(np.all(np.asarray(vals) > 0))
        consistent_positive.append(n_layers)
    return curves, deltas, head_fracs, controls, consistent_positive


def figure_echo(summary: dict) -> None:
    curves, deltas, head_fracs, controls, consistent = echo_metrics()
    layer_curves = np.array(
        [
            np.mean([curves[(layer, text)] for text in ECHO_TEXTS], axis=0)
            for layer in GLOBAL
        ]
    )
    centered = layer_curves[:, 448:577] - layer_curves[:, 496:512].mean(1, keepdims=True)
    vmax = float(np.max(np.abs(centered)))

    fig, axes = plt.subplots(1, 3, figsize=(16.1, 4.9), layout="constrained")
    fig.suptitle("Global bias tables carry a learned echo of the local 512-token horizon")

    ax = axes[0]
    image = ax.imshow(
        centered,
        aspect="auto",
        origin="lower",
        extent=(448, 576, -0.5, len(GLOBAL) - 0.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
    )
    ax.axvline(512, color=INK, ls="--", lw=1.2)
    ax.set_yticks(range(len(GLOBAL)), [f"L{x}" for x in GLOBAL])
    ax.set(
        xlabel="raw token distance d",
        ylabel="global layer",
        title="A. Bias releases suppression immediately after d=512",
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("bias relative to d=496…511")

    ax = axes[1]
    colors = [BLUE, ORANGE, TEAL]
    for j, (text, color) in enumerate(zip(ECHO_TEXTS, colors)):
        ax.scatter(GLOBAL, deltas[:, j], color=color, s=28, label=text[3:])
    ax.plot(GLOBAL, deltas.mean(1), color=INK, lw=1.5)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set(
        xlabel="global layer",
        ylabel="mean bias after − before d=512",
        title="B. All 33 layer×text steps have the same sign",
    )
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(
        0.98,
        0.97,
        f"head support\n{head_fracs.min()*100:.0f}–{head_fracs.max()*100:.0f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        bbox=NOTE_BOX,
    )

    ax = axes[2]
    colors_bar = [ORANGE if d == 512 else LIGHT_GRAY for d in controls]
    ax.bar(controls, consistent, width=40, color=colors_bar, edgecolor="white")
    ax.axhline(np.median([n for d, n in zip(controls, consistent) if d != 512]), color=GRAY, ls="--", lw=1)
    ax.set(
        xlabel="candidate boundary distance",
        ylabel="layers positive in all 3 texts (of 11)",
        title="C. The d=512 step is specific, not generic roughness",
    )
    ax.set_ylim(0, 12)
    ax.set_xticks([128, 256, 384, 512, 640, 768, 896])
    ax.grid(axis="y")

    save(fig, "global-512-echo.png")
    summary["global_512_echo"] = {
        "positive_layer_text_cells": int((deltas > 0).sum()),
        "layer_text_cells_total": int(deltas.size),
        "step_delta_min": float(deltas.min()),
        "step_delta_max": float(deltas.max()),
        "head_positive_fraction_min": float(head_fracs.min()),
        "head_positive_fraction_max": float(head_fracs.max()),
        "control_boundaries": controls,
        "layers_consistently_positive_across_three_texts": consistent,
    }


def aggregate_profile(layer: int, field: str) -> np.ndarray:
    return np.mean([tier2(layer, text)[field].mean(0) for text in TEXTS], axis=0)


def figure_terminal_wall(summary: dict) -> None:
    c1 = read_json(ROUND4 / "C1_seam_direction_act.json")
    edge = np.array([c1[str(layer)]["edge_bias_mean"] for layer in GLOBAL])
    bias59 = aggregate_profile(59, "mean_bias")
    bias65 = aggregate_profile(65, "mean_bias")
    with65 = aggregate_profile(65, "mean_mass_with")
    without65 = aggregate_profile(65, "mean_mass_without")
    in_band, out_band = slice(1008, 1024), slice(1024, 1040)
    with_ratio = float(with65[in_band].mean() / with65[out_band].mean())
    without_ratio = float(without65[in_band].mean() / without65[out_band].mean())

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), layout="constrained")
    fig.suptitle("The seam follows a depth trajectory; L65 is a qualitatively different terminal wall")

    ax = axes[0]
    ax.axhline(0, color=GRAY, lw=1)
    ax.plot(GLOBAL, edge, marker="o", color=BLUE, lw=1.8)
    ax.scatter([11], [edge[GLOBAL.index(11)]], color=RED, s=55, zorder=4)
    ax.scatter([65], [edge[-1]], color=ORANGE, s=70, zorder=4)
    labels = {5: "+0.37", 11: "−0.10", 17: "≈0", 41: "+0.36", 59: "+0.28", 65: "+2.63"}
    for layer, label in labels.items():
        i = GLOBAL.index(layer)
        below_point = layer == 65 or edge[i] < 0
        ax.annotate(
            label,
            xy=(layer, edge[i]),
            xytext=(0, -8 if below_point else 7),
            textcoords="offset points",
            ha="center",
            va="top" if below_point else "bottom",
            fontsize=8,
            bbox=NOTE_BOX,
        )
    ax.set(
        xlabel="global layer",
        ylabel="mean bias at d=1008…1023",
        title="A. Inherited seam → zero dip → rebuild → wall",
    )
    ax.grid(axis="y")

    ax = axes[1]
    d = np.arange(1100)
    ax.plot(d, bias59[:1100], color=BLUE, lw=1.8, label="L59")
    ax.plot(d, bias65[:1100], color=ORANGE, lw=2.1, label="L65")
    ax.axvline(1024, color=RED, ls="--", lw=1, label="hard extent")
    ax.set(
        xlabel="token distance d",
        ylabel="mean live positional bias (logit)",
        title="B. L65 stays huge at the boundary, then becomes zero",
    )
    ax.grid(axis="y")
    ax.legend()

    ax = axes[2]
    dzoom = np.arange(940, 1101)
    ax.plot(dzoom, with65[dzoom], color=ORANGE, lw=2, label="with bias")
    ax.plot(dzoom, without65[dzoom], color=TEAL, lw=2, label="without bias")
    ax.axvline(1024, color=RED, ls="--", lw=1)
    ax.set(
        xlabel="token distance d",
        ylabel="mean attention mass per key",
        title="C. The 8× cliff is positional, not content-driven",
    )
    ax.grid(axis="y")
    ax.legend()
    ax.text(
        0.98,
        0.96,
        f"inside / outside\nwith bias {with_ratio:.1f}×\ncontent only {without_ratio:.2f}×",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        bbox=NOTE_BOX,
    )

    save(fig, "seam-depth-and-L65-wall.png")
    summary["terminal_wall"] = {
        "edge_bias_by_global_layer": dict(zip(map(str, GLOBAL), edge.tolist())),
        "L65_near_bias_d0_7": float(bias65[:8].mean()),
        "L65_edge_bias_d1008_1023": float(bias65[in_band].mean()),
        "L65_with_bias_mass_inside_outside_ratio": with_ratio,
        "L65_without_bias_mass_inside_outside_ratio": without_ratio,
        "L65_without_bias_inside_mass": float(without65[in_band].mean()),
        "L65_without_bias_outside_mass": float(without65[out_band].mean()),
    }


def heartbeat_metrics():
    target = 2334
    ratios, profiles, selected_heads = [], {}, []
    controls = np.r_[np.arange(target - 160, target - 40), np.arange(target + 40, target + 161)]
    for layer in GLOBAL:
        without = tier2(layer, "03_templated")["mean_mass_without"]
        background = without[:, controls].mean(1)
        per_head = without[:, target] / (background + 1e-300)
        ratio = float(np.percentile(per_head, 97.5))
        head = int(np.abs(per_head - ratio).argmin())
        ratios.append(ratio)
        selected_heads.append(head)
        profiles[layer] = without[head] / (background[head] + 1e-300)
    return target, np.asarray(ratios), profiles, selected_heads


def figure_heartbeat(summary: dict) -> None:
    target, ratios, profiles, heads = heartbeat_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    fig.suptitle("Beyond the bias horizon, repeated content is still retrieved in a NoPE-like regime")

    ax = axes[0]
    ax.axhline(1, color=GRAY, ls="--", lw=1)
    ax.plot(GLOBAL, ratios, marker="o", color=PURPLE, lw=1.8)
    ax.scatter([65], [ratios[-1]], color=ORANGE, s=65, zorder=4)
    ax.set(
        xlabel="global layer",
        ylabel="97.5th-percentile head: target / local background",
        title="A. The d≈2334 heartbeat spikes without any positional bias",
    )
    ax.grid(axis="y")
    ax.annotate(
        "L65",
        xy=(65, ratios[-1]),
        xytext=(-4, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=ORANGE,
        bbox=NOTE_BOX,
    )

    ax = axes[1]
    colors = {11: BLUE, 29: TEAL, 53: PURPLE, 65: ORANGE}
    for layer in [11, 29, 53, 65]:
        d = np.arange(2260, 2410)
        ax.plot(d, profiles[layer][d], color=colors[layer], lw=1.7, label=f"L{layer} · h{heads[GLOBAL.index(layer)]}")
    ax.axvline(target, color=RED, ls="--", lw=1, label="repeat distance 2334")
    ax.set(
        xlabel="token distance d (>1024 everywhere shown)",
        ylabel="content-only mass / distance-matched background",
        title="B. A sharp content match exists where the bias is identically zero",
    )
    ax.grid(axis="y")
    ax.legend(fontsize=8)

    save(fig, "content-retrieval-beyond-horizon.png")
    summary["heartbeat_content_only"] = {
        "target_distance": target,
        "high_head_ratio_97_5pct_by_layer": dict(zip(map(str, GLOBAL), ratios.tolist())),
        "representative_head_by_layer": dict(zip(map(str, GLOBAL), heads)),
        "note": "Computed from mean_mass_without; positional bias is exactly zero at d=2334.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_style()
    summary: dict[str, object] = {}

    c4 = c4_refit()
    c5 = c5_corrected()
    figure_audit(c4, c5, summary)
    figure_interface_activation(summary)

    entities, widths = needle_setup()
    figure_needle_retrieval(summary, entities)
    recon41 = needle_reconstruction(41, entities, widths)
    figure_contrast_enhancer(summary, recon41)

    figure_echo(summary)
    figure_terminal_wall(summary)
    figure_heartbeat(summary)

    summary["C4_per_layer"] = c4
    with (OUT / "revised_mechanism_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT / 'revised_mechanism_summary.json'}")


if __name__ == "__main__":
    main()
