"""Create follow-on figures from the completed Inkling analyses.

This script is analysis-only: it reads the Round 3/4 summaries and the Tier-2
dumps, performs no model execution, and writes reproducible PNG figures plus a
small JSON file of the aggregate values used in the captions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "explanatory_figures"
TIER2 = ROOT / "dumps" / "tier2"
GLOBAL = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]

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


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
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


def tier2_dump(layer: int, text: str):
    return np.load(TIER2 / f"layer{layer:02d}_{text}_s8192.npz", allow_pickle=True)


def figure_seam(summary: dict) -> dict:
    """Show the hard-zero bias boundary and its causal attention-mass step."""
    findings = read_json(ROOT / "analysis" / "tier2" / "tier2_findings.json")
    seam = findings["seam"]
    layers = np.array(GLOBAL)

    bias_in = np.array(
        [[seam[str(layer)][text]["bias_in_mean"] for text in TEXTS] for layer in layers]
    )
    attributable = np.array(
        [
            [seam[str(layer)][text]["bias_attrib_step_mean"] for text in TEXTS]
            for layer in layers
        ]
    )
    content_only = np.array(
        [[seam[str(layer)][text]["without_step_mean"] for text in TEXTS] for layer in layers]
    )
    positive_heads = np.array(
        [
            [seam[str(layer)][text]["heads_positive_frac"] for text in TEXTS]
            for layer in layers
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), layout="constrained")
    fig.suptitle("The learned positional bias ends abruptly at d = 1,024")

    ax = axes[0]
    distance = np.arange(960, 1089)
    for layer, color in zip([5, 11, 65], [BLUE, GRAY, RED]):
        curves = []
        for text in TEXTS:
            with tier2_dump(layer, text) as data:
                curves.append(data["mean_bias"].mean(axis=0)[distance])
        curves = np.asarray(curves)
        mean = curves.mean(axis=0)
        ax.plot(distance, mean, color=color, lw=2, label=f"L{layer}")
        ax.annotate(
            f"L{layer}",
            xy=(distance[0], mean[0]),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            bbox=NOTE_BOX,
        )
        ax.fill_between(
            distance,
            mean - curves.std(axis=0),
            mean + curves.std(axis=0),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax.axvline(1024, color=INK, lw=1.2, ls="--")
    ax.axhline(0, color=LIGHT_GRAY, lw=1)
    ax.text(1027, ax.get_ylim()[1] * 0.88, "hard zero", color=INK)
    ax.set(xlabel="backward distance d", ylabel="mean signed bias (logit)", title="A. The raw bias snaps to zero")
    ax.grid(axis="y")

    ax = axes[1]
    mean = bias_in.mean(axis=1)
    std = bias_in.std(axis=1)
    ax.errorbar(layers, mean, yerr=std, color=BLUE, marker="o", lw=1.8, capsize=2)
    ax.scatter(layers, mean, c=[ORANGE if x in GLOBAL else BLUE for x in layers], zorder=3)
    ax.axhline(0, color=INK, lw=1)
    ax.annotate(
        "near-null; only majority-sign exception",
        xy=(11, mean[1]),
        xytext=(0.05, 0.72),
        textcoords="axes fraction",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": GRAY},
        color=GRAY,
        bbox=NOTE_BOX,
    )
    ax.annotate(
        "strongest late layer",
        xy=(65, mean[-1]),
        xytext=(-10, -26),
        textcoords="offset points",
        ha="right",
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        bbox=NOTE_BOX,
    )
    ax.set(xlabel="global layer", ylabel="bias averaged over d = 1008…1023", title="B. The in-window side is usually boosted")
    ax.grid(axis="y")

    ax = axes[2]
    scale = 1e4
    for values, color, marker, label in [
        (attributable, RED, "o", "bias-attributable"),
        (content_only, GRAY, "s", "without positional bias"),
    ]:
        mean = values.mean(axis=1) * scale
        std = values.std(axis=1) * scale
        ax.errorbar(layers, mean, yerr=std, color=color, marker=marker, lw=1.8, capsize=2, label=label)
    ax.axhline(0, color=INK, lw=1)
    ax.set(xlabel="global layer", ylabel="inside − outside attention mass (×10⁻⁴)", title="C. Only the biased softmax has a seam")
    ax.grid(axis="y")
    ax.legend(loc="upper left")

    save(fig, "seam-causal-depth.png")

    mean_attr = attributable.mean(axis=1)
    mean_content = content_only.mean(axis=1)
    summary["seam"] = {
        "global_layers_positive_mean_step": int((mean_attr > 0).sum()),
        "global_layers_majority_heads_positive": int((positive_heads.mean(axis=1) > 0.5).sum()),
        "global_layers_total": int(len(layers)),
        "near_null_exception": int(layers[np.argmin(mean_attr)]),
        "largest_step_layer": int(layers[np.argmax(mean_attr)]),
        "median_abs_bias_to_content_step_ratio": float(
            np.median(np.abs(mean_attr) / (np.abs(mean_content) + 1e-12))
        ),
        "mean_positive_head_fraction": float(positive_heads.mean()),
    }
    return summary


def stable_zero_crossing(curve: np.ndarray, start: int = 8, run: int = 16) -> int | None:
    for d in range(start, len(curve) - run):
        if np.all(curve[d : d + run] < 0):
            return d
    return None


def figure_rising_bias(summary: dict) -> dict:
    """Resolve the magnitude/sign ambiguity in the old 'rising' label."""
    taxonomy = read_json(ROOT / "analysis" / "round3" / "head_taxonomy_v2.json")["trunk_layers"]
    findings = read_json(ROOT / "analysis" / "tier2" / "tier2_findings.json")["pushout"]
    rising_layers = sorted(int(layer) for layer in findings)

    signed_curves = []
    for layer in rising_layers:
        rising = np.array([c == "rising" for c in taxonomy[str(layer)]["head_class"]])
        per_text = []
        for text in TEXTS:
            with tier2_dump(layer, text) as data:
                per_text.append(data["mean_bias"][rising].mean(axis=0))
        signed_curves.append(np.mean(per_text, axis=0))
    signed_curves = np.asarray(signed_curves)
    median_curve = np.median(signed_curves, axis=0)
    q10, q90 = np.percentile(signed_curves, [10, 90], axis=0)
    zero_crossing = stable_zero_crossing(median_curve)

    fields = ["content_slope", "bias_effect_slope", "net_slope"]
    slopes = {
        field: np.asarray(
            [[findings[str(layer)][text][field] for text in TEXTS] for layer in rising_layers]
        )
        for field in fields
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8), layout="constrained")
    fig.suptitle("‘Rising’ operator magnitude does not mean outward attention")

    ax = axes[0]
    d = np.arange(signed_curves.shape[1])
    ax.fill_between(d, q10, q90, color=BLUE, alpha=0.16, label="10–90% of rising layers")
    ax.plot(d, median_curve, color=BLUE, lw=2.2, label="median signed bias")
    ax.axhline(0, color=INK, lw=1)
    if zero_crossing is not None:
        ax.axvline(zero_crossing, color=RED, lw=1.2, ls="--")
        ax.text(
            0.03,
            0.06,
            f"sustained negative after d≈{zero_crossing}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=RED,
            bbox=NOTE_BOX,
        )
    ax.set(xlabel="backward distance d", ylabel="signed bias (logit)", title="A. In situ, far tokens are suppressed")
    ax.grid(axis="y")
    ax.legend(loc="upper right")

    ax = axes[1]
    x = np.asarray(rising_layers)
    for field, color, marker, label in [
        ("content_slope", GRAY, "s", "content-only slope"),
        ("bias_effect_slope", RED, "o", "bias contribution"),
        ("net_slope", BLUE, "^", "realized attention"),
    ]:
        values = slopes[field] * 1e6
        mean = values.mean(axis=1)
        lo, hi = values.min(axis=1), values.max(axis=1)
        ax.fill_between(x, lo, hi, color=color, alpha=0.08, linewidth=0)
        ax.plot(x, mean, color=color, marker=marker, ms=4, lw=1.5, label=label)
    ax.axhline(0, color=INK, lw=1)
    ax.set(xlabel="layer with ≥1 ‘rising’ head", ylabel="far-field slope (×10⁻⁶ per token)", title="B. Bias and content point inward together")
    ax.grid(axis="y")
    ax.legend(loc="lower left")

    save(fig, "rising-means-inward.png")

    all_bias_slopes = slopes["bias_effect_slope"].ravel()
    summary["rising_bias"] = {
        "rising_layers": int(len(rising_layers)),
        "layer_text_pairs_with_negative_bias_slope": int((all_bias_slopes < 0).sum()),
        "layer_text_pairs_total": int(all_bias_slopes.size),
        "median_signed_bias_sustained_zero_crossing": zero_crossing,
        "median_bias_slope_x1e6": float(np.median(all_bias_slopes) * 1e6),
        "median_content_slope_x1e6": float(np.median(slopes["content_slope"]) * 1e6),
    }
    return summary


def figure_interface(summary: dict) -> dict:
    """Show table rank and the null weight-side interface-utilization test."""
    data = read_json(ROOT / "analysis" / "round4" / "curiosity" / "C2_interface_utilization.json")
    layers = np.arange(66)
    table_energy = np.asarray([data[str(layer)]["proj_S2"] for layer in layers])
    table_share = table_energy / table_energy.sum(axis=1, keepdims=True)
    rank = np.asarray([data[str(layer)]["proj_participation_rank"] for layer in layers])
    vestigial = np.asarray([data[str(layer)]["vestigial_energy_fraction"] for layer in layers])
    baseline = np.asarray([data[str(layer)]["uniform_baseline_vestigial"] for layer in layers])

    fig = plt.figure(figsize=(13.4, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.18, 1])
    fig.suptitle("The table narrows to a bottleneck; weight geometry does not show co-adaptation")

    ax = fig.add_subplot(grid[0, :])
    image = ax.imshow(
        np.log10(np.clip(table_share.T, 1e-5, None)),
        aspect="auto",
        origin="lower",
        extent=(-0.5, 65.5, 0.5, 16.5),
        cmap="magma",
        vmin=-4,
        vmax=0,
    )
    for layer in GLOBAL:
        ax.axvline(layer, color="white", alpha=0.28, lw=0.7)
    ax.set(xlabel="trunk layer", ylabel="table mode", title="A. Share of rel_logits_proj energy (log₁₀; vertical lines are global layers)")
    ax.set_yticks([1, 4, 8, 12, 16])
    cbar = fig.colorbar(image, ax=ax, pad=0.012, fraction=0.02)
    cbar.set_label("log₁₀ energy share")

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(layers, rank, color=BLUE, lw=2)
    ax.scatter(GLOBAL, rank[GLOBAL], marker="D", color=ORANGE, s=28, label="global layer", zorder=3)
    ax.axhline(2, color=LIGHT_GRAY, lw=1, ls="--")
    ax.set(xlabel="trunk layer", ylabel="participation-ratio rank", title="B. Effective table rank collapses with depth")
    ax.set_ylim(0.8, max(4.1, rank.max() + 0.2))
    ax.grid(axis="y")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(layers, vestigial, color=RED, lw=2, label="r_proj energy in weak table modes")
    ax.plot(layers, baseline, color=GRAY, lw=1.5, ls="--", label="uniform-injection baseline")
    ax.fill_between(layers, vestigial, baseline, where=vestigial >= baseline, color=RED, alpha=0.12)
    ax.fill_between(layers, vestigial, baseline, where=vestigial < baseline, color=TEAL, alpha=0.10)
    ax.scatter(GLOBAL, vestigial[GLOBAL], marker="D", color=ORANGE, s=25, zorder=3)
    med_vest = float(np.median(vestigial))
    med_base = float(np.median(baseline))
    ax.set(xlabel="trunk layer", ylabel="fraction of injected energy", title="C. Weight-side C2 is indistinguishable from its uniform null")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y")
    ax.legend(loc="lower right")
    ax.text(1, 0.53, f"median {med_vest:.3f} vs null {med_base:.3f}", color=INK)

    save(fig, "interface-bottleneck.png")

    summary["interface"] = {
        "median_effective_rank_layers_0_28": float(np.median(rank[:29])),
        "median_effective_rank_layers_30_65": float(np.median(rank[30:])),
        "median_vestigial_input_energy_layers_30_65": float(np.median(vestigial[30:])),
        "median_vestigial_input_energy_all_layers": med_vest,
        "median_uniform_null_all_layers": med_base,
        "layers_beating_uniform_injection": int((vestigial < baseline).sum()),
        "layers_total": 66,
        "interpretation": "The weight-level utilization statistic carries no information relative to its uniform baseline; use C2-act for live co-adaptation.",
    }
    return summary


def add_phase_spans(ax: plt.Axes) -> None:
    ax.axvspan(-0.5, 13.5, color=RED, alpha=0.05)
    ax.axvspan(13.5, 29.5, color=ORANGE, alpha=0.06)
    ax.axvspan(29.5, 65.5, color=BLUE, alpha=0.045)


def figure_head_sharing(summary: dict) -> dict:
    """Contrast output-class uniformity with hidden read-subspace sharing."""
    data = read_json(ROOT / "analysis" / "round4" / "curiosity" / "C3_shared_circuit.json")
    layers = np.arange(66)
    effective = np.asarray([data[str(layer)]["eff_num_read_subspaces"] for layer in layers])
    similarity = np.asarray([data[str(layer)]["mean_offdiag_sim"] for layer in layers])
    rising_cross = data["_rising_cross_layer"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.7), sharex=True, layout="constrained")
    fig.suptitle("Heads converge on the same output shape before they share an implementation")

    ax = axes[0]
    add_phase_spans(ax)
    ax.plot(layers, effective, color=PURPLE, lw=2)
    ax.scatter(GLOBAL, effective[GLOBAL], marker="D", color=ORANGE, s=27, label="global layer", zorder=3)
    ax.set(xlabel="trunk layer", ylabel="effective number of read subspaces", title="A. Early heads use many private hidden directions")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    ax.text(1, 6, "early rising heads:\n295 effective directions\nacross 748 heads", color=PURPLE)

    ax = axes[1]
    add_phase_spans(ax)
    ax.plot(layers, similarity, color=TEAL, lw=2)
    ax.scatter(GLOBAL, similarity[GLOBAL], marker="D", color=ORANGE, s=27, zorder=3)
    ax.set(xlabel="trunk layer", ylabel="mean pairwise top-2 subspace similarity", title="B. Deep heads increasingly reuse hidden subspaces")
    ax.set_ylim(0, max(0.5, similarity.max() + 0.03))
    ax.grid(axis="y")

    for ax in axes:
        ax.text(6.5, ax.get_ylim()[1] * 0.96, "early", ha="center", va="top", color=RED)
        ax.text(21.5, ax.get_ylim()[1] * 0.96, "transition", ha="center", va="top", color=ORANGE)
        ax.text(47.5, ax.get_ylim()[1] * 0.96, "deep", ha="center", va="top", color=BLUE)

    save(fig, "head-subspace-sharing.png")

    summary["head_sharing"] = {
        "early_median_effective_subspaces_layers_0_13": float(np.median(effective[:14])),
        "deep_median_effective_subspaces_layers_30_65": float(np.median(effective[30:])),
        "early_median_pairwise_similarity_layers_0_13": float(np.median(similarity[:14])),
        "deep_median_pairwise_similarity_layers_30_65": float(np.median(similarity[30:])),
        "rising_heads_across_early_layers": int(rising_cross["n_rising_heads"]),
        "effective_directions_across_rising_heads": float(rising_cross["eff_num_directions"]),
    }
    return summary


def figure_transition_audit(summary: dict) -> dict:
    """Correct C5 for SVD sign and compare every layer on raw d=0..511."""
    trajectory = read_json(ROOT / "analysis" / "round4" / "curiosity" / "C5_flip_trajectory.json")
    taxonomy = read_json(ROOT / "analysis" / "round3" / "head_taxonomy_v2.json")["trunk_layers"]
    destination = np.asarray(trajectory["layers"])
    reported_step = np.asarray(trajectory["curve_step_mode0"])

    curves = []
    for layer in range(66):
        d = np.load(ROOT / "dumps" / "round3" / "mode_curves" / f"layer{layer:02d}.npz")
        curve = (d["S"][0] * d["Vt"][0])[:512].astype(np.float64)
        curves.append(curve / (np.linalg.norm(curve) + 1e-12))
    corrected_step = np.asarray(
        [
            min(
                np.linalg.norm(curves[i] - curves[i - 1]),
                np.linalg.norm(curves[i] + curves[i - 1]),
            )
            for i in range(1, 66)
        ]
    )
    transition = (destination >= 13) & (destination <= 28)
    corrected_ratio = float(corrected_step[transition].mean() / np.median(corrected_step))

    rising = np.asarray([taxonomy[str(layer)]["class_counts"]["rising"] / 64 for layer in range(66)])
    decay = np.asarray([taxonomy[str(layer)]["class_counts"]["decay"] / 64 for layer in range(66)])
    other = 1 - rising - decay

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.8, 7.0),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.25, 1]},
    )
    fig.suptitle("The depth transition survives after removing sign and scope artifacts")

    ax = axes[0]
    ax.axvspan(13, 28, color=ORANGE, alpha=0.08, label="registered transition band")
    ax.plot(destination, reported_step, color=LIGHT_GRAY, lw=1.2, label="reported: signed + extent-rescaled")
    ax.plot(destination, corrected_step, color=PURPLE, lw=1.8, label="corrected: sign-invariant, raw d≤511")
    ax.set(ylabel="mode-0 curve change to previous layer", title="A. The corrected transition-band mean remains 1.93× the median")
    ax.grid(axis="y")
    ax.legend(ncol=2, loc="upper right")
    ax.text(31, 1.48, f"2.75× → {corrected_ratio:.2f}×", color=PURPLE)

    ax = axes[1]
    ax.axvspan(13, 28, color=ORANGE, alpha=0.08)
    ax.plot(range(66), rising, color=RED, lw=2, label="rising head fraction")
    ax.plot(range(66), decay, color=BLUE, lw=2, label="decay head fraction")
    ax.plot(range(66), other, color=GRAY, lw=1.3, label="other / near / flat")
    ax.scatter(GLOBAL, decay[GLOBAL], marker="D", color=ORANGE, s=25, zorder=3)
    ax.set(xlabel="trunk layer", ylabel="fraction of 64 heads", title="B. The corrected spike coincides with the rising-to-decay handoff")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="center right")

    save(fig, "transition-boundary-audit.png")

    summary["transition_audit"] = {
        "round4_reported_transition_to_overall_median_ratio": float(trajectory["transition_vs_median_ratio"]),
        "corrected_sign_invariant_common_raw_support_ratio": corrected_ratio,
        "interpretation": "The original 2.75x magnitude was inflated by SVD sign canonicalization and extent rescaling. At 1.93x the phase-transition verdict still clears the registered 1.5 threshold.",
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_style()
    summary: dict[str, object] = {}
    summary = figure_seam(summary)
    summary = figure_rising_bias(summary)
    summary = figure_interface(summary)
    summary = figure_head_sharing(summary)
    summary = figure_transition_audit(summary)
    with (OUT / "figure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT / 'figure_summary.json'}")


if __name__ == "__main__":
    main()
