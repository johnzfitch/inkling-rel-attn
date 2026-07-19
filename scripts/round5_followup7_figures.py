"""Render publication figures for the certified Round 5 follow-up seven.

The script is dump-first and CPU-only.  It authenticates the certified result
against the independent verifier, checks all 35 arm manifests and the fresh
job, then renders two figures for each F7 family plus two synthesis figures.
Every figure is exported as 300-dpi PNG and vector PDF/SVG.  A machine-readable
data bundle, captions, output manifest, and contact sheets are generated beside
the figures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "round5" / "followup7"
OUT = ANALYSIS / "figures"
RESULTS_PATH = ANALYSIS / "results.json"
VERIFICATION_PATH = ANALYSIS / "verification.json"
FROZEN_PATH = ANALYSIS / "frozen_inputs.npz"
FOLLOWUP_DUMP = ROOT / "dumps" / "round5" / "followup7"
PARENT_DUMP = ROOT / "dumps" / "round5" / "r5d" / "arms"

TEXTS = (
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
)
TEXT_LABELS = ("Prose", "Code", "Templated", "Multilingual", "Needles", "Random")

# Established project palette, adjusted for colorblind-safe categorical use.
INK = "#172033"
BLUE = "#2F6BDE"
SKY = "#56B4E9"
ORANGE = "#D97706"
VERMILLION = "#D1495B"
TEAL = "#16857A"
PURPLE = "#7C5CFC"
GRAY = "#6B7280"
MID_GRAY = "#9CA3AF"
LIGHT_GRAY = "#E5E7EB"
PALE_BLUE = "#DBEAFE"
PALE_ORANGE = "#FFEDD5"
PALE_GREEN = "#D1FAE5"
WHITE = "#FFFFFF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.array(values[name], copy=True) for name in values.files}


def followup_delta(arm: str, text: str) -> np.ndarray:
    path = FOLLOWUP_DUMP / "arms" / arm / "tokens" / f"{text}.npz"
    return load_npz(path)["delta_nll"].astype(np.float64)


def parent_delta(arm: str, text: str) -> np.ndarray:
    path = PARENT_DUMP / arm / "tokens" / f"{text}.npz"
    return load_npz(path)["delta_nll"].astype(np.float64)


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.6,
            "axes.edgecolor": MID_GRAY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "grid.color": LIGHT_GRAY,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "legend.frameon": True,
            "legend.facecolor": WHITE,
            "legend.edgecolor": "none",
            "legend.framealpha": 0.94,
            "legend.fontsize": 7.4,
            "figure.titlesize": 11.2,
            "figure.titleweight": "bold",
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Convert SVG text to glyph paths.  This prevents browser/viewer
            # font substitution from changing widths and reintroducing
            # collisions in the publication vectors.
            "svg.fonttype": "path",
            "mathtext.fontset": "dejavusans",
        }
    )


def clean_axis(ax: plt.Axes, grid: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, zorder=-5)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=9.4,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.canvas.draw()
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "facecolor": WHITE,
        }
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        if suffix == "svg":
            # Matplotlib wraps path data with trailing spaces; normalize the
            # generated text so repository whitespace checks stay clean.
            svg_text = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def annotate_bars(
    ax: plt.Axes,
    bars: Any,
    formatter: Callable[[float], str],
    *,
    padding: int = 3,
) -> None:
    labels = [formatter(float(bar.get_width() if bar.get_width() else bar.get_height())) for bar in bars]
    ax.bar_label(bars, labels=labels, padding=padding, fontsize=7.5, color=INK)


def annotate_horizontal_bars(
    ax: plt.Axes,
    bars: Any,
    *,
    formatter: Callable[[float], str],
    limit: float,
    dark_threshold: float = 0.78,
) -> None:
    """Label horizontal bars without clipping or crossing the figure edge."""

    for bar in bars:
        value = float(bar.get_width())
        y = float(bar.get_y() + bar.get_height() / 2)
        if value >= dark_threshold * limit:
            ax.text(
                value - 0.02 * limit,
                y,
                formatter(value),
                ha="right",
                va="center",
                color=WHITE,
                fontsize=7.4,
                fontweight="bold",
            )
        else:
            ax.text(
                value + 0.015 * limit,
                y,
                formatter(value),
                ha="left",
                va="center",
                color=INK,
                fontsize=7.4,
            )


def annotated_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    cmap: str,
    norm: mcolors.Normalize,
    fmt: str,
    cbar_label: str,
    colorbar: bool = True,
) -> Any:
    """Adapt Matplotlib's official annotated-heatmap pattern."""

    image = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=WHITE, linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    threshold = 0.58
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = float(data[row, col])
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = WHITE if luminance < threshold else INK
            text = ax.text(col, row, format(value, fmt), ha="center", va="center", color=color, fontsize=7.2)
            if color == WHITE:
                text.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground="#00000055")])
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, pad=0.025, shrink=0.86)
        bar.set_label(cbar_label)
    return image


def authenticate() -> tuple[dict[str, Any], dict[str, Any]]:
    results = read_json(RESULTS_PATH)
    verification = read_json(VERIFICATION_PATH)
    if verification.get("passed") is not True or verification.get("errors"):
        raise RuntimeError("follow-up result is not independently verified")
    if sha256(RESULTS_PATH) != verification.get("results_sha256"):
        raise RuntimeError("results hash does not match verifier")
    manifests = sorted((FOLLOWUP_DUMP / "arms").glob("*/manifest.json"))
    if len(manifests) != 35:
        raise RuntimeError(f"expected 35 arm manifests, found {len(manifests)}")
    for path in manifests:
        manifest = read_json(path)
        if manifest.get("complete") is not True:
            raise RuntimeError(f"incomplete arm manifest: {path}")
        if manifest.get("artifact_count") != len(manifest.get("artifacts", [])):
            raise RuntimeError(f"artifact-count mismatch: {path}")
    fresh = read_json(FOLLOWUP_DUMP / "fresh" / "manifest.json")
    if fresh.get("complete") is not True or fresh.get("artifact_count") != 8:
        raise RuntimeError("fresh job is incomplete")
    return results, verification


def build_derived(results: dict[str, Any]) -> dict[str, Any]:
    families = results["families"]
    f2_specs = (
        ("L23 + L29", "bias_off_L23_L29", (23, 29)),
        ("L29 + L35", "bias_off_L29_L35", (29, 35)),
        ("L23 + L35", "bias_off_L23_L35", (23, 35)),
        ("L23 + L29 + L35", "bias_off_L23_L29_L35", (23, 29, 35)),
    )
    f2_per_text = []
    for _label, joint, layers in f2_specs:
        row = []
        for text in TEXTS:
            value = followup_delta(joint, text)
            for layer in layers:
                value = value - parent_delta(f"bias_off_L{layer:02d}", text)
            row.append(float(value.mean()))
        f2_per_text.append(row)

    f3_arms = (
        "r_remove_mean_L29",
        "r_remove_centered_L29",
        "r_remove_carrier_all_L29",
        "r_remove_noncarrier_all_L29",
        "r_remove_carrier_mean_L29",
        "r_remove_noncarrier_mean_L29",
    )
    f3_per_text = [
        [float(followup_delta(arm, text).mean()) for text in TEXTS]
        for arm in f3_arms
    ]

    with np.load(FROZEN_PATH, allow_pickle=False) as frozen:
        query_positions = frozen["patch_query_positions"].astype(np.int64)
        head_order = frozen["head_order"].astype(np.int64)
    needle_parent = parent_delta("bias_off_L29", "05_needles")[query_positions]
    needle_query = followup_delta("bias_off_L29_patch_query", "05_needles")[query_positions]
    needle_sham = followup_delta("bias_off_L29_patch_sham", "05_needles")[query_positions]

    bias = float(families["F7-1"]["certified_bias_off_cost"])
    f4_costs = families["F7-4"]["costs"]
    rescue_curve = {
        "heads": [0, 8, 16, 32, 64],
        "stencil_rescue": [
            0.0,
            1.0 - f4_costs["head_top08_stencil_only_L29"] / bias,
            1.0 - f4_costs["head_top16_stencil_only_L29"] / bias,
            1.0 - f4_costs["head_top32_stencil_only_L29"] / bias,
            float(families["F7-1"]["stencil_rescue_fraction"]),
        ],
        "top16_full_rescue": 1.0 - f4_costs["head_top16_only_L29"] / bias,
    }

    return {
        "f2_labels": [item[0] for item in f2_specs],
        "f2_per_text": f2_per_text,
        "f3_arms": list(f3_arms),
        "f3_per_text": f3_per_text,
        "patch_query_positions": query_positions.tolist(),
        "patch_parent_delta_nll": needle_parent.tolist(),
        "patch_query_delta_nll": needle_query.tolist(),
        "patch_sham_delta_nll": needle_sham.tolist(),
        "head_order": head_order.tolist(),
        "rescue_curve": rescue_curve,
    }


def figure_f7_1a(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    row = results["families"]["F7-1"]
    labels = ["d = 0", "d = 1", "d = 2", "d = 3"]
    names = [f"d{index}_off_L29" for index in range(4)]
    bias = float(row["certified_bias_off_cost"])
    effects = np.array([row["singleton_inference"][name]["effect"] for name in names]) / bias * 100
    intervals = np.array([row["singleton_inference"][name]["ci95"] for name in names]) / bias * 100
    y = np.arange(4)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.7), layout="constrained")
    colors = [BLUE, GRAY, GRAY, GRAY]
    for yi, value, ci, color in zip(y, effects, intervals, colors):
        ax.errorbar(
            value,
            yi,
            xerr=np.array([[value - ci[0]], [ci[1] - value]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=4,
            markersize=6,
            zorder=3,
        )
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.axvline(100, color=VERMILLION, linestyle="--", linewidth=1.2)
    ax.text(
        100,
        3.32,
        "full effect = 100%",
        color=VERMILLION,
        ha="right",
        va="bottom",
        fontsize=7.2,
    )
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("Damage reproduced (% of full L29 bias-off)")
    ax.set_title("F7-1a | One distance cell reproduces 78% of the full effect", loc="left")
    for yi, value, ci, color in zip(y, effects, intervals, colors):
        ax.text(ci[1] + 1.8, yi, f"{value:.1f}%", va="center", fontsize=7.3, color=color)
    ax.set_ylim(-0.7, 3.7)
    ax.set_xlim(-6, 106)
    clean_axis(ax, "x")
    save_figure(fig, "f7-1a-distance-ablation-forest")


def figure_f7_1b(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    row = results["families"]["F7-1"]
    bias = row["certified_bias_off_cost"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.35), layout="constrained")

    ablation_labels = ["Remove d = 0", "Remove d = 1–3"]
    ablation = np.array([row["costs"]["d0_off_L29"], row["costs"]["d1_3_off_L29"]]) / bias
    ablation_y = np.arange(len(ablation_labels))[::-1]
    bars = axes[0].barh(ablation_y, ablation * 100, color=[BLUE, GRAY], height=0.52)
    axes[0].set_yticks(ablation_y, labels=ablation_labels)
    axes[0].axvline(100, color=VERMILLION, linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Damage reproduced (%)")
    axes[0].set_xlim(0, 105)
    axes[0].set_title("Necessity · remove cells", loc="left")
    annotate_horizontal_bars(axes[0], bars, formatter=lambda value: f"{value:.1f}%", limit=105)
    clean_axis(axes[0], "x")

    rescue_labels = ["Restore d = 0", "Restore d = 1–3", "Keep only d = 0–3"]
    rescue = np.array(
        [
            1 - row["costs"]["restore_d0_L29"] / bias,
            1 - row["costs"]["restore_d1_3_L29"] / bias,
            row["stencil_rescue_fraction"],
        ]
    )
    rescue_y = np.arange(len(rescue_labels))[::-1]
    bars = axes[1].barh(rescue_y, rescue * 100, color=[BLUE, GRAY, TEAL], height=0.52)
    axes[1].set_yticks(rescue_y, labels=rescue_labels)
    axes[1].axvline(100, color=INK, linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Damage rescued (%)")
    axes[1].set_xlim(0, 105)
    axes[1].set_title("Sufficiency · restore or retain", loc="left")
    annotate_horizontal_bars(axes[1], bars, formatter=lambda value: f"{value:.1f}%", limit=105)
    clean_axis(axes[1], "x")
    fig.suptitle("F7-1b | d = 0 is both necessary and nearly sufficient", x=0.01, ha="left")
    save_figure(fig, "f7-1b-stencil-necessity-sufficiency")


def figure_f7_2a(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    interactions = results["families"]["F7-2"]["interactions"]
    order = ["adjacent_23_29", "adjacent_29_35", "control_23_35", "triple"]
    labels = ["L23 + L29", "L29 + L35", "L23 + L35 control", "L23 + L29 + L35"]
    effects = np.array([interactions[name]["effect"] for name in order])
    intervals = np.array([interactions[name]["ci95"] for name in order])
    y = np.arange(4)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for yi, name, value, ci in zip(y, order, effects, intervals):
        color = GRAY if name == "control_23_35" else ORANGE
        ax.errorbar(
            value,
            yi,
            xerr=np.array([[value - ci[0]], [ci[1] - value]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            elinewidth=1.6,
            markersize=6,
        )
        ax.text(ci[1] + 0.025, yi, f"{value:+.3f}", va="center", fontsize=7.5, color=color)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("Joint-removal interaction ΔNLL beyond singleton sum")
    ax.set_title("F7-2a | Backup interactions require L29; the shoulder-only control is near zero", loc="left")
    ax.set_xlim(-0.08, 1.35)
    clean_axis(ax, "x")
    save_figure(fig, "f7-2a-shoulder-interaction-forest")


def figure_f7_2b(_results: dict[str, Any], derived: dict[str, Any]) -> None:
    data = np.asarray(derived["f2_per_text"], dtype=np.float64)
    norm = mcolors.LogNorm(vmin=max(0.002, float(data.min())), vmax=float(data.max()))
    fig, ax = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
    annotated_heatmap(
        ax,
        data,
        list(derived["f2_labels"]),
        list(TEXT_LABELS),
        cmap="YlOrBr",
        norm=norm,
        fmt=".3f",
        cbar_label="Interaction ΔNLL (log color scale)",
    )
    ax.set_title("F7-2b | Every text shows positive backup interaction; strength varies by orders of magnitude", loc="left", pad=14)
    save_figure(fig, "f7-2b-shoulder-interaction-heatmap")


def figure_f7_3a(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    row = results["families"]["F7-3"]
    names = [
        "r_remove_mean_L29",
        "r_remove_noncarrier_mean_L29",
        "r_remove_centered_L29",
        "r_remove_carrier_mean_L29",
        "r_remove_carrier_all_L29",
        "r_remove_noncarrier_all_L29",
    ]
    labels = [
        "Static mean removed",
        "Non-carrier mean removed",
        "Token variation removed",
        "Carrier mean removed",
        "Carrier removed (all tokens)",
        "Carrier retained alone",
    ]
    values = np.array([row["ratios_to_bias_off"][name] for name in names]) * 100
    colors = [BLUE, TEAL, GRAY, GRAY, PURPLE, ORANGE]
    fig, ax = plt.subplots(figsize=(7.3, 4.25), layout="constrained")
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels=labels)
    ax.axvline(50, color=INK, linestyle="--", linewidth=1.0, label="registered ≥50% gate")
    ax.axvline(25, color=MID_GRAY, linestyle=":", linewidth=1.0, label="registered ≤25% gate")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Causal cost relative to full L29 bias-off (%)")
    ax.set_title("F7-3a | The static non-carrier mean carries the causal r-object", loc="left")
    annotate_horizontal_bars(ax, bars, formatter=lambda value: f"{value:.1f}%", limit=105)
    ax.text(25, -0.52, "≤25% null gate", ha="center", va="bottom", fontsize=6.6, color=GRAY)
    ax.text(50, -0.52, "≥50% load-bearing gate", ha="center", va="bottom", fontsize=6.6, color=INK)
    clean_axis(ax, "x")
    save_figure(fig, "f7-3a-r-component-causal-costs")


def figure_f7_3b(_results: dict[str, Any], derived: dict[str, Any]) -> None:
    data = np.asarray(derived["f3_per_text"], dtype=np.float64)
    labels = [
        "Remove mean",
        "Remove centered",
        "Remove carrier",
        "Keep carrier only",
        "Remove carrier mean",
        "Remove non-carrier mean",
    ]
    vmax = float(np.max(np.abs(data)))
    norm = mcolors.SymLogNorm(linthresh=0.005, linscale=0.7, vmin=-vmax, vmax=vmax, base=10)
    fig, ax = plt.subplots(figsize=(7.5, 4.35), layout="constrained")
    annotated_heatmap(
        ax,
        data,
        labels,
        list(TEXT_LABELS),
        cmap="RdBu_r",
        norm=norm,
        fmt="+.3f",
        cbar_label="Mean ΔNLL (symmetric-log color scale)",
    )
    ax.set_title("F7-3b | Static-mean costs recur across texts; the needle mixture is the exception", loc="left", pad=14)
    save_figure(fig, "f7-3b-r-components-by-text")


def figure_f7_4a(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    row = results["families"]["F7-4"]
    names = [f"head_q{index}_off_L29" for index in range(1, 5)]
    values = np.array([row["costs"][name] for name in names])
    labels = ["Q1 · highest score", "Q2", "Q3", "Q4"]
    y = np.arange(4)[::-1]
    share = float(values[0] / values.sum())
    fold_low = float(values[0] / values[1:].max())
    fold_high = float(values[0] / values[1:].min())
    fig, ax = plt.subplots(figsize=(7.2, 3.65), layout="constrained")
    for yi, value, color, size in zip(y, values, [BLUE, GRAY, GRAY, GRAY], [78, 42, 42, 42]):
        ax.hlines(yi, 4e-4, value, color=color, linewidth=1.8, alpha=0.78)
        ax.scatter(value, yi, color=color, s=size, zorder=3, edgecolor=WHITE, linewidth=0.6)
        ax.text(value * 1.16, yi, f"{value:.4f}", va="center", fontsize=7.3, color=color)
    ax.set_xscale("log")
    ax.set_xlim(4e-4, 7e-2)
    ax.set_xticks([5e-4, 1e-3, 5e-3, 1e-2, 5e-2], labels=["0.0005", "0.001", "0.005", "0.01", "0.05"])
    ax.set_yticks(y, labels=labels)
    ax.set_ylim(-0.65, 3.65)
    ax.set_xlabel("Mean ΔNLL when the quartile is removed (log scale)")
    ax.set_title(f"F7-4a | The top quartile carries {share:.0%} of summed quartile-ablation cost", loc="left")
    ax.text(
        0.98,
        0.33,
        f"Q1 is {fold_low:.0f}–{fold_high:.0f}×\nlarger than Q2–Q4",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8,
        color=BLUE,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": PALE_BLUE, "edgecolor": "none"},
    )
    clean_axis(ax, "x")
    save_figure(fig, "f7-4a-head-quartile-localization")


def figure_f7_4b(_results: dict[str, Any], derived: dict[str, Any]) -> None:
    curve = derived["rescue_curve"]
    heads = np.asarray(curve["heads"])
    rescue = np.asarray(curve["stencil_rescue"]) * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.85), layout="constrained")
    ax.plot(heads, rescue, color=BLUE, linewidth=2.0, marker="o", markersize=6, label="d = 0–3 stencil only")
    ax.scatter([16], [curve["top16_full_rescue"] * 100], color=ORANGE, s=58, marker="D", zorder=4, label="Top 16, full bias")
    ax.axvspan(0, 8, color=PALE_BLUE, alpha=0.55, zorder=-4)
    for x, y in zip(heads[1:], rescue[1:]):
        ax.text(x, min(y + 2.0, 102), f"{y:.1f}%", ha="center", va="bottom", fontsize=7.2, color=BLUE)
    ax.annotate(
        f"full-bias top 16\n{curve['top16_full_rescue']:.1%}",
        xy=(16, curve["top16_full_rescue"] * 100),
        xytext=(23, 82),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 1.0},
        color=ORANGE,
        fontsize=7.4,
        ha="center",
    )
    ax.set_xticks(heads)
    ax.set_xlim(-2, 66)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Number of retained heads, ranked before outcomes")
    ax.set_ylabel("Full bias-off damage rescued (%)")
    ax.set_title("F7-4b | Eight heads rescue 86%; gains are nearly saturated by 16", loc="left")
    ax.text(4, 8, "highest-ranked\n8 heads", ha="center", va="center", fontsize=7.0, color=BLUE)
    clean_axis(ax, "y")
    save_figure(fig, "f7-4b-head-rescue-saturation")


def figure_f7_5a(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    row = results["families"]["F7-5"]
    labels = ["Registered query patch", "Seeded sham patch"]
    records = [row["query_patch"], row["sham_patch"]]
    effects = np.array([record["rescue_fraction"] for record in records]) * 100
    parent_costs = np.array(
        [record["absolute_rescue"] / record["rescue_fraction"] for record in records],
        dtype=np.float64,
    )
    intervals = np.array(
        [
            np.asarray(record["absolute_rescue_ci95"], dtype=np.float64) / parent_cost * 100
            for record, parent_cost in zip(records, parent_costs)
        ]
    )
    y = np.array([1, 0])
    fig, ax = plt.subplots(figsize=(7.2, 3.45), layout="constrained")
    for yi, value, ci, color in zip(y, effects, intervals, [TEAL, GRAY]):
        ax.errorbar(
            value,
            yi,
            xerr=np.array([[value - ci[0]], [ci[1] - value]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            elinewidth=1.7,
            markersize=6.5,
        )
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("Original retrieval damage rescued (%)")
    ax.set_title("F7-5a | Patching 24 L29 query states rescues 54%; the sham rescues none", loc="left")
    for yi, record, color in zip(y, records, [TEAL, GRAY]):
        ax.text(
            max(intervals[1 - yi]) + 3.0,
            yi,
            f"{record['rescue_fraction']:+.1%}",
            va="center",
            color=color,
            fontsize=8,
        )
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(min(-18, float(intervals.min()) - 5), max(82, float(intervals.max()) + 12))
    clean_axis(ax, "x")
    save_figure(fig, "f7-5a-query-patch-rescue-forest")


def figure_f7_5b(_results: dict[str, Any], derived: dict[str, Any]) -> None:
    parent = np.asarray(derived["patch_parent_delta_nll"], dtype=np.float64)
    query = np.asarray(derived["patch_query_delta_nll"], dtype=np.float64)
    sham = np.asarray(derived["patch_sham_delta_nll"], dtype=np.float64)
    lo = float(min(parent.min(), query.min(), sham.min())) - 0.08
    hi = float(max(parent.max(), query.max(), sham.max())) + 0.08
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.65), sharex=True, sharey=True, layout="constrained")
    for ax, patched, title, color in (
        (axes[0], query, "Query-state patch", TEAL),
        (axes[1], sham, "Seeded sham patch", GRAY),
    ):
        ax.plot([lo, hi], [lo, hi], color=MID_GRAY, linestyle="--", linewidth=1.0, zorder=-2)
        ax.scatter(parent, patched, s=30, color=color, alpha=0.78, edgecolor=WHITE, linewidth=0.45)
        ax.scatter(parent.mean(), patched.mean(), s=100, marker="*", color=ORANGE, edgecolor=INK, linewidth=0.6, zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, loc="left")
        clean_axis(ax, "both")
    axes[0].set_ylabel("Damage after patch (ΔNLL)")
    axes[0].text(0.03, 0.94, "pooled damage ↓54%", transform=axes[0].transAxes, color=TEAL, fontsize=7.4, va="top", fontweight="bold")
    axes[1].text(0.03, 0.94, "pooled change −3%", transform=axes[1].transAxes, color=GRAY, fontsize=7.4, va="top", fontweight="bold")
    axes[0].text(0.03, 0.87, "below diagonal = rescue", transform=axes[0].transAxes, color=GRAY, fontsize=6.8, va="top")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.supxlabel("Damage without patch at the 24 registered query tokens (ΔNLL)")
    fig.suptitle("F7-5b | Query-state patching moves damaged tokens toward baseline; sham patching does not", x=0.01, ha="left")
    save_figure(fig, "f7-5b-query-token-paired-scatter")


def figure_f7_6a(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    cells = results["families"]["F7-6"]["kernel_gain_correlation"]["clock_loto_L53_L59"]["cells"]
    data = np.array(
        [[cells[f"L{layer:02d}:{text}"] for text in TEXTS] for layer in (53, 59)],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.25), layout="constrained")
    annotated_heatmap(
        ax,
        data,
        ["L53", "L59"],
        list(TEXT_LABELS),
        cmap="magma_r",
        norm=mcolors.Normalize(vmin=0, vmax=0.5),
        fmt=".3f",
        cbar_label="Held-out |gain/log-position correlation|",
    )
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if data[row, col] < 0.20:
                patch = FancyBboxPatch(
                    (col - 0.47, row - 0.47),
                    0.94,
                    0.94,
                    boxstyle="round,pad=0.01",
                    fill=False,
                    edgecolor=TEAL,
                    linewidth=2.2,
                )
                ax.add_patch(patch)
    ax.set_title("F7-6a | Only 1 of 12 held-out clock correlations passes the registered <0.20 gate", loc="left", pad=14)
    save_figure(fig, "f7-6a-clock-loto-transfer-heatmap")


def figure_f7_6b(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    row = results["families"]["F7-6"]
    stats = row["kernel_gain_correlation"]
    order = [
        "clock_union_L53",
        "clock_union_L59",
        "clock_union_L53_L59",
        "clock_pertext_L53_L59",
        "clock_loto_L53_L59",
        "clock_sham6_L53_L59",
    ]
    labels = ["Union L53", "Union L59", "Union joint", "Per-text joint", "LOTO joint", "6-D sham"]
    medians = np.array([stats[name]["median"] for name in order])
    maxima = np.array([stats[name]["maximum"] for name in order])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), layout="constrained", gridspec_kw={"width_ratios": [1.15, 0.85]})
    y = np.arange(len(order))[::-1]
    for yi, med, maximum, name in zip(y, medians, maxima, order):
        color = GRAY if "sham" in name else (VERMILLION if "loto" in name else BLUE)
        axes[0].plot([med, maximum], [yi, yi], color=color, linewidth=1.8)
        axes[0].scatter([med], [yi], color=color, s=34, zorder=3)
        axes[0].scatter([maximum], [yi], color=color, marker="|", s=95, zorder=3)
    axes[0].axvline(0.20, color=TEAL, linestyle="--", linewidth=1.0, label="LOTO pass gate")
    axes[0].axvline(0.50, color=GRAY, linestyle=":", linewidth=1.0, label="sham median gate")
    axes[0].set_yticks(y, labels=labels)
    axes[0].set_xlabel("Held-out |gain/log-position correlation|\n(point = median; cap = maximum)")
    axes[0].set_title("Geometry does not transfer", loc="left")
    axes[0].legend(loc="upper right")
    clean_axis(axes[0], "x")
    panel_label(axes[0], "a")

    behavior_order = ["clock_union_L53_L59", "clock_pertext_L53_L59", "clock_loto_L53_L59"]
    behavior_labels = ["Union joint", "Per-text joint", "LOTO joint"]
    null_ceiling = 0.005
    costs = np.array([row["behavior_costs"][name] for name in behavior_order]) / null_ceiling * 100
    behavior_y = np.arange(len(behavior_labels))[::-1]
    bars = axes[1].barh(behavior_y, costs, color=TEAL, height=0.55)
    axes[1].set_yticks(behavior_y, labels=behavior_labels)
    axes[1].axvline(100, color=VERMILLION, linestyle="--", linewidth=1.0)
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Behavioral cost (% of null ceiling)")
    axes[1].set_title("Behavior remains negligible", loc="left")
    annotate_horizontal_bars(axes[1], bars, formatter=lambda value: f"{value:.1f}%", limit=105)
    clean_axis(axes[1], "x")
    panel_label(axes[1], "b")
    fig.suptitle("F7-6b | Clock directions are readable, but not load-bearing at 8k", x=0.01, ha="left")
    save_figure(fig, "f7-6b-clock-geometry-behavior-dissociation")


def draw_single_effect(
    ax: plt.Axes,
    effect: float,
    interval: list[float],
    *,
    color: str,
    xlabel: str,
    title: str,
    threshold: tuple[float, float] | None = None,
) -> None:
    ax.errorbar(
        effect,
        0,
        xerr=np.array([[effect - interval[0]], [interval[1] - effect]]),
        fmt="o",
        color=color,
        ecolor=color,
        capsize=5,
        elinewidth=1.8,
        markersize=7,
    )
    ax.axvline(0, color=INK, linewidth=1.0)
    if threshold:
        ax.axvspan(threshold[0], threshold[1], color=PALE_ORANGE, alpha=0.7, zorder=-4)
        ax.axvline(threshold[0], color=ORANGE, linestyle="--", linewidth=0.9)
        ax.axvline(threshold[1], color=ORANGE, linestyle="--", linewidth=0.9)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.text(0.5, 0.82, f"{effect:+.4f}", transform=ax.transAxes, ha="center", color=color, fontweight="bold")
    clean_axis(ax, "x")


def figure_f7_7a(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    row = results["families"]["F7-7"]
    rank = row["ranking"]
    accuracy = rank["accuracy"]
    calibration = row["calibration"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.05), layout="constrained")
    draw_single_effect(
        axes[0],
        rank["effect"],
        rank["ci95"],
        color=VERMILLION,
        xlabel="Δ mean log1p(target rank)",
        title="Target rank worsens",
    )
    draw_single_effect(
        axes[1],
        accuracy["effect"] * 100,
        [value * 100 for value in accuracy["ci95"]],
        color=VERMILLION,
        xlabel="Δ top-1 accuracy (percentage points)",
        title="Top-1 accuracy falls",
    )
    draw_single_effect(
        axes[2],
        calibration["effect"],
        calibration["ci95"],
        color=ORANGE,
        xlabel="Δ ECE20",
        title="ECE20 shift",
        threshold=(-0.01, 0.01),
    )
    fig.suptitle("F7-7a | Bias-off harms ranking and accuracy; the calibration shift stays below 0.01", x=0.01, ha="left")
    save_figure(fig, "f7-7a-ranking-versus-calibration")


def class_forest_panel(ax: plt.Axes, rows: list[tuple[str, dict[str, Any]]], title: str, color: str) -> None:
    expanded: list[tuple[str, float, list[float], bool]] = []
    for label, record in rows:
        primary = float(record["matched_contrast"])
        primary_ci = record["ci95"]
        secondary = record["target_aligned_secondary"]
        sec_value = float(secondary["matched_contrast"])
        sec_ci = secondary["ci95"]
        expanded.extend(
            [
                (f"{label} · primary", primary, primary_ci, True),
                (f"{label} · secondary", sec_value, sec_ci, False),
            ]
        )
    y = np.arange(len(expanded))[::-1]
    for yi, (_label, value, ci, primary) in zip(y, expanded):
        ax.errorbar(
            value,
            yi,
            xerr=np.array([[value - ci[0]], [ci[1] - value]]),
            fmt="o",
            color=color,
            markerfacecolor=color if primary else WHITE,
            markeredgecolor=color,
            ecolor=color,
            capsize=3,
            markersize=5.2,
        )
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y, labels=[label for label, *_ in expanded])
    ax.set_xlabel("Matched class contrast in ΔNLL")
    ax.set_title(title, loc="left")
    clean_axis(ax, "x")


def figure_f7_7b(results: dict[str, Any], _derived: dict[str, Any]) -> None:
    classes = results["families"]["F7-7"]["fresh_classes"]
    slack = [
        ("First content", classes["07b_slack_multi:first_content"]),
        ("Pronouns", classes["07b_slack_multi:pronouns"]),
    ]
    math = [
        ("Provider-unit starts", classes["08_math_llm:unit_starts"]),
        ("Pronouns", classes["08_math_llm:pronouns"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.15), layout="constrained")
    class_forest_panel(axes[0], slack, "Fresh Slack · green = registered >0", BLUE)
    class_forest_panel(axes[1], math, "Fresh assistant math · green = registered >0", ORANGE)
    axes[0].set_xlim(-0.36, 0.36)
    axes[1].set_xlim(-1.35, 2.05)
    for ax, upper in zip(axes, (0.36, 2.05)):
        ax.axvspan(0, upper, color=PALE_GREEN, alpha=0.45, zorder=-6)
    fig.suptitle("F7-7b | Fresh class effects are null or reversed—not the registered positive effect", x=0.01, ha="left")
    save_figure(fig, "f7-7b-fresh-class-replication-forest")


def node(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    textcolor: str = INK,
    linewidth: float = 1.2,
) -> FancyBboxPatch:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", color=textcolor, fontsize=8.2, linespacing=1.25)
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str, style: str = "-") -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.5,
        color=color,
        linestyle=style,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)


def figure_synthesis_a(_results: dict[str, Any], _derived: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.5), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    node(ax, (0.03, 0.43), 0.17, 0.18, "Static non-carrier mean\n78–80% of causal cost", facecolor=PALE_BLUE, edgecolor=BLUE)
    node(ax, (0.27, 0.43), 0.16, 0.18, "Top stencil heads\n8 / 16 / 32 rescue\n86% / 93% / 97%", facecolor=PALE_BLUE, edgecolor=BLUE)
    node(ax, (0.50, 0.43), 0.14, 0.18, "L29 d = 0 update\n98% rescue", facecolor=PALE_GREEN, edgecolor=TEAL)
    node(ax, (0.71, 0.43), 0.13, 0.18, "Query state\n54% recall rescue", facecolor=PALE_GREEN, edgecolor=TEAL)
    node(ax, (0.90, 0.43), 0.08, 0.18, "Content\nmatch", facecolor=PALE_GREEN, edgecolor=TEAL)
    arrow(ax, (0.20, 0.52), (0.27, 0.52), color=BLUE)
    arrow(ax, (0.43, 0.52), (0.50, 0.52), color=BLUE)
    arrow(ax, (0.64, 0.52), (0.71, 0.52), color=TEAL)
    arrow(ax, (0.84, 0.52), (0.90, 0.52), color=TEAL)

    node(ax, (0.43, 0.76), 0.11, 0.11, "L23 shoulder", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    node(ax, (0.58, 0.76), 0.11, 0.11, "L35 shoulder", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    arrow(ax, (0.49, 0.76), (0.55, 0.62), color=ORANGE)
    arrow(ax, (0.63, 0.76), (0.60, 0.62), color=ORANGE)
    ax.text(0.56, 0.91, "redundant backups: joint interactions +0.35 to +1.04", ha="center", color=ORANGE, fontsize=8)

    node(ax, (0.06, 0.10), 0.15, 0.12, "Carrier mean\n0.2%", facecolor="#F3F4F6", edgecolor=MID_GRAY, textcolor=GRAY)
    node(ax, (0.25, 0.10), 0.15, 0.12, "Centered variation\n1.9%", facecolor="#F3F4F6", edgecolor=MID_GRAY, textcolor=GRAY)
    node(ax, (0.44, 0.10), 0.15, 0.12, "d = 1–3 joint\n0.6%", facecolor="#F3F4F6", edgecolor=MID_GRAY, textcolor=GRAY)
    node(ax, (0.63, 0.10), 0.17, 0.12, "Late clock\n≤ 0.000585 NLL", facecolor="#F3F4F6", edgecolor=MID_GRAY, textcolor=GRAY)
    ax.text(0.43, 0.03, "measurable but not load-bearing at 8k", ha="center", color=GRAY, fontsize=8, style="italic")
    ax.set_title("Synthesis A | At 8k, a static d=0 signal builds query states used by later content retrieval", loc="left", pad=8)
    save_figure(fig, "synthesis-a-revised-mechanism-map")


def figure_synthesis_b(_results: dict[str, Any], _derived: dict[str, Any]) -> None:
    rows = ["F7-1", "F7-2", "F7-3", "F7-4", "F7-5", "F7-6", "F7-7"]
    columns = ["Primary clause", "Secondary clause", "Control / transfer", "Family verdict"]
    values = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, -1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=int,
    )
    labels = np.array(
        [
            ["d=0\nnecessary", "d=0–3\nsufficient", "d=1–3\nnull", "PASS"],
            ["L23+29\ninteraction", "L29+35\ninteraction", "triple\ninteraction", "PASS"],
            ["static mean\nload-bearing", "centered\nnull", "carrier\nnull", "PASS"],
            ["top quartile\nlocalized", "top 16\nrescues 93%", "—", "PASS"],
            ["query patch\nrescues 54%", "sham patch\nnull", "—", "PASS"],
            ["LOTO transfer\nfails", "sham control\npasses", "behavior\nnull", "FAIL"],
            ["ranking\npasses", "calibration\nfails", "fresh classes\nfail", "FAIL"],
        ],
        dtype=object,
    )
    cmap = mcolors.ListedColormap(["#F3F4F6", "#FEE2E2", "#D1FAE5"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.2, 5.1), layout="constrained")
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            weight = "bold" if col == 3 else "normal"
            color = GRAY if values[row, col] == -1 else INK
            ax.text(col, row, labels[row, col], ha="center", va="center", fontsize=7.3, color=color, fontweight=weight, linespacing=1.15)
    legend = [
        Line2D([0], [0], marker="s", linestyle="", color="#D1FAE5", markersize=10, label="registered pass"),
        Line2D([0], [0], marker="s", linestyle="", color="#FEE2E2", markersize=10, label="registered fail"),
        Line2D([0], [0], marker="s", linestyle="", color="#F3F4F6", markersize=10, label="not applicable"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=3)
    ax.set_title("Synthesis B | Five mechanistic families pass; transfer and class-generalization families fail", loc="left", pad=14)
    save_figure(fig, "synthesis-b-registered-evidence-matrix")


FIGURE_INFO: dict[str, dict[str, Any]] = {
    "f7-1a-distance-ablation-forest": {
        "title": "F7-1a — signed distance ablation forest",
        "caption": "Registered singleton effects and 95% block-bootstrap intervals. Removing d=0 is the only Holm-confirmed singleton and reproduces 77.8% of full L29 bias-off damage.",
        "sources": ["results.json:F7-1.singleton_inference", "results.json:F7-1.certified_bias_off_cost"],
    },
    "f7-1b-stencil-necessity-sufficiency": {
        "title": "F7-1b — stencil necessity and sufficiency",
        "caption": "Complementary ablation and restoration arms resolve the causal near field to d=0. Restoring d=0 rescues 98.1%; retaining only d=0..3 rescues 99.0%.",
        "sources": ["results.json:F7-1.costs", "results.json:F7-1.stencil_rescue_fraction"],
    },
    "f7-2a-shoulder-interaction-forest": {
        "title": "F7-2a — shoulder interaction forest",
        "caption": "Registered joint-minus-singleton interactions with 95% block-bootstrap intervals. Every L29-containing set is strongly super-additive; the L23+L35 control is small.",
        "sources": ["results.json:F7-2.interactions"],
    },
    "f7-2b-shoulder-interaction-heatmap": {
        "title": "F7-2b — shoulder interactions by text",
        "caption": "Raw per-text joint-minus-singleton mean ΔNLL. A log color scale reveals positive redundancy across all 24 cells while preserving the 800-fold range.",
        "sources": ["followup arm token dumps", "certified R5-D singleton token dumps"],
    },
    "f7-3a-r-component-causal-costs": {
        "title": "F7-3a — r-component causal costs",
        "caption": "Pooled component costs relative to full L29 bias-off. Static mean and non-carrier mean reproduce most damage; centered and carrier removals are null-scale.",
        "sources": ["results.json:F7-3.ratios_to_bias_off"],
    },
    "f7-3b-r-components-by-text": {
        "title": "F7-3b — r components by text",
        "caption": "Raw per-text mean ΔNLL for all six registered r-space interventions, shown with a symmetric-log color scale and exact cell labels. The needle mixture is the clear low-cost exception to the otherwise stable static-mean mechanism.",
        "sources": ["followup r-component token dumps"],
    },
    "f7-4a-head-quartile-localization": {
        "title": "F7-4a — head-quartile localization",
        "caption": "Pooled cost of removing each frozen 16-head quartile on a logarithmic ΔNLL axis. The highest pre-outcome stencil-score quartile carries 94% of the summed quartile-ablation cost and costs 30–67 times more than Q2–Q4.",
        "sources": ["results.json:F7-4.costs"],
    },
    "f7-4b-head-rescue-saturation": {
        "title": "F7-4b — nested head rescue",
        "caption": "Damage rescued by retaining the nested top-8/16/32/64 d=0..3 stencils. Rescue reaches 93.4% with 16 heads and 97.5% with 32.",
        "sources": ["results.json:F7-4.costs", "frozen_inputs.npz:head_order"],
    },
    "f7-5a-query-patch-rescue-forest": {
        "title": "F7-5a — query-state rescue forest",
        "caption": "Needle-recall rescue as a percentage of the common parent damage, with registered 95% bootstrap intervals transformed to the same scale. Query patching rescues 54.1%; the seeded sham interval spans zero.",
        "sources": ["results.json:F7-5"],
    },
    "f7-5b-query-token-paired-scatter": {
        "title": "F7-5b — token-level query mediation",
        "caption": "Each point is one of 24 registered needle queries. Query-state patching moves the pooled mean below the equality line; sham patching does not.",
        "sources": ["frozen_inputs.npz:patch_query_positions", "parent and patch token dumps"],
    },
    "f7-6a-clock-loto-transfer-heatmap": {
        "title": "F7-6a — held-out clock transfer",
        "caption": "Residual median-head clock gain correlation after projecting the five-text LOTO basis. Only one of twelve cells meets the registered <0.20 transfer criterion.",
        "sources": ["results.json:F7-6.kernel_gain_correlation.clock_loto_L53_L59"],
    },
    "f7-6b-clock-geometry-behavior-dissociation": {
        "title": "F7-6b — clock geometry versus behavior",
        "caption": "Fitted union/per-text bases flatten kernel drift, whereas LOTO transfer fails. Behavioral costs are normalized to the registered 0.005 NLL null ceiling; every real joint intervention remains below 12% of it.",
        "sources": ["results.json:F7-6.kernel_gain_correlation", "results.json:F7-6.behavior_costs"],
    },
    "f7-7a-ranking-versus-calibration": {
        "title": "F7-7a — ranking versus calibration",
        "caption": "Registered full-vocabulary effects and 95% intervals. Rank and accuracy worsen decisively; ECE changes significantly but remains inside the ±0.01 promotion band.",
        "sources": ["results.json:F7-7.ranking", "results.json:F7-7.calibration"],
    },
    "f7-7b-fresh-class-replication-forest": {
        "title": "F7-7b — fresh class replication",
        "caption": "Matched query- and target-aligned class contrasts on fresh Slack and assistant-math text. No registered primary contrast is positive; Slack pronouns invert significantly in the secondary alignment.",
        "sources": ["results.json:F7-7.fresh_classes"],
    },
    "synthesis-a-revised-mechanism-map": {
        "title": "Synthesis A — revised 8k mechanism",
        "caption": "Post-verdict graphical interpretation: a static non-carrier mean drives a sparse d=0 stencil in top heads, improving query construction for downstream content matching; shoulders provide redundancy.",
        "sources": ["certified F7-1 through F7-6 results"],
    },
    "synthesis-b-registered-evidence-matrix": {
        "title": "Synthesis B — registered evidence matrix",
        "caption": "Clause-level disposition of all seven families. F7-1 through F7-5 pass; F7-6 fails transfer despite behavioral nullity; F7-7 passes ranking but fails calibration promotion and fresh replication.",
        "sources": ["verification.json:verdicts", "results.json:families"],
    },
}


FIGURE_FUNCTIONS = [
    figure_f7_1a,
    figure_f7_1b,
    figure_f7_2a,
    figure_f7_2b,
    figure_f7_3a,
    figure_f7_3b,
    figure_f7_4a,
    figure_f7_4b,
    figure_f7_5a,
    figure_f7_5b,
    figure_f7_6a,
    figure_f7_6b,
    figure_f7_7a,
    figure_f7_7b,
    figure_synthesis_a,
    figure_synthesis_b,
]


def source_provenance() -> dict[str, str]:
    sources = [RESULTS_PATH, VERIFICATION_PATH, FROZEN_PATH, Path(__file__)]
    followup_arms = (
        "bias_off_L23_L29",
        "bias_off_L29_L35",
        "bias_off_L23_L35",
        "bias_off_L23_L29_L35",
        "r_remove_mean_L29",
        "r_remove_centered_L29",
        "r_remove_carrier_all_L29",
        "r_remove_noncarrier_all_L29",
        "r_remove_carrier_mean_L29",
        "r_remove_noncarrier_mean_L29",
        "bias_off_L29_patch_query",
        "bias_off_L29_patch_sham",
    )
    parent_arms = ("bias_off_L23", "bias_off_L29", "bias_off_L35")
    sources.extend(FOLLOWUP_DUMP / "arms" / name / "manifest.json" for name in followup_arms)
    sources.extend(PARENT_DUMP / name / "manifest.json" for name in parent_arms)
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sources}


def write_figure_data(results: dict[str, Any], verification: dict[str, Any], derived: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "kind": "round5_followup7_publication_figure_data",
        "certified": True,
        "results_sha256": sha256(RESULTS_PATH),
        "verification_sha256": sha256(VERIFICATION_PATH),
        "artifact_count_rehashed": verification["artifact_count_rehashed"],
        "verdicts": verification["verdicts"],
        "source_sha256": source_provenance(),
        "certified_families": results["families"],
        "derived_raw_views": derived,
    }
    with (OUT / "figure_data.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_captions() -> None:
    lines = [
        "# Round 5 follow-up seven publication figures",
        "",
        "Status: **independently verified; post-verdict visualizations**",
        "",
        "Each numbered discovery has two standalone figures. PNG files are",
        "300 dpi; PDF and SVG are vector exports from the same Matplotlib",
        "artists. SVG typography is stored as glyph paths for cross-viewer",
        "stability. Interpretive synthesis panels are labeled post-verdict and",
        "do not change any registered disposition.",
        "",
    ]
    for index, (stem, info) in enumerate(FIGURE_INFO.items(), start=1):
        lines.extend(
            [
                f"## {index}. {info['title']}",
                "",
                f"![{info['title']}](./{stem}.png)",
                "",
                info["caption"],
                "",
                "Sources: " + "; ".join(f"`{source}`" for source in info["sources"]) + ".",
                "",
                f"Exports: [`PNG`](./{stem}.png) · [`PDF`](./{stem}.pdf) · [`SVG`](./{stem}.svg)",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproduction",
            "",
            "```powershell",
            ".\\.venv-tier2\\Scripts\\python.exe scripts/round5_followup7_figures.py",
            "```",
            "",
            "`figure_data.json` preserves the plotted summary and raw-derived",
            "values; `figure_manifest.json` records source/output hashes and",
            "pixel dimensions. `contact-sheet-1.png` and `contact-sheet-2.png`",
            "are QA views only, not manuscript figures.",
            "",
        ]
    )
    (OUT / "FIGURES.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def make_contact_sheets(stems: list[str]) -> None:
    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 26)
    label_font = ImageFont.truetype(font_path, 22)
    for sheet_index, start in enumerate(range(0, len(stems), 8), start=1):
        group = stems[start : start + 8]
        thumb_w, thumb_h = 920, 500
        pad, label_h = 28, 42
        canvas = Image.new("RGB", (2 * thumb_w + 3 * pad, 4 * (thumb_h + label_h + pad) + 3 * pad), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 6), f"Round 5 follow-up figures · QA contact sheet {sheet_index}", font=font, fill=INK)
        y_offset = pad + 30
        for local_index, stem in enumerate(group):
            row, col = divmod(local_index, 2)
            x = pad + col * (thumb_w + pad)
            y = y_offset + row * (thumb_h + label_h + pad)
            with Image.open(OUT / f"{stem}.png") as source:
                image = source.convert("RGB")
                image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                px = x + (thumb_w - image.width) // 2
                py = y + label_h + (thumb_h - image.height) // 2
                canvas.paste(image, (px, py))
            draw.text((x, y), f"{start + local_index + 1:02d} · {stem}", font=label_font, fill=INK)
        canvas.save(OUT / f"contact-sheet-{sheet_index}.png", dpi=(150, 150))


def write_manifest() -> None:
    records = []
    for path in sorted(OUT.iterdir()):
        if not path.is_file() or path.name == "figure_manifest.json":
            continue
        record: dict[str, Any] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                record["pixels"] = [image.width, image.height]
                record["mode"] = image.mode
        records.append(record)
    payload = {
        "schema_version": 1,
        "kind": "round5_followup7_publication_figure_manifest",
        "figure_count": len(FIGURE_INFO),
        "formats_per_figure": ["png", "pdf", "svg"],
        "png_dpi": 300,
        "matplotlib_version": matplotlib.__version__,
        "source_sha256": source_provenance(),
        "outputs": records,
    }
    with (OUT / "figure_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_outputs() -> None:
    for stem in FIGURE_INFO:
        for suffix in ("png", "pdf", "svg"):
            path = OUT / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size < 1_000:
                raise RuntimeError(f"missing or implausibly small output: {path}")
    if len(FIGURE_FUNCTIONS) != 16 or len(FIGURE_INFO) != 16:
        raise RuntimeError("expected exactly 16 registered figure definitions")


def main() -> None:
    base_style()
    OUT.mkdir(parents=True, exist_ok=True)
    results, verification = authenticate()
    derived = build_derived(results)
    for function in FIGURE_FUNCTIONS:
        function(results, derived)
    validate_outputs()
    write_figure_data(results, verification, derived)
    write_captions()
    make_contact_sheets(list(FIGURE_INFO))
    write_manifest()
    print(
        json.dumps(
            {
                "passed": True,
                "figures": len(FIGURE_INFO),
                "exports": len(FIGURE_INFO) * 3,
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
