"""Build the five corrected corpus-v2 aperture figures from sealed artifacts.

The script validates the committed result reports, private class freezes, and
corrected aperture manifest before deriving any display-only values. It never
executes the model and does not write raw token-level data to the analysis
directory.
"""

from __future__ import annotations

import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "analysis" / "round5" / "corpus_v2_corrected"
OUT = REPORT_DIR / "figures"
DUMP = ROOT / "dumps" / "round5" / "corpus_v2_corrected_aperture"
MANIFEST_PATH = DUMP / "manifest.json"
READOUTS_PATH = REPORT_DIR / "readouts.json"
DEPTH_PATH = REPORT_DIR / "depth_readouts.json"
V20_CLASSES_PATH = ROOT / "corpus_v2" / "classes.json"
V21_CLASSES_PATH = ROOT / "corpus_v2" / "depth_classes.json"

LAYERS = np.array([17, 23, 29, 35, 41, 47, 53, 59], dtype=np.int64)
SHALLOW = np.array([17, 23, 29], dtype=np.int64)
DEEP = np.array([35, 41, 47, 53, 59], dtype=np.int64)
MID = np.array([23, 29, 35, 41, 47], dtype=np.int64)
TRACE_START = 4096
TRACE_STOP = 5632
BIN_SIZE = 256

V20_TEXT = "07_slack_human"
V21_TEXT = "07b_slack_multi"
MATH_TEXT = "08_math_llm"
CLASS_ORDER = ["speaker_labels", "first_content", "pronouns"]
CLASS_LABELS = {
    "speaker_labels": "Speaker labels",
    "first_content": "First content",
    "pronouns": "Pronouns",
}

SPEAKER = "#D55E00"
FIRST = "#009E73"
PRONOUN = "#6F5BD7"
CLASS_COLORS = {
    "speaker_labels": SPEAKER,
    "first_content": FIRST,
    "pronouns": PRONOUN,
}
V20_COLOR = "#4C78A8"
V21_COLOR = "#F58518"
SLACK_COLOR = "#2F6BDE"
MATH_COLOR = "#7C5CFC"
NEGATIVE = "#3B82F6"
POSITIVE = "#D95F02"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
GRID = "#E5E7EB"
INK = "#172033"
DISPLAY_QUANTUM = Decimal("0.001")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-14):
        raise AssertionError(f"{label}: {actual} != {expected}")


def format_effect(value: float) -> str:
    """Format signed effects to three decimals using decimal half-up rounding."""
    rounded = Decimal(str(value)).quantize(DISPLAY_QUANTUM, rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return f"{rounded:+.3f}"


def require_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    readouts = read_json(READOUTS_PATH)
    depth = read_json(DEPTH_PATH)
    manifest = read_json(MANIFEST_PATH)
    v20_classes = read_json(V20_CLASSES_PATH)
    v21_classes = read_json(V21_CLASSES_PATH)

    manifest_sha = sha256_file(MANIFEST_PATH)
    if (
        readouts.get("kind") != "corpus_v2_a6_corrected_registered_readouts"
        or depth.get("kind") != "round5_a6_depth_resolved_readouts"
        or manifest.get("kind") != "corpus_v2_a6_corrected_aperture_dump"
        or manifest.get("complete") is not True
        or readouts.get("aperture_manifest_sha256") != manifest_sha
        or depth.get("aperture_manifest_sha256") != manifest_sha
        or readouts.get("classes_sha256") != sha256_file(V20_CLASSES_PATH)
        or depth.get("depth_classes_sha256") != sha256_file(V21_CLASSES_PATH)
        or manifest.get("v20_classes_sha256") != sha256_file(V20_CLASSES_PATH)
        or manifest.get("depth_classes_sha256") != sha256_file(V21_CLASSES_PATH)
        or manifest.get("layers") != LAYERS.tolist()
    ):
        raise RuntimeError("corrected reports, class freezes, or manifest are stale")

    for text in (V20_TEXT, V21_TEXT, MATH_TEXT):
        for layer in LAYERS:
            key = f"L{layer:02d}_{text}"
            record = manifest["files"].get(key)
            if record is None:
                raise RuntimeError(f"missing manifest record: {key}")
            path = DUMP / record["path"]
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"aperture hash mismatch: {path}")

    return readouts, depth, manifest, v20_classes, v21_classes


def midrank_percentiles(values: np.ndarray, bin_size: int = BIN_SIZE) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for start in range(0, len(values), bin_size):
        stop = min(start + bin_size, len(values))
        block = values[start:stop]
        order = np.argsort(block, kind="mergesort")
        sorted_values = block[order]
        ranks = np.empty(len(block), dtype=np.float64)
        cursor = 0
        while cursor < len(block):
            end = cursor + 1
            while end < len(block) and sorted_values[end] == sorted_values[cursor]:
                end += 1
            ranks[order[cursor:end]] = ((cursor + 1) + end) / 2.0
            cursor = end
        result[start:stop] = (ranks - 0.5) / len(block)
    return result


def load_scores(text: str, manifest: dict[str, Any]) -> dict[int, np.ndarray]:
    scores: dict[int, np.ndarray] = {}
    for layer in LAYERS:
        key = f"L{layer:02d}_{text}"
        path = DUMP / manifest["files"][key]["path"]
        with np.load(path, allow_pickle=False) as data:
            aperture = data["aperture_full"]
        if aperture.shape != (8192,) or not np.isfinite(aperture).all():
            raise RuntimeError(f"invalid aperture array: {path}")
        scores[int(layer)] = midrank_percentiles(aperture)
    return scores


def effect_profile(
    scores: dict[int, np.ndarray], positions: np.ndarray
) -> dict[int, float]:
    return {
        int(layer): float(np.median(scores[int(layer)][positions]) - 0.5)
        for layer in LAYERS
    }


def band_median(profile: dict[int, float], band: np.ndarray) -> float:
    return float(np.median([profile[int(layer)] for layer in band]))


def class_positions(
    v20_classes: dict[str, Any], v21_classes: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    v20_starts = np.asarray(v20_classes["classes"]["message_starts"], dtype=np.int64)
    v20 = {
        "speaker_labels": v20_starts,
        "first_content": v20_starts[v20_starts + 2 < 8192] + 2,
        "pronouns": np.asarray(v20_classes["classes"]["pronouns"], dtype=np.int64),
    }
    v21 = {
        name: np.asarray(v21_classes["classes"][name], dtype=np.int64)
        for name in CLASS_ORDER
    }
    for arm, classes in (("v2.0", v20), ("v2.1", v21)):
        for name, positions in classes.items():
            if (
                positions.size == 0
                or np.unique(positions).size != positions.size
                or np.any(positions < 0)
                or np.any(positions >= 8192)
            ):
                raise RuntimeError(f"invalid {arm} positions for {name}")
    return v20, v21


def derive_data() -> tuple[dict[str, Any], dict[str, Any]]:
    readouts, depth, manifest, v20_freeze, v21_freeze = require_inputs()
    positions20, positions21 = class_positions(v20_freeze, v21_freeze)
    scores = {
        V20_TEXT: load_scores(V20_TEXT, manifest),
        V21_TEXT: load_scores(V21_TEXT, manifest),
        MATH_TEXT: load_scores(MATH_TEXT, manifest),
    }

    profiles20 = {
        name: effect_profile(scores[V20_TEXT], positions20[name]) for name in CLASS_ORDER
    }
    profiles21 = {
        name: effect_profile(scores[V21_TEXT], positions21[name]) for name in CLASS_ORDER
    }
    profiles_math = {
        name: effect_profile(scores[MATH_TEXT], positions21[name]) for name in CLASS_ORDER
    }

    primary = {record["class_name"]: record for record in depth["primary"]}
    controls = {
        next(
            record["class_name"]
            for record in depth["primary"]
            if record["prediction"] == control["prediction"]
        ): control
        for control in depth["crossed_math_controls"]
    }
    for name in CLASS_ORDER:
        record = primary[name]
        for layer_text, expected in record["per_layer_effects"].items():
            assert_close(
                profiles21[name][int(layer_text)], expected, f"{name} L{layer_text}"
            )
        assert_close(
            band_median(profiles21[name], SHALLOW),
            record["shallow_band_effect"],
            f"{name} shallow band",
        )
        assert_close(
            band_median(profiles21[name], DEEP),
            record["deep_band_effect"],
            f"{name} deep band",
        )
        control = controls[name]
        for layer_text, expected in control["per_layer_effects"].items():
            assert_close(
                profiles_math[name][int(layer_text)],
                expected,
                f"{name} math L{layer_text}",
            )
        assert_close(
            band_median(
                profiles_math[name],
                SHALLOW if name == "pronouns" else DEEP,
            ),
            control["effect"],
            f"{name} registered math control band",
        )

    averaged_mid_v20 = np.mean(
        np.stack([scores[V20_TEXT][int(layer)] for layer in MID]), axis=0
    )
    v20_registered = {record["name"]: record for record in readouts["class_primary"]}
    assert_close(
        float(np.median(averaged_mid_v20[positions20["speaker_labels"]]) - 0.5),
        v20_registered["message_starts"]["effect"],
        "v2.0 message-start registered effect",
    )
    assert_close(
        float(np.median(averaged_mid_v20[positions20["pronouns"]]) - 0.5),
        v20_registered["pronouns"]["effect"],
        "v2.0 pronoun registered effect",
    )

    null_low = min(primary[name]["null_quantiles"]["q025"] for name in CLASS_ORDER)
    null_high = max(primary[name]["null_quantiles"]["q975"] for name in CLASS_ORDER)
    deep_trace = np.mean(
        np.stack([scores[V21_TEXT][int(layer)] for layer in DEEP]), axis=0
    )
    trace_boundaries = positions21["speaker_labels"]
    trace_boundaries = trace_boundaries[
        (trace_boundaries >= TRACE_START) & (trace_boundaries < TRACE_STOP)
    ]

    deep20 = {name: band_median(profiles20[name], DEEP) for name in CLASS_ORDER}
    deep21 = {name: band_median(profiles21[name], DEEP) for name in CLASS_ORDER}
    math_controls = {name: float(controls[name]["effect"]) for name in CLASS_ORDER}
    correlations = readouts["p_v2_2"]["primary"]

    figure_data = {
        "schema_version": 1,
        "kind": "corpus_v2_corrected_figure_data",
        "generated_by_sha256": sha256_file(Path(__file__)),
        "display": {
            "effect_format": "signed fixed-point with 3 decimal places",
            "effect_rounding": "decimal ROUND_HALF_UP",
        },
        "sources": {
            "readouts_sha256": sha256_file(READOUTS_PATH),
            "depth_readouts_sha256": sha256_file(DEPTH_PATH),
            "aperture_manifest_sha256": sha256_file(MANIFEST_PATH),
            "v20_classes_sha256": sha256_file(V20_CLASSES_PATH),
            "v21_classes_sha256": sha256_file(V21_CLASSES_PATH),
        },
        "figure_1": {
            "arm": V21_TEXT,
            "layers": LAYERS.tolist(),
            "shallow_band": SHALLOW.tolist(),
            "deep_band": DEEP.tolist(),
            "effects": {
                name: [profiles21[name][int(layer)] for layer in LAYERS]
                for name in CLASS_ORDER
            },
            "null_envelope_95": [float(null_low), float(null_high)],
            "null_definition": "envelope of the three committed registered band-test 95% permutation-null intervals",
            "thresholds": {"widening": 0.10, "narrowing": -0.05},
            "unmeasured_bridge": {
                "measured_endpoints": [29, 35],
                "intervening_layers": [30, 31, 32, 33, 34],
                "display": "faint dashed segment; no crossing point is measured",
            },
        },
        "figure_2": {
            "band": DEEP.tolist(),
            "v20_single_thread": deep20,
            "v21_multi_conversation": deep21,
            "v20_first_content_definition": "frozen message-start position + 2",
        },
        "figure_3": {
            "arm": V21_TEXT,
            "slice": [TRACE_START, TRACE_STOP],
            "deep_band": DEEP.tolist(),
            "token_reducer": "mean of five per-layer within-256-token percentile ranks",
            "message_boundary_count": int(trace_boundaries.size),
            "slice_median_percentile": float(np.median(deep_trace[TRACE_START:TRACE_STOP])),
            "boundary_median_percentile": float(np.median(deep_trace[trace_boundaries])),
            "boundary_fraction_above_0_5": float(
                np.mean(deep_trace[trace_boundaries] > 0.5)
            ),
        },
        "figure_4": {
            text: {
                "bin_correlations": correlations[text]["bin_correlations"],
                "median_spearman": correlations[text]["median_spearman"],
                "positive_bins": correlations[text]["positive_bins"],
            }
            for text in (V20_TEXT, MATH_TEXT)
        },
        "figure_5": {
            "layers": LAYERS.tolist(),
            "classes": CLASS_ORDER,
            "v21_effect_matrix": [
                [profiles21[name][int(layer)] for name in CLASS_ORDER]
                for layer in LAYERS
            ],
            "math_control_registered_band": math_controls,
            "math_control_bands": {
                "speaker_labels": DEEP.tolist(),
                "first_content": DEEP.tolist(),
                "pronouns": SHALLOW.tolist(),
            },
        },
    }
    runtime = {
        "profiles20": profiles20,
        "profiles21": profiles21,
        "deep20": deep20,
        "deep21": deep21,
        "deep_trace": deep_trace,
        "trace_boundaries": trace_boundaries,
        "null_low": float(null_low),
        "null_high": float(null_high),
        "math_controls": math_controls,
        "correlations": correlations,
    }
    return figure_data, runtime


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#9CA3AF",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "none",
            "legend.framealpha": 0.94,
            "figure.titlesize": 15,
            "figure.titleweight": "normal",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def figure_1(runtime: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 6.6), layout="constrained")
    ax.axvspan(14, 32, color="#DBEAFE", alpha=0.55, zorder=-4)
    ax.axvspan(32, 62, color="#FDE7D3", alpha=0.48, zorder=-4)
    ax.fill_between(
        [14, 62],
        [runtime["null_low"]] * 2,
        [runtime["null_high"]] * 2,
        color="#9CA3AF",
        alpha=0.24,
        linewidth=0,
        zorder=-2,
    )
    ax.axhline(0, color=INK, linewidth=1.1, zorder=0)
    ax.axhline(0.10, color=POSITIVE, linestyle="--", linewidth=1.1, alpha=0.75)
    ax.axhline(-0.05, color=NEGATIVE, linestyle="--", linewidth=1.1, alpha=0.75)

    for name in CLASS_ORDER:
        values = np.array(
            [runtime["profiles21"][name][int(layer)] for layer in LAYERS]
        )
        ax.plot(
            LAYERS[:3],
            values[:3],
            marker="o",
            markersize=7,
            linewidth=2.4,
            color=CLASS_COLORS[name],
            label=CLASS_LABELS[name],
            zorder=3,
        )
        ax.plot(
            LAYERS[3:],
            values[3:],
            marker="o",
            markersize=7,
            linewidth=2.4,
            color=CLASS_COLORS[name],
            zorder=3,
        )
        ax.plot(
            LAYERS[2:4],
            values[2:4],
            linestyle=(0, (2, 2.4)),
            linewidth=1.35,
            color=CLASS_COLORS[name],
            alpha=0.32,
            zorder=2,
        )

    speaker = runtime["profiles21"]["speaker_labels"]
    ax.annotate(
        "speaker-label sign reversal\nbracketed by measured L29 and L35",
        xy=(32, (speaker[29] + speaker[35]) / 2),
        xytext=(23.5, 0.178),
        arrowprops={"arrowstyle": "->", "color": SPEAKER, "linewidth": 1.3},
        color=SPEAKER,
        fontsize=9.0,
        ha="center",
    )
    ax.text(23, 0.215, "registered shallow band", ha="center", color="#365E9D")
    ax.text(47, 0.215, "registered deep band", ha="center", color="#9A4D13")
    ax.text(
        32,
        -0.302,
        "L30–L34 not measured",
        ha="center",
        va="bottom",
        color=GRAY,
        fontsize=8.0,
    )
    ax.text(61.4, 0.104, "+0.10 widening target", ha="right", va="bottom", color=POSITIVE, fontsize=8.5)
    ax.text(61.4, -0.054, "−0.05 narrowing target", ha="right", va="top", color=NEGATIVE, fontsize=8.5)

    null_patch = Patch(
        facecolor="#9CA3AF", alpha=0.24, edgecolor="none", label="95% permutation-null envelope"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [null_patch], labels + [null_patch.get_label()], loc="lower right", ncol=2)
    ax.set(
        xlim=(14.5, 61.5),
        ylim=(-0.31, 0.235),
        xticks=LAYERS,
        xticklabels=[f"L{layer}" for layer in LAYERS],
        xlabel="Global transformer layer",
        ylabel="Aperture effect (median percentile − 0.5)",
    )
    clean_axis(ax)
    fig.suptitle("The delimiter aperture effect flips sign with depth")
    ax.set_title(
        "Fresh v2.1 multi-conversation arm · 0 means no different from a typical token",
        loc="left",
        color=GRAY,
        fontsize=10,
    )
    save(fig, "figure-01-depth-flip-profile.png")


def figure_2(runtime: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.9), layout="constrained")
    ax.axvspan(-0.09, 0, color="#DBEAFE", alpha=0.35, zorder=-3)
    ax.axvspan(0, 0.29, color="#FDE7D3", alpha=0.30, zorder=-3)
    ax.axvline(0, color=INK, linewidth=1.1, zorder=0)
    y_positions = np.arange(len(CLASS_ORDER))
    for y, name in zip(y_positions, CLASS_ORDER, strict=True):
        left = runtime["deep20"][name]
        right = runtime["deep21"][name]
        ax.plot([left, right], [y, y], color="#9CA3AF", linewidth=2.0, zorder=1)
        ax.scatter(left, y, s=85, color=V20_COLOR, edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(right, y, s=85, color=V21_COLOR, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(format_effect(left), (left, y), xytext=(0, 11), textcoords="offset points", ha="center", color=V20_COLOR, fontsize=8.5)
        ax.annotate(format_effect(right), (right, y), xytext=(0, -16), textcoords="offset points", ha="center", color=V21_COLOR, fontsize=8.5)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=V20_COLOR, markeredgecolor="white", markersize=8, label="v2.0 single-thread"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=V21_COLOR, markeredgecolor="white", markersize=8, label="v2.1 multi-conversation"),
    ]
    ax.legend(handles=legend, loc="lower right")
    ax.set(
        xlim=(-0.09, 0.29),
        ylim=(len(CLASS_ORDER) - 0.45, -0.55),
        yticks=y_positions,
        yticklabels=[CLASS_LABELS[name] for name in CLASS_ORDER],
        xlabel="Deep-band aperture effect",
    )
    clean_axis(ax)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Boundary widening replicates; first-content reverses")
    ax.set_title(
        "Same corrected arithmetic · deep band = median across L35, L41, L47, L53, L59",
        loc="left",
        color=GRAY,
        fontsize=10,
    )
    save(fig, "figure-02-replication-reversal-dumbbell.png")


def figure_3(runtime: dict[str, Any]) -> None:
    profile = runtime["deep_trace"]
    boundaries = runtime["trace_boundaries"]
    x = np.arange(TRACE_START, TRACE_STOP)
    fig, ax = plt.subplots(figsize=(13.8, 5.2), layout="constrained")
    ax.vlines(
        boundaries,
        0,
        1,
        colors=SPEAKER,
        linestyles=(0, (3, 4)),
        linewidth=0.7,
        alpha=0.27,
        zorder=0,
    )
    ax.plot(x, profile[TRACE_START:TRACE_STOP], color="#355C7D", linewidth=0.72, alpha=0.86, zorder=1)
    ax.scatter(
        boundaries,
        profile[boundaries],
        s=24,
        color=SPEAKER,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.axhline(0.5, color=INK, linestyle="--", linewidth=1.0, alpha=0.70)
    boundary_median = float(np.median(profile[boundaries]))
    slice_median = float(np.median(profile[TRACE_START:TRACE_STOP]))
    ax.text(
        0.012,
        0.965,
        f"message-boundary median {boundary_median:.3f}  ·  slice median {slice_median:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
    )
    legend = [
        Line2D([0], [0], color="#355C7D", linewidth=1.2, label="deep-band aperture percentile"),
        Line2D([0], [0], color=SPEAKER, linestyle=(0, (3, 4)), marker="o", markersize=5, label="message boundary"),
        Line2D([0], [0], color=INK, linestyle="--", linewidth=1.0, label="typical token (0.5)"),
    ]
    ax.legend(handles=legend, loc="lower right", ncol=3)
    ax.set(
        xlim=(TRACE_START, TRACE_STOP - 1),
        ylim=(-0.02, 1.02),
        xlabel="Token position in v2.1 chat arm",
        ylabel="Deep-band aperture percentile",
    )
    clean_axis(ax)
    fig.suptitle("Message boundaries tend to coincide with a wider deep-global aperture")
    ax.set_title(
        "Fixed middle 1,536-token slice · thin line is unsmoothed · five deep-global percentile ranks averaged per token",
        loc="left",
        color=GRAY,
        fontsize=10,
    )
    save(fig, "figure-03-aperture-along-text.png")


def figure_4(runtime: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True, layout="constrained")
    panels = [
        (V20_TEXT, "Slack human", SLACK_COLOR),
        (MATH_TEXT, "Math-LLM", MATH_COLOR),
    ]
    for ax, (text, title, color) in zip(axes, panels, strict=True):
        record = runtime["correlations"][text]
        values = np.asarray(record["bin_correlations"], dtype=np.float64)
        x = np.arange(1, len(values) + 1)
        colors = [color if value > 0 else "#B85C5C" for value in values]
        ax.bar(x, values, width=0.78, color=colors, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color=INK, linewidth=1.0)
        ax.axhline(record["median_spearman"], color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            0.98,
            0.95,
            f"median ρ = {record['median_spearman']:+.3f}\n{record['positive_bins']}/16 bins positive",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=INK,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        )
        ax.set(
            title=title,
            xlim=(0.35, 16.65),
            ylim=(-0.10, 0.56),
            xticks=x,
            xlabel="512-token position bin",
        )
        ax.tick_params(axis="x", labelsize=8)
        clean_axis(ax)
    axes[0].set_ylabel("Within-bin Spearman ρ")
    fig.suptitle("The aperture–surprisal association is small but consistent")
    save(fig, "figure-04-surprisal-law-bars.png")


def figure_5(figure_data: dict[str, Any], runtime: dict[str, Any]) -> None:
    source = np.asarray(figure_data["figure_5"]["v21_effect_matrix"], dtype=np.float64)
    control = np.array([runtime["math_controls"][name] for name in CLASS_ORDER])
    matrix = np.vstack([source, control])
    limit = float(np.ceil(np.max(np.abs(matrix)) * 20) / 20)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, ax = plt.subplots(figsize=(8.4, 7.0), layout="constrained")
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.axhline(2.5, color="white", linewidth=3.0)
    ax.axhline(7.5, color=INK, linewidth=1.6)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if abs(value) >= 0.14 else INK
            ax.text(col, row, format_effect(value), ha="center", va="center", color=text_color, fontsize=9)

    row_labels = [f"L{layer}" for layer in LAYERS] + ["Math control†"]
    ax.set(
        xticks=np.arange(len(CLASS_ORDER)),
        xticklabels=[CLASS_LABELS[name] for name in CLASS_ORDER],
        yticks=np.arange(len(row_labels)),
        yticklabels=row_labels,
        xlabel="Frozen token class",
        ylabel="v2.1 layer (plus registered cross-control)",
    )
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    cbar = fig.colorbar(image, ax=ax, pad=0.025, shrink=0.88)
    cbar.set_label("Aperture effect  ·  blue narrows, red widens")
    fig.suptitle("Depth and token class jointly determine aperture direction")
    ax.set_title(
        "White is the null neighborhood · separator marks shallow/deep bands",
        loc="left",
        color=GRAY,
        fontsize=10,
        pad=10,
    )
    fig.text(
        0.5,
        -0.012,
        "† Math control uses each class's registered band: deep (L35–L59) for speaker labels and first content; shallow (L17–L29) for pronouns.",
        ha="center",
        va="top",
        color=GRAY,
        fontsize=8.1,
    )
    save(fig, "figure-05-depth-class-heatmap.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_style()
    figure_data, runtime = derive_data()
    data_path = OUT / "figure_data.json"
    with data_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(figure_data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {data_path}")
    figure_1(runtime)
    figure_2(runtime)
    figure_3(runtime)
    figure_4(runtime)
    figure_5(figure_data, runtime)


if __name__ == "__main__":
    main()
