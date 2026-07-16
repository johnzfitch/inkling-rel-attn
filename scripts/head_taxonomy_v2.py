"""Round 3 A3: full-resolution head magnitude taxonomy from complete D0 dumps.

The analyzed profile is ``abs(S[h,0] * U[h,:,0])``.  It measures the
magnitude of the leading composed-operator mode, not signed attention logits,
attention probabilities, or realized token-to-token behavior.  Class names
therefore describe magnitude shape and deliberately avoid behavioral labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
D0_DIR = ROOT / "dumps" / "round3" / "perhead_svd"
ROUND2_SUMMARY = ROOT / "analysis" / "round2" / "per_head_summary.json"
OUT_DIR = ROOT / "analysis" / "round3"
FIG_DIR = OUT_DIR / "figs"
CLASSES = ("near_concentrated", "magnitude_decay", "magnitude_rising", "flat", "other")
OLD_CLASSES = ("prev_token", "decay", "flat", "other")
LEGACY_CLASS_ALIAS = {
    "near_concentrated": "near_focused",
    "magnitude_decay": "decay",
    "magnitude_rising": "rising",
    "flat": "flat",
    "other": "other",
}
COLORS = {
    "near_concentrated": "#7c3aed",
    "magnitude_decay": "#2563eb",
    "magnitude_rising": "#dc2626",
    "flat": "#059669",
    "other": "#9ca3af",
}


def dump_path(kind: str, layer: int) -> Path:
    return D0_DIR / (f"layer{layer:02d}.npz" if kind == "trunk" else f"mtp{layer}.npz")


def rho_value(p: np.ndarray) -> float:
    result = spearmanr(p, np.arange(len(p))).statistic
    return 0.0 if not np.isfinite(result) else float(result)


def classify_new(p: np.ndarray) -> dict[str, object]:
    extent = len(p)
    mass = float(np.sum(p**2))
    near_fraction = float(np.sum(p[:4] ** 2) / (mass + 1e-300))
    near_mass_ratio = float(near_fraction / (4.0 / extent))
    d_peak = int(np.argmax(p))
    rho = rho_value(p)
    if near_mass_ratio > 20.0 and d_peak < 4:
        class_name = "near_concentrated"
    elif rho < -0.5:
        class_name = "magnitude_decay"
    elif rho > 0.5:
        class_name = "magnitude_rising"
    elif (float(p.max()) - float(p.min())) < 0.1 * float(p.max()):
        class_name = "flat"
    else:
        class_name = "other"
    cumulative = np.cumsum(p**2)
    effective_range = int(np.searchsorted(cumulative, 0.9 * mass, side="left"))
    effective_range = min(effective_range, extent - 1)
    return {
        "class": class_name,
        "near_mass_ratio": near_mass_ratio,
        "near4_mass_fraction": near_fraction,
        "spearman_rho": rho,
        "d_peak": d_peak,
        "effective_range": effective_range,
    }


def classify_old(p: np.ndarray) -> str:
    if float(np.sum(p[:4] ** 2)) > 0.5 * float(np.sum(p**2)):
        return "prev_token"
    rho = rho_value(p)
    if rho < -0.5:
        return "decay"
    if (float(p.max()) - float(p.min())) < 0.1 * float(p.max()):
        return "flat"
    return "other"


def analyze_family(kind: str, count: int) -> tuple[dict[str, object], list[list[str]], np.ndarray]:
    layers: dict[str, object] = {}
    all_classes: list[list[str]] = []
    all_peaks = np.zeros((count, 64), dtype=np.int32)
    for layer in range(count):
        path = dump_path(kind, layer)
        with np.load(path) as dump:
            singular_values = dump["S"].astype(np.float64)
            u = dump["U"].astype(np.float64)
        if singular_values.shape != (64, 16) or u.shape[0] != 64 or u.shape[2] != 16:
            raise ValueError(f"bad D0 shapes in {path}")
        classes: list[str] = []
        near_ratios = []
        near_fractions = []
        rhos = []
        peaks = []
        ranges = []
        old_classes = []
        for head in range(64):
            p = np.abs(singular_values[head, 0] * u[head, :, 0])
            measured = classify_new(p)
            classes.append(str(measured["class"]))
            near_ratios.append(float(measured["near_mass_ratio"]))
            near_fractions.append(float(measured["near4_mass_fraction"]))
            rhos.append(float(measured["spearman_rho"]))
            peaks.append(int(measured["d_peak"]))
            ranges.append(int(measured["effective_range"]))
            old_classes.append(classify_old(p))
        counts = {name: int(classes.count(name)) for name in CLASSES}
        legacy_classes = [LEGACY_CLASS_ALIAS[name] for name in classes]
        legacy_counts = {
            name: int(legacy_classes.count(name))
            for name in ("near_focused", "decay", "rising", "flat", "other")
        }
        old_counts = {name: int(old_classes.count(name)) for name in OLD_CLASSES}
        layers[str(layer)] = {
            "dump": str(path.relative_to(ROOT)),
            "extent": int(u.shape[1]),
            "class_counts": counts,
            "head_class": classes,
            "legacy_class_counts": legacy_counts,
            "legacy_head_class": legacy_classes,
            "near_mass_ratio": near_ratios,
            "near4_mass_fraction": near_fractions,
            "spearman_rho": rhos,
            "d_peak": peaks,
            "effective_range": ranges,
            "old_rule_class_counts": old_counts,
        }
        all_classes.append(classes)
        all_peaks[layer] = peaks
        print(
            f"{kind:5s} {layer:02d}: "
            + " ".join(f"{name}={counts[name]:2d}" for name in CLASSES)
        )
        if counts["magnitude_rising"] > 32:
            print(
                f"[FLAG] {kind} layer {layer:02d}: "
                f"magnitude_rising={counts['magnitude_rising']}/64"
            )
    return layers, all_classes, all_peaks


def cross_check_round2(trunk: dict[str, object]) -> dict[str, object]:
    with ROUND2_SUMMARY.open(encoding="utf-8") as f:
        round2 = json.load(f)
    runs = round2["runs"]
    reference_run = "A" if "A" in runs else str(round2["reshape_test"]["winner"])
    reference = runs[reference_run]["class_counts_per_layer"]
    mismatches = []
    for layer in range(66):
        measured = trunk[str(layer)]["old_rule_class_counts"]
        expected = reference[str(layer)]
        if any(int(measured[name]) != int(expected[name]) for name in OLD_CLASSES):
            mismatch = {
                "layer": layer,
                "measured": {name: int(measured[name]) for name in OLD_CLASSES},
                "round2": {name: int(expected[name]) for name in OLD_CLASSES},
            }
            mismatches.append(mismatch)
            print(
                f"[CONTRADICTION] D0 old-rule counts differ from Round 2 at layer {layer}: "
                f"measured={mismatch['measured']} expected={mismatch['round2']}"
            )
    passed = len(mismatches) == 0
    print(
        f"Round 2 old-rule cross-check: {66 - len(mismatches)}/66 layers match "
        f"({'PASS' if passed else 'FAIL'})"
    )
    return {
        "reference_file": str(ROUND2_SUMMARY.relative_to(ROOT)),
        "reference_run": reference_run,
        "matching_layers": 66 - len(mismatches),
        "total_layers": 66,
        "mismatches": mismatches,
        "passed": passed,
    }


def counts_matrix(layers: dict[str, object]) -> np.ndarray:
    matrix = np.zeros((len(layers), len(CLASSES)), dtype=np.int32)
    for layer in range(len(layers)):
        matrix[layer] = [layers[str(layer)]["class_counts"][name] for name in CLASSES]
    return matrix


def make_figures(
    trunk: dict[str, object],
    mtp: dict[str, object],
    trunk_classes: list[list[str]],
    mtp_classes: list[list[str]],
    trunk_peaks: np.ndarray,
    mtp_peaks: np.ndarray,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    for ax, layers, label in ((axes[0], trunk, "trunk layer"), (axes[1], mtp, "MTP layer")):
        matrix = counts_matrix(layers)
        x = np.arange(len(layers))
        ax.stackplot(
            x,
            *[matrix[:, i] for i in range(len(CLASSES))],
            labels=CLASSES,
            colors=[COLORS[name] for name in CLASSES],
            alpha=0.9,
        )
        ax.set_xlim(0, len(layers) - 1)
        ax.set_ylim(0, 64)
        ax.set_xlabel(label)
        ax.set_ylabel("head count")
        ax.grid(True, axis="y", color="#e5e7eb", lw=0.5)
    axes[0].set_title("Leading-mode relative-position magnitude taxonomy")
    axes[0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "taxonomy_depth_stack.png", dpi=150)
    plt.close(fig)

    class_values: dict[str, list[int]] = {name: [] for name in CLASSES}
    for classes, peaks in ((trunk_classes, trunk_peaks), (mtp_classes, mtp_peaks)):
        for layer in range(len(classes)):
            for head in range(64):
                class_values[classes[layer][head]].append(int(peaks[layer, head]))
    present = [name for name in CLASSES if class_values[name]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(
        [class_values[name] for name in present],
        bins=np.arange(0, 1024 + 17, 16),
        stacked=True,
        label=present,
        color=[COLORS[name] for name in present],
        alpha=0.9,
    )
    ax.set_xlim(0, 1024)
    ax.set_xlabel("d_peak")
    ax.set_ylabel("head count")
    ax.set_title("Exact peak distance across trunk and MTP heads")
    ax.legend(frameon=False, ncol=5)
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d_peak_hist.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trunk, trunk_classes, trunk_peaks = analyze_family("trunk", 66)
    mtp, mtp_classes, mtp_peaks = analyze_family("mtp", 8)
    cross_check = cross_check_round2(trunk)
    make_figures(trunk, mtp, trunk_classes, mtp_classes, trunk_peaks, mtp_peaks)

    output = {
        "dump_only_analysis_input": str(D0_DIR.relative_to(ROOT)),
        "round2_summary_used_only_as_cross_check_reference": str(
            ROUND2_SUMMARY.relative_to(ROOT)
        ),
        "definitions": {
            "profile": "p=abs(S[h,0]*U[h,:,0]) at full resolution",
            "interpretation_limit": (
                "Classes describe the magnitude shape of one weight-level SVD mode. "
                "They do not establish signed attention preference, attention "
                "probability, or behavior on tokens."
            ),
            "near_mass_ratio": "(sum(p[:4]^2)/sum(p^2))/(4/extent)",
            "near_concentrated": "near_mass_ratio>20 and argmax(p)<4",
            "magnitude_decay": "Spearman rho(p,d)<-0.5",
            "magnitude_rising": "Spearman rho(p,d)>0.5",
            "flat": "max(p)-min(p)<0.1*max(p)",
            "effective_range": "minimum d with cumulative squared mass >= 0.9 total",
            "rule_order": list(CLASSES),
            "legacy_class_alias": LEGACY_CLASS_ALIAS,
        },
        "round2_old_rule_cross_check": cross_check,
        "trunk_layers": trunk,
        "mtp_layers": mtp,
    }
    out_path = OUT_DIR / "head_taxonomy_v2.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Written {out_path}")
    print(f"Written {FIG_DIR / 'taxonomy_depth_stack.png'}")
    print(f"Written {FIG_DIR / 'd_peak_hist.png'}")


if __name__ == "__main__":
    main()
