"""Round 3 A4: analyze causal SConv taps from the D2 raw dump only.

D2 shapes are [channels, 1, kernel=4], the PyTorch depthwise Conv1d checkpoint
layout (out_channel, in_channel_per_group, kernel).  With left causal padding
and Conv1d cross-correlation, stored kernel index 3 is t-0 and index 0 is t-3;
all reported tap arrays are therefore reversed into lag order [t-0..t-3].
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
D2_DIR = ROOT / "dumps" / "round3" / "sconv"
OUT_DIR = ROOT / "analysis" / "round3"
FIG_DIR = OUT_DIR / "figs"
GROUPS = ("k_sconv", "v_sconv", "attn_sconv", "mlp_sconv")
TRUNK_GLOBAL = set(range(5, 66, 6))
MTP_GLOBAL = {1, 3}
LAG_COLORS = ("#1d4ed8", "#60a5fa", "#f59e0b", "#dc2626")


def is_global(kind: str, layer: int) -> bool:
    return layer in (TRUNK_GLOBAL if kind == "trunk" else MTP_GLOBAL)


def raw_path(kind: str, layer: int, group: str) -> Path:
    prefix = f"layer{layer:02d}" if kind == "trunk" else f"mtp{layer}"
    return D2_DIR / f"{prefix}_{group}.npy"


def lag_taps(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[1] != 1 or array.shape[2] != 4:
        raise ValueError(f"expected [channels,1,4], got {array.shape}")
    return np.asarray(array[:, 0, ::-1], dtype=np.float64)


def summarize(taps: np.ndarray) -> dict[str, object]:
    abs_taps = np.abs(taps)
    l1 = abs_taps.sum(axis=1)
    energy = np.sum(taps**2, axis=0)
    energy_profile = energy / (energy.sum() + 1e-300)
    per_channel_delay = (abs_taps * np.arange(4)[None, :]).sum(axis=1) / (l1 + 1e-300)
    effective_delay = float(
        np.sum(abs_taps * np.arange(4)[None, :]) / (np.sum(abs_taps) + 1e-300)
    )
    near_identity = taps[:, 0] > 0.9 * l1

    dc_gain = np.abs(np.sum(taps, axis=1))
    nyquist_gain = np.abs(np.sum(taps * np.array([1.0, -1.0, 1.0, -1.0]), axis=1))
    smoother = (~near_identity) & (dc_gain > 1.5 * nyquist_gain)
    differentiator = (~near_identity) & (nyquist_gain > 1.5 * dc_gain)
    other = ~(near_identity | smoother | differentiator)
    fractions = {
        "passthrough": float(np.mean(near_identity)),
        "smoother": float(np.mean(smoother)),
        "differentiator": float(np.mean(differentiator)),
        "other": float(np.mean(other)),
    }
    dominant = max(fractions, key=fractions.get)
    return {
        "channels": int(taps.shape[0]),
        "tap_energy_profile_t0_to_t3": [float(v) for v in energy_profile],
        "effective_delay_l1_weighted": effective_delay,
        "per_channel_effective_delay_mean": float(np.mean(per_channel_delay)),
        "per_channel_effective_delay_std": float(np.std(per_channel_delay)),
        "per_channel_effective_delay_median": float(np.median(per_channel_delay)),
        "near_identity_fraction": float(np.mean(near_identity)),
        "channel_character_fraction": fractions,
        "dominant_character": dominant,
        "mean_signed_taps_t0_to_t3": [float(v) for v in np.mean(taps, axis=0)],
        "mean_abs_taps_t0_to_t3": [float(v) for v in np.mean(abs_taps, axis=0)],
    }


def make_figure(depth_data: dict[str, list[list[float]]], global_x: list[int]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    x = np.arange(74)
    for ax, group in zip(axes, GROUPS):
        matrix = np.asarray(depth_data[group], dtype=np.float64)
        ax.stackplot(
            x,
            *[matrix[:, lag] for lag in range(4)],
            colors=LAG_COLORS,
            labels=["t-0", "t-1", "t-2", "t-3"],
            alpha=0.9,
        )
        ax.scatter(global_x, np.full(len(global_x), 1.025), marker="D", s=18,
                   color="#111827", clip_on=False, zorder=5)
        ax.axvline(65.5, color="#111827", ls="--", lw=1)
        ax.text(65.2, 0.05, "trunk", ha="right", va="bottom", fontsize=8)
        ax.text(65.8, 0.05, "MTP", ha="left", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.06)
        ax.set_ylabel(group.replace("_", "\n"))
        ax.grid(True, axis="y", color="#e5e7eb", lw=0.5)
    axes[0].set_title("SConv tap-energy allocation vs depth (diamonds mark global layers)")
    axes[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    axes[-1].set_xlabel("combined depth: trunk 0-65, then MTP 0-7")
    axes[-1].set_xticks([0, 10, 20, 30, 40, 50, 60, 66, 67, 68, 69, 70, 71, 72, 73])
    axes[-1].set_xticklabels(
        ["0", "10", "20", "30", "40", "50", "60", "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sconv_taps_depth.png", dpi=150)
    plt.close(fig)


def main() -> None:
    with (D2_DIR / "_meta.json").open(encoding="utf-8") as f:
        dump_meta = json.load(f)
    if int(dump_meta["tensor_count"]) != 296:
        raise ValueError("D2 metadata does not contain all 296 tensors")

    layers = {"trunk": {}, "mtp": {}}
    aggregates_raw: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    depth_data: dict[str, list[list[float]]] = {group: [] for group in GROUPS}
    global_x = []

    for kind, count in (("trunk", 66), ("mtp", 8)):
        for layer in range(count):
            global_layer = is_global(kind, layer)
            combined_x = layer if kind == "trunk" else 66 + layer
            if global_layer:
                global_x.append(combined_x)
            entry = {
                "is_local": not global_layer,
                "combined_depth": combined_x,
                "kernels": {},
            }
            for group in GROUPS:
                path = raw_path(kind, layer, group)
                array = np.load(path, mmap_mode="r")
                taps = lag_taps(array)
                measured = summarize(taps)
                measured["dump"] = str(path.relative_to(ROOT))
                measured["dump_shape"] = [int(v) for v in array.shape]
                entry["kernels"][group] = measured
                locality = "global" if global_layer else "local"
                aggregates_raw[(kind, locality, group)].append(taps)
                depth_data[group].append(measured["tap_energy_profile_t0_to_t3"])
            layers[kind][str(layer)] = entry
            chars = ", ".join(
                f"{group}={entry['kernels'][group]['dominant_character']}"
                for group in GROUPS
            )
            print(f"{kind:5s} {layer:02d} ({'global' if global_layer else 'local'}): {chars}")

    aggregates: dict[str, object] = {}
    for (kind, locality, group), pieces in aggregates_raw.items():
        aggregates.setdefault(kind, {}).setdefault(locality, {})[group] = summarize(
            np.concatenate(pieces, axis=0)
        )

    make_figure(depth_data, sorted(global_x))
    output = {
        "dump_only_input": str(D2_DIR.relative_to(ROOT)),
        "axis_inference": {
            "dumped_shape": "[channels, 1, 4]",
            "channel_axis": 0,
            "singleton_in_channel_per_group_axis": 1,
            "stored_kernel_axis": 2,
            "reported_lag_order": "reverse stored axis: index 3 is t-0, then t-1, t-2, t-3",
            "reason": (
                "depthwise PyTorch Conv1d stores [out_channel,in_channel/groups,kernel]; "
                "causal left padding plus cross-correlation makes the final stored tap current-time"
            ),
        },
        "character_heuristic": {
            "passthrough": "signed tap_t0 > 0.9 * channel L1 mass",
            "smoother": "not passthrough and |DC gain| > 1.5 * |Nyquist gain|",
            "differentiator": "not passthrough and |Nyquist gain| > 1.5 * |DC gain|",
            "other": "remainder",
        },
        "layers": layers,
        "aggregates": aggregates,
    }
    out_path = OUT_DIR / "sconv_summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Written {out_path}")
    print(f"Written {FIG_DIR / 'sconv_taps_depth.png'}")


if __name__ == "__main__":
    main()
