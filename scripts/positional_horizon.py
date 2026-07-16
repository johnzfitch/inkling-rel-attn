"""Round 3 A2: extent-boundary analysis from D0/D1 dumps only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
D0_DIR = ROOT / "dumps" / "round3" / "perhead_svd"
D1_DIR = ROOT / "dumps" / "round3" / "mode_curves"
OUT_DIR = ROOT / "analysis" / "round3"
FIG_DIR = OUT_DIR / "figs"
TRUNK_GLOBAL = set(range(5, 66, 6))
MTP_GLOBAL = {1, 3}
TAU_POS = 1_000_000
TAU = float(1.0 + 0.1 * np.log((TAU_POS + 1) / 128_000.0))


def dump_path(directory: Path, kind: str, layer: int) -> Path:
    return directory / (f"layer{layer:02d}.npz" if kind == "trunk" else f"mtp{layer}.npz")


def is_global(kind: str, layer: int) -> bool:
    return layer in (TRUNK_GLOBAL if kind == "trunk" else MTP_GLOBAL)


def metrics(curve: np.ndarray) -> dict[str, float]:
    extent = len(curve)
    edge_amp = float(np.mean(np.abs(curve[extent - 16 : extent])))
    interior_amp = float(np.mean(np.abs(curve[extent // 4 : extent // 2])))
    return {
        "edge_amp": edge_amp,
        "interior_amp": interior_amp,
        "cliff_ratio": float(edge_amp / (interior_amp + 1e-12)),
    }


def operator_gain(mode_factors: np.ndarray) -> np.ndarray:
    """Basis-invariant row norm from orthogonal SVD mode contributions.

    mode_factors has shape [rank, extent] and contains S[k]*Vt[k,d] for a
    layer projection, or S[h,k]*U[h,d,k] transposed into the same layout for a
    composed per-head operator.  Orthogonality of the complementary SVD factor
    makes the L2 norm across modes equal to the full operator row norm.
    """
    return np.sqrt(np.sum(np.square(mode_factors), axis=0))


def analyze_layer(kind: str, layer: int) -> dict[str, object]:
    d1_path = dump_path(D1_DIR, kind, layer)
    d0_path = dump_path(D0_DIR, kind, layer)
    with np.load(d1_path) as mode_dump:
        singular_values = mode_dump["S"].astype(np.float64)
        vt = mode_dump["Vt"].astype(np.float64)
    if singular_values.shape != (16,) or vt.shape[0] != 16:
        raise ValueError(f"bad D1 shapes in {d1_path}")
    global_layer = is_global(kind, layer)
    mode_factors = singular_values[:, None] * vt
    modes: dict[str, dict[str, float]] = {}
    for mode in range(16):
        entry = {
            "singular_value": float(singular_values[mode]),
            **metrics(singular_values[mode] * vt[mode]),
        }
        if global_layer:
            entry["tau_scaled_edge_amp_at_1m"] = float(TAU * entry["edge_amp"])
        modes[str(mode)] = entry
    full_rank = metrics(operator_gain(mode_factors))
    full_rank["rank"] = int(len(singular_values))
    if global_layer:
        full_rank["tau_scaled_edge_amp_at_1m"] = float(TAU * full_rank["edge_amp"])

    with np.load(d0_path) as head_dump:
        head_s = head_dump["S"].astype(np.float64)
        head_u = head_dump["U"].astype(np.float64)
    if head_s.shape != (64, 16) or head_u.shape[0] != 64 or head_u.shape[2] != 16:
        raise ValueError(f"bad D0 shapes in {d0_path}")
    head_edge = np.zeros(64, dtype=np.float64)
    head_interior = np.zeros(64, dtype=np.float64)
    head_cliff = np.zeros(64, dtype=np.float64)
    head_full_edge = np.zeros(64, dtype=np.float64)
    head_full_interior = np.zeros(64, dtype=np.float64)
    head_full_cliff = np.zeros(64, dtype=np.float64)
    for head in range(64):
        measured = metrics(head_s[head, 0] * head_u[head, :, 0])
        head_edge[head] = measured["edge_amp"]
        head_interior[head] = measured["interior_amp"]
        head_cliff[head] = measured["cliff_ratio"]
        full_gain = operator_gain(
            (head_u[head] * head_s[head][None, :]).T
        )
        full_measured = metrics(full_gain)
        head_full_edge[head] = full_measured["edge_amp"]
        head_full_interior[head] = full_measured["interior_amp"]
        head_full_cliff[head] = full_measured["cliff_ratio"]
    per_head: dict[str, object] = {
        "singular_value_mode0": [float(v) for v in head_s[:, 0]],
        "edge_amp": [float(v) for v in head_edge],
        "interior_amp": [float(v) for v in head_interior],
        "cliff_ratio": [float(v) for v in head_cliff],
        "median_cliff_ratio": float(np.median(head_cliff)),
        "max_cliff_ratio": float(np.max(head_cliff)),
    }
    if global_layer:
        per_head["tau_scaled_edge_amp_at_1m"] = [float(TAU * v) for v in head_edge]
    per_head_full_rank: dict[str, object] = {
        "edge_amp": [float(v) for v in head_full_edge],
        "interior_amp": [float(v) for v in head_full_interior],
        "cliff_ratio": [float(v) for v in head_full_cliff],
        "median_cliff_ratio": float(np.median(head_full_cliff)),
        "max_cliff_ratio": float(np.max(head_full_cliff)),
    }
    if global_layer:
        per_head_full_rank["tau_scaled_edge_amp_at_1m"] = [
            float(TAU * v) for v in head_full_edge
        ]

    return {
        "is_local": not global_layer,
        "extent": int(vt.shape[1]),
        "mode_dump": str(d1_path.relative_to(ROOT)),
        "head_dump": str(d0_path.relative_to(ROOT)),
        "modes": modes,
        "full_rank_operator": full_rank,
        "per_head_mode0": per_head,
        "per_head_full_rank": per_head_full_rank,
    }


def make_figure(trunk: dict[str, object], mtp: dict[str, object]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    for ax, family, label in ((axes[0], trunk, "trunk layer"), (axes[1], mtp, "MTP layer")):
        xs = np.array(sorted(int(k) for k in family))
        ys = np.array(
            [float(family[str(i)]["full_rank_operator"]["cliff_ratio"]) for i in xs]
        )
        globals_ = np.array([not bool(family[str(i)]["is_local"]) for i in xs])
        ax.scatter(
            xs[~globals_], np.maximum(ys[~globals_], 1e-7), s=20,
            color="#9ca3af", edgecolor="white", linewidth=0.4,
            label="local endpoint (not a live seam)", zorder=2,
        )
        ax.plot(
            xs[globals_], np.maximum(ys[globals_], 1e-7),
            color="#b45309", lw=1.7, zorder=3,
        )
        ax.scatter(
            xs[globals_], np.maximum(ys[globals_], 1e-7), s=42, marker="D",
            color="#b45309", edgecolor="white", linewidth=0.6,
            label="global live seam", zorder=4,
        )
        ax.axhline(0.05, color="#059669", ls="--", lw=1, label="smooth threshold")
        ax.axhline(0.25, color="#dc2626", ls=":", lw=1.2, label="cliff threshold")
        ax.set_yscale("log")
        ax.set_ylabel("full-rank edge/interior ratio")
        ax.set_xlabel(label)
        ax.grid(True, which="both", color="#e5e7eb", lw=0.5)
        ax.legend(frameon=False, ncol=4, fontsize=8)
    axes[0].set_title(
        "Basis-invariant operator gain at each extent boundary "
        "(only global diamonds are live attention seams)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cliff_ratio_depth.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trunk: dict[str, object] = {}
    mtp: dict[str, object] = {}
    for kind, count, target in (("trunk", 66, trunk), ("mtp", 8, mtp)):
        for layer in range(count):
            target[str(layer)] = analyze_layer(kind, layer)
            m0 = target[str(layer)]["modes"]["0"]
            print(
                f"{kind:5s} {layer:02d} "
                f"({'local' if target[str(layer)]['is_local'] else 'global'}): "
                f"mode0 edge={m0['edge_amp']:.5g} interior={m0['interior_amp']:.5g} "
                f"cliff={m0['cliff_ratio']:.4f}"
            )

    # Legacy mode-0/all-boundary statistic is retained for comparability, but
    # it is not the positional-horizon verdict: local attention ends exactly
    # where its relative table ends, and a single SVD mode is basis-dependent.
    all_mode0 = [
        (f"trunk:{layer}", float(entry["modes"]["0"]["cliff_ratio"]))
        for layer, entry in trunk.items()
    ] + [
        (f"mtp:{layer}", float(entry["modes"]["0"]["cliff_ratio"]))
        for layer, entry in mtp.items()
    ]
    legacy_median_mode0 = float(np.median([value for _, value in all_mode0]))
    legacy_worst = sorted(all_mode0, key=lambda item: item[1], reverse=True)[:5]

    trunk_global_full_rank = [
        (f"trunk:{layer}", float(entry["full_rank_operator"]["cliff_ratio"]))
        for layer, entry in trunk.items()
        if not bool(entry["is_local"])
    ]
    mtp_global_full_rank = [
        (f"mtp:{layer}", float(entry["full_rank_operator"]["cliff_ratio"]))
        for layer, entry in mtp.items()
        if not bool(entry["is_local"])
    ]
    median_global = float(np.median([value for _, value in trunk_global_full_rank]))
    verdict = "smooth" if median_global < 0.05 else "cliff" if median_global > 0.25 else "mixed"
    worst = sorted(trunk_global_full_rank, key=lambda item: item[1], reverse=True)[:5]
    print("Five worst LIVE global-trunk full-rank seams:")
    for name, value in worst:
        print(f"  {name:10s} {value:.5f}")
    print(f"global-trunk full-rank median={median_global:.5f} verdict={verdict}")
    print(
        f"legacy all-boundary mode-0 median={legacy_median_mode0:.5f} "
        "(not used for verdict)"
    )

    make_figure(trunk, mtp)
    output = {
        "dump_only_inputs": [
            str(D0_DIR.relative_to(ROOT)),
            str(D1_DIR.relative_to(ROOT)),
        ],
        "definitions": {
            "edge_amp": "mean |curve| over extent-16:extent",
            "interior_amp": "mean |curve| over extent//4:extent//2",
            "cliff_ratio": "edge_amp/(interior_amp+1e-12)",
            "layer_curve": "S[k]*Vt[k] for every k in 0..15",
            "head_curve": "S[h,0]*U[h,:,0]",
            "full_rank_layer_gain": "sqrt(sum_k (S[k]*Vt[k,d])^2)",
            "full_rank_head_gain": "sqrt(sum_k (S[h,k]*U[h,d,k])^2)",
            "live_seam_scope": "global trunk layers only; local extent equals attention window",
        },
        "tau_position": TAU_POS,
        "tau_at_1m": TAU,
        "primary_statistic": "median full-rank edge/interior ratio over global trunk layers",
        "median_global_trunk_full_rank_cliff_ratio": median_global,
        "median_global_mtp_full_rank_cliff_ratio": float(
            np.median([value for _, value in mtp_global_full_rank])
        ),
        "verdict": verdict,
        "five_worst_global_trunk_full_rank_seams": [
            {"layer": name, "cliff_ratio": float(value)} for name, value in worst
        ],
        "legacy_mode0_all_boundaries": {
            "warning": (
                "Includes local window endpoints and a single SVD mode; retained only "
                "for historical comparability and not used for the horizon verdict."
            ),
            "median_cliff_ratio": legacy_median_mode0,
            "five_worst": [
                {"layer": name, "cliff_ratio": float(value)}
                for name, value in legacy_worst
            ],
        },
        "trunk_layers": trunk,
        "mtp_layers": mtp,
    }
    out_path = OUT_DIR / "positional_horizon.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Written {out_path}")
    print(f"Written {FIG_DIR / 'cliff_ratio_depth.png'}")


if __name__ == "__main__":
    main()
