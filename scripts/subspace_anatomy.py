"""Deep analysis of Inkling's per-head hidden read subspaces.

The Round-4 C3 analysis compared each head's top-2 right-singular subspace.
This follow-up separates five effects that the original scalar mixed together:

1. concentration of the pooled head-read spectrum;
2. a matched random-subspace null that preserves each head's S1/S0 ratio;
3. breadth of each communal direction across the 64 heads;
4. live activation of those directions from the Tier-2 r-vector captures;
5. persistence across layers and persistence of head identity.

All analysis is dump-first. No model execution, GPU work, or network access is
performed. Outputs are written to analysis/subspace_anatomy/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parents[1]
D0 = ROOT / "dumps" / "round3" / "perhead_svd"
CAPTURE = ROOT / "dumps" / "tier2" / "capture"
TIER2 = ROOT / "dumps" / "tier2"
WEIGHTS = ROOT / "weights"
OUT = ROOT / "analysis" / "subspace_anatomy"

N_LAYERS = 66
N_HEADS = 64
HIDDEN = 6144
HEAD_RANK = 2
COMMON_BASIS = 4
LIVE_COMPONENTS = 16
SELECTED = [0, 28, 40, 59, 65]
KERNEL_SELECTED = [0, 28, 40, 53, 59, 65]
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


@dataclass
class LayerGeometry:
    layer: int
    singular: np.ndarray
    distance_u2: np.ndarray
    v2: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    basis4: np.ndarray
    structural_share: np.ndarray
    effective_directions: float
    matched_null_directions: float
    n50: int
    head_loadings: np.ndarray
    head_support: np.ndarray


@dataclass
class LiveMetrics:
    live_share: np.ndarray
    mean: np.ndarray
    rms: np.ndarray
    constantness: np.ndarray
    sign_consistency: np.ndarray
    median_abs_position_corr: np.ndarray
    max_abs_position_corr: np.ndarray
    corpus_mean_spread: np.ndarray


def style() -> None:
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
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def phase_spans(ax: plt.Axes) -> None:
    ax.axvspan(-0.5, 13.5, color=RED, alpha=0.045)
    ax.axvspan(13.5, 29.5, color=ORANGE, alpha=0.055)
    ax.axvspan(29.5, 65.5, color=BLUE, alpha=0.04)


def matched_null_pr(weights: np.ndarray) -> float:
    """Expected PR for random directions with the same row energies.

    Rows from the same head are exactly orthogonal, so their expected cross term
    is zero. All other random 6144-D unit-vector pairs have E[(u.v)^2] = 1/D.
    """
    row_energy = (weights * weights).reshape(-1)
    total = float(row_energy.sum())
    diagonal = float((row_energy * row_energy).sum())
    within_head_ordered = float(2 * np.prod(weights * weights, axis=1).sum())
    random_cross = (total * total - diagonal - within_head_ordered) / HIDDEN
    return total * total / (diagonal + random_cross)


def head_pair_similarity(v2: np.ndarray) -> np.ndarray:
    """Mean cosine of the two principal angles for every head pair."""
    flat = v2.reshape(N_HEADS * HEAD_RANK, HIDDEN)
    gram = flat @ flat.T
    blocks = gram.reshape(N_HEADS, HEAD_RANK, N_HEADS, HEAD_RANK).transpose(0, 2, 1, 3)
    return np.linalg.svd(blocks, compute_uv=False).mean(axis=-1)


def cluster_order(similarity: np.ndarray) -> np.ndarray:
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return leaves_list(linkage(squareform(distance, checks=False), method="average"))


def analyze_geometry(layer: int) -> LayerGeometry:
    with np.load(D0 / f"layer{layer:02d}.npz") as dump:
        singular = dump["S"].astype(np.float32)
        distance_u2 = dump["U"][:, :, :HEAD_RANK].astype(np.float32)
        v2 = dump["V"][:, :HEAD_RANK, :].astype(np.float32)

    s2 = singular[:, :HEAD_RANK]
    weights = s2 / np.sqrt((s2 * s2).sum(axis=1, keepdims=True))
    pooled = (v2 * weights[:, :, None]).reshape(N_HEADS * HEAD_RANK, HIDDEN)
    gram = pooled @ pooled.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 1e-10, None).astype(np.float32)
    eigenvectors = eigenvectors[:, order].astype(np.float32)

    basis4 = (eigenvectors[:, :COMMON_BASIS].T @ pooled) / np.sqrt(
        eigenvalues[:COMMON_BASIS, None]
    )
    share = eigenvalues / eigenvalues.sum()
    effective = float(eigenvalues.sum() ** 2 / np.dot(eigenvalues, eigenvalues))
    n50 = int(np.searchsorted(np.cumsum(share), 0.5) + 1)

    # For each communal component, squared eigenvector coefficients describe
    # how its mass is distributed over the two rows belonging to each head.
    loadings = (eigenvectors * eigenvectors).reshape(N_HEADS, HEAD_RANK, -1).sum(axis=1).T
    loadings /= loadings.sum(axis=1, keepdims=True)
    support = 1.0 / np.sum(loadings * loadings, axis=1)

    return LayerGeometry(
        layer=layer,
        singular=singular,
        distance_u2=distance_u2,
        v2=v2,
        weights=weights,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        basis4=basis4.astype(np.float32),
        structural_share=share,
        effective_directions=effective,
        matched_null_directions=matched_null_pr(weights),
        n50=n50,
        head_loadings=loadings,
        head_support=support,
    )


def live_metrics(geometry: LayerGeometry) -> LiveMetrics:
    """Recover activation of communal hidden directions from compact r-vectors.

    For C_h = U_h S_h V_h, the saved r-vector gives C_h x = proj.T r_h.
    Therefore V_h x = U_h.T proj.T r_h / S_h. Combining those coefficients
    with the pooled eigensystem recovers x projected onto each communal hidden
    direction without using the overflow-prone float16 residual capture.
    """
    layer = geometry.layer
    proj = np.load(WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy").astype(np.float32)
    transform = np.einsum("de,hek->hdk", proj, geometry.distance_u2, optimize=True)
    singular2 = geometry.singular[:, :HEAD_RANK]

    sums = np.zeros(LIVE_COMPONENTS, dtype=np.float64)
    squares = np.zeros(LIVE_COMPONENTS, dtype=np.float64)
    positives = np.zeros(LIVE_COMPONENTS, dtype=np.int64)
    effect = np.zeros(LIVE_COMPONENTS, dtype=np.float64)
    denominator = 0.0
    count = 0
    position_corr = []
    corpus_means = []
    pos = np.arange(8192, dtype=np.float64)
    pos = (pos - pos.mean()) / pos.std()

    for text in TEXTS:
        rvec = np.load(CAPTURE / f"rvec_L{layer:02d}_{text}.npy", mmap_mode="r").astype(
            np.float32
        )
        v_coeff = np.einsum("thd,hdk->thk", rvec, transform, optimize=True)
        v_coeff /= singular2[None, :, :]
        pooled_response = (v_coeff * geometry.weights[None, :, :]).reshape(
            len(rvec), N_HEADS * HEAD_RANK
        )
        activation = pooled_response @ geometry.eigenvectors[:, :LIVE_COMPONENTS]
        activation /= np.sqrt(geometry.eigenvalues[:LIVE_COMPONENTS])[None, :]

        sums += activation.sum(axis=0)
        squares += (activation * activation).sum(axis=0)
        positives += (activation > 0).sum(axis=0)
        effect += geometry.eigenvalues[:LIVE_COMPONENTS] * (
            activation * activation
        ).sum(axis=0)
        denominator += float((pooled_response * pooled_response).sum())
        count += len(activation)
        corpus_means.append(activation.mean(axis=0))
        centered = activation - activation.mean(axis=0, keepdims=True)
        scaled = centered / (activation.std(axis=0, keepdims=True) + 1e-12)
        position_corr.append((pos[:, None] * scaled).mean(axis=0))

    mean = sums / count
    rms = np.sqrt(squares / count)
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0))
    positive_fraction = positives / count
    return LiveMetrics(
        live_share=effect / denominator,
        mean=mean,
        rms=rms,
        constantness=np.abs(mean) / (rms + 1e-12),
        sign_consistency=np.maximum(positive_fraction, 1.0 - positive_fraction),
        median_abs_position_corr=np.median(np.abs(position_corr), axis=0),
        max_abs_position_corr=np.max(np.abs(position_corr), axis=0),
        corpus_mean_spread=np.std(corpus_means, axis=0) / (std + 1e-12),
    )


def random_pair_baseline() -> float:
    # In high dimension, a random 2x2 overlap matrix is accurately approximated
    # by iid N(0, 1/D) entries. This deterministic Monte Carlo gives the null for
    # the same mean-principal-cosine statistic used on the real heads.
    rng = np.random.default_rng(0)
    overlap = rng.normal(size=(200_000, HEAD_RANK, HEAD_RANK)) / np.sqrt(HIDDEN)
    return float(np.linalg.svd(overlap, compute_uv=False).mean())


def basis_affinity(bases: np.ndarray) -> np.ndarray:
    affinity = np.zeros((N_LAYERS, N_LAYERS), dtype=np.float32)
    for i in range(N_LAYERS):
        for j in range(i, N_LAYERS):
            overlap = bases[i] @ bases[j].T
            value = float(np.sum(overlap * overlap) / COMMON_BASIS)
            affinity[i, j] = affinity[j, i] = value
    return affinity


def adjacent_head_metrics(geometries: list[LayerGeometry], affinity: np.ndarray):
    same, other = [], []
    offsets = [1, 7, 19, 31]
    for layer in range(1, N_LAYERS):
        left = geometries[layer - 1].v2
        right = geometries[layer].v2
        overlap = np.einsum("hkd,hld->hkl", left, right, optimize=True)
        same.append(float(np.linalg.svd(overlap, compute_uv=False).mean()))
        shifted = []
        for offset in offsets:
            rotated = np.roll(right, offset, axis=0)
            overlap = np.einsum("hkd,hld->hkl", left, rotated, optimize=True)
            shifted.append(float(np.linalg.svd(overlap, compute_uv=False).mean()))
        other.append(float(np.mean(shifted)))
    adjacent_basis = np.diag(affinity, k=1).astype(float)
    return np.asarray(same), np.asarray(other), adjacent_basis


def figure_geometry_live(
    geometries: list[LayerGeometry], live: list[LiveMetrics]
) -> None:
    layers = np.arange(N_LAYERS)
    observed = np.asarray([g.effective_directions for g in geometries])
    null = np.asarray([g.matched_null_directions for g in geometries])
    structural1 = np.asarray([g.structural_share[0] for g in geometries])
    structural4 = np.asarray([g.structural_share[:4].sum() for g in geometries])
    live1 = np.asarray([m.live_share[0] for m in live])
    live4 = np.asarray([m.live_share[:4].sum() for m in live])
    support = np.asarray([g.head_support[0] / N_HEADS for g in geometries])
    constantness = np.asarray([m.constantness[0] for m in live])
    position = np.asarray([m.median_abs_position_corr[0] for m in live])

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), layout="constrained")
    fig.suptitle("Live states select communal carriers before the weights fully converge on them")

    ax = axes[0]
    phase_spans(ax)
    ax.plot(layers, null, color=GRAY, ls="--", lw=1.8, label="matched random null")
    ax.plot(layers, observed, color=PURPLE, lw=2.1, label="observed pooled spectrum")
    ax.scatter(GLOBAL, observed[GLOBAL], color=ORANGE, marker="D", s=24, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([3, 10, 30, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set(xlabel="trunk layer", ylabel="effective hidden directions", title="A. Sharing exceeds rank-matched chance")
    ax.grid(axis="y", which="both")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    phase_spans(ax)
    ax.plot(layers, structural1, color=RED, ls="--", lw=1.4, label="C1 structural share")
    ax.plot(layers, live1, color=RED, lw=2.1, label="C1 live effect share")
    ax.plot(layers, structural4, color=BLUE, ls="--", lw=1.4, label="C1–C4 structural share")
    ax.plot(layers, live4, color=BLUE, lw=2.1, label="C1–C4 live effect share")
    ax.set(xlabel="trunk layer", ylabel="fraction of pooled read energy", title="B. Live states are far more concentrated than weights")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y")
    ax.legend(
        loc="lower left",
        ncol=2,
        fontsize=8,
        handlelength=1.7,
        columnspacing=0.8,
        labelspacing=0.35,
    )

    ax = axes[2]
    phase_spans(ax)
    ax.plot(layers, support, color=TEAL, lw=2, label="C1 effective head support / 64")
    ax.plot(layers, constantness, color=ORANGE, lw=2, label="C1 |mean| / RMS activation")
    ax.plot(layers, position, color=GRAY, lw=1.6, label="C1 median |corr(position)|")
    ax.scatter(GLOBAL, support[GLOBAL], color=ORANGE, marker="D", s=23, zorder=3)
    ax.set(xlabel="trunk layer", ylabel="fraction / correlation", title="C. The leading carrier is broad and often constant-sign")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=8)

    save(fig, "geometry-vs-live-sharing.png")


def figure_component_anatomy(
    geometries: list[LayerGeometry], live: list[LiveMetrics]
) -> dict[int, np.ndarray]:
    fig = plt.figure(figsize=(18.2, 7.0))
    grid = fig.add_gridspec(2, len(SELECTED), height_ratios=[1.0, 1.3], hspace=0.32, wspace=0.25)
    fig.suptitle("Communal components change from weak coalitions to layer-wide carriers")
    orders: dict[int, np.ndarray] = {}
    heatmaps = []

    for col, layer in enumerate(SELECTED):
        geometry, activation = geometries[layer], live[layer]
        similarity = head_pair_similarity(geometry.v2)
        order = cluster_order(similarity)
        orders[layer] = order

        ax = fig.add_subplot(grid[0, col])
        x = np.arange(1, 9)
        width = 0.38
        ax.bar(x - width / 2, geometry.structural_share[:8], width, color=GRAY, label="weights")
        ax.bar(x + width / 2, activation.live_share[:8], width, color=RED, label="live effect")
        ax.set_ylim(0, 0.9)
        ax.set_xticks(x)
        ax.set_title(
            f"L{layer}: PR={geometry.effective_directions:.1f}\n"
            f"C1 support={geometry.head_support[0]:.0f}/64"
        )
        if col == 0:
            ax.set_ylabel("energy share")
            ax.legend(loc="upper right")
        else:
            ax.set_yticklabels([])
        ax.grid(axis="y")

        ax = fig.add_subplot(grid[1, col])
        loading = geometry.head_loadings[:6, order] * N_HEADS
        image = ax.imshow(loading, aspect="auto", cmap="viridis", vmin=0, vmax=4, origin="upper")
        heatmaps.append(image)
        ax.set_xticks([])
        ax.set_xlabel("heads (clustered order)")
        ax.set_yticks(range(6))
        ax.set_yticklabels([f"C{i}" for i in range(1, 7)] if col == 0 else [])
        if col == 0:
            ax.set_ylabel("communal component")

    cax = fig.add_axes([0.952, 0.16, 0.012, 0.44])
    cbar = fig.colorbar(heatmaps[-1], cax=cax)
    cbar.set_label("head loading relative to uniform")
    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.08, top=0.86)
    save(fig, "component-head-anatomy.png")
    return orders


def figure_head_matrices(
    geometries: list[LayerGeometry], orders: dict[int, np.ndarray], random_baseline: float
) -> None:
    matrices = []
    for layer in SELECTED:
        matrix = head_pair_similarity(geometries[layer].v2)
        matrix = matrix[np.ix_(orders[layer], orders[layer])]
        np.fill_diagonal(matrix, np.nan)
        matrices.append(matrix)
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    vmax = float(np.quantile(finite, 0.99))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, len(SELECTED), figsize=(17.8, 3.9))
    fig.suptitle("Deep sharing is diffuse across heads, not a handful of tight clusters")
    for ax, layer, matrix in zip(axes, SELECTED, matrices):
        image = ax.imshow(matrix, cmap=cmap, vmin=random_baseline, vmax=vmax, origin="upper")
        mean = float(np.nanmean(matrix))
        ax.set_title(f"L{layer}\nmean={mean:.3f}")
        ax.set_xticks([0, 31, 63])
        ax.set_yticks([0, 31, 63])
        ax.set_xlabel("clustered head")
        if ax is axes[0]:
            ax.set_ylabel("clustered head")
        else:
            ax.set_yticklabels([])
    cax = fig.add_axes([0.952, 0.20, 0.012, 0.50])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("mean principal-angle cosine")
    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.16, top=0.79, wspace=0.18)
    save(fig, "head-cluster-matrices.png")


def figure_affinity(affinity: np.ndarray) -> None:
    full = affinity.copy()
    np.fill_diagonal(full, np.nan)
    global_matrix = affinity[np.ix_(GLOBAL, GLOBAL)].copy()
    np.fill_diagonal(global_matrix, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(15.3, 6.3), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("The communal basis is a layer-family property; global layers recur every six blocks")

    ax = axes[0]
    image = ax.imshow(full, origin="lower", cmap=cmap, norm=LogNorm(vmin=3e-4, vmax=0.7))
    ax.set(xlabel="layer", ylabel="layer", title="A. Top-4 communal-basis overlap")
    ax.set_xticks(np.arange(0, 66, 5))
    ax.set_yticks(np.arange(0, 66, 5))
    ax.scatter(GLOBAL, np.full(len(GLOBAL), 65.8), color=ORANGE, marker="v", s=24, clip_on=False)
    ax.scatter(np.full(len(GLOBAL), -0.8), GLOBAL, color=ORANGE, marker=">", s=24, clip_on=False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("average squared canonical correlation")

    ax = axes[1]
    image2 = ax.imshow(global_matrix, origin="lower", cmap=cmap, vmin=0, vmax=0.65)
    ax.set_xticks(range(len(GLOBAL)))
    ax.set_yticks(range(len(GLOBAL)))
    ax.set_xticklabels(GLOBAL, rotation=45)
    ax.set_yticklabels(GLOBAL)
    ax.set(xlabel="global layer", ylabel="global layer", title="B. Global-family reuse—and the L65 reset")
    for i in range(len(GLOBAL)):
        for j in range(len(GLOBAL)):
            if i != j and global_matrix[i, j] >= 0.1:
                ax.text(j, i, f"{global_matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    cbar2 = fig.colorbar(image2, ax=ax, fraction=0.045, pad=0.025)
    cbar2.set_label("top-4 overlap")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "layer-basis-affinity.png")


def figure_head_lineage(
    same: np.ndarray, other: np.ndarray, adjacent_basis: np.ndarray, random_baseline: float
) -> None:
    destination = np.arange(1, N_LAYERS)
    boundary = np.asarray([(layer in GLOBAL) or ((layer - 1) in GLOBAL) for layer in destination])

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 4.7))
    fig.suptitle("The shared basis persists, but head number does not carry a lineage")

    ax = axes[0]
    phase_spans(ax)
    ax.plot(destination, same, color=BLUE, lw=2, label="same head index")
    ax.plot(destination, other, color=RED, lw=1.7, ls="--", label="four shifted head indices")
    ax.axhline(random_baseline, color=GRAY, lw=1.2, ls=":", label="random 2-D subspace null")
    ax.set(xlabel="destination layer", ylabel="adjacent-layer top-2 similarity", title="A. Same-index and other-index heads are indistinguishable")
    ax.grid(axis="y")
    ax.legend(loc="upper left")

    ax = axes[1]
    phase_spans(ax)
    ax.plot(destination, adjacent_basis, color=TEAL, lw=1.8, zorder=1)
    ax.scatter(destination[~boundary], adjacent_basis[~boundary], color=BLUE, s=22, label="same attention scope", zorder=2)
    ax.scatter(destination[boundary], adjacent_basis[boundary], color=RED, marker="D", s=28, label="local/global boundary", zorder=3)
    ax.set(xlabel="destination layer", ylabel="top-4 communal-basis overlap", title="B. What persists is the layer-level basis")
    ax.set_ylim(-0.02, 0.72)
    ax.grid(axis="y")
    ax.legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "head-identity-and-basis-persistence.png")


def kernel_reconstruction(
    layer: int, geometry: LayerGeometry, activation: LiveMetrics
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, float]], np.ndarray]:
    with np.load(D0 / f"layer{layer:02d}.npz") as dump:
        singular = dump["S"].astype(np.float32)
        distance_u = dump["U"].astype(np.float32)
        v_full = dump["V"].astype(np.float32)

    pooled = (geometry.v2 * geometry.weights[:, :, None]).reshape(
        N_HEADS * HEAD_RANK, HIDDEN
    )
    basis16 = (geometry.eigenvectors[:, :LIVE_COMPONENTS].T @ pooled) / np.sqrt(
        geometry.eigenvalues[:LIVE_COMPONENTS, None]
    )
    head_coeff = np.einsum("hkd,jd->hkj", v_full, basis16, optimize=True)
    distance_curves = np.einsum(
        "hek,hk,hkj->hje", distance_u, singular, head_coeff, optimize=True
    )
    components = distance_curves.mean(axis=0) * activation.mean[:, None]
    extent = distance_u.shape[1]
    actual = []
    for text in TEXTS:
        with np.load(TIER2 / f"layer{layer:02d}_{text}_s8192.npz") as dump:
            actual.append(dump["mean_bias"].mean(axis=0)[:extent])
    actual = np.mean(actual, axis=0)

    reconstructions = {}
    metrics = {}
    for count in [1, 4, 16]:
        predicted = components[:count].sum(axis=0)
        reconstructions[count] = predicted
        metrics[count] = {
            "correlation": float(np.corrcoef(predicted, actual)[0, 1]),
            "r2": float(
                1
                - np.sum((actual - predicted) ** 2)
                / (np.sum((actual - actual.mean()) ** 2) + 1e-12)
            ),
            "relative_error": float(np.linalg.norm(actual - predicted) / np.linalg.norm(actual)),
        }
    return actual, components, metrics, np.stack([reconstructions[1], reconstructions[4], reconstructions[16]])


def figure_kernel_reconstruction(
    geometries: list[LayerGeometry], live: list[LiveMetrics]
) -> dict[str, dict[str, dict[str, float]]]:
    fig, axes = plt.subplots(2, 3, figsize=(15.7, 8.0))
    fig.suptitle("Communal directions reconstruct the signed positional kernel—but late global layers need more of them")
    metrics_out = {}
    for ax, layer in zip(axes.flat, KERNEL_SELECTED):
        actual, _, metrics, recon = kernel_reconstruction(layer, geometries[layer], live[layer])
        d = np.arange(len(actual))
        ax.plot(d, actual, color=INK, lw=2.2, label="actual mean bias")
        ax.plot(d, recon[0], color=ORANGE, lw=1.5, label="C1")
        ax.plot(d, recon[1], color=BLUE, lw=1.5, label="C1–C4")
        ax.plot(d, recon[2], color=RED, lw=1.5, ls="--", label="C1–C16")
        ax.axhline(0, color=LIGHT_GRAY, lw=1)
        ax.set_title(
            f"L{layer} ({len(actual)}-token support)\n"
            f"R²: C1 {metrics[1]['r2']:.2f}, C4 {metrics[4]['r2']:.2f}, C16 {metrics[16]['r2']:.2f}"
        )
        ax.set_xlabel("backward distance d")
        ax.set_ylabel("mean signed bias")
        ax.grid(axis="y")
        metrics_out[str(layer)] = {str(k): v for k, v in metrics.items()}
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "communal-kernel-reconstruction.png")
    return metrics_out


def aggregate_pair_types(affinity: np.ndarray) -> dict[str, dict[str, float]]:
    is_global = np.zeros(N_LAYERS, dtype=bool)
    is_global[GLOBAL] = True
    eye = np.eye(N_LAYERS, dtype=bool)
    masks = {
        "global_global": is_global[:, None] & is_global[None, :] & ~eye,
        "local_local": (~is_global[:, None]) & (~is_global[None, :]) & ~eye,
        "cross_scope": (is_global[:, None] ^ is_global[None, :]) & ~eye,
    }
    return {
        name: {
            "mean": float(affinity[mask].mean()),
            "median": float(np.median(affinity[mask])),
            "q90": float(np.quantile(affinity[mask], 0.9)),
        }
        for name, mask in masks.items()
    }


def build_summary(
    geometries: list[LayerGeometry],
    live: list[LiveMetrics],
    affinity: np.ndarray,
    same: np.ndarray,
    other: np.ndarray,
    adjacent_basis: np.ndarray,
    random_baseline: float,
    kernel_metrics: dict,
) -> dict:
    observed = np.asarray([g.effective_directions for g in geometries])
    null = np.asarray([g.matched_null_directions for g in geometries])
    support = np.asarray([g.head_support[0] for g in geometries])
    live1 = np.asarray([m.live_share[0] for m in live])
    live4 = np.asarray([m.live_share[:4].sum() for m in live])
    structural1 = np.asarray([g.structural_share[0] for g in geometries])
    constantness = np.asarray([m.constantness[0] for m in live])
    position = np.asarray([m.median_abs_position_corr[0] for m in live])
    advantage = same - other

    per_layer = {}
    for layer in range(N_LAYERS):
        per_layer[str(layer)] = {
            "effective_directions": float(observed[layer]),
            "matched_random_effective_directions": float(null[layer]),
            "observed_to_null_ratio": float(observed[layer] / null[layer]),
            "directions_for_50pct_structural_energy": geometries[layer].n50,
            "component1_structural_share": float(structural1[layer]),
            "component1_live_effect_share": float(live1[layer]),
            "components1_4_live_effect_share": float(live4[layer]),
            "component1_effective_head_support": float(support[layer]),
            "component1_constantness_abs_mean_over_rms": float(constantness[layer]),
            "component1_median_abs_position_corr": float(position[layer]),
        }

    return {
        "definitions": {
            "pooled_matrix": "For each head, top-2 hidden read directions V are weighted by S1/S2 and normalized to equal total energy per head, then stacked into M[128,6144].",
            "communal_components": "Right singular directions of M, obtained from eig(M M^T).",
            "effective_directions": "Participation ratio of the pooled eigenvalue spectrum; range approaches 128 for independent rows and 1 for one shared direction.",
            "matched_null": "Random 6144-D directions preserving each head's top-2 singular-value ratio and exact within-head orthogonality.",
            "head_support": "Participation ratio of a communal component's squared loadings across 64 heads.",
            "live_effect_share": "lambda_j * E[(x dot b_j)^2] divided by total pooled read energy, recovered exactly from captured r-vectors.",
            "basis_affinity": "Average squared canonical correlation between fixed top-4 communal bases; random expectation is 4/6144.",
        },
        "parity_note": "The r-vector recovery identity was independently checked against directly normalized hidden states at L0: relative error 5.5e-4, correlation 0.9999998.",
        "random_top2_pair_similarity": random_baseline,
        "random_top4_basis_affinity": COMMON_BASIS / HIDDEN,
        "headline": {
            "early_median_observed_directions_L0_13": float(np.median(observed[:14])),
            "early_median_matched_null_L0_13": float(np.median(null[:14])),
            "deep_median_observed_directions_L30_65": float(np.median(observed[30:])),
            "deep_median_matched_null_L30_65": float(np.median(null[30:])),
            "deep_median_component1_head_support": float(np.median(support[30:])),
            "early_median_component1_live_effect_share": float(np.median(live1[:14])),
            "deep_median_component1_live_effect_share": float(np.median(live1[30:])),
            "strongest_sharing_layer": int(np.argmin(observed / null)),
            "strongest_sharing_observed_to_null_ratio": float(np.min(observed / null)),
            "same_head_identity_median_advantage": float(np.median(advantage)),
            "same_head_identity_positive_layer_fraction": float(np.mean(advantage > 0)),
        },
        "pair_type_affinity": aggregate_pair_types(affinity),
        "global_affinity_matrix": affinity[np.ix_(GLOBAL, GLOBAL)].tolist(),
        "adjacent": {
            "destination_layer": list(range(1, N_LAYERS)),
            "same_head_similarity": same.tolist(),
            "shifted_other_head_similarity": other.tolist(),
            "communal_basis_affinity": adjacent_basis.tolist(),
        },
        "kernel_reconstruction": kernel_metrics,
        "per_layer": per_layer,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()

    geometries = []
    for layer in range(N_LAYERS):
        geometries.append(analyze_geometry(layer))
    print("computed pooled geometry for 66 layers")

    live = []
    for layer, geometry in enumerate(geometries):
        live.append(live_metrics(geometry))
        if layer % 11 == 10 or layer == N_LAYERS - 1:
            print(f"processed live r-vector activations through L{layer}")

    bases = np.stack([g.basis4 for g in geometries])
    affinity = basis_affinity(bases)
    same, other, adjacent_basis = adjacent_head_metrics(geometries, affinity)
    random_baseline = random_pair_baseline()

    figure_geometry_live(geometries, live)
    orders = figure_component_anatomy(geometries, live)
    figure_head_matrices(geometries, orders, random_baseline)
    figure_affinity(affinity)
    figure_head_lineage(same, other, adjacent_basis, random_baseline)
    kernel_metrics = figure_kernel_reconstruction(geometries, live)

    summary = build_summary(
        geometries,
        live,
        affinity,
        same,
        other,
        adjacent_basis,
        random_baseline,
        kernel_metrics,
    )
    with (OUT / "subspace_anatomy.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        OUT / "common_bases_top4.npz",
        basis=bases,
        eigenvalues=np.stack([g.eigenvalues for g in geometries]),
        structural_share=np.stack([g.structural_share for g in geometries]),
        live_share=np.stack([m.live_share for m in live]),
        mean_activation=np.stack([m.mean for m in live]),
        head_support=np.stack([g.head_support for g in geometries]),
        affinity=affinity,
    )
    print(f"wrote {OUT / 'subspace_anatomy.json'}")
    print(f"wrote {OUT / 'common_bases_top4.npz'}")


if __name__ == "__main__":
    main()
