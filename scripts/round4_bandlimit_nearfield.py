"""R4-W (Whittaker band-limit) + R4-N (near-field battery) — the two
never-run ROUND4_SPEC.md sections, implemented as registered.

R4-W (spec): for every mode curve, rFFT; effective bandwidth f90 (fraction of
spectral energy below f); decimate 2x and 4x, reconstruct by sinc (Whittaker)
interpolation, report reconstruction error. Output: bandlimit.json.

R4-N (spec): per (layer, head), the 8 raw near-field values (d in [0,8)) of
the dominant per-head mode curve (dumps/round3/perhead_svd, same object the
family battery used). Tests: (1) discontinuity = distance of the near-field
vector from the far-field fit extrapolated back to d<8; (2) k-means over the
4224 vectors, k selected by silhouette over 2..8, seed 0. Output:
nearfield.json.

Method details frozen here before outcomes:
  - Mode curves for R4-W: modes 0-2 per layer (99%+ of energy at rank ~2).
  - f90 = smallest FFT bin index f such that cumulative power spectrum
    (excluding DC) reaches 90%, reported as cycles/token (f / extent).
  - Decimation: keep every 2nd (4th) sample starting at 0; sinc-reconstruct
    on the full grid (Whittaker sum over kept samples with the decimated
    sampling interval); error = RMSE(reconstruction, original) / RMS(original)
    evaluated on interior points d in [16, extent-16] to avoid edge lobes.
  - R4-N far-field fit: a*exp(-r*d)+c on d in [8,128] per head curve
    (LF6: far field is single-exp), extrapolated to d in [0,8);
    discontinuity = ||near - pred||2 / (RMS of curve on d in [8,128]).
  - Clustering: vectors L2-normalized after subtracting nothing (raw shape
    incl. scale sign; normalization makes motifs scale-free), k-means
    (n_init=10, seed 0), silhouette on the normalized vectors.

Dump-first: reads only existing round3 dumps + weights meta; writes JSONs to
analysis/round4/ (the spec's registered output location).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
D0 = ROOT / "dumps" / "round3" / "perhead_svd"
D1 = ROOT / "dumps" / "round3" / "mode_curves"
META = ROOT / "weights" / "_meta.json"
OUT = ROOT / "analysis" / "round4"
GLOBALS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def f90_cycles(curve: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(curve)) ** 2
    spectrum[0] = 0.0
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    cum = np.cumsum(spectrum) / total
    f = int(np.searchsorted(cum, 0.90) + 1)
    return f / len(curve)


def sinc_reconstruct(curve: np.ndarray, stride: int) -> np.ndarray:
    kept = np.arange(0, len(curve), stride)
    grid = np.arange(len(curve), dtype=np.float64)
    return np.array([
        float(np.sum(curve[kept] * np.sinc((g - kept) / stride))) for g in grid])


def bandlimit() -> dict:
    meta = json.loads(META.read_text(encoding="utf-8"))
    layers = {}
    for key in sorted(meta, key=int):
        layer = int(key)
        extent = meta[key]["extent"]
        data = np.load(D1 / f"layer{layer:02d}.npz")
        modes = {}
        for mode in range(3):
            curve = (data["S"][mode] * data["Vt"][mode, :]).astype(np.float64)
            interior = slice(16, extent - 16)
            entry = {"f90_cycles_per_token": f90_cycles(curve)}
            for stride in (2, 4):
                recon = sinc_reconstruct(curve, stride)
                err = (np.sqrt(np.mean((recon[interior] - curve[interior]) ** 2))
                       / max(np.sqrt(np.mean(curve[interior] ** 2)), 1e-300))
                entry[f"decimate_{stride}x_rel_rmse"] = float(err)
            modes[str(mode)] = entry
        layers[f"L{layer:02d}"] = modes
    return layers


def nearfield() -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    meta = json.loads(META.read_text(encoding="utf-8"))
    vectors, keys, discontinuity = [], [], {}
    for key in sorted(meta, key=int):
        layer = int(key)
        data = np.load(D0 / f"layer{layer:02d}.npz")
        for head in range(64):
            curve = (data["S"][head, 0] * data["U"][head, :, 0]).astype(np.float64)
            near = curve[:8]
            d_far = np.arange(8, 129, dtype=np.float64)
            y_far = curve[8:129]
            scale = max(float(np.sqrt(np.mean(y_far ** 2))), 1e-300)
            amp = 5.0 * float(np.max(np.abs(y_far)))
            try:
                # C4 lesson: components unresolvable on the fit window (fast
                # rates, free amplitudes) explode when extrapolated backward.
                # Bound r so exp(-8r) stays appreciable (r <= 0.35) and
                # amplitudes to 5x the window peak.
                params, _ = curve_fit(
                    lambda dd, a, r, c: a * np.exp(-r * dd) + c,
                    d_far, y_far,
                    p0=(float(y_far[0] - y_far[-1]), 0.03, float(y_far[-1])),
                    bounds=([-amp, 0.0, -amp], [amp, 0.35, amp]),
                    maxfev=20000)
                pred = params[0] * np.exp(-params[1] * np.arange(8.0)) + params[2]
                disc = float(np.linalg.norm(near - pred) / (np.sqrt(8) * scale))
            except Exception:
                disc = float("nan")
            discontinuity[f"L{layer:02d}_h{head}"] = disc
            norm = np.linalg.norm(near)
            vectors.append(near / norm if norm > 0 else near)
            keys.append((layer, head))
    matrix = np.stack(vectors)
    best_k, best_sil, best_labels = None, -2.0, None
    sils = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(matrix)
        sil = float(silhouette_score(matrix, km.labels_))
        sils[str(k)] = sil
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, km.labels_
    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(matrix)
    centroids = km.cluster_centers_
    labels = km.labels_
    cluster_summary = {}
    for c in range(best_k):
        members = [keys[i] for i in range(len(keys)) if labels[i] == c]
        by_scope = {"global": sum(1 for l, _ in members if l in GLOBALS),
                    "local": sum(1 for l, _ in members if l not in GLOBALS)}
        cluster_summary[str(c)] = {
            "n": len(members), "scope": by_scope,
            "centroid": [round(float(x), 4) for x in centroids[c]],
            "depth_quartiles": [float(np.percentile([l for l, _ in members], q))
                                for q in (25, 50, 75)]}
    per_layer_motif = {}
    for c in range(best_k):
        for i, (layer, _head) in enumerate(keys):
            name = f"L{layer:02d}"
            per_layer_motif.setdefault(name, [0] * best_k)
            if labels[i] == c:
                per_layer_motif[name][c] += 1
    disc_values = np.array([v for v in discontinuity.values() if np.isfinite(v)])
    return {"discontinuity_summary": {
                "median": float(np.median(disc_values)),
                "q90": float(np.percentile(disc_values, 90)),
                "max": float(disc_values.max()),
                "n_fit_failures": int(sum(1 for v in discontinuity.values()
                                          if not np.isfinite(v)))},
            "per_layer_motif_counts": per_layer_motif,
            "silhouettes": sils, "chosen_k": int(best_k),
            "chosen_silhouette": best_sil,
            "clusters": cluster_summary,
            "per_head_discontinuity_q": {
                "hi": sorted((k for k, v in discontinuity.items() if np.isfinite(v)),
                             key=lambda k: -discontinuity[k])[:12]}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    sources = {"_meta.json": sha256_file(META),
               "script": sha256_file(Path(__file__))}
    band = {"kind": "round4_bandlimit", "schema_version": 1,
            "created_at_utc": stamp, "sources": sources, "layers": bandlimit()}
    (OUT / "bandlimit.json").write_text(
        json.dumps(band, indent=2, sort_keys=True), encoding="utf-8")
    values = [(name, modes["0"]["f90_cycles_per_token"],
               modes["0"]["decimate_2x_rel_rmse"], modes["0"]["decimate_4x_rel_rmse"])
              for name, modes in band["layers"].items()]
    f90s = [v[1] for v in values]
    print(f"mode-0 f90 cycles/token: median {np.median(f90s):.4f} "
          f"min {min(f90s):.4f} max {max(f90s):.4f}")
    print(f"decimate-2x rel RMSE: median {np.median([v[2] for v in values]):.4f}; "
          f"4x: median {np.median([v[3] for v in values]):.4f}")

    near = {"kind": "round4_nearfield", "schema_version": 1,
            "created_at_utc": stamp, "sources": sources, **nearfield()}
    (OUT / "nearfield.json").write_text(
        json.dumps(near, indent=2, sort_keys=True), encoding="utf-8")
    print("discontinuity:", near["discontinuity_summary"])
    print("silhouettes:", {k: round(v, 3) for k, v in near["silhouettes"].items()})
    print("chosen k:", near["chosen_k"], "clusters:",
          {c: v["n"] for c, v in near["clusters"].items()})


if __name__ == "__main__":
    main()
