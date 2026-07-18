"""LF7 — MTP parentage: did the drafter fork the trunk?

Registered prediction (ROUND5_LEFTFIELD_SPEC.md, blind): MTP layers' nearest
trunk parents cluster in the deep trunk (L30+), and all 8 drafters share one
parent neighborhood. Null = distances among unrelated trunk pairs.

Method choices FROZEN here, before any outcome is computed:
  - Hidden-side object: the row space of wr_du [1024, 6144] — the directions
    of the hidden state that feed the positional table.
  - PRIMARY metric: chordal subspace distance between energy-90% right
    singular subspaces, using k = min(k_i, k_j) leading directions of each.
  - SECONDARY metric: same with fixed k = 64.
  - Curve-side check: chordal distance between proj row spaces restricted to
    the shared d < 512 window (extent-mismatch pairs flagged, not excluded).
  - Verdict thresholds: prediction clause (a) passes if >= 6/8 primary-metric
    parents have depth >= 30 AND the median parent depth >= 30; clause (b)
    passes if the max pairwise spread of the 8 parent depths is <= 12.
  - Fork-vs-fresh: an MTP layer is a "fork" of its parent if its parent
    distance is below the 5th percentile of the trunk-trunk null (all
    unordered trunk pairs, both metrics reported; primary decides).
  - Independent re-derivation (promotion rule): nearest parents recomputed
    from sketched Gram matrices G_i = (W_i R)^T (W_i R), R fixed-seed
    [6144 x 256] Gaussian; Frobenius-cosine similarity. Primary parents must
    match the sketch parents for >= 6/8 drafters for any claim promotion.

Dump-first: stage `dump` writes all spectra, subspaces, and distance matrices
to dumps/round5/lf7/ before stage `analyze` reads ONLY the dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
MTP = WEIGHTS / "mtp"
DUMP_DIR = ROOT / "dumps" / "round5" / "lf7"
OUT_DIR = ROOT / "analysis" / "round5" / "lf7"
GLOBALS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}
ENERGY = 0.90
K_FIXED = 64
# K_STORE must cover every unit's energy-90% rank. History of this constant:
# 64 (bug: silently degraded the primary to k=64), then 640 (bug: TRUNK k90
# reaches 776 - MTP tops out at 616, so MTP-trunk distances were intact, but
# 406/2145 trunk-trunk null pairs were computed on truncated bases with an
# unmatched sqrt(k) normalization, corrupting the null percentiles). Now the
# full row-space bound; chordal() also refuses k beyond the stored basis.
K_STORE = 1024
SKETCH_SEED = 0x1F7
DEEP_MIN = 30
MIN_DEEP_PARENTS = 6
NEIGHBORHOOD_SPAN = 12
FORK_PERCENTILE = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_files() -> list[tuple[str, Path, Path]]:
    units = []
    for layer in range(66):
        units.append((f"L{layer:02d}", WEIGHTS / f"layer{layer:02d}_wr_du.npy",
                      WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"))
    for drafter in range(8):
        units.append((f"M{drafter}", MTP / f"mtp{drafter}_wr_du.npy",
                      MTP / f"mtp{drafter}_rel_logits_proj.npy"))
    return units


def chordal(u: np.ndarray, v: np.ndarray, k: int) -> float:
    """Chordal distance between the leading-k column subspaces of u, v."""
    if k > u.shape[1] or k > v.shape[1]:
        raise ValueError(f"k={k} exceeds stored basis width {u.shape[1]}")
    a, b = u[:, :k], v[:, :k]
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.sqrt(max(k - float(np.sum(s * s)), 0.0)) / np.sqrt(k))


def dump_command() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    units = unit_files()
    names = [name for name, _, _ in units]
    spectra: dict[str, np.ndarray] = {}
    bases: dict[str, np.ndarray] = {}
    k90: dict[str, int] = {}
    projs: dict[str, np.ndarray] = {}
    sketches: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(SKETCH_SEED)
    sketch = rng.standard_normal((6144, 256)).astype(np.float64)
    sources = {}
    for name, wpath, ppath in units:
        w = np.load(wpath).astype(np.float64)
        sources[wpath.name] = sha256_file(wpath)
        sources[ppath.name] = sha256_file(ppath)
        _, s, vt = np.linalg.svd(w, full_matrices=False)
        energy = np.cumsum(s * s) / np.sum(s * s)
        k90[name] = int(np.searchsorted(energy, ENERGY) + 1)
        spectra[name] = s.astype(np.float32)
        bases[name] = vt[:K_STORE].T.astype(np.float32)          # [6144, 640]
        projs[name] = np.load(ppath).astype(np.float32)[:, :512]  # shared window
        ws = w @ sketch                                           # [1024, 256]
        g = ws.T @ ws
        sketches[name] = (g / np.linalg.norm(g)).astype(np.float32)
        print(f"{name}: k90={k90[name]}", flush=True)

    n = len(names)
    d_primary = np.zeros((n, n)); d_fixed = np.zeros((n, n))
    d_curve = np.zeros((n, n)); s_sketch = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = bases[names[i]].astype(np.float64), bases[names[j]].astype(np.float64)
            kp = min(k90[names[i]], k90[names[j]])
            d_primary[i, j] = d_primary[j, i] = chordal(bi, bj, kp)
            d_fixed[i, j] = d_fixed[j, i] = chordal(bi, bj, K_FIXED)
            pi = projs[names[i]].astype(np.float64)
            pj = projs[names[j]].astype(np.float64)
            qi = np.linalg.qr(pi.T)[0]; qj = np.linalg.qr(pj.T)[0]
            d_curve[i, j] = d_curve[j, i] = chordal(qi, qj, 16)
            s_sketch[i, j] = s_sketch[j, i] = float(
                np.sum(sketches[names[i]].astype(np.float64) * sketches[names[j]].astype(np.float64)))
        s_sketch[i, i] = 1.0

    np.savez(DUMP_DIR / "lf7_dump.npz",
             names=np.array(names), k90=np.array([k90[m] for m in names]),
             d_primary=d_primary, d_fixed=d_fixed, d_curve=d_curve,
             s_sketch=s_sketch,
             spectra=np.stack([np.pad(spectra[m], (0, 1024 - len(spectra[m]))) for m in names]),
             # full bases so the dump is self-contained for independent
             # re-derivation (audit finding: earlier dumps stored only the
             # finished matrices)
             **{f"basis_{m}": bases[m] for m in names})
    manifest = {
        "kind": "round5_lf7_parentage_dump", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_frozen": {"energy": ENERGY, "k_fixed": K_FIXED,
                          "sketch_seed": SKETCH_SEED, "curve_window": 512,
                          "primary": "chordal, energy-90% subspaces, k=min(ki,kj)"},
        "source_sha256": sha256_file(Path(__file__)),
        "input_sha256": sources,
        "dump_sha256": sha256_file(DUMP_DIR / "lf7_dump.npz"),
    }
    with (DUMP_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print("dump complete")


def analyze_command() -> None:
    manifest = json.loads((DUMP_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dump_sha256"] != sha256_file(DUMP_DIR / "lf7_dump.npz"):
        raise RuntimeError("dump hash mismatch")
    z = np.load(DUMP_DIR / "lf7_dump.npz", allow_pickle=False)
    names = [str(x) for x in z["names"]]
    trunk = [i for i, m in enumerate(names) if m.startswith("L")]
    mtps = [i for i, m in enumerate(names) if m.startswith("M")]

    def parents(matrix: np.ndarray, largest: bool = False) -> dict[str, dict]:
        result = {}
        for i in mtps:
            row = matrix[i, trunk]
            j = int(np.argmax(row) if largest else np.argmin(row))
            result[names[i]] = {"parent": names[trunk[j]], "depth": trunk[j],
                                "distance": float(row[j])}
        return result

    d_primary = z["d_primary"]; d_fixed = z["d_fixed"]; d_curve = z["d_curve"]
    s_sketch = z["s_sketch"]
    primary = parents(d_primary)
    fixed = parents(d_fixed)
    curve = parents(d_curve)
    sketch = parents(s_sketch, largest=True)

    null = np.array([d_primary[i, j] for a, i in enumerate(trunk)
                     for j in trunk[a + 1:]])
    threshold = float(np.percentile(null, FORK_PERCENTILE))
    curve_null = np.array([d_curve[i, j] for a, i in enumerate(trunk)
                           for j in trunk[a + 1:]])
    for name, record in curve.items():
        record["null_percentile"] = float(
            100.0 * np.mean(curve_null <= record["distance"]))
        record["below_trunk_minimum"] = bool(record["distance"] < curve_null.min())
    for name, record in primary.items():
        record["null_percentile"] = float(100.0 * np.mean(null <= record["distance"]))
        record["fork"] = bool(record["distance"] < threshold)
        record["sketch_agrees"] = sketch[name]["parent"] == record["parent"]

    depths = [record["depth"] for record in primary.values()]
    agree = sum(record["sketch_agrees"] for record in primary.values())
    clause_a = (sum(d >= DEEP_MIN for d in depths) >= MIN_DEEP_PARENTS
                and float(np.median(depths)) >= DEEP_MIN)
    clause_b = (max(depths) - min(depths)) <= NEIGHBORHOOD_SPAN
    report = {
        "kind": "round5_lf7_parentage", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dump_sha256": manifest["dump_sha256"],
        "source_sha256": sha256_file(Path(__file__)),
        "primary_parents": primary,
        "fixed_k_parents": fixed,
        "curve_parents": curve,
        "sketch_parents": sketch,
        "sketch_agreement": f"{agree}/8",
        "trunk_null": {"n_pairs": int(len(null)), "p05": threshold,
                       "median": float(np.median(null)),
                       "min": float(null.min()), "max": float(null.max())},
        "curve_trunk_null": {"median": float(np.median(curve_null)),
                             "p05": float(np.percentile(curve_null, 5)),
                             "min": float(curve_null.min())},
        "curve_note": ("secondary observation, NOT promoted: all 8 MTP banks "
                       "sit below the trunk-trunk minimum, nearest L51 (deep "
                       "LOCAL) x7 / L47 (global) x1; the mode-0 cosine "
                       "re-derivation (lf7_rederivation.json) does not confirm "
                       "the neighborhood (1/8) - its null is saturated "
                       "(p95=0.996), so a discriminating re-derivation is "
                       "still owed before any promotion"),
        "prediction": {
            "clause_a_deep_cluster": bool(clause_a),
            "clause_b_shared_neighborhood": bool(clause_b),
            "registered": "parents cluster L30+ and share one neighborhood",
            "passed": bool(clause_a and clause_b),
        },
        "promotion_gate": {
            "requires": "sketch parents match primary for >= 6/8",
            "met": bool(agree >= 6),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "lf7_parentage.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({m: (r["parent"], round(r["distance"], 4), r["fork"],
                          r["sketch_agrees"]) for m, r in primary.items()},
                     indent=2))
    print("prediction:", report["prediction"])
    print("promotion gate met:", report["promotion_gate"]["met"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["dump", "analyze"])
    args = parser.parse_args()
    if args.stage == "dump":
        dump_command()
    else:
        analyze_command()


if __name__ == "__main__":
    main()
