"""LF6 — power-law mimicry: is the 2-exp kernel a quadrature of corpus MI(d)?

Registered prediction (ROUND5_LEFTFIELD_SPEC.md; peek disclosure: 2-exp fits
and their rates exist from Round 4; k=1,3, the power law, and the MI
comparison are new): BIC order 2-exp > power > 1-exp on most globals; 3-exp
not decisively better than 2-exp; kernel shape tracks MI(d) with rank
correlation > 0.9 on prose.

Method frozen before outcomes:
  - MI estimator: token-level plug-in MI (nats) with Miller-Madow correction,
    alphabet = top-63 most frequent token ids + OTHER, distances d = 1..1024.
    Prose source: corpus/_moby.txt (the long source text, tokenized fully);
    code source: corpus/02_code.txt (8,192 tokens — small-sample caveat
    frozen here: its far-field MI is noise-dominated and reported
    descriptively only). Noise floor: mean MI of 5 seeded circular-shift
    surrogates at each d in {16, 64, 256, 1024}, reported alongside.
  - Kernel object: |mode-0| of each global layer's proj bank.
  - Comparison: Spearman(log MI(d), log |mode-0(d)|) over d in [16, 1023],
    prose primary. Prediction threshold: rho > 0.9 on >= 6/11 globals.
  - Family race on d in [32, extent-1], log-space least squares:
    1-exp, 2-exp, 3-exp (positive amplitudes/rates, seeded multistart),
    power law a*d^-alpha. BIC = n*ln(SSE/n) + k*ln(n).
    Clause A: BIC(2exp) < BIC(power) < BIC(1exp) on >= 6/11 globals.
    Clause B: BIC(3exp) >= BIC(2exp) - 6 (no decisive win) on >= 6/11.

Dump-first: `mi` and `fits` write dumps before `analyze` reads only dumps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
CORPUS = ROOT / "corpus"
DUMP_DIR = ROOT / "dumps" / "round5" / "lf6"
OUT_DIR = ROOT / "analysis" / "round5" / "lf6"
GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
ALPHABET = 64
D_MAX = 1024
FIT_LO = 32
SEED = 0x1F6
FLOOR_DS = (16, 64, 256, 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mode0(proj: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(proj.astype(np.float64), full_matrices=False)
    curve = vt[0]
    if np.mean(curve[:32]) < 0:
        curve = -curve
    return curve


def coarse_ids(text: str) -> np.ndarray:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(CORPUS / "tokenizer.json"))
    ids = np.asarray(tok.encode(text).ids, dtype=np.int64)
    values, counts = np.unique(ids, return_counts=True)
    top = values[np.argsort(-counts)[:ALPHABET - 1]]
    mapping = {int(v): i for i, v in enumerate(top)}
    return np.asarray([mapping.get(int(t), ALPHABET - 1) for t in ids],
                      dtype=np.int64)


def mi_curve(x: np.ndarray) -> np.ndarray:
    n_total = len(x)
    out = np.empty(D_MAX)
    for d in range(1, D_MAX + 1):
        a, b = x[:-d], x[d:]
        n = len(a)
        joint = np.bincount(a * ALPHABET + b, minlength=ALPHABET * ALPHABET
                            ).astype(np.float64)
        joint /= n
        px = joint.reshape(ALPHABET, ALPHABET).sum(1)
        py = joint.reshape(ALPHABET, ALPHABET).sum(0)
        nz = joint > 0
        mi = float(np.sum(joint[nz] * np.log(
            joint[nz] / (np.outer(px, py).ravel()[nz]))))
        k_xy = int(nz.sum()); k_x = int((px > 0).sum()); k_y = int((py > 0).sum())
        out[d - 1] = mi - (k_xy - k_x - k_y + 1) / (2.0 * n)
    return out


def mi_command() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    record: dict[str, np.ndarray] = {}
    meta: dict = {}
    for label, path in (("prose", CORPUS / "_moby.txt"),
                        ("code", CORPUS / "02_code.txt")):
        text = path.read_text(encoding="utf-8")
        x = coarse_ids(text)
        meta[label] = {"file": path.name, "n_tokens": int(len(x)),
                       "sha256": sha256_file(path)}
        record[f"mi_{label}"] = mi_curve(x)
        floors = {}
        for d in FLOOR_DS:
            vals = []
            for _ in range(5):
                shift = int(rng.integers(d + 1, len(x) - d - 1))
                y = np.roll(x, shift)
                a, b = x[:-d], y[d:]
                n = len(a)
                joint = np.bincount(a * ALPHABET + b,
                                    minlength=ALPHABET * ALPHABET).astype(np.float64) / n
                px = joint.reshape(ALPHABET, ALPHABET).sum(1)
                py = joint.reshape(ALPHABET, ALPHABET).sum(0)
                nz = joint > 0
                vals.append(float(np.sum(joint[nz] * np.log(
                    joint[nz] / (np.outer(px, py).ravel()[nz])))))
            floors[str(d)] = float(np.mean(vals))
        meta[label]["shuffle_floor"] = floors
        print(f"{label}: {len(x)} tokens, MI(1)={record[f'mi_{label}'][0]:.4f}, "
              f"MI(1024)={record[f'mi_{label}'][-1]:.5f}, floors={floors}", flush=True)
    np.savez(DUMP_DIR / "lf6_mi.npz", **record)
    manifest = {"kind": "round5_lf6_mi_dump", "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "alphabet": ALPHABET, "seed": SEED, "texts": meta,
                "source_sha256": sha256_file(Path(__file__)),
                "dump_sha256": sha256_file(DUMP_DIR / "lf6_mi.npz")}
    (DUMP_DIR / "mi_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("mi dump complete")


def k_exp(k):
    def model(d, *p):
        total = np.zeros_like(d, dtype=np.float64)
        for i in range(k):
            total = total + p[2 * i] * np.exp(-p[2 * i + 1] * d)
        return total
    return model


def fit_family(d: np.ndarray, y: np.ndarray, k: int, rng,
               warm: list | None = None) -> tuple[float, int, list]:
    """Staged fitting: k-exp is seeded from the (k-1)-exp solution plus a
    slow/faster residual component, alongside seeded random restarts. The
    first pass's pure-random restarts collapsed to the 1-exp solution on
    10/11 layers (BIC gap exactly the parameter penalty) — an optimizer
    failure, fixed BEFORE any verdict was recorded."""
    model = k_exp(k)
    log_y = np.log(np.maximum(y, 1e-12))
    starts = []
    if warm is not None:
        for ratio in (0.2, 5.0):
            for amp in (0.3, 0.03):
                slow_rate = max(warm[1] * ratio, 1e-5)
                starts.append(list(warm) + [float(y[0] * amp), float(slow_rate)])
    for _ in range(8):
        p0 = []
        for i in range(k):
            p0 += [float(y[0] * rng.uniform(0.05, 1.0)),
                   float(10 ** rng.uniform(-3.5, -0.7))]
        starts.append(p0)
    best_sse, best_params = np.inf, None
    for p0 in starts:
        if len(p0) != 2 * k:
            continue
        try:
            params, _ = curve_fit(
                lambda dd, *pp: np.log(np.maximum(model(dd, *pp), 1e-12)),
                d, log_y, p0=p0, maxfev=60000, bounds=(0, np.inf))
            sse = float(np.sum((np.log(np.maximum(model(d, *params), 1e-12))
                                - log_y) ** 2))
            if sse < best_sse:
                best_sse, best_params = sse, [float(x) for x in params]
        except Exception:
            continue
    return best_sse, 2 * k, best_params or []


def fits_command() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {}
    sources = {}
    for layer in GLOBALS:
        path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        sources[path.name] = sha256_file(path)
        curve = np.abs(mode0(np.load(path)))
        extent = len(curve)
        d = np.arange(FIT_LO, extent, dtype=np.float64)
        y = curve[FIT_LO:extent]
        log_y = np.log(np.maximum(y, 1e-12))
        n = len(d)
        entry = {}
        warm = None
        for k in (1, 2, 3):
            sse, n_params, params = fit_family(d, y, k, rng, warm=warm)
            warm = params if params else warm
            entry[f"exp{k}"] = {"sse": sse, "params": params,
                                "bic": float(n * np.log(sse / n) + n_params * np.log(n))}
        design = np.stack([np.log(d), np.ones_like(d)], 1)
        coeffs, res, *_ = np.linalg.lstsq(design, log_y, rcond=None)
        sse = float(res[0]) if len(res) else float(
            np.sum((design @ coeffs - log_y) ** 2))
        entry["power"] = {"sse": sse, "alpha": float(-coeffs[0]),
                          "bic": float(n * np.log(sse / n) + 2 * np.log(n))}
        results[f"L{layer:02d}"] = entry
        print(f"L{layer:02d}: " + " ".join(
            f"{m}={entry[m]['bic']:.1f}" for m in ("exp1", "exp2", "exp3", "power")),
            flush=True)
    manifest = {"kind": "round5_lf6_fit_dump", "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fit_lo": FIT_LO, "restarts": 8, "seed": SEED,
                "fits": results,
                "source_sha256": sha256_file(Path(__file__)),
                "input_sha256": sources}
    (DUMP_DIR / "fits_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("fits dump complete")


def analyze_command() -> None:
    mi_manifest = json.loads((DUMP_DIR / "mi_manifest.json").read_text(encoding="utf-8"))
    if mi_manifest["dump_sha256"] != sha256_file(DUMP_DIR / "lf6_mi.npz"):
        raise RuntimeError("MI dump hash mismatch")
    fits_manifest_sha = sha256_file(DUMP_DIR / "fits_manifest.json")
    fits = json.loads((DUMP_DIR / "fits_manifest.json").read_text(encoding="utf-8"))["fits"]
    z = np.load(DUMP_DIR / "lf6_mi.npz", allow_pickle=False)
    mi_prose = z["mi_prose"]; mi_code = z["mi_code"]

    correlations = {}
    for layer in GLOBALS:
        curve = np.abs(mode0(np.load(WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy")))
        extent = len(curve)
        hi = min(extent - 1, 1023)
        dd = np.arange(16, hi + 1)
        rho_p = float(spearmanr(np.log(np.maximum(mi_prose[dd - 1], 1e-9)),
                                np.log(np.maximum(curve[dd], 1e-12))).statistic)
        rho_c = float(spearmanr(np.log(np.maximum(mi_code[dd - 1], 1e-9)),
                                np.log(np.maximum(curve[dd], 1e-12))).statistic)
        correlations[f"L{layer:02d}"] = {"prose": rho_p, "code_descriptive": rho_c}

    clause_a = clause_b = 0
    order = {}
    for name, entry in fits.items():
        a = entry["exp2"]["bic"] < entry["power"]["bic"] < entry["exp1"]["bic"]
        b = entry["exp3"]["bic"] >= entry["exp2"]["bic"] - 6.0
        clause_a += a; clause_b += b
        order[name] = {"bic_order_2exp_power_1exp": bool(a),
                       "exp3_not_decisive": bool(b),
                       "bic": {m: entry[m]["bic"] for m in entry}}
    n_rho = sum(v["prose"] > 0.9 for v in correlations.values())
    report = {"kind": "round5_lf6_mi_mimicry", "schema_version": 1,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "mi_dump_sha256": mi_manifest["dump_sha256"],
              "fits_manifest_sha256": fits_manifest_sha,
              "source_sha256": sha256_file(Path(__file__)),
              "provenance_notes": [
                  "the correlation stage reloads proj weights (mode-0 is "
                  "recomputed live, not read from a dump) - a deviation from "
                  "the dump-first docstring, disclosed",
                  "only the staged-init fits manifest is on disk; the "
                  "collapsed first-pass BIC table exists in the session "
                  "record, not as an artifact"],
              "correlations": correlations,
              "family_race": order,
              "prediction": {
                  "registered": ("BIC 2exp > power > 1exp on most globals; "
                                 "3exp not decisive; prose rank-corr > 0.9"),
                  "clause_bic_order": f"{clause_a}/11 (needs >=6)",
                  "clause_exp3": f"{clause_b}/11 (needs >=6)",
                  "clause_rho": f"{n_rho}/11 (needs >=6)",
                  "passed": bool(clause_a >= 6 and clause_b >= 6 and n_rho >= 6)}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lf6_mi_mimicry.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["prediction"], indent=2))
    print({k: round(v["prose"], 3) for k, v in correlations.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["mi", "fits", "analyze"])
    args = parser.parse_args()
    {"mi": mi_command, "fits": fits_command, "analyze": analyze_command}[args.stage]()


if __name__ == "__main__":
    main()
