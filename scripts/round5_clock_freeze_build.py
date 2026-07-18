"""Build the frozen clock-freeze inputs for the R5-D clock arms.

Registered in ROUND5_R5D_CLOCK_AMENDMENT.md. Reads only the certified
widened-capture dumps; writes analysis/round5/r5d_clock/clock_freeze.npz and
a manifest with every input hash. Refuses overwrite.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
AMENDMENT = ROOT / "registrations" / "ROUND5_R5D_CLOCK_AMENDMENT.md"
OUT_DIR = ROOT / "analysis" / "round5" / "r5d_clock"
OUT_NPZ = OUT_DIR / "clock_freeze.npz"
OUT_MANIFEST = OUT_DIR / "clock_freeze_manifest.json"

TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
LAYERS = [53, 59, 65]
SEQ = 8192


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT_NPZ.exists() or OUT_MANIFEST.exists():
        raise FileExistsError("refusing to overwrite the sealed clock freeze")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    starts = np.arange(64, SEQ, 64)
    x = np.log1p(starts + 31.5)
    xc = x - x.mean()

    hashes: dict[str, str] = {"amendment": sha256_file(AMENDMENT)}
    arrays: dict[str, np.ndarray] = {}
    baseline_stats: dict[str, float] = {}

    for L in LAYERS:
        path = CAP / "replay" / f"rvec_L{L:02d}_06_random.npy"
        hashes[path.name] = sha256_file(path)
        r = np.load(path).astype(np.float32).reshape(SEQ, 1024).astype(np.float64)
        B = r[64:].reshape(127, 64, 1024).mean(1)
        Bc = B - B.mean(0)
        slope = (xc @ Bc) / (xc @ xc)
        g = slope / np.linalg.norm(slope)
        arrays[f"G_L{L}"] = g

        # anchor mean over the six certified v1 texts
        acc = np.zeros(1024, dtype=np.float64)
        for t in TEXTS:
            tp = CAP / "replay" / f"rvec_L{L:02d}_{t}.npy"
            if tp.name not in hashes:
                hashes[tp.name] = sha256_file(tp)
            acc += np.load(tp).astype(np.float32).reshape(SEQ, 1024).astype(np.float64).mean(0)
        arrays[f"rbar_L{L}"] = acc / len(TEXTS)

        # peeked baseline |corr| median for the record (06_random)
        proj = np.asarray(np.load(ROOT / "weights" / f"layer{L:02d}_rel_logits_proj.npy"), dtype=np.float64)
        hashes[f"layer{L:02d}_rel_logits_proj.npy"] = sha256_file(
            ROOT / "weights" / f"layer{L:02d}_rel_logits_proj.npy")
        curves = B.reshape(127, 64, 16) @ proj
        meanc = curves.mean(0)
        gain = (curves * meanc).sum(2) / (meanc * meanc).sum(1)
        corr = np.array([np.corrcoef(gain[:, h], x)[0, 1] for h in range(64)])
        baseline_stats[f"baseline_abs_gain_corr_median_L{L}"] = float(np.median(np.abs(corr)))

    # sham: seeded Gaussian orthogonal to G_59
    seed_payload = f"{hashes['amendment']}:clock_sham_L59".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    z = rng.standard_normal(1024)
    g59 = arrays["G_L59"]
    z -= (z @ g59) * g59
    sham = z / np.linalg.norm(z)
    if abs(float(sham @ g59)) > 1e-12:
        raise RuntimeError("sham orthogonalization failed")
    arrays["sham_L59"] = sham

    np.savez(OUT_NPZ, **arrays)
    manifest = {
        "schema_version": 1,
        "kind": "round5_r5d_clock_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_source_sha256": sha256_file(Path(__file__)),
        "sham_seed_unsigned_be": seed,
        "input_sha256": hashes,
        "artifact_sha256": sha256_file(OUT_NPZ),
        "layers": LAYERS,
        "peeked_baseline_stats": baseline_stats,
        "capture_manifest_sha256": sha256_file(CAP / "manifest.json"),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "input_sha256"}, indent=1))


if __name__ == "__main__":
    main()
