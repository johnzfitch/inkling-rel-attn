"""Round 3 D1: dump complete layer-level relative-position SVDs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
OUT_DIR = ROOT / "dumps" / "round3" / "mode_curves"
NEAR = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_path(kind: str, layer: int) -> Path:
    if kind == "layer":
        return WEIGHTS_DIR / f"layer{layer:02d}_rel_logits_proj.npy"
    return WEIGHTS_DIR / "mtp" / f"mtp{layer}_rel_logits_proj.npy"


def output_path(kind: str, layer: int) -> Path:
    return OUT_DIR / (f"layer{layer:02d}.npz" if kind == "layer" else f"mtp{layer}.npz")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for kind, count in (("layer", 66), ("mtp", 8)):
        for layer in range(count):
            proj = np.load(input_path(kind, layer)).astype(np.float32, copy=False)
            u, singular_values, vt = np.linalg.svd(proj, full_matrices=False)
            for mode in range(16):
                if float(vt[mode, :NEAR].sum()) < 0.0:
                    vt[mode] *= -1.0
                    u[:, mode] *= -1.0
            u = u.astype(np.float32, copy=False)
            singular_values = singular_values.astype(np.float32, copy=False)
            vt = vt.astype(np.float32, copy=False)
            out_path = output_path(kind, layer)
            np.savez(out_path, S=singular_values, U=u, Vt=vt)
            reconstruction = (u * singular_values[None, :]) @ vt
            max_abs_diff = float(np.max(np.abs(proj - reconstruction)))
            files[out_path.name] = {
                "sha256": sha256(out_path),
                "bytes": int(out_path.stat().st_size),
                "S_shape": [int(x) for x in singular_values.shape],
                "U_shape": [int(x) for x in u.shape],
                "Vt_shape": [int(x) for x in vt.shape],
                "dtype": "float32",
                "max_abs_reconstruction_error": max_abs_diff,
            }
            print(
                f"{kind:5s} {layer:02d}: extent={proj.shape[1]} "
                f"reconstruction={max_abs_diff:.3e} -> {out_path.name}"
            )
            if max_abs_diff >= 1e-5:
                print(
                    f"[CONTRADICTION] D1 {kind}:{layer} reconstruction error "
                    f"{max_abs_diff:.6g} >= 1e-5"
                )

    checksums_path = OUT_DIR / "checksums.json"
    with checksums_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": "sha256",
                "sign_convention": "flip U[:,k] and Vt[k,:] when sum(Vt[k,:8]) < 0",
                "files": files,
            },
            f,
            indent=2,
        )
    print(f"Written {checksums_path}")


if __name__ == "__main__":
    main()
