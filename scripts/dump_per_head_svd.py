"""Round 3 D0: dump complete rank-16 per-head SVDs for trunk and MTP.

For C_h = proj.T @ Wr_h, materializing C_h before every SVD is unnecessary:
QR(proj.T)=Qa Ra and QR(Wr_h.T)=Qb Rb reduce the problem to the exact 16x16
core Ra @ Rb.T.  The resulting U/S/V are the complete nonzero economy SVD of
C_h, not a truncation or randomized approximation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
OUT_DIR = ROOT / "dumps" / "round3" / "perhead_svd"
NUM_HEADS = 64
D_REL = 16
HIDDEN = 6144
VALIDATION_TOL = 1e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(kind: str, layer: int) -> tuple[Path, Path]:
    if kind == "layer":
        return (
            WEIGHTS_DIR / f"layer{layer:02d}_wr_du.npy",
            WEIGHTS_DIR / f"layer{layer:02d}_rel_logits_proj.npy",
        )
    if kind == "mtp":
        return (
            WEIGHTS_DIR / "mtp" / f"mtp{layer}_wr_du.npy",
            WEIGHTS_DIR / "mtp" / f"mtp{layer}_rel_logits_proj.npy",
        )
    raise ValueError(kind)


def output_path(kind: str, layer: int) -> Path:
    return OUT_DIR / (f"layer{layer:02d}.npz" if kind == "layer" else f"mtp{layer}.npz")


@torch.no_grad()
def dump_layer(kind: str, layer: int, device: torch.device) -> dict[str, object]:
    wr_path, proj_path = input_paths(kind, layer)
    wr = np.load(wr_path)
    proj = np.load(proj_path)
    if wr.shape != (NUM_HEADS * D_REL, HIDDEN):
        raise ValueError(f"{wr_path}: unexpected shape {wr.shape}")
    if proj.shape[0] != D_REL:
        raise ValueError(f"{proj_path}: unexpected shape {proj.shape}")

    blocks = torch.from_numpy(
        np.ascontiguousarray(wr.reshape(NUM_HEADS, D_REL, HIDDEN))
    ).to(device=device, dtype=torch.float32)
    proj_t = torch.from_numpy(np.ascontiguousarray(proj.T)).to(
        device=device, dtype=torch.float32
    )

    qa, ra = torch.linalg.qr(proj_t, mode="reduced")
    qb, rb = torch.linalg.qr(blocks.transpose(1, 2), mode="reduced")
    core = torch.matmul(ra.unsqueeze(0), rb.transpose(-1, -2))
    uc, singular_values, vhc = torch.linalg.svd(core, full_matrices=False)
    u = torch.matmul(qa.unsqueeze(0), uc)
    v = torch.matmul(vhc, qb.transpose(-1, -2))

    signs = torch.where(
        u[:, :8, :].sum(dim=1) < 0,
        torch.tensor(-1.0, device=device),
        torch.tensor(1.0, device=device),
    )
    u = u * signs[:, None, :]
    v = v * signs[:, :, None]

    s_np = singular_values.cpu().numpy().astype(np.float32, copy=False)
    u_np = u.cpu().numpy().astype(np.float32, copy=False)
    v_np = v.cpu().numpy().astype(np.float32, copy=False)
    out_path = output_path(kind, layer)
    np.savez(out_path, S=s_np, U=u_np, V=v_np)
    digest = sha256(out_path)
    print(
        f"{kind:5s} {layer:02d}: S {s_np.shape} U {u_np.shape} V {v_np.shape} "
        f"-> {out_path.name} ({out_path.stat().st_size / 2**20:.1f} MiB)"
    )

    del blocks, proj_t, qa, ra, qb, rb, core, uc, singular_values, vhc, u, v
    torch.cuda.empty_cache()
    return {
        "sha256": digest,
        "bytes": int(out_path.stat().st_size),
        "S_shape": [int(x) for x in s_np.shape],
        "U_shape": [int(x) for x in u_np.shape],
        "V_shape": [int(x) for x in v_np.shape],
        "dtype": "float32",
    }


def validate_dump(kind: str, layer: int, head: int) -> dict[str, object]:
    wr_path, proj_path = input_paths(kind, layer)
    wr = np.load(wr_path, mmap_mode="r")
    proj = np.load(proj_path, mmap_mode="r")
    block = np.asarray(wr[head * D_REL : (head + 1) * D_REL], dtype=np.float32)
    fresh = np.asarray(proj.T, dtype=np.float32) @ block
    with np.load(output_path(kind, layer)) as dump:
        s = dump["S"][head].astype(np.float32, copy=False)
        u = dump["U"][head].astype(np.float32, copy=False)
        v = dump["V"][head].astype(np.float32, copy=False)
    reconstructed = (u * s[None, :]) @ v
    max_abs_diff = float(np.max(np.abs(fresh - reconstructed)))
    passed = bool(max_abs_diff < VALIDATION_TOL)
    print(
        f"validation {kind}:{layer}:head{head}: max abs diff={max_abs_diff:.3e} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        print(
            f"[CONTRADICTION] D0 reconstruction {kind}:{layer}:head{head} "
            f"diff {max_abs_diff:.6g} >= {VALIDATION_TOL}"
        )
    return {
        "kind": kind,
        "layer": int(layer),
        "head": int(head),
        "max_abs_diff": max_abs_diff,
        "tolerance": float(VALIDATION_TOL),
        "passed": passed,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("D0 requires CUDA; no CUDA device is available")
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("layout: confirmed head-major wr.reshape(64,16,6144)")

    files: dict[str, dict[str, object]] = {}
    for kind, count in (("layer", 66), ("mtp", 8)):
        for layer in range(count):
            info = dump_layer(kind, layer, device)
            files[output_path(kind, layer).name] = info

    population = [
        (kind, layer, head)
        for kind, count in (("layer", 66), ("mtp", 8))
        for layer in range(count)
        for head in range(NUM_HEADS)
    ]
    rng = np.random.default_rng(0)
    selected = rng.choice(len(population), size=3, replace=False)
    validations = [validate_dump(*population[int(i)]) for i in selected]
    all_passed = bool(all(item["passed"] for item in validations))

    checksums_path = OUT_DIR / "checksums.json"
    with checksums_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": "sha256",
                "layout": "head-major wr.reshape(64,16,6144)",
                "factorization": "QR(proj.T), QR(Wr_h.T), then SVD of 16x16 core",
                "files": files,
                "validation_seed": 0,
                "reconstruction_validations": validations,
                "passed": all_passed,
            },
            f,
            indent=2,
        )
    print(f"Written {checksums_path}")
    if not all_passed:
        raise AssertionError("D0 reconstruction validation failed")


if __name__ == "__main__":
    main()
