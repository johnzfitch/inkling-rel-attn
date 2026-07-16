"""Round 3 A1: validate official relative-logit gather/mask semantics."""
from __future__ import annotations

import json
import hashlib
import inspect
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
OUT_DIR = ROOT / "analysis" / "round3"
LAYERS = (0, 5, 33, 65)
NUM_HEADS = 64
D_REL = 16
NUM_DRAWS = 5
Q_LEN = 64
TOL = 1e-5
SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/main/"
    "src/transformers/models/inkling/modular_inkling.py"
)


def load_official_relative_logits():
    """Import the installed Transformers implementation and identify its source.

    The repository keeps the Inkling-capable environment in ``.venv-tier2``.
    Adding that environment's site-packages also makes this validation runnable
    from another Python executable without copying the official implementation
    into this script.  Failure to import is fatal: a local fallback would make
    the validation self-referential again.
    """
    site_packages = ROOT / ".venv-tier2" / "Lib" / "site-packages"
    if site_packages.is_dir() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))

    try:
        import transformers
        from transformers.models.inkling.modeling_inkling import InklingRelativeLogits
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The official Transformers InklingRelativeLogits implementation is "
            "required; run this script with .venv-tier2 or install a Transformers "
            "release that contains the Inkling model."
        ) from exc

    source_name = inspect.getsourcefile(InklingRelativeLogits)
    if source_name is None:
        raise RuntimeError("Could not locate the official InklingRelativeLogits source")
    source_path = Path(source_name).resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    try:
        recorded_path = str(source_path.relative_to(ROOT))
    except ValueError:
        recorded_path = str(source_path)
    identity = {
        "package": "transformers",
        "version": str(transformers.__version__),
        "class": f"{InklingRelativeLogits.__module__}.{InklingRelativeLogits.__name__}",
        "source_path": recorded_path,
        "source_sha256": source_sha256,
    }
    return InklingRelativeLogits, identity


def relative_logits_forward(
    relative_states: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    proj: torch.Tensor,
) -> torch.Tensor:
    """Local copy of InklingRelativeLogits.forward semantics."""
    rel_extent = proj.shape[1]
    rel_logits = (relative_states @ proj).transpose(1, 2)
    distance = (query_positions[:, None] - key_positions[None, :])[None, None, :, :]
    gather_index = distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    position_bias = rel_logits.gather(-1, gather_index)
    return position_bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)


def explicit_bias_path(
    unit_vectors: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    proj: torch.Tensor,
) -> torch.Tensor:
    profiles = unit_vectors @ proj
    distance = query_positions[:, None] - key_positions[None, :]
    flat_distance = distance.reshape(-1)
    valid = (flat_distance >= 0) & (flat_distance < proj.shape[1])
    flat_bias = torch.zeros(
        (*profiles.shape[:2], flat_distance.numel()), dtype=profiles.dtype
    )
    flat_bias[:, :, valid] = profiles[:, :, flat_distance[valid]]
    return flat_bias.reshape(
        profiles.shape[0], profiles.shape[1], query_positions.numel(), key_positions.numel()
    )


def log_scaling_tau(query_position: int) -> float:
    effective_n = float(query_position + 1)
    return float(1.0 + 0.1 * math.log(max(effective_n / 128_000.0, 1.0)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official_class, official_identity = load_official_relative_logits()
    print(
        "Official implementation: "
        f"{official_identity['class']} (transformers {official_identity['version']})"
    )
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((NUM_DRAWS, NUM_HEADS, D_REL)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    unit_vectors = torch.from_numpy(vectors)

    layer_results: dict[str, dict[str, object]] = {}
    for layer in LAYERS:
        proj_np = np.load(WEIGHTS_DIR / f"layer{layer:02d}_rel_logits_proj.npy")
        proj = torch.from_numpy(np.asarray(proj_np, dtype=np.float32))
        extent = int(proj.shape[1])
        query_positions = torch.arange(extent, extent + Q_LEN, dtype=torch.long)
        key_positions = torch.arange(0, extent + Q_LEN, dtype=torch.long)
        relative_states = unit_vectors[:, None, :, :].expand(-1, Q_LEN, -1, -1)
        official_module = official_class(D_REL, extent).eval()
        with torch.no_grad():
            official_module.proj.copy_(proj)
            official = official_module(
                relative_states, query_positions, key_positions
            )
        local_copy = relative_logits_forward(
            relative_states, query_positions, key_positions, proj
        )
        explicit = explicit_bias_path(unit_vectors, query_positions, key_positions, proj)
        official_vs_explicit = float(torch.max(torch.abs(official - explicit)).item())
        local_vs_official = float(torch.max(torch.abs(local_copy - official)).item())
        passed = bool(official_vs_explicit < TOL and local_vs_official < TOL)
        layer_results[str(layer)] = {
            "extent": extent,
            "distance_min": int(query_positions[0] - key_positions[-1]),
            "distance_max": int(query_positions[-1] - key_positions[0]),
            "official_vs_independent_explicit_max_abs_diff": official_vs_explicit,
            "local_copy_vs_official_max_abs_diff": local_vs_official,
            "passed": passed,
        }
        print(
            f"layer {layer:02d}: max |official-explicit|={official_vs_explicit:.3e}; "
            f"max |local-official|={local_vs_official:.3e} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            print(
                f"[CONTRADICTION] layer {layer}: official/explicit diff="
                f"{official_vs_explicit:.6g}, local/official diff={local_vs_official:.6g}"
            )

    q_pos = 1_000_000
    tau = log_scaling_tau(q_pos)
    expected_formula = float(
        1.0 + 0.1 * math.log(max((q_pos + 1) / 128_000.0, 1.0))
    )
    spec_approximation = 1.2058
    formula_passed = bool(abs(tau - expected_formula) < 1e-12)
    spec_approximation_passed = bool(abs(tau - spec_approximation) < 1e-4)
    contradictions = []
    if not spec_approximation_passed:
        message = (
            "Spec says tau(10^6) is approximately 1.2058, but the official "
            f"formula evaluates to {tau:.9f} (rounds to 1.2056)."
        )
        contradictions.append(message)
        print(f"[CONTRADICTION] {message}")
    print(
        f"tau({q_pos})={tau:.9f} official-formula "
        f"{'PASS' if formula_passed else 'FAIL'}"
    )

    passed = bool(all(v["passed"] for v in layer_results.values()) and formula_passed)
    output = {
        "official_source": SOURCE_URL,
        "official_implementation": official_identity,
        "comparison": (
            "installed official InklingRelativeLogits versus an independently "
            "indexed explicit construction; local forward copy checked separately"
        ),
        "seed": 0,
        "random_unit_vectors_per_head": NUM_DRAWS,
        "q_len": Q_LEN,
        "tolerance": float(TOL),
        "layers": layer_results,
        "tau": {
            "query_position": q_pos,
            "effective_n": q_pos + 1,
            "value": tau,
            "official_formula_expected": expected_formula,
            "formula_passed": formula_passed,
            "spec_approximation": spec_approximation,
            "spec_approximation_abs_diff": float(abs(tau - spec_approximation)),
            "spec_approximation_passed_at_1e-4": spec_approximation_passed,
            "applies_to": ["query", "position_bias"],
        },
        "contradictions": contradictions,
        "passed": passed,
    }
    out_path = OUT_DIR / "mechanism_validation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Written {out_path}")
    if not passed:
        raise AssertionError("mechanism validation failed")


if __name__ == "__main__":
    main()
