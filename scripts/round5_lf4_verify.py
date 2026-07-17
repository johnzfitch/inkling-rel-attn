"""Independent scalar verification and confirmation for LF4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "dumps" / "tier2" / "capture"
WEIGHTS = ROOT / "weights"
DUMP = ROOT / "dumps" / "round5" / "lf4"
CONTROLS = ROOT / "analysis" / "round5" / "lf4_verifier_controls.json"
MAIN_REPORT = ROOT / "analysis" / "round5" / "lf4" / "zoom_lens.json"
INPUT_VALIDATION = ROOT / "analysis" / "round5" / "lf4" / "input_validation.json"
LF5_CONFIRMATION = ROOT / "analysis" / "round5" / "lf5" / "confirmation.json"
VERIFY_REPORT = ROOT / "analysis" / "round5" / "lf4" / "verification.json"
CONFIRMATION = ROOT / "analysis" / "round5" / "lf4" / "confirmation.json"
REGISTRATION_COMMIT = "34278b4"
PLAN_COMMIT = "d4e2579"
MID_GLOBALS = [23, 29, 35, 41, 47]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def scalar_apertures(layer: int, text: str, positions: list[int]) -> dict[int, float]:
    projection = np.load(WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy").astype(np.float64)
    rvec = np.load(CAPTURE / f"rvec_L{layer:02d}_{text}.npy", mmap_mode="r")
    result = {}
    for q in positions:
        curves = np.asarray(rvec[q], dtype=np.float64) @ projection
        absolute = np.abs(curves)
        result[q] = float(absolute[:, 129:].sum(dtype=np.float64) / absolute.sum(dtype=np.float64))
    return result


def sample_midranks(values: dict[int, float], positions: list[int]) -> dict[int, float]:
    result = {}
    position_set = set(positions)
    for start in range(0, 8192, 256):
        block = sorted(q for q in position_set if start <= q < start + 256)
        if not block:
            continue
        data = np.array([values[q] for q in block], dtype=np.float64)
        order = np.argsort(data, kind="mergesort")
        ranks = np.empty(len(data), dtype=np.float64)
        cursor = 0
        while cursor < len(data):
            end = cursor + 1
            while end < len(data) and data[order[end]] == data[order[cursor]]:
                end += 1
            ranks[order[cursor:end]] = ((cursor + 1) + end) / 2.0
            cursor = end
        percentiles = (ranks - 0.5) / len(data)
        result.update({q: float(value) for q, value in zip(block, percentiles)})
    return result


def independent_contrast(
    name: str,
    spec: dict[str, Any],
    main: dict[str, Any],
    permutations: int,
) -> dict[str, Any]:
    text = spec["text"]
    class_positions = [int(x) for x in spec["class_positions"]]
    control_positions = [int(x) for x in spec["control_positions"]]
    all_positions = sorted(class_positions + control_positions)
    averaged = {q: 0.0 for q in all_positions}
    max_dump_delta = 0.0
    per_layer_effects = {}

    for layer in MID_GLOBALS:
        scalar = scalar_apertures(layer, text, all_positions)
        with np.load(DUMP / f"aperture_L{layer:02d}_{text}.npz") as dumped:
            aperture = dumped["aperture_full"]
            delta = max(abs(scalar[q] - float(aperture[q])) for q in all_positions)
            max_dump_delta = max(max_dump_delta, delta)
        ranks = sample_midranks(scalar, all_positions)
        for q in all_positions:
            averaged[q] += ranks[q] / len(MID_GLOBALS)
        per_layer_effects[str(layer)] = float(
            np.median([ranks[q] for q in class_positions])
            - np.median([ranks[q] for q in control_positions])
        )

    observed = float(
        np.median([averaged[q] for q in class_positions])
        - np.median([averaged[q] for q in control_positions])
    )
    seed = int(
        hashlib.sha256(f"{REGISTRATION_COMMIT}|LF4|scalar-verify|{name}".encode()).hexdigest()[:16],
        16,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    class_set = set(class_positions)
    blocks = []
    counts = []
    for start in range(0, 8192, 256):
        block = np.array([q for q in all_positions if start <= q < start + 256], dtype=np.int64)
        blocks.append(block)
        counts.append(sum(q in class_set for q in block))
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        pseudo_class = []
        pseudo_control = []
        for block, count in zip(blocks, counts):
            if not len(block):
                continue
            chosen = set(int(x) for x in rng.choice(block, size=count, replace=False))
            pseudo_class.extend(q for q in block if int(q) in chosen)
            pseudo_control.extend(q for q in block if int(q) not in chosen)
        null[iteration] = np.median([averaged[int(q)] for q in pseudo_class]) - np.median(
            [averaged[int(q)] for q in pseudo_control]
        )
    direction = "negative" if name == "function_words" else "positive"
    exceed = np.count_nonzero(null <= observed) if direction == "negative" else np.count_nonzero(null >= observed)
    p_value = float((1 + exceed) / (permutations + 1))
    main_effect = float(main["effect"])
    sign_agreement = (observed < 0 and main_effect < 0) if direction == "negative" else (
        observed > 0 and main_effect > 0
    )
    checks = {
        "raw_dump_max_abs_delta_le_1e-12": max_dump_delta <= 1e-12,
        "effect_sign_agreement": sign_agreement,
    }
    return {
        "name": name,
        "text": text,
        "class_count": len(class_positions),
        "control_count": len(control_positions),
        "max_abs_delta_vs_blocked_dump": max_dump_delta,
        "matched_sample_effect": observed,
        "main_rank_effect": main_effect,
        "direction": direction,
        "permutations": permutations,
        "p_one_sided": p_value,
        "seed": seed,
        "per_layer_effects": per_layer_effects,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--verify-report", type=Path, default=VERIFY_REPORT)
    parser.add_argument("--confirmation", type=Path, default=CONFIRMATION)
    args = parser.parse_args()

    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    main_report = json.loads(MAIN_REPORT.read_text(encoding="utf-8"))
    main_by_name = {item["name"]: item for item in main_report["primary"]}
    results = [
        independent_contrast(name, spec, main_by_name[name], args.permutations)
        for name, spec in controls["contrasts"].items()
    ]
    verification = {
        "schema_version": 1,
        "kind": "round5_lf4_independent_verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "source_sha256": sha256_file(Path(__file__)),
        "controls_sha256": sha256_file(CONTROLS),
        "main_report_sha256": sha256_file(MAIN_REPORT),
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    atomic_json(args.verify_report, verification)

    lf5 = json.loads(LF5_CONFIRMATION.read_text(encoding="utf-8"))
    input_validation = json.loads(INPUT_VALIDATION.read_text(encoding="utf-8"))
    gates = {
        "lf5_methodology_gate": bool(
            lf5.get("methodology_passed")
            and lf5.get("production_backend") == "replay"
            and lf5.get("registered_cpu_gate_passed") is False
            and lf5.get("registered_cpu_failure_preserved_gate")
        ),
        "input_validation_gate": bool(
            input_validation.get("passed")
            and not input_validation.get("errors")
            and len(input_validation.get("records", {})) == 462
            and input_validation.get("lf5_confirmation_sha256")
            == sha256_file(LF5_CONFIRMATION)
        ),
        "position_matched_control_gate": all(
            "p_holm" in item and "prediction_passed" in item for item in main_report["primary"]
        ),
        "negative_control_gate": bool(main_report.get("negative_controls_passed")),
        "independent_rederivation_gate": bool(verification["passed"]),
    }
    confirmation = {
        "schema_version": 1,
        "kind": "round5_lf4_confirmation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "lf5_production_backend": "replay",
        "registered_cpu_gate_passed": False,
        **gates,
        "methodology_passed": all(gates.values()),
        "all_three_primary_predictions_passed": bool(
            main_report.get("all_three_primary_predictions_passed")
        ),
        "source_hashes": {
            "lf5_confirmation": sha256_file(LF5_CONFIRMATION),
            "input_validation": sha256_file(INPUT_VALIDATION),
            "controls": sha256_file(CONTROLS),
            "main_report": sha256_file(MAIN_REPORT),
            "verification": sha256_file(args.verify_report),
        },
    }
    atomic_json(args.confirmation, confirmation)
    print(json.dumps(confirmation, indent=2, sort_keys=True))
    if not confirmation["methodology_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
