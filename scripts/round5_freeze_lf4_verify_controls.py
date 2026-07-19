"""Freeze position-matched controls for LF4's independent scalar verifier."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCI = ROOT / "analysis" / "round5" / "loci.json"
OUT = ROOT / "analysis" / "round5" / "lf4_verifier_controls.json"
REGISTRATION_COMMIT = "34278b4"
PLAN_COMMIT = "d4e2579"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def positions(loci: dict[str, Any], text: str, token_class: str) -> list[int]:
    if token_class == "closers":
        return [int(x) for x in loci["texts"][text]["loci"]["bracket_query_positions"]]
    return [
        int(x["token_pos"])
        for x in loci["texts"][text]["classes"][token_class]
    ]


def matched_controls(class_positions: list[int], seed: int) -> list[int]:
    class_set = set(class_positions)
    rng = np.random.Generator(np.random.PCG64(seed))
    controls: list[int] = []
    for start in range(0, 8192, 256):
        stop = start + 256
        count = sum(start <= q < stop for q in class_positions)
        candidates = np.array(
            [q for q in range(start, stop) if q not in class_set], dtype=np.int64
        )
        if count > len(candidates):
            raise RuntimeError((start, count, len(candidates)))
        if count:
            controls.extend(int(x) for x in rng.choice(candidates, size=count, replace=False))
    controls.sort()
    if len(controls) != len(class_positions) or class_set.intersection(controls):
        raise AssertionError("matched-control construction failed")
    return controls


def main() -> None:
    loci = json.loads(LOCI.read_text(encoding="utf-8"))
    specs = [
        ("closers", "02_code", "closers"),
        ("pronouns", "01_prose_en", "pronouns"),
        ("function_words", "01_prose_en", "function_words"),
    ]
    contrasts = {}
    for name, text, token_class in specs:
        class_pos = sorted(set(positions(loci, text, token_class)))
        seed = int(
            hashlib.sha256(
                f"{REGISTRATION_COMMIT}|LF4|independent-controls|{name}".encode()
            ).hexdigest()[:16],
            16,
        )
        controls = matched_controls(class_pos, seed)
        contrasts[name] = {
            "text": text,
            "class_name": token_class,
            "class_positions": class_pos,
            "control_positions": controls,
            "seed": seed,
            "bin_size": 256,
            "sampling": "equal class count per bin, without replacement from non-class positions",
        }
    payload = {
        "schema_version": 1,
        "kind": "round5_lf4_independent_control_freeze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "loci_sha256": sha256_file(LOCI),
        "source_sha256": sha256_file(Path(__file__)),
        "contrasts": contrasts,
    }
    atomic_json(OUT, payload)
    print(json.dumps({name: len(item["control_positions"]) for name, item in contrasts.items()}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
