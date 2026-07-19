"""Freeze A6 depth-resolved token classes before corrected model measurement.

The output stays in gitignored ``corpus_v2/``. This selector reads only token
IDs, sidecars, the already-frozen Round-5 prose loci, and the tokenizer. It does
not read activations, NLLs, apertures, or any prior model outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from corpus_v2_freeze_classes import PRONOUNS, normalized_word


ROOT = Path(__file__).resolve().parents[1]
CORPUS_V2 = ROOT / "corpus_v2"
CORPUS_V1 = ROOT / "corpus"
TOKENIZER = CORPUS_V1 / "tokenizer.json"
LOCI = ROOT / "analysis" / "round5" / "loci.json"
DEFAULT_OUT = CORPUS_V2 / "depth_classes.json"
SLACK = "07b_slack_multi"
MATH = "08_math_llm"
PROSE = "01_prose_en"
SEQ = 8192
BIN_SIZE = 256
A6_COMMIT = "061bb04dffef05eae33f9fffc430394faa4052c5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def verified_ids(root: Path, name: str, manifest: dict[str, Any]) -> np.ndarray:
    path = root / f"{name}.ids.npy"
    if sha256_file(path) != manifest["texts"][name]["ids_sha256"]:
        raise RuntimeError(f"{name} ID hash mismatch")
    values = np.load(path, allow_pickle=False)
    if values.shape != (SEQ,) or values.dtype != np.int32:
        raise RuntimeError(f"invalid IDs for {name}: {values.shape}, {values.dtype}")
    return values


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"{A6_COMMIT}|depth-classes|{label}".encode()).hexdigest()[:16], 16)


def stratified_random_mask(reference: list[int], *, seed: int) -> list[int]:
    reference_array = np.asarray(sorted(set(reference)), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    selected: list[int] = []
    for start in range(0, SEQ, BIN_SIZE):
        stop = min(start + BIN_SIZE, SEQ)
        count = int(np.count_nonzero((reference_array >= start) & (reference_array < stop)))
        if count:
            selected.extend(
                int(value)
                for value in rng.choice(np.arange(start, stop), size=count, replace=False)
            )
    selected.sort()
    if len(selected) != len(reference_array) or len(selected) != len(set(selected)):
        raise RuntimeError("invalid stratified random mask")
    return selected


def build_manifest() -> dict[str, Any]:
    v2_manifest_path = CORPUS_V2 / "manifest.json"
    v1_manifest_path = CORPUS_V1 / "manifest.json"
    v2_manifest = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    if v2_manifest.get("private") is not True or v2_manifest.get("seq") != SEQ:
        raise RuntimeError("invalid corpus-v2 manifest")
    if v1_manifest.get("seq") != SEQ:
        raise RuntimeError("invalid corpus manifest")

    slack_ids = verified_ids(CORPUS_V2, SLACK, v2_manifest)
    verified_ids(CORPUS_V2, MATH, v2_manifest)
    verified_ids(CORPUS_V1, PROSE, v1_manifest)

    sidecar_path = CORPUS_V2 / f"{SLACK}.sidecar.json"
    if sha256_file(sidecar_path) != v2_manifest["texts"][SLACK]["sidecar_sha256"]:
        raise RuntimeError("v2.1 Slack sidecar hash mismatch")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("seq") != SEQ
        or sidecar.get("unit") != "message"
        or len(sidecar.get("token_unit_index", [])) != SEQ
    ):
        raise RuntimeError("invalid v2.1 Slack sidecar")
    labels = [int(value) for value in sidecar["token_unit_index"]]
    starts = [0] + [
        position for position in range(1, SEQ) if labels[position] != labels[position - 1]
    ]
    stored_starts = sorted(int(value) for value in sidecar["unit_start_tokens"])
    missing_stored_starts = sorted(set(starts) - set(stored_starts))
    extra_stored_starts = sorted(set(stored_starts) - set(starts))
    if (
        len(starts) != int(sidecar["n_units_used"])
        or len(starts) != len(set(starts))
        or any(value < 0 or value >= SEQ for value in starts)
        or labels[0] != 0
        or labels[-1] != int(sidecar["n_units_used"]) - 1
        or any(right < left or right - left > 1 for left, right in zip(labels, labels[1:]))
        or extra_stored_starts
        or missing_stored_starts not in ([], [1024])
    ):
        raise RuntimeError("invalid v2.1 message starts")

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    fragments = tokenizer.decode_batch(
        [[int(token_id)] for token_id in slack_ids], skip_special_tokens=False
    )
    pronouns = [
        position
        for position, fragment in enumerate(fragments)
        if normalized_word(fragment) in PRONOUNS
    ]
    first_content = [position + 2 for position in starts if position + 2 < SEQ]
    excluded_truncated = [position for position in starts if position + 2 >= SEQ]
    if not pronouns or not first_content:
        raise RuntimeError("empty v2.1 depth class")

    loci = json.loads(LOCI.read_text(encoding="utf-8"))
    prose_entry = loci["texts"][PROSE]
    if prose_entry["ids_sha256"] != v1_manifest["texts"][PROSE]["ids_sha256"]:
        raise RuntimeError("stale prose sentence-start loci")
    sentence_starts = sorted(
        int(record["token_pos"]) for record in prose_entry["loci"]["sentence_starts"]
    )
    classes = {
        "speaker_labels": starts,
        "pronouns": pronouns,
        "first_content": first_content,
        "sentence_starts": sentence_starts,
    }
    for name, positions in classes.items():
        if (
            not positions
            or len(positions) != len(set(positions))
            or any(value < 0 or value >= SEQ for value in positions)
        ):
            raise RuntimeError(f"invalid depth class {name}")

    prediction_classes = {
        "P-d1": {"source_text": SLACK, "class": "speaker_labels"},
        "P-d2": {"source_text": SLACK, "class": "pronouns"},
        "P-d3": {"source_text": SLACK, "class": "first_content"},
        "P-d4": {"source_text": PROSE, "class": "sentence_starts"},
    }
    random_masks: dict[str, dict[str, list[int]]] = {}
    random_seeds: dict[str, dict[str, int]] = {}
    for prediction, specification in prediction_classes.items():
        positions = classes[specification["class"]]
        random_masks[prediction] = {}
        random_seeds[prediction] = {}
        for text in (specification["source_text"], MATH):
            seed = seed_for(f"{prediction}|{text}")
            random_seeds[prediction][text] = seed
            random_masks[prediction][text] = stratified_random_mask(positions, seed=seed)

    dependency = ROOT / "scripts" / "corpus_v2_freeze_classes.py"
    return {
        "schema_version": 1,
        "kind": "round5_a6_depth_private_class_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_a6_commit": A6_COMMIT,
        "selector_source_sha256": sha256_file(Path(__file__)),
        "selector_dependency_sha256": sha256_file(dependency),
        "inputs": {
            "corpus_v2_manifest_sha256": sha256_file(v2_manifest_path),
            "corpus_manifest_sha256": sha256_file(v1_manifest_path),
            "slack_ids_sha256": sha256_file(CORPUS_V2 / f"{SLACK}.ids.npy"),
            "slack_sidecar_sha256": sha256_file(sidecar_path),
            "math_ids_sha256": sha256_file(CORPUS_V2 / f"{MATH}.ids.npy"),
            "prose_ids_sha256": sha256_file(CORPUS_V1 / f"{PROSE}.ids.npy"),
            "prose_loci_sha256": sha256_file(LOCI),
            "tokenizer_sha256": sha256_file(TOKENIZER),
        },
        "definitions": {
            "speaker_labels": "all transitions in 07b_slack_multi sidecar token_unit_index, including token 0",
            "pronouns": "exact corpus-v2 single-token selector and pronoun lexicon",
            "first_content": "speaker-label position + 2; out-of-range truncated tails excluded",
            "sentence_starts": "frozen Round-5 01_prose_en sentence-start loci",
            "position_bins": BIN_SIZE,
            "random_masks": "deterministic within-bin samples matching each class count per bin",
        },
        "excluded_truncated_message_starts": excluded_truncated,
        "sidecar_start_disposition": {
            "stored_count": len(stored_starts),
            "derived_transition_count": len(starts),
            "missing_stored_starts": missing_stored_starts,
            "extra_stored_starts": extra_stored_starts,
            "ids_or_text_changed": False,
        },
        "counts": {name: len(positions) for name, positions in classes.items()},
        "classes": classes,
        "prediction_classes": prediction_classes,
        "random_seeds": random_seeds,
        "random_masks": random_masks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    if not args.check:
        if args.out.exists():
            raise SystemExit(f"REFUSING to overwrite immutable class freeze {args.out}")
        atomic_json(args.out, manifest)
        print(f"wrote private depth class freeze: {args.out}")


if __name__ == "__main__":
    main()
