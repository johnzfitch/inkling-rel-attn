"""Freeze private corpus-v2 class positions before model measurement.

The output stays under gitignored corpus_v2/. Only selector code is public.
No activations, weights, NLL values, or Round-5 results are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus_v2"
TOKENIZER_PATH = ROOT / "corpus" / "tokenizer.json"
DEFAULT_OUT = CORPUS / "classes.json"
REGISTRATION_COMMIT = "7fb84ab4e5d164bc759c32e38b0e58803abd5299"
PUBLIC_BOUNDARY_COMMIT = "65b220c2d185829dfc4c8e617a67e673d2fa9cd2"

PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
}
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "as",
    "at", "by", "for", "from", "in", "into", "of", "on", "onto", "to",
    "with", "without", "is", "am", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "have", "has", "had", "can", "could",
    "may", "might", "must", "shall", "should", "will", "would", "not",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def normalized_word(fragment: str) -> str | None:
    word = fragment.strip().casefold()
    return word if re.fullmatch(r"[a-z]+", word) else None


def load_verified_ids(name: str, private_manifest: dict[str, Any]) -> np.ndarray:
    path = CORPUS / f"{name}.ids.npy"
    expected = private_manifest["texts"][name]["ids_sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} ID hash mismatch")
    ids = np.load(path, allow_pickle=False)
    if ids.shape != (8192,) or ids.dtype != np.int32:
        raise RuntimeError(f"{name} IDs have unexpected shape/dtype: {ids.shape}, {ids.dtype}")
    return ids


def build_manifest() -> dict[str, Any]:
    private_manifest_path = CORPUS / "manifest.json"
    private_manifest = json.loads(private_manifest_path.read_text(encoding="utf-8"))
    if private_manifest.get("private") is not True or private_manifest.get("seq") != 8192:
        raise RuntimeError("invalid private corpus-v2 manifest")

    slack_ids = load_verified_ids("07_slack_human", private_manifest)
    load_verified_ids("08_math_llm", private_manifest)
    sidecar_path = CORPUS / "07_slack_human.sidecar.json"
    sidecar_expected = private_manifest["texts"]["07_slack_human"]["sidecar_sha256"]
    if sha256_file(sidecar_path) != sidecar_expected:
        raise RuntimeError("Slack sidecar hash mismatch")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("seq") != 8192
        or sidecar.get("unit") != "message"
        or len(sidecar.get("token_unit_index", [])) != 8192
    ):
        raise RuntimeError("invalid Slack sidecar schema")

    starts = [int(value) for value in sidecar["unit_start_tokens"]]
    if (
        len(starts) != int(sidecar["n_units_used"])
        or len(starts) != len(set(starts))
        or any(value < 0 or value >= 8192 for value in starts)
    ):
        raise RuntimeError("invalid message-start positions")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    fragments = tokenizer.decode_batch(
        [[int(token_id)] for token_id in slack_ids], skip_special_tokens=False
    )
    pronouns: list[int] = []
    function_words: list[int] = []
    for position, fragment in enumerate(fragments):
        word = normalized_word(fragment)
        if word in PRONOUNS:
            pronouns.append(position)
        if word in FUNCTION_WORDS:
            function_words.append(position)
    if not pronouns or not function_words:
        raise RuntimeError("a registered token class is empty")

    classes = {
        "message_starts": sorted(starts),
        "pronouns": pronouns,
        "function_words": function_words,
    }
    for name, positions in classes.items():
        if len(positions) != len(set(positions)) or any(
            value < 0 or value >= 8192 for value in positions
        ):
            raise RuntimeError(f"invalid {name} positions")

    return {
        "schema_version": 1,
        "kind": "corpus_v2_private_class_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        "selector_source_sha256": sha256_file(Path(__file__)),
        "inputs": {
            "private_manifest_sha256": sha256_file(private_manifest_path),
            "slack_ids_sha256": sha256_file(CORPUS / "07_slack_human.ids.npy"),
            "slack_sidecar_sha256": sha256_file(sidecar_path),
            "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        },
        "definitions": {
            "normalization": "decode one token ID; strip outer whitespace; casefold; require [a-z]+",
            "message_starts": "all Slack sidecar unit_start_tokens",
            "pronouns": sorted(PRONOUNS),
            "function_words": sorted(FUNCTION_WORDS),
            "demonstratives_in_pronouns": False,
        },
        "counts": {name: len(positions) for name, positions in classes.items()},
        "classes": classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    if not args.check:
        atomic_json(args.out, manifest)
        print(f"wrote private class freeze: {args.out}")


if __name__ == "__main__":
    main()
