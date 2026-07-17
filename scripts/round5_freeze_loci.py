"""Freeze Round-5 LF5 query loci and LF4 token classes before outcomes.

This script reads only the committed corpus IDs, tokenizer, and needle sidecar.
It does not read activations, attention, weights, or Round-5 result files.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import os
import re
import tokenize
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
DEFAULT_OUT = ROOT / "analysis" / "round5" / "loci.json"
REGISTRATION_COMMIT = "34278b4"
PLAN_COMMIT = "d4e2579"
TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
NON_RANDOM_TEXTS = TEXTS[:-1]

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
DEMONSTRATIVES_EXCLUDED = {"this", "that", "these", "those"}


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
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


class DecodedText:
    def __init__(self, name: str, ids: np.ndarray, tokenizer: Tokenizer):
        self.name = name
        self.ids = np.asarray(ids, dtype=np.int64)
        id_lists = [[int(x)] for x in self.ids]
        self.fragments = tokenizer.decode_batch(id_lists, skip_special_tokens=False)
        full = tokenizer.decode(self.ids.tolist(), skip_special_tokens=False)
        self.decode_concatenates = "".join(self.fragments) == full
        self.text = full

        self.starts: list[int] = []
        self.ends: list[int] = []
        cursor = 0
        if self.decode_concatenates:
            for fragment in self.fragments:
                self.starts.append(cursor)
                cursor += len(fragment)
                self.ends.append(cursor)
            if cursor != len(self.text):
                raise AssertionError(f"offset construction failed for {name}")

    def token_at_char(self, char_pos: int) -> int:
        if not self.decode_concatenates:
            raise RuntimeError(f"{self.name} has no unique per-token Unicode character offsets")
        if not 0 <= char_pos < len(self.text):
            raise IndexError((self.name, char_pos, len(self.text)))
        token_pos = bisect.bisect_right(self.ends, char_pos)
        if token_pos >= len(self.ids):
            raise IndexError((self.name, char_pos, token_pos))
        if not self.starts[token_pos] <= char_pos < self.ends[token_pos]:
            raise AssertionError((self.name, char_pos, token_pos))
        return token_pos

    def record(self, token_pos: int, **extra: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "token_pos": int(token_pos),
            "token_id": int(self.ids[token_pos]),
            "decoded_fragment": self.fragments[token_pos],
            "char_start": int(self.starts[token_pos]) if self.decode_concatenates else None,
            "char_end": int(self.ends[token_pos]) if self.decode_concatenates else None,
        }
        record.update(extra)
        return record


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    cursor = 0
    for line in text.splitlines(keepends=True):
        cursor += len(line)
        offsets.append(cursor)
    return offsets


def python_bracket_pairs(decoded: DecodedText) -> list[dict[str, Any]]:
    offsets = line_offsets(decoded.text)

    def absolute(pos: tuple[int, int]) -> int:
        row, col = pos
        return offsets[row - 1] + col

    stack: list[tuple[str, int]] = []
    pairs: list[dict[str, Any]] = []
    expected = {")": "(", "]": "[", "}": "{"}
    try:
        lexical_tokens = list(tokenize.generate_tokens(io.StringIO(decoded.text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise RuntimeError(f"Python lexical failure in {decoded.name}: {exc}") from exc

    for tok in lexical_tokens:
        if tok.type != tokenize.OP or tok.string not in "()[]{}":
            continue
        char_pos = absolute(tok.start)
        if tok.string in "([{":
            stack.append((tok.string, char_pos))
            continue
        if not stack or stack[-1][0] != expected[tok.string]:
            raise RuntimeError(
                f"typed bracket mismatch at {tok.start}: {tok.string!r}, "
                f"stack top={stack[-1] if stack else None}"
            )
        opener, opener_char = stack.pop()
        q = decoded.token_at_char(char_pos)
        k = decoded.token_at_char(opener_char)
        pairs.append(
            decoded.record(
                q,
                locus="bracket_closer",
                bracket=tok.string,
                opener=opener,
                closer_char=int(char_pos),
                opener_char=int(opener_char),
                opener_token_pos=int(k),
                token_distance=int(q - k),
            )
        )
    if stack:
        raise RuntimeError(f"{len(stack)} unmatched opening brackets at EOF")
    if not pairs:
        raise RuntimeError("bracket selector produced no pairs")
    return pairs


def prose_sentence_starts(decoded: DecodedText) -> list[dict[str, Any]]:
    starts: dict[int, int] = {}
    first = re.search(r"[^\W_]", decoded.text, flags=re.UNICODE)
    if first:
        starts[decoded.token_at_char(first.start())] = first.start()

    boundary = re.compile(
        r"[.!?][\"'’”\)\]]*\s+[\"'“‘\(\[]*(?P<start>[^\W_])",
        flags=re.UNICODE,
    )
    for match in boundary.finditer(decoded.text):
        char_pos = match.start("start")
        starts.setdefault(decoded.token_at_char(char_pos), char_pos)

    records = [
        decoded.record(q, locus="sentence_start", lexical_char=int(char_pos))
        for q, char_pos in sorted(starts.items())
    ]
    if not records:
        raise RuntimeError("sentence-start selector produced no loci")
    return records


def heartbeat_line_ends(decoded: DecodedText) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cursor = 0
    for line_no, line in enumerate(decoded.text.splitlines(keepends=True), start=1):
        body = line.rstrip("\r\n").rstrip()
        if re.search(r"(?<!\S)HEARTBEAT(?!\S)", body):
            if not body:
                raise RuntimeError(f"empty heartbeat line {line_no}")
            char_pos = cursor + len(body) - 1
            q = decoded.token_at_char(char_pos)
            records.append(
                decoded.record(
                    q,
                    locus="heartbeat_line_end",
                    line_number=int(line_no),
                    final_char=int(char_pos),
                )
            )
        cursor += len(line)
    if not records:
        raise RuntimeError("heartbeat selector produced no loci")
    return records


def random_controls(decoded: DecodedText, excluded: set[int]) -> tuple[int, list[dict[str, Any]]]:
    digest = hashlib.sha256(f"{REGISTRATION_COMMIT}|LF5|{decoded.name}".encode()).hexdigest()
    seed = int(digest[:16], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    positions: list[int] = []
    for block in range(8):
        lo, hi = block * 1024, (block + 1) * 1024
        candidates = np.array([q for q in range(lo, hi) if q not in excluded], dtype=np.int64)
        if len(candidates) < 8:
            raise RuntimeError(f"too few controls in {decoded.name} block {block}")
        positions.extend(int(x) for x in rng.choice(candidates, size=8, replace=False))
    positions.sort()
    if len(positions) != 64 or len(set(positions)) != 64:
        raise AssertionError(f"random-control selection failed for {decoded.name}")
    return seed, [decoded.record(q, locus="random_control") for q in positions]


def normalized_word(fragment: str) -> str | None:
    word = fragment.strip().casefold()
    return word if re.fullmatch(r"[a-z]+", word) else None


def class_records(decoded: DecodedText, positions: list[int], class_name: str) -> list[dict[str, Any]]:
    records = []
    for q in positions:
        word = normalized_word(decoded.fragments[q])
        records.append(decoded.record(q, token_class=class_name, normalized_word=word))
    return records


def special_and_control_ids(tokenizer_path: Path) -> set[int]:
    raw = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    ids = {int(t["id"]) for t in raw.get("added_tokens", []) if t.get("special")}
    for token, token_id in raw.get("model", {}).get("vocab", {}).items():
        if token.startswith("<|") and token.endswith("|>"):
            ids.add(int(token_id))
    return ids


def planted_codeword_positions(tokenizer: Tokenizer, sidecar: dict[str, Any], seq: int) -> set[int]:
    excluded: set[int] = set()
    for entity in sidecar["entities"]:
        width = len(tokenizer.encode(" " + entity["codeword"], add_special_tokens=False).ids)
        if width <= 0:
            raise RuntimeError(f"zero token width for {entity['codeword']}")
        for start in entity.get("token_positions", []):
            excluded.update(range(int(start), min(int(start) + width, seq)))
    return excluded


def build_manifest() -> dict[str, Any]:
    tokenizer_path = CORPUS / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    corpus_manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    sidecar_path = CORPUS / "05_needles.sidecar.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    decoded: dict[str, DecodedText] = {}
    ids_by_text: dict[str, np.ndarray] = {}
    for name in TEXTS:
        path = CORPUS / f"{name}.ids.npy"
        actual_sha = sha256_file(path)
        expected_sha = corpus_manifest["texts"][name]["ids_sha256"]
        if actual_sha != expected_sha:
            raise RuntimeError(f"ID hash mismatch for {name}: {actual_sha} != {expected_sha}")
        ids = np.load(path)
        if ids.shape != (8192,):
            raise RuntimeError(f"unexpected ID shape for {name}: {ids.shape}")
        ids_by_text[name] = ids
        decoded[name] = DecodedText(name, ids, tokenizer)

    brackets = python_bracket_pairs(decoded["02_code"])
    sentence_starts = prose_sentence_starts(decoded["01_prose_en"])
    heartbeats = heartbeat_line_ends(decoded["03_templated"])

    named_by_text: dict[str, set[int]] = {name: set() for name in TEXTS}
    named_by_text["02_code"].update(x["token_pos"] for x in brackets)
    named_by_text["01_prose_en"].update(x["token_pos"] for x in sentence_starts)
    named_by_text["03_templated"].update(x["token_pos"] for x in heartbeats)

    random_by_text: dict[str, list[dict[str, Any]]] = {}
    random_seeds: dict[str, int] = {}
    for name in TEXTS:
        seed, records = random_controls(decoded[name], named_by_text[name])
        random_seeds[name] = seed
        random_by_text[name] = records

    counts = Counter(int(x) for name in NON_RANDOM_TEXTS for x in ids_by_text[name])
    rare_ids = {token_id for token_id, count in counts.items() if count <= 2}
    excluded_ids = special_and_control_ids(tokenizer_path)
    needle_excluded = planted_codeword_positions(tokenizer, sidecar, seq=8192)

    texts: dict[str, Any] = {}
    for name in TEXTS:
        item = decoded[name]
        pronoun_pos: list[int] = []
        function_pos: list[int] = []
        demonstrative_pos: list[int] = []
        rare_with_needles: list[int] = []
        for q, token_id in enumerate(item.ids):
            word = normalized_word(item.fragments[q])
            if word in PRONOUNS:
                pronoun_pos.append(q)
            if word in FUNCTION_WORDS:
                function_pos.append(q)
            if word in DEMONSTRATIVES_EXCLUDED:
                demonstrative_pos.append(q)
            if (
                int(token_id) in rare_ids
                and int(token_id) not in excluded_ids
                and bool(item.fragments[q].strip())
            ):
                rare_with_needles.append(q)
        rare_primary = [
            q for q in rare_with_needles
            if not (name == "05_needles" and q in needle_excluded)
        ]

        loci: dict[str, Any] = {"random_controls": random_by_text[name]}
        if name == "02_code":
            loci["bracket_pairs"] = brackets
            loci["bracket_query_positions"] = sorted(named_by_text[name])
        if name == "01_prose_en":
            loci["sentence_starts"] = sentence_starts
        if name == "03_templated":
            loci["heartbeat_line_ends"] = heartbeats

        classes: dict[str, Any] = {
            "pronouns": class_records(item, pronoun_pos, "pronoun"),
            "function_words": class_records(item, function_pos, "function_word"),
            "demonstratives_excluded": class_records(
                item, demonstrative_pos, "demonstrative_excluded"
            ),
            "rare_bpe_primary": class_records(item, rare_primary, "rare_bpe_primary"),
            "rare_bpe_with_needles_secondary": class_records(
                item, rare_with_needles, "rare_bpe_with_needles_secondary"
            ),
        }
        texts[name] = {
            "ids_sha256": sha256_file(CORPUS / f"{name}.ids.npy"),
            "decoded_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
            "n_tokens": int(len(item.ids)),
            "n_chars": int(len(item.text)),
            "per_token_decode_concatenates": item.decode_concatenates,
            "loci": loci,
            "classes": classes,
            "counts": {
                "random_controls": 64,
                "bracket_pairs": len(brackets) if name == "02_code" else 0,
                "bracket_query_positions": len(named_by_text[name]) if name == "02_code" else 0,
                "sentence_starts": len(sentence_starts) if name == "01_prose_en" else 0,
                "heartbeat_line_ends": len(heartbeats) if name == "03_templated" else 0,
                "pronouns": len(pronoun_pos),
                "function_words": len(function_pos),
                "demonstratives_excluded": len(demonstrative_pos),
                "rare_bpe_primary": len(rare_primary),
                "rare_bpe_with_needles_secondary": len(rare_with_needles),
            },
        }

    return {
        "schema_version": 1,
        "kind": "round5_lf5_lf4_preoutcome_freeze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "selector_source_sha256": sha256_file(Path(__file__)),
        "inputs": {
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "corpus_manifest_sha256": sha256_file(CORPUS / "manifest.json"),
            "needle_sidecar_sha256": sha256_file(sidecar_path),
        },
        "definitions": {
            "decode_contract": "decode committed IDs verbatim; never re-tokenize text files",
            "random_char_offsets": "null when arbitrary IDs split a Unicode code point; token ID remains authoritative",
            "pronouns": sorted(PRONOUNS),
            "demonstratives_excluded": sorted(DEMONSTRATIVES_EXCLUDED),
            "function_words": sorted(FUNCTION_WORDS),
            "rare_bpe": "ID frequency <=2 across texts 01-05; non-whitespace; special/control IDs excluded",
            "rare_bpe_needles": "primary excludes both planted codeword mention spans in text 05",
            "position_bins": 256,
            "position_bin_sensitivity": [128, 512],
            "random_control_seed": 'int(SHA256("34278b4|LF5|<text>")[:16], 16)',
            "random_control_sampling": "8 without replacement per fixed 1024-token block",
            "brackets": "Python tokenize.OP outside strings/comments plus strict typed stack",
            "sentence_starts": "stream start or first lexical character after .?!, closers, whitespace, and optional opener",
            "heartbeat": "final non-newline token on lines containing exact whitespace-delimited HEARTBEAT",
        },
        "random_seeds": random_seeds,
        "rare_token_id_count": len(rare_ids),
        "excluded_special_control_ids": sorted(excluded_ids),
        "needle_codeword_excluded_positions": sorted(needle_excluded),
        "texts": texts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="build and validate without writing")
    args = parser.parse_args()
    manifest = build_manifest()
    summary = {name: item["counts"] for name, item in manifest["texts"].items()}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.check:
        atomic_json(args.out, manifest)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
