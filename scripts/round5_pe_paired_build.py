"""Build and freeze the fresh paired P-e/P-f corpus before GPU capture.

The private text, token IDs, sidecars, and class positions stay below the
gitignored ``corpus_v2/`` tree.  The only public output is a compact,
content-free freeze containing hashes, counts, definitions, and gates.

The builder reads only the registered private archive and tokenizer.  It does
not read model weights, activations, attention, losses, apertures, or any prior
outcome artifact.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import hashlib
import html
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path(r"D:\windows\slack-archive\slack-archive\data")
DEFAULT_OUT = ROOT / "corpus_v2"
TOKENIZER_PATH = ROOT / "corpus" / "tokenizer.json"
PUBLIC_FREEZE = ROOT / "analysis" / "round5" / "pe" / "corpus_freeze.json"

SINGLE = "09_pe_single_thread"
MULTI = "10_pe_multi_conversation"
ARMS = [SINGLE, MULTI]
SEQ = 8192
BIN_SIZE = 256
PARTITION_TARGETS = list(range(1024, SEQ, 1024))
PSEUDONYMS = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz"

PE_COMMIT = "71f0ad3efff199a83c333340ddd8c8f9a8d7f228"
PF_COMMIT = PE_COMMIT
D1_COMMIT = "51b8c00fe9b632086d0745221578be452f76f60c"
EXPECTED_NONEMPTY_CHANNELS = 19
EXPECTED_HUMAN_CHARS = 1_008_273
EXPECTED_LARGEST = {"human_chars": 807_211, "human_messages": 11_599, "users": 15}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_output(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, path)


def clean_slack(text: str, pseudonyms: dict[str, str]) -> str:
    """Remove Slack transport markup without retaining raw user/channel IDs."""
    value = html.unescape(str(text)).replace("�", "'")
    replacements = {
        "â€™": "’",
        "â€˜": "‘",
        "â€œ": "“",
        "â€\u009d": "”",
        "â€“": "–",
        "â€”": "—",
        "Â ": " ",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(
        r"<@([A-Za-z0-9]+)>",
        lambda match: "@" + pseudonyms.get(match.group(1), "user"),
        value,
    )
    value = re.sub(r"<#(?:[A-Za-z0-9]+)\|([^>]+)>", r"#\1", value)
    value = re.sub(r"<#[A-Za-z0-9]+>", "#channel", value)
    value = re.sub(
        r"<!subteam\^[^|>]+(?:\|([^>]+))?>",
        lambda match: match.group(1) or "@group",
        value,
    )
    value = re.sub(
        r"<!([^>|]+)(?:\|([^>]+))?>",
        lambda match: match.group(2) or "@" + match.group(1),
        value,
    )
    value = re.sub(r"<(https?://[^|>]+)\|([^>]*)>", r"\2", value)
    value = re.sub(r"<(mailto:[^|>]+)\|([^>]*)>", r"\2", value)
    value = re.sub(r"<(https?://[^>]+)>", r"\1", value)
    value = re.sub(r"<mailto:([^>]+)>", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    return value.strip()


def human_messages(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [
        message
        for message in payload
        if isinstance(message, dict)
        and isinstance(message.get("user"), str)
        and "bot_id" not in message
        and not message.get("subtype")
        and bool(message.get("text"))
    ]


def select_registered_source(archive: Path) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[int, str, Path, list[dict[str, Any]], int]] = []
    total_chars = 0
    total_messages = 0
    for filename in sorted(glob.glob(str(archive / "C*.json"))):
        path = Path(filename)
        messages = human_messages(path)
        chars = sum(len(str(message["text"])) for message in messages)
        if chars:
            source_hash = sha256_file(path)
            candidates.append((chars, source_hash, path, messages, len({m["user"] for m in messages})))
            total_chars += chars
            total_messages += len(messages)
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) != EXPECTED_NONEMPTY_CHANNELS or total_chars != EXPECTED_HUMAN_CHARS:
        raise RuntimeError(
            "private C-channel inventory differs from the preregistered feasibility snapshot: "
            f"channels={len(candidates)}, chars={total_chars}"
        )
    chars, source_hash, path, messages, users = candidates[0]
    largest = {"human_chars": chars, "human_messages": len(messages), "users": users}
    if largest != EXPECTED_LARGEST:
        raise RuntimeError(f"largest C-channel differs from preregistration: {largest}")
    inventory = {
        "glob": "C*.json",
        "nonempty_channels": len(candidates),
        "human_messages": total_messages,
        "human_chars": total_chars,
        "selection": "maximum human-message character count; SHA-256 tie break",
        "selected_rank": 1,
        "selected_file_sha256": source_hash,
        "selected_human_chars": chars,
        "selected_human_messages": len(messages),
        "selected_distinct_users": users,
    }
    return path, messages, inventory


def ordered_records(messages: list[dict[str, Any]], source_hash: str) -> list[dict[str, Any]]:
    indexed = list(enumerate(messages))
    indexed.sort(key=lambda item: (float(item[1].get("ts", 0.0)), item[0]))
    records: list[dict[str, Any]] = []
    for source_ordinal, message in indexed:
        if not clean_slack(str(message["text"]), {}):
            continue
        pair_id = hashlib.sha256(
            f"{source_hash}|{source_ordinal}|{message.get('ts', '')}".encode("utf-8")
        ).hexdigest()[:24]
        records.append(
            {
                "source_ordinal": int(source_ordinal),
                "pair_id": pair_id,
                "user": str(message["user"]),
                "raw_text": str(message["text"]),
            }
        )
    if not records:
        raise RuntimeError("registered source produced no clean human messages")
    return records


def pseudonym_map(records: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        user = record["user"]
        if user not in result:
            if len(result) >= len(PSEUDONYMS):
                raise RuntimeError("pseudonym alphabet exhausted")
            result[user] = PSEUDONYMS[len(result)]
    return result


def render_records(
    records: list[dict[str, Any]], conversation_starts: list[int]
) -> tuple[str, list[tuple[int, int]], list[dict[str, Any]]]:
    starts = sorted(set(int(value) for value in conversation_starts))
    if not starts or starts[0] != 0 or any(value < 0 or value >= len(records) for value in starts):
        raise ValueError("invalid conversation partition")
    ranges = starts + [len(records)]
    maps = [pseudonym_map(records[ranges[i] : ranges[i + 1]]) for i in range(len(starts))]
    pieces: list[str] = []
    bounds: list[tuple[int, int]] = []
    metadata: list[dict[str, Any]] = []
    character = 0
    for message_index, record in enumerate(records):
        conversation = bisect.bisect_right(starts, message_index) - 1
        mapping = maps[conversation]
        cleaned = clean_slack(record["raw_text"], mapping)
        if not cleaned:
            raise RuntimeError("message became empty only after pseudonymization")
        speaker = mapping[record["user"]]
        piece = f"{speaker}: {cleaned}\n"
        pieces.append(piece)
        bounds.append((character, character + len(piece)))
        character += len(piece)
        metadata.append(
            {
                "message_index": message_index,
                "source_ordinal": record["source_ordinal"],
                "pair_id": record["pair_id"],
                "speaker": speaker,
                "conversation_index": conversation,
            }
        )
    return "".join(pieces), bounds, metadata


def token_labels(offsets: list[tuple[int, int]], bounds: list[tuple[int, int]]) -> list[int]:
    labels: list[int] = []
    unit = 0
    for start, _stop in offsets:
        while unit + 1 < len(bounds) and start >= bounds[unit][1]:
            unit += 1
        labels.append(unit)
    return labels


def transition_starts(labels: list[int]) -> list[int]:
    return [0] + [index for index in range(1, len(labels)) if labels[index] != labels[index - 1]]


def choose_partitions(
    tokenizer: Tokenizer, records: list[dict[str, Any]]
) -> tuple[list[int], dict[str, Any]]:
    text, bounds, _metadata = render_records(records, [0])
    encoding = tokenizer.encode(text)
    labels = token_labels(encoding.offsets, bounds)
    starts = transition_starts(labels)
    partitions = [0]
    realized: list[dict[str, int]] = []
    for target in PARTITION_TARGETS:
        offset = int(np.searchsorted(np.asarray(starts), target, side="left"))
        if offset >= len(starts):
            raise RuntimeError(f"source is too short for partition target {target}")
        token_start = starts[offset]
        message_index = labels[token_start]
        if message_index <= partitions[-1]:
            raise RuntimeError("partition targets collapsed onto one message")
        partitions.append(message_index)
        realized.append(
            {
                "target_token": target,
                "single_thread_token": token_start,
                "message_index": message_index,
            }
        )
    return partitions, {
        "rule": "first complete message start at or after each 1024-token target in the single-thread draft",
        "targets": PARTITION_TARGETS,
        "realized": realized,
        "conversation_count": len(partitions),
    }


def assemble_arm(
    tokenizer: Tokenizer,
    name: str,
    records: list[dict[str, Any]],
    partitions: list[int],
    *,
    sequence: int = SEQ,
) -> dict[str, Any]:
    text, bounds, metadata = render_records(records, partitions)
    encoding = tokenizer.encode(text)
    if len(encoding.ids) < sequence:
        raise RuntimeError(f"{name} has only {len(encoding.ids)} tokens")
    ids = np.asarray(encoding.ids[:sequence], dtype=np.int32)
    offsets = [(int(a), int(b)) for a, b in encoding.offsets[:sequence]]
    labels = token_labels(offsets, bounds)
    starts = transition_starts(labels)
    if labels[0] != 0 or any(right - left != 1 for left, right in zip(labels, labels[1:]) if right != left):
        raise RuntimeError(f"{name} has skipped or nonmonotone message labels")
    used_messages = int(labels[-1]) + 1
    conversation_labels = [int(metadata[label]["conversation_index"]) for label in labels]
    conversation_starts = transition_starts(conversation_labels)
    text_end = offsets[-1][1]
    rendered_text = text[:text_end]
    used_metadata: list[dict[str, Any]] = []
    start_by_message = {labels[position]: position for position in starts}
    for message_index in range(used_messages):
        record = dict(metadata[message_index])
        record["start_token"] = int(start_by_message[message_index])
        record["complete_in_capture"] = bool(bounds[message_index][1] <= text_end)
        used_metadata.append(record)
    sidecar = {
        "schema_version": 1,
        "kind": "round5_pe_paired_arm_sidecar",
        "arm": name,
        "seq": sequence,
        "token_message_index": labels,
        "token_conversation_index": conversation_labels,
        "message_start_tokens": starts,
        "conversation_start_tokens": conversation_starts,
        "messages": used_metadata,
        "n_messages_used": used_messages,
        "n_conversations_used": int(conversation_labels[-1]) + 1,
        "token_offsets": offsets,
    }
    return {
        "ids": ids,
        "text": rendered_text,
        "sidecar": sidecar,
        "fragments": tokenizer.decode_batch(
            [[int(token_id)] for token_id in ids], skip_special_tokens=False
        ),
    }


def boundary_records(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    messages = {int(item["start_token"]): item for item in sidecar["messages"]}
    conversation_starts = [int(value) for value in sidecar["conversation_start_tokens"]]
    conversation_set = set(conversation_starts[1:])
    output: list[dict[str, Any]] = []
    for boundary in sidecar["message_start_tokens"]:
        boundary = int(boundary)
        if boundary == 0:
            continue
        if boundary in conversation_set:
            prior_openers = [value for value in conversation_starts if value < boundary]
            opener = prior_openers[-1]
            boundary_type = "higher_scope"
        else:
            opener = max(value for value in conversation_starts if value <= boundary)
            boundary_type = "ordinary_message"
        output.append(
            {
                "token": boundary,
                "pair_id": messages[boundary]["pair_id"],
                "boundary_type": boundary_type,
                "segment_open_token": opener,
                "retired_context_dose": boundary - opener,
            }
        )
    return output


def stratified_random_mask(
    reference: list[int], excluded: set[int], *, seed: int, sequence: int = SEQ
) -> list[int]:
    reference_array = np.asarray(sorted(set(reference)), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    selected: list[int] = []
    for start in range(0, sequence, BIN_SIZE):
        stop = min(start + BIN_SIZE, sequence)
        count = int(np.count_nonzero((reference_array >= start) & (reference_array < stop)))
        if not count:
            continue
        candidates = np.asarray(
            [position for position in range(start, stop) if position not in excluded],
            dtype=np.int64,
        )
        if len(candidates) < count:
            raise RuntimeError("insufficient random-control candidates in a position bin")
        selected.extend(int(value) for value in rng.choice(candidates, size=count, replace=False))
    selected.sort()
    if len(selected) != len(reference_array) or set(selected) & excluded:
        raise RuntimeError("invalid stratified random mask")
    return selected


def seed_for(label: str, ids_hash: str) -> int:
    payload = f"{PE_COMMIT}|{PF_COMMIT}|{ids_hash}|{label}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def freeze_classes(arm: dict[str, Any], name: str) -> dict[str, Any]:
    ids = arm["ids"]
    text = arm["text"]
    sidecar = arm["sidecar"]
    fragments = arm["fragments"]
    offsets = sidecar["token_offsets"]
    message_starts = [int(value) for value in sidecar["message_start_tokens"]]
    start_set = set(message_starts)
    message_start_for_token: list[int] = []
    current = 0
    for position in range(len(ids)):
        if position in start_set:
            current = position
        message_start_for_token.append(current)

    url_spans = [(match.start(), match.end()) for match in re.finditer(r"https?://\S+", text)]
    url_tokens = [
        position
        for position, (start, stop) in enumerate(offsets)
        if stop > start and any(start >= left and stop <= right for left, right in url_spans)
    ]
    proper_nouns: list[int] = []
    gratitude: list[int] = []
    gratitude_words = {"thank", "thanks", "thx", "ty"}
    for position, fragment in enumerate(fragments):
        stripped = fragment.strip()
        folded = stripped.casefold()
        if folded in gratitude_words or folded.startswith("thank"):
            gratitude.append(position)
        message_offset = position - message_start_for_token[position]
        previous = fragments[position - 1] if position else ""
        after_terminal = bool(
            previous.endswith("\n") or re.search(r"[.!?][\"')\]]*\s*$", previous)
        )
        if (
            re.fullmatch(r" ?[A-Z][a-z]{2,}", fragment)
            and message_offset not in (0, 1, 2)
            and not after_terminal
        ):
            proper_nouns.append(position)
    label_colons = [
        start + 1
        for start in message_starts
        if start + 1 < SEQ and fragments[start + 1].strip() == ":"
    ]
    boundaries = boundary_records(sidecar)
    ordinary = [item["token"] for item in boundaries if item["boundary_type"] == "ordinary_message"]
    higher_scope = [item["token"] for item in boundaries if item["boundary_type"] == "higher_scope"]
    classes = {
        "url_tokens": sorted(set(url_tokens)),
        "proper_noun_proxy": sorted(set(proper_nouns)),
        "gratitude_tokens": sorted(set(gratitude)),
        "label_colons": sorted(set(label_colons)),
        "ordinary_message_starts": ordinary,
        "higher_scope_starts": higher_scope,
    }
    excluded = set().union(*(set(values) for values in classes.values()))
    ids_hash = sha256_bytes(ids.tobytes())
    random_masks: dict[str, list[int]] = {}
    seeds: dict[str, int] = {}
    for class_name, positions in classes.items():
        seed = seed_for(f"{name}|{class_name}", ids_hash)
        seeds[class_name] = seed
        random_masks[class_name] = stratified_random_mask(positions, excluded, seed=seed)
    return {
        "boundaries": boundaries,
        "classes": classes,
        "counts": {key: len(value) for key, value in classes.items()},
        "voided_below_8": {
            key: len(classes[key]) < 8
            for key in ("url_tokens", "proper_noun_proxy", "gratitude_tokens", "label_colons")
        },
        "random_seeds": seeds,
        "random_masks": random_masks,
        "definitions": {
            "position_bin": BIN_SIZE,
            "url_tokens": "token character span wholly inside https?://\\S+",
            "proper_noun_proxy": "^ ?[A-Z][a-z]{2,}$; message offsets 0-2 and sentence/newline starts excluded",
            "gratitude_tokens": "stripped casefold in {thank,thanks,thx,ty} or begins thank",
            "label_colons": "message-start +1 when decoded fragment stripped equals ':'",
            "random_masks": "without-replacement within 256-token bins; union of all frozen classes excluded",
        },
    }


def scrub_audit(arm: dict[str, Any], raw_users: set[str]) -> dict[str, Any]:
    text = arm["text"]
    return {
        "replacement_character_count": text.count("�"),
        "slack_angle_markup_count": len(re.findall(r"<[^>]+>", text)),
        "common_mojibake_count": sum(text.count(value) for value in ("â€™", "â€œ", "â€", "Â ")),
        "raw_user_id_hits": sum(text.count(user) for user in raw_users),
    }


def build_payloads(archive: Path, tokenizer_path: Path) -> dict[str, Any]:
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_path, messages, inventory = select_registered_source(archive)
    records = ordered_records(messages, inventory["selected_file_sha256"])
    partitions, partition_record = choose_partitions(tokenizer, records)
    single = assemble_arm(tokenizer, SINGLE, records, [0])
    multi = assemble_arm(tokenizer, MULTI, records, partitions)
    arms = {SINGLE: single, MULTI: multi}
    classes = {name: freeze_classes(arms[name], name) for name in ARMS}

    pair_positions = {
        name: {item["pair_id"]: int(item["start_token"]) for item in arms[name]["sidecar"]["messages"]}
        for name in ARMS
    }
    union = set(pair_positions[SINGLE]) | set(pair_positions[MULTI])
    intersection = set(pair_positions[SINGLE]) & set(pair_positions[MULTI])
    preservation = len(intersection) / max(len(union), 1)
    if preservation < 0.80:
        raise RuntimeError(f"paired message-start preservation {preservation:.3f} is below 0.80")

    ordinary_pair_ids = {
        name: {
            item["pair_id"]
            for item in classes[name]["boundaries"]
            if item["boundary_type"] == "ordinary_message"
        }
        for name in ARMS
    }
    paired_ordinary = sorted(ordinary_pair_ids[SINGLE] & ordinary_pair_ids[MULTI])
    position_deltas = [
        pair_positions[MULTI][pair_id] - pair_positions[SINGLE][pair_id]
        for pair_id in sorted(intersection)
    ]
    position_alignment = {
        "paired_count": len(position_deltas),
        "exact_position_matches": int(np.count_nonzero(np.asarray(position_deltas) == 0)),
        "mismatched_positions": int(np.count_nonzero(np.asarray(position_deltas) != 0)),
        "median_multi_minus_single": float(np.median(position_deltas)),
        "min_multi_minus_single": int(min(position_deltas)),
        "max_multi_minus_single": int(max(position_deltas)),
    }
    class_payload = {
        "schema_version": 1,
        "kind": "round5_pe_pf_private_class_freeze",
        "arms": classes,
        "paired_messages": [
            {
                "pair_id": pair_id,
                "single_token": pair_positions[SINGLE][pair_id],
                "multi_token": pair_positions[MULTI][pair_id],
            }
            for pair_id in sorted(intersection)
        ],
        "paired_ordinary_messages": [
            {
                "pair_id": pair_id,
                "single_token": pair_positions[SINGLE][pair_id],
                "multi_token": pair_positions[MULTI][pair_id],
            }
            for pair_id in paired_ordinary
        ],
        "pair_preservation": {
            "single_message_starts": len(pair_positions[SINGLE]),
            "multi_message_starts": len(pair_positions[MULTI]),
            "intersection": len(intersection),
            "union": len(union),
            "intersection_over_union": preservation,
            "minimum": 0.80,
            "passed": preservation >= 0.80,
        },
        "pair_position_alignment": position_alignment,
    }
    raw_users = {str(message["user"]) for message in messages}
    scrub = {name: scrub_audit(arms[name], raw_users) for name in ARMS}
    if any(any(value != 0 for value in audit.values()) for audit in scrub.values()):
        raise RuntimeError(f"privacy/scrub audit failed: {scrub}")
    return {
        "arms": arms,
        "classes": class_payload,
        "inventory": inventory,
        "partition": partition_record,
        "scrub": scrub,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "source_path": source_path,
    }


def output_paths(out_root: Path) -> list[Path]:
    paths = [out_root / "pe_manifest.json", out_root / "pe_classes.json", PUBLIC_FREEZE]
    for name in ARMS:
        paths.extend(
            [
                out_root / f"{name}.ids.npy",
                out_root / f"{name}.txt",
                out_root / f"{name}.sidecar.json",
            ]
        )
    return paths


def write_build(payloads: dict[str, Any], out_root: Path) -> None:
    existing = [path for path in output_paths(out_root) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite paired freeze outputs: {existing}")
    out_root.mkdir(parents=True, exist_ok=True)
    arm_records: dict[str, Any] = {}
    for name in ARMS:
        arm = payloads["arms"][name]
        ids_path = out_root / f"{name}.ids.npy"
        text_path = out_root / f"{name}.txt"
        sidecar_path = out_root / f"{name}.sidecar.json"
        atomic_npy(ids_path, arm["ids"])
        atomic_bytes(text_path, arm["text"].encode("utf-8"))
        atomic_bytes(sidecar_path, canonical_json_bytes(arm["sidecar"]))
        arm_records[name] = {
            "ids": {"sha256": sha256_file(ids_path), "bytes": ids_path.stat().st_size},
            "text": {"sha256": sha256_file(text_path), "bytes": text_path.stat().st_size},
            "sidecar": {"sha256": sha256_file(sidecar_path), "bytes": sidecar_path.stat().st_size},
            "messages": arm["sidecar"]["n_messages_used"],
            "conversations": arm["sidecar"]["n_conversations_used"],
        }
    classes_path = out_root / "pe_classes.json"
    atomic_bytes(classes_path, canonical_json_bytes(payloads["classes"]))
    manifest = {
        "schema_version": 1,
        "kind": "round5_pe_paired_private_corpus",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "private": True,
        "never_commit": "token IDs and sidecars reconstruct private workplace text",
        "builder_source_sha256": sha256_file(Path(__file__)),
        "build_git_head": git_output("rev-parse", "HEAD"),
        "registration_commits": {"P-e": PE_COMMIT, "P-f": PF_COMMIT, "D1": D1_COMMIT},
        "registration_sha256": {
            "P-e": sha256_file(ROOT / "registrations" / "ROUND5_APERTURE_CONTEXT_DOSE_PREREG.md"),
            "P-f": sha256_file(ROOT / "registrations" / "ROUND5_APERTURE_ANCHOR_PREREG.md"),
            "D1": sha256_file(ROOT / "registrations" / "ROUND5_CAPTURE_SCOPE_D1.md"),
        },
        "seq": SEQ,
        "tokenizer_sha256": payloads["tokenizer_sha256"],
        "source_inventory": payloads["inventory"],
        "partition": payloads["partition"],
        "scrub_audit": payloads["scrub"],
        "arms": arm_records,
        "classes": {
            "path": classes_path.name,
            "sha256": sha256_file(classes_path),
            "bytes": classes_path.stat().st_size,
        },
    }
    manifest_path = out_root / "pe_manifest.json"
    atomic_bytes(manifest_path, canonical_json_bytes(manifest))
    public = {
        "schema_version": 1,
        "kind": "round5_pe_pf_public_corpus_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_data_opened": False,
        "registration_commits": manifest["registration_commits"],
        "registration_sha256": manifest["registration_sha256"],
        "builder_source_sha256": manifest["builder_source_sha256"],
        "private_manifest_sha256": sha256_file(manifest_path),
        "private_classes_sha256": manifest["classes"]["sha256"],
        "tokenizer_sha256": payloads["tokenizer_sha256"],
        "source_inventory": payloads["inventory"],
        "prior_arm_overlap": {
            "message_or_conversation_overlap": False,
            "reason": "fresh C-channel family; prior Slack arms use D-channel family",
            "speaker_overlap_excluded": False,
        },
        "partition": payloads["partition"],
        "dose_definition": {
            "ordinary_message": "boundary token minus current conversation opening token",
            "higher_scope": "boundary token minus opening token of the conversation just closed",
            "outcome_dependent": False,
        },
        "pair_preservation": payloads["classes"]["pair_preservation"],
        "pair_position_alignment": payloads["classes"]["pair_position_alignment"],
        "scrub_audit": payloads["scrub"],
        "arms": {
            name: {
                "ids_sha256": arm_records[name]["ids"]["sha256"],
                "text_sha256": arm_records[name]["text"]["sha256"],
                "sidecar_sha256": arm_records[name]["sidecar"]["sha256"],
                "messages": arm_records[name]["messages"],
                "conversations": arm_records[name]["conversations"],
                "class_counts": payloads["classes"]["arms"][name]["counts"],
                "voided_below_8": payloads["classes"]["arms"][name]["voided_below_8"],
            }
            for name in ARMS
        },
        "paired_ordinary_message_count": len(payloads["classes"]["paired_ordinary_messages"]),
        "control_freeze": {
            "crossed_math_arm": "08_math_llm corrected A6/A8 capture at identical absolute positions",
            "random_masks": "stored privately; deterministic, 256-token-position-stratified",
        },
        "permutation_seed_rule": "derive from SHA-256 of this committed public freeze",
    }
    atomic_bytes(PUBLIC_FREEZE, canonical_json_bytes(public))
    print(json.dumps({"arms": public["arms"], "pair_preservation": public["pair_preservation"]}, indent=2))
    print(f"wrote private paired corpus under {out_root}")
    print(f"wrote public freeze {PUBLIC_FREEZE}")


def check_existing(payloads: dict[str, Any], out_root: Path) -> None:
    manifest = json.loads((out_root / "pe_manifest.json").read_text(encoding="utf-8"))
    classes = json.loads((out_root / "pe_classes.json").read_text(encoding="utf-8"))
    if classes != payloads["classes"]:
        raise RuntimeError("private P-e/P-f classes do not reproduce")
    for name in ARMS:
        arm = payloads["arms"][name]
        ids = np.load(out_root / f"{name}.ids.npy", allow_pickle=False)
        sidecar_bytes = (out_root / f"{name}.sidecar.json").read_bytes()
        text_bytes = (out_root / f"{name}.txt").read_bytes()
        if (
            not np.array_equal(ids, arm["ids"])
            or sidecar_bytes != canonical_json_bytes(arm["sidecar"])
            or text_bytes != arm["text"].encode("utf-8")
        ):
            raise RuntimeError(f"paired arm does not reproduce: {name}")
        paths = {
            "ids": out_root / f"{name}.ids.npy",
            "text": out_root / f"{name}.txt",
            "sidecar": out_root / f"{name}.sidecar.json",
        }
        for field, path in paths.items():
            if manifest["arms"][name][field]["sha256"] != sha256_file(path):
                raise RuntimeError(f"stale private manifest record: {name} {field}")
    public = json.loads(PUBLIC_FREEZE.read_text(encoding="utf-8"))
    if public.get("private_manifest_sha256") != sha256_file(out_root / "pe_manifest.json"):
        raise RuntimeError("public freeze does not bind the private manifest")
    print("paired corpus and class freeze reproduce exactly")


def self_test() -> None:
    cleaned = clean_slack(
        "hi <@U1> <https://example.test|site> <!here> <#C1|room>", {"U1": "A"}
    )
    if cleaned != "hi @A site @here #room" or re.search(r"<[^>]+>", cleaned):
        raise AssertionError(f"Slack scrub self-test failed: {cleaned!r}")
    reference = [1, 4, 260, 270]
    first = stratified_random_mask(reference, set(reference), seed=7, sequence=512)
    second = stratified_random_mask(reference, set(reference), seed=7, sequence=512)
    if first != second or len(first) != len(reference) or set(first) & set(reference):
        raise AssertionError("stratified random-mask self-test failed")
    labels = token_labels([(0, 1), (1, 3), (4, 5)], [(0, 3), (3, 6)])
    if labels != [0, 0, 1] or transition_starts(labels) != [0, 2]:
        raise AssertionError("transition-label self-test failed")
    print("paired builder self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "dry-run", "check", "self-test"])
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return
    payloads = build_payloads(args.archive.resolve(), args.tokenizer.resolve())
    summary = {
        "source_inventory": payloads["inventory"],
        "partition": payloads["partition"],
        "pair_preservation": payloads["classes"]["pair_preservation"],
        "pair_position_alignment": payloads["classes"]["pair_position_alignment"],
        "counts": {name: payloads["classes"]["arms"][name]["counts"] for name in ARMS},
    }
    if args.command == "dry-run":
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "check":
        check_existing(payloads, args.out.resolve())
    else:
        write_build(payloads, args.out.resolve())


if __name__ == "__main__":
    main()
