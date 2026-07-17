"""Build corpus v2 per CORPUS_V2_SPEC.md (frozen recipe; data stays private).

Arms:
  07_slack_human -- DM channels, human messages only, pseudonymized speakers.
  08_math_llm    -- assistant turns, providers claude/chatgpt/gemini in thirds.

Writes to R:\\inkling\\corpus_v2\\ (gitignored): {name}.ids.npy, {name}.txt,
{name}.sidecar.json, manifest.json. Never touches corpus/ (v1).
"""
import glob
import hashlib
import html
import json
import os
import re

import numpy as np
from tokenizers import Tokenizer

SLACK = r"D:\windows\slack-archive\slack-archive\data"
CONV = r"C:\Users\johnz\Projects\big-math-bigger-models\conversations"
OUT = r"R:\inkling\corpus_v2"
SEQ = 8192
PROVIDER_BUDGETS = {"claude": 2730, "chatgpt": 2730, "gemini": 2732}
TOK = Tokenizer.from_file(r"R:\inkling\corpus\tokenizer.json")


def clean_text(s):
    s = s.replace("�", "'")
    s = html.unescape(s)
    return s


def slack_clean(s, pseud):
    s = clean_text(s)
    s = re.sub(r"<@(\w+)>", lambda m: pseud.get(m.group(1), "@user"), s)
    s = re.sub(r"<(https?://[^|>]+)\|([^>]*)>", r"\2", s)
    s = re.sub(r"<(https?://[^>]+)>", r"\1", s)
    return s


def build_slack():
    chans = []
    for f in glob.glob(os.path.join(SLACK, "D*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        msgs = [m for m in d if isinstance(m, dict) and "user" in m and "bot_id" not in m
                and not m.get("subtype") and m.get("text")]
        chars = sum(len(m["text"]) for m in msgs)
        if chars:
            chans.append((chars, os.path.basename(f)[:-5], msgs))
    chans.sort(key=lambda t: (-t[0], t[1]))
    parts, meta = [], []           # meta: (channel, speaker) per message
    for _, cid, msgs in chans:
        msgs = sorted(msgs, key=lambda m: float(m["ts"]))
        pseud, order = {}, iter("ABCDEFGHJKLMNPQRSTUVWXYZ")
        for m in msgs:
            if m["user"] not in pseud:
                pseud[m["user"]] = next(order, "Z")
        for m in msgs:
            txt = slack_clean(m["text"], pseud).strip()
            if not txt:
                continue
            parts.append(f"{pseud[m['user']]}: {txt}\n")
            meta.append((cid, pseud[m["user"]]))
    return parts, meta


def build_slack_v21():
    """A6/v2.1 recipe: top 8 DM channels by human chars, up to 1,024 tokens each,
    concatenated in rank order (channel blocks); continue down the ranking if a
    channel underfills. Eight conversations instead of one; v2.0 arm untouched."""
    chans = []
    for f in glob.glob(os.path.join(SLACK, "D*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        msgs = [m for m in d if isinstance(m, dict) and "user" in m and "bot_id" not in m
                and not m.get("subtype") and m.get("text")]
        chars = sum(len(m["text"]) for m in msgs)
        if chars:
            chans.append((chars, os.path.basename(f)[:-5], msgs))
    chans.sort(key=lambda t: (-t[0], t[1]))
    parts, meta = [], []
    total = 0
    target = SEQ + 64          # small surplus: BPE joins at channel boundaries
    for _, cid, msgs in chans:
        if total >= target:
            break
        msgs = sorted(msgs, key=lambda m: float(m["ts"]))
        pseud, order = {}, iter("ABCDEFGHJKLMNPQRSTUVWXYZ")
        for m in msgs:
            if m["user"] not in pseud:
                pseud[m["user"]] = next(order, "Z")
        chan_budget = min(1024, target - total)
        used = 0
        for m in msgs:
            txt = slack_clean(m["text"], pseud).strip()
            if not txt:
                continue
            piece = f"{pseud[m['user']]}: {txt}\n"
            piece, n = truncate_to_tokens(piece, chan_budget - used)
            if n == 0:
                break
            parts.append(piece)
            meta.append((cid, pseud[m["user"]]))
            used += n
            if used >= chan_budget:
                break
        total += used
        print(f"  {cid}: {used} tokens")
    assert total >= SEQ + 32, f"only {total} tokens gathered"
    return parts, meta


def truncate_to_tokens(text, budget):
    """Cut text at the character offset of its budget-th token (exact per-stream)."""
    enc = TOK.encode(text)
    if len(enc.ids) <= budget:
        return text, len(enc.ids)
    return text[: enc.offsets[budget][0]], budget


def build_mathllm():
    assert sum(PROVIDER_BUDGETS.values()) == SEQ
    parts, meta = [], []                    # meta: (provider, file, turn_idx) per turn
    for prov in ["claude", "chatgpt", "gemini"]:
        toks = 0
        done = False
        for f in sorted(glob.glob(os.path.join(CONV, prov, "*.json"))):
            if "(" in os.path.basename(f):
                continue
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            msgs = d.get("messages") or (d.get("clean") or {}).get("messages") or []
            for i, m in enumerate(msgs):
                if str(m.get("role", "")).lower() != "assistant":
                    continue
                txt = clean_text(str(m.get("content", ""))).strip()
                if len(txt) < 40:
                    continue
                piece = txt + "\n\n"
                remaining = PROVIDER_BUDGETS[prov] - toks
                piece, used = truncate_to_tokens(piece, remaining)
                if used == 0:
                    done = True
                    break
                parts.append(piece)
                meta.append((prov, os.path.basename(f), i))
                toks += used
                if toks >= PROVIDER_BUDGETS[prov]:
                    done = True
                    break
            if done:
                break
        assert toks == PROVIDER_BUDGETS[prov], (
            f"{prov}: built {toks} tokens, expected {PROVIDER_BUDGETS[prov]}"
        )
        print(f"  {prov}: {toks} tokens (budget {PROVIDER_BUDGETS[prov]})")
    return parts, meta


def assemble(name, parts, meta, unit_label):
    text = "".join(parts)
    enc = TOK.encode(text)
    ids = np.asarray(enc.ids[:SEQ], dtype=np.int32)
    assert len(ids) == SEQ, f"{name}: only {len(enc.ids)} tokens"
    offsets = enc.offsets[:SEQ]
    # per-token unit labels + unit-start token positions, measured post-tokenization
    bounds, pos = [], 0
    for p in parts:
        bounds.append((pos, pos + len(p)))
        pos += len(p)
    labels = []
    ui = 0
    for t, (a, b) in enumerate(offsets):
        while ui + 1 < len(bounds) and a >= bounds[ui][1]:
            ui += 1
        labels.append(ui)
    # A7: a BPE token can straddle a character boundary between concatenated
    # units. Unit starts are therefore defined by the authoritative per-token
    # label transition, not by whether a token's character span contains the
    # raw join. This preserves all IDs/text and returns one start per used unit.
    starts = [0] + [t for t in range(1, len(labels)) if labels[t] != labels[t - 1]]
    np.save(os.path.join(OUT, f"{name}.ids.npy"), ids)
    with open(os.path.join(OUT, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(text[: offsets[-1][1]])
    side = dict(seq=SEQ, unit=unit_label, n_units_used=int(labels[-1]) + 1,
                token_unit_index=labels, unit_start_tokens=starts,
                unit_meta=[list(m) for m in meta[: labels[-1] + 1]])
    json.dump(side, open(os.path.join(OUT, f"{name}.sidecar.json"), "w"))
    print(f"{name}: {SEQ} tokens, {labels[-1]+1} {unit_label}s, {len(starts)} unit-start tokens")
    return ids


def sha(p):
    h = hashlib.sha256()
    h.update(open(p, "rb").read())
    return h.hexdigest()


def main():
    import sys
    os.makedirs(OUT, exist_ok=True)
    if "v21" in sys.argv[1:]:
        print("building 07b_slack_multi (v2.1) ...")
        assemble("07b_slack_multi", *build_slack_v21(), "message")
        manifest = json.load(open(os.path.join(OUT, "manifest.json")))
        p = os.path.join(OUT, "07b_slack_multi.ids.npy")
        manifest["texts"]["07b_slack_multi"] = dict(
            ids_sha256=sha(p),
            sidecar_sha256=sha(os.path.join(OUT, "07b_slack_multi.sidecar.json")),
            recipe="v2.1 per ROUND5_AMENDMENT_A6.md")
        json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
        print("manifest updated")
        return
    print("building 07_slack_human ...")
    assemble("07_slack_human", *build_slack(), "message")
    print("building 08_math_llm ...")
    assemble("08_math_llm", *build_mathllm(), "turn")
    manifest = {"spec": "CORPUS_V2_SPEC.md", "seq": SEQ, "private": True,
                "never_commit": "ids reconstruct private text",
                "texts": {}}
    for name in ["07_slack_human", "08_math_llm"]:
        p = os.path.join(OUT, f"{name}.ids.npy")
        manifest["texts"][name] = dict(ids_sha256=sha(p),
                                       sidecar_sha256=sha(os.path.join(OUT, f"{name}.sidecar.json")))
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print("manifest written ->", OUT)


if __name__ == "__main__":
    main()
