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


def truncate_to_tokens(text, budget):
    """Cut text at the character offset of its budget-th token (exact per-stream)."""
    enc = TOK.encode(text)
    if len(enc.ids) <= budget:
        return text, len(enc.ids)
    return text[: enc.offsets[budget][0]], budget


def build_mathllm():
    per = SEQ // 3                          # exact thirds; gemini (last) absorbs
    budgets = {"claude": per, "chatgpt": per, "gemini": per + (SEQ - 3 * per) + 512}
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
                remaining = budgets[prov] - toks
                piece, used = truncate_to_tokens(piece, remaining)
                if used == 0:
                    done = True
                    break
                parts.append(piece)
                meta.append((prov, os.path.basename(f), i))
                toks += used
                if toks >= budgets[prov]:
                    done = True
                    break
            if done:
                break
        print(f"  {prov}: {toks} tokens (budget {budgets[prov]})")
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
    labels, starts = [], []
    ui = 0
    for t, (a, b) in enumerate(offsets):
        while ui + 1 < len(bounds) and a >= bounds[ui][1]:
            ui += 1
        labels.append(ui)
        if a <= bounds[ui][0] < b or (t == 0):
            starts.append(t)
        elif t > 0 and offsets[t - 1][1] <= bounds[ui][0] < a + 1 and labels[t - 1] != ui:
            starts.append(t)
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
    os.makedirs(OUT, exist_ok=True)
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
