"""
Build the Tier-2 measurement corpus: 6 texts x 8192 tokens, pre-tokenized.

The corpus IS the experiment for question (b) -- content decay is a property
of the data -- so each text targets a distinct content-match regime:

  01_prose_en     natural English prose (Moby Dick body). Baseline content decay.
  02_code         real Python source (local site-packages). Strong local syntax
                  + genuine long-range structure (def <-> use, imports).
  03_templated    synthetic server logs, seeded RNG. Near-degenerate content
                  similarity everywhere; includes an exact-repeat heartbeat
                  line every 64 lines => known-period induction targets, so
                  bias-driven vs induction-driven long-range attention are
                  separable via the with/without decomposition.
  04_multilingual Spanish + German prose halves. Generality check.
  05_needles      prose scaffold with planted rare entities re-mentioned at
                  distances straddling d=1024 -- the designed instrument for
                  the seam question (a). Ground-truth mention positions are
                  MEASURED post-tokenization into 05_needles.sidecar.json
                  (dump-first: measure, don't assume).
  06_random       uniform random token ids (seeded, specials excluded),
                  stored as ids directly. The control arm: content-match has
                  no distance structure in expectation, so mass_without(d)
                  should be ~flat and mass_with(d) reads the bias's in-situ
                  strength against a known-flat baseline. Off-distribution --
                  interpret early layers with more confidence than deep ones.

Artifacts (dump-first): per text, {name}.txt (raw, except 06), {name}.ids.npy
(int32, exactly SEQ tokens), plus manifest.json (sha256, token counts,
tokenizer provenance, no-BOS note) and 05_needles.sidecar.json.

Runner contract: feed ids.npy verbatim; do NOT re-tokenize the .txt files
(BPE round-trips are not guaranteed, especially for 06).
"""
import hashlib
import json
import os
import random
import re

import numpy as np
from tokenizers import Tokenizer

CORPUS = r"R:\inkling\corpus"
SEQ = 8192
TOK = Tokenizer.from_file(os.path.join(CORPUS, "tokenizer.json"))


def strip_gutenberg(raw):
    s = raw.find("*** START")
    e = raw.find("*** END")
    body = raw[s:e] if s != -1 and e != -1 else raw
    body = body.split("\n", 1)[1] if "\n" in body else body
    return re.sub(r"\r\n", "\n", body).strip()


def take_tokens(text, n=SEQ):
    ids = TOK.encode(text).ids
    assert len(ids) >= n, f"only {len(ids)} tokens, need {n}"
    return ids[:n]


def save(name, ids, raw_text=None):
    ids = np.asarray(ids, dtype=np.int32)
    assert ids.shape == (SEQ,)
    np.save(os.path.join(CORPUS, f"{name}.ids.npy"), ids)
    if raw_text is not None:
        with open(os.path.join(CORPUS, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
    print(f"{name}: {len(ids)} tokens saved")
    return ids


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_templated():
    rng = random.Random(0)
    methods = ["GET", "POST", "PUT", "DELETE"]
    paths = ["/api/v2/users", "/api/v2/orders", "/api/v2/items", "/healthz", "/metrics"]
    codes = [200, 200, 200, 201, 204, 301, 400, 403, 404, 500]
    lines = []
    for i in range(3000):
        if i % 64 == 0:
            # exact-repeat heartbeat: a known-period induction target
            lines.append("2026-07-15T00:00:00Z worker-0 HEARTBEAT status=OK queue_depth=0 uptime_check=PASSED")
        else:
            lines.append(
                f"2026-07-15T{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}Z "
                f"worker-{rng.randrange(8)} {rng.choice(methods)} {rng.choice(paths)} "
                f"status={rng.choice(codes)} latency_ms={rng.randrange(1, 900)} "
                f"req_id={rng.randrange(16**8):08x}"
            )
    return "\n".join(lines)


def build_needles():
    """Prose scaffold + planted entities whose re-mention distances straddle 1024.

    Entities are unique nonsense codewords (tokenizer-hostile on purpose: they
    fragment into rare subtokens, giving sharp content-match targets). Each is
    introduced once and re-mentioned once; target token-distances alternate
    below/above the seam: ~{896..1000} vs ~{1048..1152}, jittered.
    True positions are measured after tokenization and written to the sidecar.
    """
    rng = random.Random(1)
    fillers = [
        "The survey team recorded the readings and moved on to the next station without incident.",
        "Weather over the ridge stayed calm through the afternoon, which the log described as fortunate.",
        "Provisions were counted twice, as the quartermaster insisted, and the tally matched.",
        "A minor dispute about the route was settled by consulting the older of the two charts.",
        "The instruments were recalibrated at dusk according to the standing procedure.",
        "Nothing in the samples suggested contamination, though the analyst noted the point for review.",
        "The relief crew arrived a day early and was put to work cataloguing the specimens.",
        "By the third week the routine had settled into a rhythm that the diary calls almost pleasant.",
    ]
    syll = ["zor", "vek", "tal", "mun", "qir", "bex", "dov", "lys", "fen", "gac"]
    def codeword(k):
        rng2 = random.Random(100 + k)
        return "".join(rng2.choice(syll) for _ in range(3)).upper() + f"-{rng2.randrange(10, 99)}"

    n_entities = 24
    intro_tmpl = "At this point the expedition first catalogued the artifact designated {} in the manifest."
    recall_tmpl = "Only much later did anyone connect the anomaly back to the artifact designated {} from before."

    # build in sentence units, tracking approximate token counts as we go
    sents, events = [], []   # events: (kind, entity_idx, sentence_idx)
    tok_count = 0
    ent = 0
    pending = []             # (entity_idx, due_token_pos)
    while tok_count < SEQ + 512:
        due = [p for p in pending if p[1] <= tok_count]
        if due:
            k, _ = due[0]
            pending.remove(due[0])
            s = recall_tmpl.format(codeword(k))
            events.append(("recall", k, len(sents)))
        elif ent < n_entities and tok_count > 200 and rng.random() < 0.25:
            k = ent; ent += 1
            s = intro_tmpl.format(codeword(k))
            events.append(("intro", k, len(sents)))
            below = (k % 2 == 0)
            gap = rng.randrange(896, 1001) if below else rng.randrange(1048, 1153)
            pending.append((k, tok_count + gap))
        else:
            s = rng.choice(fillers)
        sents.append(s)
        tok_count += len(TOK.encode(" " + s).ids)
    text = " ".join(sents)

    ids = take_tokens(text)
    # measure true token positions of each codeword's mentions in the trimmed ids
    id_str_offsets = TOK.encode(text).offsets[:SEQ]
    sidecar = {"seq": SEQ, "entities": []}
    for k in range(n_entities):
        cw = codeword(k)
        char_hits = [m.start() for m in re.finditer(re.escape(cw), text)]
        tok_hits = []
        for ch in char_hits:
            pos = next((t for t, (a, b) in enumerate(id_str_offsets) if a <= ch < b), None)
            if pos is not None:
                tok_hits.append(pos)
        e = {"codeword": cw, "token_positions": tok_hits}
        if len(tok_hits) >= 2:
            e["distance"] = tok_hits[1] - tok_hits[0]
            e["side_of_seam"] = "below" if e["distance"] < 1024 else "above"
        sidecar["entities"].append(e)
    n_below = sum(1 for e in sidecar["entities"] if e.get("side_of_seam") == "below")
    n_above = sum(1 for e in sidecar["entities"] if e.get("side_of_seam") == "above")
    print(f"  needles: {n_below} below seam, {n_above} above seam "
          f"(of {n_entities}; unpaired ones fell past the trim)")
    json.dump(sidecar, open(os.path.join(CORPUS, "05_needles.sidecar.json"), "w"), indent=2)
    return text, ids


def build_code():
    import numpy as _np, scipy as _sp
    roots = [os.path.dirname(_np.__file__), os.path.dirname(_sp.__file__)]
    picks, total = [], 0
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if "tests" in dirpath:
                continue
            for fn in sorted(files):
                if fn.endswith(".py") and total < 400_000:
                    p = os.path.join(dirpath, fn)
                    try:
                        src = open(p, encoding="utf-8").read()
                    except (UnicodeDecodeError, OSError):
                        continue
                    if len(src) > 20_000:
                        picks.append(f"# ==== {fn} ====\n" + src)
                        total += len(src)
        if total >= 400_000:
            break
    return "\n\n".join(picks)


def build_random():
    """Uniform ids over the regular vocab, specials excluded. Seed 0."""
    cfg = json.load(open(os.path.join(CORPUS, "tokenizer_config.json")))
    special_ids = set()
    for tok_id, info in cfg.get("added_tokens_decoder", {}).items():
        special_ids.add(int(tok_id))
    rng = np.random.default_rng(0)
    ids, out = rng.integers(0, 200_000, size=SEQ * 2), []
    for t in ids:
        if int(t) not in special_ids:
            out.append(int(t))
        if len(out) == SEQ:
            break
    return np.array(out, dtype=np.int32)


def main():
    manifest = {"seq": SEQ, "tokenizer": "tokenizer.json (from thinkingmachines/Inkling)",
                "bos": "none prepended — model config defines no bos_token; raw prefill",
                "runner_contract": "consume {name}.ids.npy verbatim; never re-tokenize the .txt",
                "texts": {}}

    moby = strip_gutenberg(open(os.path.join(CORPUS, "_moby.txt"), encoding="utf-8").read())
    save("01_prose_en", take_tokens(moby[10_000:]), moby[10_000:120_000])

    code = build_code()
    save("02_code", take_tokens(code), code[:150_000])

    tmpl = build_templated()
    save("03_templated", take_tokens(tmpl), tmpl)

    es = strip_gutenberg(open(os.path.join(CORPUS, "_quijote.txt"), encoding="utf-8").read())
    de = strip_gutenberg(open(os.path.join(CORPUS, "_faust.txt"), encoding="utf-8").read())
    ml_es, ml_de = es[20_000:120_000], de[5_000:105_000]
    ids_es = TOK.encode(ml_es).ids[: SEQ // 2]
    ids_de = TOK.encode(ml_de).ids[: SEQ - len(ids_es)]
    save("04_multilingual", ids_es + ids_de, ml_es + "\n\n" + ml_de)

    ntext, nids = build_needles()
    save("05_needles", nids, ntext)

    save("06_random", build_random())  # ids only, no meaningful .txt

    for name in ["01_prose_en", "02_code", "03_templated", "04_multilingual",
                 "05_needles", "06_random"]:
        p = os.path.join(CORPUS, f"{name}.ids.npy")
        manifest["texts"][name] = {"ids_sha256": sha256(p),
                                   "n_tokens": int(np.load(p).shape[0])}
    json.dump(manifest, open(os.path.join(CORPUS, "manifest.json"), "w"), indent=2)
    print("manifest written")


if __name__ == "__main__":
    main()
