"""
Download the Inkling-NVFP4 checkpoint (592GB) to R:\inkling\nvfp4.

Why not `hf download`: R: is exFAT, which does not support the byte-range lock
semantics `filelock` needs. huggingface_hub livelocks on
`.cache/huggingface/.gitignore.lock` and never starts. This downloader uses the
same HTTP Range idiom as extract_rel_attn.py, with no lock files.

Resumable: progress is a per-shard sidecar `<shard>.chunks` bitmap. Re-running
skips completed chunks. Verifies final size against the Hub's content-length.
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "thinkingmachines/Inkling-NVFP4"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
OUT = r"R:\inkling\nvfp4"
CHUNK = 64 << 20  # 64MB
WORKERS = 16

_lock = threading.Lock()
_done_bytes = 0
_t0 = time.time()


def _open(url, rng=None, timeout=180):
    h = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def content_length(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_chunk(path, url, idx, lo, hi):
    global _done_bytes
    for attempt in range(6):
        try:
            with _open(url, (lo, hi)) as r:
                buf = r.read(hi - lo + 1)
            if len(buf) != hi - lo + 1:
                raise IOError(f"short read {len(buf)} != {hi - lo + 1}")
            with open(path, "r+b") as f:
                f.seek(lo)
                f.write(buf)
            with _lock:
                _done_bytes += len(buf)
            return idx
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))


def shard_chunks_path(path):
    return path + ".chunks"


def load_done(path, n):
    p = shard_chunks_path(path)
    if not os.path.exists(p):
        return set()
    try:
        return set(json.load(open(p)))
    except Exception:
        return set()


def save_done(path, done):
    json.dump(sorted(done), open(shard_chunks_path(path), "w"))


def download_shard(name):
    url = BASE + name
    path = os.path.join(OUT, name)
    size = content_length(url)
    ranges = [(i, o, min(o + CHUNK - 1, size - 1)) for i, o in enumerate(range(0, size, CHUNK))]

    if os.path.exists(path) and os.path.getsize(path) == size:
        done = load_done(path, len(ranges))
        if len(done) == len(ranges):
            with _lock:
                globals()["_done_bytes"] = _done_bytes + size
            print(f"  [skip] {name} complete ({size/1e9:.1f}GB)", flush=True)
            return
    else:
        with open(path, "wb") as f:
            f.truncate(size)
        done = set()

    todo = [r for r in ranges if r[0] not in done]
    with _lock:
        globals()["_done_bytes"] = _done_bytes + (len(ranges) - len(todo)) * CHUNK
    print(f"  {name}: {size/1e9:.1f}GB, {len(todo)}/{len(ranges)} chunks to fetch", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch_chunk, path, url, i, lo, hi) for i, lo, hi in todo]
        for k, fu in enumerate(futs):
            done.add(fu.result())
            if k % 32 == 0:
                save_done(path, done)
                el = time.time() - _t0
                print(f"    {_done_bytes/1e9:7.1f}GB  {_done_bytes/el/1e6:6.0f} MB/s", flush=True)
    save_done(path, done)
    assert os.path.getsize(path) == size, f"{name} size mismatch"


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(_open(BASE + "model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    # small config/tokenizer files too
    extras = ["config.json", "model.safetensors.index.json", "hf_quant_config.json"]
    for e in extras:
        p = os.path.join(OUT, e)
        if not os.path.exists(p):
            with _open(BASE + e) as r, open(p, "wb") as f:
                f.write(r.read())
    total = 0
    print(f"{len(shards)} shards -> {OUT}", flush=True)
    for i, s in enumerate(shards):
        print(f"[{i+1}/{len(shards)}] {s}", flush=True)
        download_shard(s)
    el = time.time() - _t0
    print(f"DONE {_done_bytes/1e9:.1f}GB in {el/60:.1f} min ({_done_bytes/el/1e6:.0f} MB/s)", flush=True)


if __name__ == "__main__":
    main()
