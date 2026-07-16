"""
Pull only the relative-attention tensors (wr_du.weight, rel_logits_proj.proj)
for every LLM layer of thinkingmachines/Inkling directly via HTTP Range
requests against the safetensors shards on the Hub -- no full-checkpoint
download (checkpoint is ~1.9TB; the tensors we need total well under 1GB).

Safetensors layout: first 8 bytes = little-endian uint64 header length N,
next N bytes = JSON header {tensor_name: {dtype, shape, data_offsets:[start,end]}},
then raw tensor bytes starting at offset 8+N. data_offsets are relative to
that base.
"""
import json
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO = "thinkingmachines/Inkling"
ROOT = Path(__file__).resolve().parents[1]
REVISION = os.environ.get("INKLING_REVISION", "main")
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
INDEX_PATH = Path(
    os.environ.get(
        "INKLING_INDEX_PATH",
        ROOT / "dumps" / "round3" / "model.safetensors.index.json",
    )
)
OUT_DIR = str(ROOT / "weights")
NUM_LAYERS = 66

_header_cache = {}


def load_weight_map():
    """Load the checkpoint index from a durable path, downloading if absent."""
    index_path = Path(INDEX_PATH)
    if not index_path.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        url = BASE + "model.safetensors.index.json"
        print(f"downloading index: {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            raw = response.read()
        # Parse before publishing the cache so an interrupted/error response
        # never becomes the next run's trusted index.
        parsed = json.loads(raw)
        if not isinstance(parsed.get("weight_map"), dict):
            raise ValueError(f"checkpoint index at {url} has no weight_map")
        temporary = index_path.with_name(index_path.name + ".tmp")
        temporary.write_bytes(raw)
        temporary.replace(index_path)
    with index_path.open(encoding="utf-8") as f:
        parsed = json.load(f)
    weight_map = parsed.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"checkpoint index at {index_path} has no weight_map")
    return weight_map


def get_header(shard):
    if shard in _header_cache:
        return _header_cache[shard]
    url = BASE + shard
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=30) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        hdr = json.loads(r.read(n))
    _header_cache[shard] = (hdr, 8 + n)
    return _header_cache[shard]


def fetch_tensor(shard, tensor_name):
    hdr, base_offset = get_header(shard)
    info = hdr[tensor_name]
    dtype, shape, (start, end) = info["dtype"], info["shape"], info["data_offsets"]
    assert dtype == "BF16", dtype
    url = BASE + shard
    lo, hi = base_offset + start, base_offset + end - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    expected_bytes = end - start
    if len(raw) != expected_bytes:
        raise IOError(
            f"range fetch for {tensor_name} returned {len(raw)} bytes; "
            f"expected {expected_bytes}"
        )
    shape_bytes = int(np.prod(shape, dtype=np.int64)) * 2
    if expected_bytes != shape_bytes:
        raise ValueError(
            f"header byte count for {tensor_name} is {expected_bytes}, "
            f"but BF16 shape {shape} requires {shape_bytes}"
        )
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
    return t.float().numpy()


def main():
    weight_map = load_weight_map()
    local_layer_ids = set(
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25,
         26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49,
         50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64]
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {}
    for i in range(NUM_LAYERS):
        is_local = i in local_layer_ids
        wr_key = f"model.llm.layers.{i}.attn.wr_du.weight"
        proj_key = f"model.llm.layers.{i}.attn.rel_logits_proj.proj"
        wr_shard = weight_map[wr_key]
        proj_shard = weight_map[proj_key]

        wr = fetch_tensor(wr_shard, wr_key)          # [num_heads*d_rel, hidden] = [1024, 6144]
        proj = fetch_tensor(proj_shard, proj_key)     # [d_rel, extent] = [16, 512 or 1024]

        np.save(os.path.join(OUT_DIR, f"layer{i:02d}_wr_du.npy"), wr)
        np.save(os.path.join(OUT_DIR, f"layer{i:02d}_rel_logits_proj.npy"), proj)
        meta[i] = {
            "is_local": is_local,
            "extent": proj.shape[1],
            "wr_du_shape": list(wr.shape),
            "proj_shape": list(proj.shape),
        }
        print(f"layer {i:02d} ({'local' if is_local else 'global'}): "
              f"wr_du {wr.shape}, proj {proj.shape}")

    json.dump(meta, open(os.path.join(OUT_DIR, "_meta.json"), "w"), indent=2)
    print("Done. Metadata written to", os.path.join(OUT_DIR, "_meta.json"))


if __name__ == "__main__":
    main()
