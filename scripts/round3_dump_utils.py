"""Shared raw checkpoint range-dump helpers for Round 3 D2/D3."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from extract_rel_attn import BASE, INDEX_PATH, fetch_tensor, get_header


ROOT = Path(__file__).resolve().parents[1]
INDEX_CACHE = ROOT / "dumps" / "round3" / "model.safetensors.index.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weight_map() -> dict[str, str]:
    """Load the index from a durable cache, prior cache, or the Hub."""
    INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_CACHE.exists():
        index_path = INDEX_CACHE
    elif os.environ.get("INKLING_INDEX_PATH") and Path(
        os.environ["INKLING_INDEX_PATH"]
    ).exists():
        index_path = Path(os.environ["INKLING_INDEX_PATH"])
        shutil.copyfile(index_path, INDEX_CACHE)
        index_path = INDEX_CACHE
    elif Path(INDEX_PATH).exists():
        shutil.copyfile(INDEX_PATH, INDEX_CACHE)
        index_path = INDEX_CACHE
    else:
        url = BASE + "model.safetensors.index.json"
        print(f"downloading index: {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            INDEX_CACHE.write_bytes(response.read())
        index_path = INDEX_CACHE
    with index_path.open(encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def inspect_all(tasks: list[dict[str, object]], workers: int = 8) -> None:
    """Populate source dtype/shape for every task before any tensor fetch."""
    by_shard: dict[str, list[dict[str, object]]] = defaultdict(list)
    for task in tasks:
        by_shard[str(task["shard"])].append(task)

    def inspect_shard(shard: str) -> tuple[str, dict[str, object]]:
        header, _ = get_header(shard)
        return shard, header

    headers: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect_shard, shard): shard for shard in by_shard}
        for future in as_completed(futures):
            shard, header = future.result()
            headers[shard] = header
            print(f"inspected header {shard}")

    for task in tasks:
        shard = str(task["shard"])
        key = str(task["tensor_name"])
        info = headers[shard][key]
        task["source_dtype"] = str(info["dtype"])
        task["source_shape"] = [int(v) for v in info["shape"]]
        task["data_offsets"] = [int(v) for v in info["data_offsets"]]


def dump_tasks(
    tasks: list[dict[str, object]], out_dir: Path, workers: int = 8
) -> dict[str, dict[str, object]]:
    """Fetch missing raw tensors, preserving checkpoint shape, then describe them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list[dict[str, object]]] = defaultdict(list)
    for task in tasks:
        path = out_dir / str(task["file"])
        expected_shape = tuple(int(v) for v in task["source_shape"])
        valid_existing = False
        if path.exists():
            try:
                existing = np.load(path, mmap_mode="r")
                valid_existing = existing.shape == expected_shape and existing.dtype == np.float32
            except Exception:
                valid_existing = False
        if not valid_existing:
            by_shard[str(task["shard"])].append(task)
        else:
            print(f"reuse {path.name}: shape={expected_shape}")

    def fetch_shard(shard: str, shard_tasks: list[dict[str, object]]) -> list[str]:
        completed = []
        for task in shard_tasks:
            key = str(task["tensor_name"])
            array = fetch_tensor(shard, key)
            expected_shape = tuple(int(v) for v in task["source_shape"])
            if array.shape != expected_shape:
                raise ValueError(f"{key}: fetched {array.shape}, header says {expected_shape}")
            np.save(out_dir / str(task["file"]), array)
            completed.append(str(task["file"]))
        return completed

    if by_shard:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_shard, shard, shard_tasks): shard
                for shard, shard_tasks in by_shard.items()
            }
            for future in as_completed(futures):
                shard = futures[future]
                for name in future.result():
                    print(f"fetched {name} from {shard}")

    meta: dict[str, dict[str, object]] = {}
    for task in tasks:
        path = out_dir / str(task["file"])
        array = np.load(path, mmap_mode="r")
        meta[str(task["file"])] = {
            "tensor_name": str(task["tensor_name"]),
            "shard": str(task["shard"]),
            "source_shape": [int(v) for v in task["source_shape"]],
            "source_dtype": str(task["source_dtype"]),
            "saved_shape": [int(v) for v in array.shape],
            "saved_dtype": str(array.dtype),
            "storage_note": "BF16 values losslessly upcast to float32 for NumPy",
            "sha256": sha256(path),
            "bytes": int(path.stat().st_size),
        }
    return meta
