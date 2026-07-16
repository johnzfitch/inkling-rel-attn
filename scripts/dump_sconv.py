"""Round 3 D2: fetch every trunk and MTP SConv tensor, with no analysis."""
from __future__ import annotations

import json
from pathlib import Path

from round3_dump_utils import ROOT, dump_tasks, inspect_all, load_weight_map


OUT_DIR = ROOT / "dumps" / "round3" / "sconv"
GROUPS = (
    ("k_sconv", "attn.k_sconv.weight"),
    ("v_sconv", "attn.v_sconv.weight"),
    ("attn_sconv", "attn_sconv.weight"),
    ("mlp_sconv", "mlp_sconv.weight"),
)


def build_tasks(weight_map: dict[str, str]) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for layer in range(66):
        for short, suffix in GROUPS:
            key = f"model.llm.layers.{layer}.{suffix}"
            if key not in weight_map:
                raise KeyError(key)
            tasks.append(
                {
                    "file": f"layer{layer:02d}_{short}.npy",
                    "tensor_name": key,
                    "shard": weight_map[key],
                    "kind": "trunk",
                    "layer": layer,
                    "group": short,
                }
            )
    for layer in range(8):
        for short, suffix in GROUPS:
            key = f"model.mtp.layers.{layer}.transformer_block.{suffix}"
            if key not in weight_map:
                raise KeyError(key)
            tasks.append(
                {
                    "file": f"mtp{layer}_{short}.npy",
                    "tensor_name": key,
                    "shard": weight_map[key],
                    "kind": "mtp",
                    "layer": layer,
                    "group": short,
                }
            )
    return tasks


def main() -> None:
    weight_map = load_weight_map()
    tasks = build_tasks(weight_map)
    if len(tasks) != (66 + 8) * 4:
        raise AssertionError(f"expected 296 tensors, got {len(tasks)}")

    # Binding order: all exact names and shapes are resolved from index+headers
    # before the first raw tensor payload is requested.
    inspect_all(tasks)
    print("all 296 SConv names and shapes inspected; starting raw fetch")
    files = dump_tasks(tasks, OUT_DIR)
    meta = {
        "tensor_count": len(tasks),
        "header_shapes_checked_before_fetch": True,
        "derived_quantities": None,
        "files": files,
    }
    meta_path = OUT_DIR / "_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Written {meta_path}")


if __name__ == "__main__":
    main()
