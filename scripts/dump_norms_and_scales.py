"""Round 3 D3: fetch all registered norm/scale tensors, with no analysis."""
from __future__ import annotations

import json

from round3_dump_utils import ROOT, dump_tasks, inspect_all, load_weight_map


OUT_DIR = ROOT / "dumps" / "round3" / "norms"
GROUPS = (
    ("q_norm", "attn.q_norm.weight"),
    ("k_norm", "attn.k_norm.weight"),
    ("attn_norm", "attn_norm.weight"),
    ("mlp_norm", "mlp_norm.weight"),
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
    inspect_all(tasks)
    print("all 296 norm/scale names and shapes inspected; starting raw fetch")
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
