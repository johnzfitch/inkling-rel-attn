"""A6-corrected GPU capture for corpus v2.0, v2.1, and the prose arm.

Runs the validated streaming Inkling forward over all 66 layers and stores only
the registered r-vectors plus per-target-token NLL. The fresh output is separate
from the provisional v2.0 capture. Private inputs and raw dumps remain gitignored.
This script deliberately does not compute registered means or aperture readouts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tokenizers

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import tier2_stream as T  # noqa: E402
from tier2_run import build_layer  # noqa: E402
from tier2_stream import ShardReader, measuring_attention  # noqa: E402

from transformers import AutoConfig  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.inkling.modeling_inkling import (  # noqa: E402
    InklingRMSNorm,
    InklingRelativeLogits,
    eager_attention_forward,
)


NVFP4 = ROOT / "nvfp4"
CORPUS_V2 = ROOT / "corpus_v2"
CORPUS_V1 = ROOT / "corpus"
TOKENIZER = ROOT / "corpus" / "tokenizer.json"
DEFAULT_OUT = ROOT / "dumps" / "round5" / "corpus_v2_corrected_capture"
TEXTS = ["07_slack_human", "08_math_llm", "07b_slack_multi", "01_prose_en"]
TEXT_ROOT = {
    "07_slack_human": CORPUS_V2,
    "08_math_llm": CORPUS_V2,
    "07b_slack_multi": CORPUS_V2,
    "01_prose_en": CORPUS_V1,
}
LAYERS = list(range(66))
SEQ = 8192
REGISTRATION_COMMIT = "7fb84ab4e5d164bc759c32e38b0e58803abd5299"
A6_COMMIT = "061bb04dffef05eae33f9fffc430394faa4052c5"
PUBLIC_BOUNDARY_COMMIT = A6_COMMIT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_output(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def artifact_record(
    out_root: Path,
    path: Path,
    *,
    kind: str,
    dtype: str,
    shape: tuple[int, ...],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(out_root).as_posix(),
        "kind": kind,
        "dtype": dtype,
        "shape": [int(value) for value in shape],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if extra:
        record.update(extra)
    return record


def load_input_manifests() -> dict[Path, dict[str, Any]]:
    manifests = {
        root: json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for root in sorted(set(TEXT_ROOT.values()), key=str)
    }
    if manifests[CORPUS_V2].get("private") is not True:
        raise RuntimeError("invalid private corpus-v2 manifest")
    for root, manifest in manifests.items():
        if manifest.get("seq") != SEQ:
            raise RuntimeError(f"invalid sequence length in {root / 'manifest.json'}")
    return manifests


def verified_ids(name: str, manifests: dict[Path, dict[str, Any]]) -> np.ndarray:
    root = TEXT_ROOT[name]
    path = root / f"{name}.ids.npy"
    expected = manifests[root]["texts"][name]["ids_sha256"]
    if sha256_file(path) != expected:
        raise RuntimeError(f"{name} ID hash mismatch")
    ids = np.load(path, allow_pickle=False)
    if ids.shape != (SEQ,) or ids.dtype != np.int32:
        raise RuntimeError(f"{name} IDs have unexpected shape/dtype")
    return ids


def package_record(module: Any) -> dict[str, str]:
    return {
        "version": str(module.__version__),
        "module_path": str(Path(module.__file__).resolve()),
    }


def checkpoint_shard_records() -> dict[str, dict[str, Any]]:
    index = json.loads((NVFP4 / "model.safetensors.index.json").read_text(encoding="utf-8"))
    indexed_files = sorted(set(index["weight_map"].values()))
    shards = [name for name in indexed_files if name.startswith("model-")]
    nontrunk = [name for name in indexed_files if not name.startswith("model-")]
    if len(shards) != 33 or nontrunk != ["mtp.safetensors"]:
        raise RuntimeError(
            f"unexpected checkpoint index inventory: trunk={len(shards)}, nontrunk={nontrunk}"
        )
    # The directory also contains the unused MTP checkpoint; production trunk
    # provenance is exactly the 33 model shards referenced by the index.
    on_disk = sorted(p.name for p in NVFP4.glob("model-*-of-00033.safetensors"))
    if shards != on_disk:
        raise RuntimeError(
            f"index/directory shard mismatch: {len(shards)} in index, {len(on_disk)} on disk; "
            f"only-index={sorted(set(shards) - set(on_disk))} "
            f"only-disk={sorted(set(on_disk) - set(shards))}"
        )
    records: dict[str, dict[str, Any]] = {}
    for ordinal, name in enumerate(shards, start=1):
        path = NVFP4 / name
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"hashing checkpoint shard {ordinal:02d}/{len(shards)}: {name}", flush=True)
        records[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return records


def stock_attention_parity_gate() -> dict[str, Any]:
    """Bitwise-check the production compact path against stock eager BF16 math."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the stock attention parity gate")

    class Dummy:
        num_key_value_groups = 2
        training = False

    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xA6)
    cases: dict[str, Any] = {}
    prior_active = dict(T._ACTIVE)
    prior_capture = dict(T._CAPTURE)
    try:
        for label, sliding, window, extent in (
            ("global", False, 64, 32),
            ("sliding", True, 8, 8),
        ):
            heads, kv_heads, seq, dim = 4, 2, 17, 8
            query = torch.randn(
                1, heads, seq, dim, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            key = torch.randn(
                1, kv_heads, seq, dim, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            value = torch.randn(
                1, kv_heads, seq, dim, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            compact = torch.randn(
                1, heads, seq, extent, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            qpos = torch.arange(seq, device="cuda")
            kpos = torch.arange(seq, device="cuda")
            distance = qpos[:, None] - kpos[None, :]
            in_extent = (distance >= 0) & (distance < extent)
            gather_index = distance.clamp(0, extent - 1)[None, None].expand(1, heads, -1, -1)
            dense_bias = torch.gather(compact, 3, gather_index).masked_fill(
                ~in_extent[None, None], 0.0
            )
            valid = distance >= 0
            if sliding:
                valid &= distance < window
            mask = torch.zeros(
                1, 1, seq, seq, device="cuda", dtype=torch.bfloat16
            ).masked_fill(~valid[None, None], torch.finfo(torch.bfloat16).min)
            scaling = 1.0 / dim
            T._ACTIVE.update(
                meter=None,
                sliding=sliding,
                window=window,
                qchunk=7,
                needle_qpos=None,
                needle_rows=None,
            )
            T._CAPTURE["enabled"] = False
            measured, _ = measuring_attention(
                Dummy(), query, key, value, None, scaling, position_bias=compact
            )
            stock, _ = eager_attention_forward(
                Dummy(), query, key, value, mask, scaling, position_bias=dense_bias
            )
            equal = bool(torch.equal(measured, stock))
            max_delta = float((measured.float() - stock.float()).abs().max().item())
            cases[label] = {
                "bitwise_equal": equal,
                "max_output_delta": max_delta,
                "shape": list(measured.shape),
            }
            if not equal:
                raise RuntimeError(f"stock attention parity failed for {label}: {max_delta}")
    finally:
        T._ACTIVE.clear()
        T._ACTIVE.update(prior_active)
        T._CAPTURE.clear()
        T._CAPTURE.update(prior_capture)
    print("stock attention parity passed: global + sliding bitwise equal", flush=True)
    return {"passed": True, "cases": cases}


@torch.no_grad()
def compute_nll(
    hidden: torch.Tensor,
    ids: np.ndarray,
    final_norm: InklingRMSNorm,
    unembed: torch.Tensor,
    *,
    mup_multiplier: float,
    unpadded_vocab: int,
    token_chunk: int,
) -> np.ndarray:
    states = final_norm(hidden) / mup_multiplier
    targets = torch.from_numpy(ids[1:].astype(np.int64)).to(states.device)
    if int(targets.max()) >= unpadded_vocab:
        raise RuntimeError("target token exceeds unpadded vocabulary")
    losses: list[torch.Tensor] = []
    for start in range(0, len(ids) - 1, token_chunk):
        stop = min(start + token_chunk, len(ids) - 1)
        logits = torch.nn.functional.linear(states[:, start:stop, :], unembed)[0]
        logits32 = logits[:, :unpadded_vocab].float()
        target_logits = logits32.gather(1, targets[start:stop, None])[:, 0]
        losses.append((torch.logsumexp(logits32, dim=-1) - target_logits).cpu())
        del logits, logits32, target_logits
    result = torch.cat(losses).numpy().astype(np.float32, copy=False)
    if result.shape != (SEQ - 1,) or not np.isfinite(result).all():
        raise RuntimeError("invalid NLL output")
    return result


def provenance(
    config: Any,
    args: argparse.Namespace,
    manifests: dict[Path, dict[str, Any]],
    shard_records: dict[str, dict[str, Any]],
    parity: dict[str, Any],
) -> dict[str, Any]:
    import transformers

    public_sources = [
        "corpus_v2_capture.py",
        "tier2_run.py",
        "tier2_stream.py",
        "tier2_nvfp4.py",
    ]
    return {
        "registration_commit": REGISTRATION_COMMIT,
        "amendment_a6_commit": A6_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        "git_head": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "spec_sha256": sha256_file(ROOT / "registrations" / "CORPUS_V2_SPEC.md"),
        "amendment_a1_sha256": sha256_file(ROOT / "registrations" / "CORPUS_V2_AMENDMENT_A1.md"),
        "amendment_a6_sha256": sha256_file(ROOT / "registrations" / "ROUND5_AMENDMENT_A6.md"),
        "amendment_a7_execution_sha256": sha256_file(
            ROOT / "registrations" / "ROUND5_AMENDMENT_A7_EXECUTION.md"
        ),
        "amendment_a7_execution_commit": git_output(
            "log", "-1", "--follow", "--format=%H", "--",
            "registrations/ROUND5_AMENDMENT_A7_EXECUTION.md"
        ),
        "depth_prereg_sha256": sha256_file(
            ROOT / "registrations" / "ROUND5_DEPTH_RESOLVED_PREREG.md"
        ),
        "execution_plan_sha256": sha256_file(
            ROOT / "registrations" / "CORPUS_V2_EXECUTION_PLAN.md"
        ),
        "input_manifest_sha256": {
            root.relative_to(ROOT).as_posix(): sha256_file(root / "manifest.json")
            for root in manifests
        },
        "input_ids_sha256": {
            name: sha256_file(TEXT_ROOT[name] / f"{name}.ids.npy") for name in TEXTS
        },
        "tokenizer_sha256": sha256_file(TOKENIZER),
        "checkpoint_index_sha256": sha256_file(NVFP4 / "model.safetensors.index.json"),
        "checkpoint_shards": shard_records,
        "checkpoint_index_nontrunk_files": ["mtp.safetensors"],
        "config_sha256": sha256_file(NVFP4 / "config.json"),
        "source_sha256": {
            name: sha256_file(SCRIPT_DIR / name) for name in public_sources
        },
        "packages": {
            "numpy": package_record(np),
            "tokenizers": package_record(tokenizers),
            "torch": package_record(torch),
            "transformers": package_record(transformers),
        },
        "modeling_inkling_sha256": sha256_file(
            Path(sys.modules[InklingRMSNorm.__module__].__file__).resolve()
        ),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "stock_attention_parity": parity,
        "attention_dtype_boundary": "BF16 content+bias add, then FP32 softmax",
        "texts": TEXTS,
        "layers": LAYERS,
        "seq": SEQ,
        "qchunk": args.qchunk,
        "nll_token_chunk": args.nll_token_chunk,
        "normalized_inputs_captured": False,
        "attention_meter_enabled": False,
        "model": {
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "num_attention_heads": int(config.num_attention_heads),
            "unpadded_vocab_size": int(config.unpadded_vocab_size),
            "logits_mup_width_multiplier": float(config.logits_mup_width_multiplier),
        },
    }


def self_test() -> None:
    manifests = load_input_manifests()
    for name in TEXTS:
        verified_ids(name, manifests)
    test = np.arange(17, dtype=np.float16)
    if not np.isfinite(test).all():
        raise AssertionError("unexpected NumPy failure")
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qchunk", type=int, default=512)
    parser.add_argument("--nll-token-chunk", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for corpus-v2 capture")
    if args.qchunk != 512:
        raise ValueError("registered qchunk is 512")

    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    torch.set_grad_enabled(False)

    config = AutoConfig.from_pretrained(NVFP4).text_config
    if int(config.num_hidden_layers) != len(LAYERS):
        raise RuntimeError("unexpected checkpoint layer count")
    ALL_ATTENTION_FUNCTIONS.register("tier2_measure", measuring_attention)
    config._attn_implementation = "tier2_measure"
    InklingRelativeLogits.forward = T.compact_relative_logits_forward
    input_manifests = load_input_manifests()
    parity = stock_attention_parity_gate()
    shard_records = checkpoint_shard_records()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "corpus_v2_a6_corrected_capture",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "production_capture": True,
        "artifacts": [],
    }
    manifest.update(provenance(config, args, input_manifests, shard_records, parity))
    atomic_json(manifest_path, manifest)
    started = time.time()

    try:
        reader = ShardReader(str(NVFP4))
        embed_weight = reader.get("model.llm.embed.weight", "cuda").to(torch.bfloat16)
        embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        embed_norm.weight = torch.nn.Parameter(
            reader.get("model.llm.embed_norm.weight", "cuda").to(torch.bfloat16),
            requires_grad=False,
        )
        embed_norm.eval()

        ids_by_text: dict[str, np.ndarray] = {}
        hidden: dict[str, torch.Tensor] = {}
        for name in TEXTS:
            ids = verified_ids(name, input_manifests)
            ids_by_text[name] = ids
            ids_cuda = torch.from_numpy(ids.astype(np.int64)).to("cuda")
            hidden[name] = embed_norm(
                torch.nn.functional.embedding(ids_cuda, embed_weight).unsqueeze(0)
            )
        del embed_weight, embed_norm
        torch.cuda.empty_cache()
        print(f"embedded {len(TEXTS)} private arms at seq={SEQ}", flush=True)

        T._CAPTURE["enabled"] = True
        for layer_index in LAYERS:
            layer_started = time.time()
            layer = build_layer(config, layer_index, reader, "cuda")
            is_sliding = config.layer_types[layer_index] == "hybrid_sliding"
            for name in TEXTS:
                T._ACTIVE.update(
                    meter=None,
                    sliding=is_sliding,
                    window=config.sliding_window_size,
                    qchunk=args.qchunk,
                    needle_qpos=None,
                    needle_rows=None,
                )
                T._CAPTURE["rvec"] = None
                hidden[name] = layer(
                    hidden[name],
                    attention_mask=None,
                    conv_mask=None,
                    past_key_values=None,
                )
                captured = T._CAPTURE["rvec"]
                if captured is None:
                    raise RuntimeError(f"missing r-vector at L{layer_index:02d}, {name}")
                rvec = captured.numpy()
                if rvec.shape != (SEQ, 64, 16) or rvec.dtype != np.float16:
                    raise RuntimeError(f"invalid r-vector at L{layer_index:02d}, {name}")
                if not np.isfinite(rvec).all():
                    raise RuntimeError(f"non-finite r-vector at L{layer_index:02d}, {name}")
                path = out_root / "rvec" / f"rvec_L{layer_index:02d}_{name}.npy"
                atomic_npy(path, rvec)
                manifest["artifacts"].append(
                    artifact_record(
                        out_root,
                        path,
                        kind="rvec",
                        dtype="float16",
                        shape=rvec.shape,
                        extra={"layer": layer_index, "text": name},
                    )
                )
                T._CAPTURE["rvec"] = None
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            manifest["last_completed_layer"] = layer_index
            manifest["elapsed_seconds"] = round(time.time() - started, 3)
            atomic_json(manifest_path, manifest)
            print(
                f"layer {layer_index:02d} {time.time() - layer_started:.1f}s",
                flush=True,
            )

        T._CAPTURE["enabled"] = False
        T._ACTIVE["meter"] = None
        final_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        final_norm.weight = torch.nn.Parameter(
            reader.get("model.llm.norm.weight", "cuda").to(torch.bfloat16),
            requires_grad=False,
        )
        final_norm.eval()
        unembed = reader.get("model.llm.unembed.weight", "cuda").to(torch.bfloat16)
        for name in TEXTS:
            nll = compute_nll(
                hidden[name],
                ids_by_text[name],
                final_norm,
                unembed,
                mup_multiplier=float(config.logits_mup_width_multiplier),
                unpadded_vocab=int(config.unpadded_vocab_size),
                token_chunk=args.nll_token_chunk,
            )
            path = out_root / "nll" / f"nll_{name}.npz"
            atomic_npz(
                path,
                target_position=np.arange(1, SEQ, dtype=np.int32),
                target_id=ids_by_text[name][1:].astype(np.int32),
                nll=nll,
            )
            manifest["artifacts"].append(
                artifact_record(
                    out_root,
                    path,
                    kind="next_token_nll",
                    dtype="target_position:int32,target_id:int32,nll:float32",
                    shape=nll.shape,
                    extra={"text": name},
                )
            )
            print(f"captured NLL {name}: {len(nll)} finite values", flush=True)
        del unembed, final_norm

        expected_artifacts = len(TEXTS) * (len(LAYERS) + 1)
        if len(manifest["artifacts"]) != expected_artifacts:
            raise RuntimeError(f"unexpected artifact count: {len(manifest['artifacts'])}")
        manifest["complete"] = True
        manifest["artifact_count"] = len(manifest["artifacts"])
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_seconds"] = round(time.time() - started, 3)
        atomic_json(manifest_path, manifest)
        print(
            f"DONE {expected_artifacts} artifacts in {manifest['wall_seconds'] / 60:.2f} min",
            flush=True,
        )
    except Exception as exc:
        T._CAPTURE["enabled"] = False
        T._CAPTURE["rvec"] = None
        T._ACTIVE["meter"] = None
        manifest["complete"] = False
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = repr(exc)
        manifest["traceback"] = traceback.format_exc()
        manifest["wall_seconds"] = round(time.time() - started, 3)
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
