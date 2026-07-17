"""Minimal registered GPU capture for the two private corpus-v2 arms.

Runs the validated streaming Inkling forward over all 66 layers and stores only
the registered r-vectors plus per-target-token NLL. Private inputs and raw dumps
remain gitignored. This script deliberately does not compute registered means or
aperture readouts.
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
)


NVFP4 = ROOT / "nvfp4"
CORPUS = ROOT / "corpus_v2"
TOKENIZER = ROOT / "corpus" / "tokenizer.json"
DEFAULT_OUT = ROOT / "dumps" / "round5" / "corpus_v2_capture"
TEXTS = ["07_slack_human", "08_math_llm"]
LAYERS = list(range(66))
SEQ = 8192
REGISTRATION_COMMIT = "7fb84ab4e5d164bc759c32e38b0e58803abd5299"
PUBLIC_BOUNDARY_COMMIT = "65b220c2d185829dfc4c8e617a67e673d2fa9cd2"


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


def verified_ids(name: str, private_manifest: dict[str, Any]) -> np.ndarray:
    path = CORPUS / f"{name}.ids.npy"
    if sha256_file(path) != private_manifest["texts"][name]["ids_sha256"]:
        raise RuntimeError(f"{name} ID hash mismatch")
    ids = np.load(path, allow_pickle=False)
    if ids.shape != (SEQ,) or ids.dtype != np.int32:
        raise RuntimeError(f"{name} IDs have unexpected shape/dtype")
    return ids


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


def provenance(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    public_sources = [
        "corpus_v2_capture.py",
        "tier2_run.py",
        "tier2_stream.py",
        "tier2_nvfp4.py",
    ]
    return {
        "registration_commit": REGISTRATION_COMMIT,
        "public_boundary_commit": PUBLIC_BOUNDARY_COMMIT,
        "git_head": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "spec_sha256": sha256_file(ROOT / "CORPUS_V2_SPEC.md"),
        "amendment_a1_sha256": sha256_file(ROOT / "CORPUS_V2_AMENDMENT_A1.md"),
        "execution_plan_sha256": sha256_file(ROOT / "CORPUS_V2_EXECUTION_PLAN.md"),
        "private_manifest_sha256": sha256_file(CORPUS / "manifest.json"),
        "tokenizer_sha256": sha256_file(TOKENIZER),
        "checkpoint_index_sha256": sha256_file(NVFP4 / "model.safetensors.index.json"),
        "config_sha256": sha256_file(NVFP4 / "config.json"),
        "source_sha256": {
            name: sha256_file(SCRIPT_DIR / name) for name in public_sources
        },
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
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
    private_manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    if private_manifest.get("private") is not True or private_manifest.get("seq") != SEQ:
        raise AssertionError("invalid private corpus manifest")
    for name in TEXTS:
        verified_ids(name, private_manifest)
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

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "corpus_v2_registered_capture",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "production_capture": True,
        "artifacts": [],
    }
    manifest.update(provenance(config, args))
    atomic_json(manifest_path, manifest)
    started = time.time()

    try:
        private_manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
        if private_manifest.get("private") is not True or private_manifest.get("seq") != SEQ:
            raise RuntimeError("invalid private corpus-v2 manifest")
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
            ids = verified_ids(name, private_manifest)
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

        if len(manifest["artifacts"]) != 134:
            raise RuntimeError(f"unexpected artifact count: {len(manifest['artifacts'])}")
        manifest["complete"] = True
        manifest["artifact_count"] = len(manifest["artifacts"])
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_seconds"] = round(time.time() - started, 3)
        atomic_json(manifest_path, manifest)
        print(f"DONE 134 artifacts in {manifest['wall_seconds'] / 60:.2f} min", flush=True)
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
