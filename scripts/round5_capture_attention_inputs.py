"""Round-5 capture amendment: lossless attention inputs and integrity meters.

This repeats the validated Tier-2 forward without changing model outputs. It
captures the actual BF16 output of each layer's input RMSNorm as uint16 bits,
replays r-vectors/needle rows, inventories massive residual coordinates, and
computes a true next-token NLL baseline after the final layer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
from tier2_stream import GLOBAL_LAYERS, Meter, ShardReader, measuring_attention  # noqa: E402

from transformers import AutoConfig  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.inkling.modeling_inkling import (  # noqa: E402
    InklingRMSNorm,
    InklingRelativeLogits,
)


NVFP4_DIR = ROOT / "nvfp4"
CORPUS = ROOT / "corpus"
OLD_CAPTURE = ROOT / "dumps" / "tier2" / "capture"
DEFAULT_OUT = ROOT / "dumps" / "round5" / "attention_inputs"
TEXTS = [
    "01_prose_en",
    "02_code",
    "03_templated",
    "04_multilingual",
    "05_needles",
    "06_random",
]
REGISTRATION_COMMIT = "34278b4"
PLAN_COMMIT = "d4e2579"
MASSIVE_THRESHOLD = 30000.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:  # provenance must not break a capture
        return f"unavailable: {exc}"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def artifact_record(
    out_root: Path,
    path: Path,
    *,
    kind: str,
    logical_dtype: str | None = None,
    shape: tuple[int, ...] | list[int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(out_root).as_posix(),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if logical_dtype is not None:
        record["logical_dtype"] = logical_dtype
    if shape is not None:
        record["shape"] = [int(x) for x in shape]
    if extra:
        record.update(extra)
    return record


def verify_ids(name: str, seq: int) -> np.ndarray:
    path = CORPUS / f"{name}.ids.npy"
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["texts"][name]["ids_sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} ID hash mismatch: {actual} != {expected}")
    ids = np.load(path)
    if seq > len(ids):
        raise ValueError(f"requested seq={seq} exceeds {name} length={len(ids)}")
    return np.asarray(ids[:seq], dtype=np.int64)


def bf16_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 capture, got {tensor.dtype}")
    cpu = tensor.detach().contiguous().to("cpu")
    return cpu.view(torch.uint16).numpy().copy()


def uint16_to_bf16_float32(array: np.ndarray) -> torch.Tensor:
    if array.dtype != np.uint16:
        raise TypeError(array.dtype)
    return torch.from_numpy(np.ascontiguousarray(array)).view(torch.bfloat16).float()


def bitwise_equal_fp16(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, int]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False, max(actual.size, expected.size)
    if actual.dtype != np.float16:
        raise TypeError(f"bitwise replay expects float16, got {actual.dtype}")
    mismatch = int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
    return mismatch == 0, mismatch


def save_bf16_capture(
    out_root: Path,
    path: Path,
    tensor: torch.Tensor,
    *,
    layer: int,
    text: str,
) -> dict[str, Any]:
    payload = bf16_to_uint16(tensor)
    decoded = uint16_to_bf16_float32(payload)
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError(f"non-finite normalized input at layer {layer}, text {text}")
    atomic_npy(path, payload)
    return artifact_record(
        out_root,
        path,
        kind="normalized_attention_input",
        logical_dtype="bfloat16 (uint16 bit payload)",
        shape=payload.shape,
        extra={"physical_dtype": "uint16", "layer": layer, "text": text},
    )


def save_massive_census(
    out_root: Path,
    path: Path,
    hidden: torch.Tensor,
    *,
    layer: int,
    text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = hidden.detach()[0]
    indices = torch.nonzero(state.float().abs() > MASSIVE_THRESHOLD, as_tuple=False)
    if len(indices):
        positions = indices[:, 0].to("cpu", torch.int32).numpy()
        channels = indices[:, 1].to("cpu", torch.int32).numpy()
        values = state[indices[:, 0], indices[:, 1]].float().to("cpu").numpy()
        max_abs = float(np.max(np.abs(values)))
    else:
        positions = np.empty(0, dtype=np.int32)
        channels = np.empty(0, dtype=np.int32)
        values = np.empty(0, dtype=np.float32)
        max_abs = 0.0
    atomic_npz(path, position=positions, channel=channels, value=values)
    record = artifact_record(
        out_root,
        path,
        kind="massive_activation_census",
        logical_dtype="(int32, int32, float32)",
        shape=(len(values),),
        extra={
            "layer": layer,
            "text": text,
            "threshold_abs_gt": MASSIVE_THRESHOLD,
            "count": int(len(values)),
            "max_abs": max_abs,
        },
    )
    return record, {"count": int(len(values)), "max_abs": max_abs}


def save_replay_array(
    out_root: Path,
    path: Path,
    array: np.ndarray,
    *,
    kind: str,
    layer: int,
    text: str | None,
    original: Path | None,
    verify: bool,
) -> dict[str, Any]:
    array = np.asarray(array)
    if array.dtype != np.float16:
        raise TypeError(f"{kind} replay must be float16, got {array.dtype}")
    mismatch = None
    if verify:
        if original is None or not original.exists():
            raise FileNotFoundError(original)
        expected = np.load(original, mmap_mode="r")
        equal, mismatch = bitwise_equal_fp16(array, expected)
        if not equal:
            raise RuntimeError(
                f"{kind} replay mismatch at layer={layer}, text={text}: {mismatch} FP16 words"
            )
    atomic_npy(path, array)
    return artifact_record(
        out_root,
        path,
        kind=kind,
        logical_dtype="float16",
        shape=array.shape,
        extra={
            "layer": layer,
            "text": text,
            "original_path": str(original) if original is not None else None,
            "bitwise_verified": bool(verify),
            "mismatch_words": mismatch,
        },
    )


def save_replay_rows(
    out_root: Path,
    path: Path,
    qpos: np.ndarray,
    rows: np.ndarray,
    *,
    layer: int,
    original: Path | None,
    verify: bool,
) -> dict[str, Any]:
    qpos = np.asarray(qpos, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.float16)
    mismatch = None
    if verify:
        if original is None or not original.exists():
            raise FileNotFoundError(original)
        with np.load(original) as expected:
            if not np.array_equal(qpos, expected["qpos"]):
                raise RuntimeError(f"needle qpos replay mismatch at layer {layer}")
            equal, mismatch = bitwise_equal_fp16(rows, expected["rows"])
        if not equal:
            raise RuntimeError(f"needle-row replay mismatch at layer {layer}: {mismatch} words")
    atomic_npz(path, qpos=qpos, rows=rows)
    return artifact_record(
        out_root,
        path,
        kind="needle_rows_replay",
        logical_dtype="qpos:int64, rows:float16",
        shape=rows.shape,
        extra={
            "layer": layer,
            "original_path": str(original) if original is not None else None,
            "bitwise_verified": bool(verify),
            "mismatch_words": mismatch,
        },
    )


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
    losses: list[torch.Tensor] = []
    targets = torch.from_numpy(ids[1:].astype(np.int64)).to(states.device)
    if int(targets.max()) >= unpadded_vocab:
        raise RuntimeError(
            f"target ID {int(targets.max())} exceeds unpadded vocab {unpadded_vocab}"
        )
    for start in range(0, len(ids) - 1, token_chunk):
        end = min(start + token_chunk, len(ids) - 1)
        logits = torch.nn.functional.linear(states[:, start:end, :], unembed)[0]
        logits = logits[:, :unpadded_vocab]
        logits32 = logits.float()
        target = targets[start:end]
        target_logits = logits32.gather(1, target[:, None])[:, 0]
        nll = torch.logsumexp(logits32, dim=-1) - target_logits
        losses.append(nll.to("cpu"))
        del logits, logits32, target_logits, nll
    result = torch.cat(losses).numpy().astype(np.float32, copy=False)
    if result.shape != (len(ids) - 1,) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid NLL result: shape={result.shape}, finite={np.isfinite(result).all()}")
    return result


def provenance(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    source_files = [
        "round5_capture_attention_inputs.py",
        "tier2_run.py",
        "tier2_stream.py",
        "tier2_nvfp4.py",
    ]
    return {
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "git_head": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "checkpoint_index_sha256": sha256_file(NVFP4_DIR / "model.safetensors.index.json"),
        "config_sha256": sha256_file(NVFP4_DIR / "config.json"),
        "tokenizer_sha256": sha256_file(CORPUS / "tokenizer.json"),
        "corpus_manifest_sha256": sha256_file(CORPUS / "manifest.json"),
        "source_sha256": {name: sha256_file(SCRIPT_DIR / name) for name in source_files},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "seq": args.seq,
        "layers": args.layer_ids,
        "texts": args.texts,
        "qchunk": args.qchunk,
        "massive_threshold": MASSIVE_THRESHOLD,
        "nll_token_chunk": args.nll_token_chunk,
        "model": {
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "head_dim": config.head_dim,
            "unpadded_vocab_size": config.unpadded_vocab_size,
            "logits_mup_width_multiplier": config.logits_mup_width_multiplier,
        },
    }


def self_test() -> None:
    original = torch.tensor([0.0, -0.0, 1.0, -2.5, 123.0], dtype=torch.bfloat16)
    payload = bf16_to_uint16(original)
    restored = torch.from_numpy(payload.copy()).view(torch.bfloat16)
    if not torch.equal(original.view(torch.uint16), restored.view(torch.uint16)):
        raise AssertionError("BF16 bit round-trip failed")
    a = np.array([0.0, -0.0, 1.0], dtype=np.float16)
    b = a.copy()
    equal, mismatch = bitwise_equal_fp16(a, b)
    if not equal or mismatch:
        raise AssertionError("FP16 equality self-test failed")
    b.view(np.uint16)[1] = np.float16(0.0).view(np.uint16)
    equal, mismatch = bitwise_equal_fp16(a, b)
    if equal or mismatch != 1:
        raise AssertionError("signed-zero distinction self-test failed")
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--texts", default=",".join(TEXTS))
    parser.add_argument("--qchunk", type=int, default=512)
    parser.add_argument("--nll-token-chunk", type=int, default=256)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-nll", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    args.texts = args.texts.split(",")
    args.layer_ids = (
        list(range(66)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    )
    if args.layer_ids[0] != 0 or args.layer_ids != list(range(args.layer_ids[-1] + 1)):
        raise ValueError("layers must start at zero and be contiguous")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the amended capture")
    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    full_run = args.seq == 8192 and args.layer_ids == list(range(66)) and args.texts == TEXTS
    if full_run and args.skip_nll:
        raise ValueError("the full amendment run may not skip NLL")
    verify_replay = full_run

    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(NVFP4_DIR).text_config
    if config.num_hidden_layers != 66:
        raise RuntimeError(f"unexpected layer count: {config.num_hidden_layers}")
    ALL_ATTENTION_FUNCTIONS.register("tier2_measure", measuring_attention)
    config._attn_implementation = "tier2_measure"
    InklingRelativeLogits.forward = T.compact_relative_logits_forward

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_lossless_attention_input_capture",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "full_registered_capture": full_run,
        "replay_verification_required": verify_replay,
        "artifacts": [],
        "massive_summary": {},
        "nll_summary": {},
    }
    manifest.update(provenance(config, args))
    manifest_path = out_root / "manifest.json"
    atomic_json(manifest_path, manifest)

    start_time = time.time()
    try:
        rd = ShardReader(str(NVFP4_DIR))
        embed_weight = rd.get("model.llm.embed.weight", "cuda").to(torch.bfloat16)
        embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        embed_norm.weight = torch.nn.Parameter(
            rd.get("model.llm.embed_norm.weight", "cuda").to(torch.bfloat16),
            requires_grad=False,
        )
        embed_norm.eval()

        ids_by_text: dict[str, np.ndarray] = {}
        hidden: dict[str, torch.Tensor] = {}
        for name in args.texts:
            ids = verify_ids(name, args.seq)
            ids_by_text[name] = ids
            ids_cuda = torch.from_numpy(ids).to("cuda", torch.long)
            hidden[name] = embed_norm(
                torch.nn.functional.embedding(ids_cuda, embed_weight).unsqueeze(0)
            )
        del embed_weight, embed_norm
        torch.cuda.empty_cache()
        print(f"embedded {len(args.texts)} texts at seq={args.seq}", flush=True)

        sidecar = json.loads((CORPUS / "05_needles.sidecar.json").read_text(encoding="utf-8"))
        needle_qpos = sorted(
            {
                int(entity["token_positions"][1])
                for entity in sidecar["entities"]
                if len(entity.get("token_positions", [])) >= 2
                and int(entity["token_positions"][1]) < args.seq
            }
        )

        for layer_idx in args.layer_ids:
            layer_start = time.time()
            layer = build_layer(config, layer_idx, rd, "cuda")
            is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
            normalized_holder: dict[str, torch.Tensor] = {}

            def capture_norm(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
                if "value" in normalized_holder:
                    raise RuntimeError("input-layernorm hook fired more than once before save")
                normalized_holder["value"] = output.detach()[0].to("cpu")

            hook = layer.input_layernorm.register_forward_hook(capture_norm)
            manifest["massive_summary"][f"L{layer_idx:02d}"] = {}
            try:
                for name in args.texts:
                    normalized_holder.clear()
                    qpos = needle_qpos if name == "05_needles" else None
                    dmax = config.sliding_window_size if is_sliding else args.seq
                    meter = Meter(layer.self_attn.num_heads, dmax, "cuda")
                    T._ACTIVE.update(
                        meter=meter,
                        sliding=is_sliding,
                        window=config.sliding_window_size,
                        qchunk=args.qchunk,
                        needle_qpos=qpos,
                        needle_rows=({} if qpos else None),
                    )
                    T._CAPTURE["enabled"] = True
                    T._CAPTURE["rvec"] = None
                    hidden[name] = layer(
                        hidden[name],
                        attention_mask=None,
                        conv_mask=None,
                        past_key_values=None,
                    )
                    if "value" not in normalized_holder:
                        raise RuntimeError(f"input-layernorm hook did not fire at L{layer_idx}, {name}")

                    norm_path = (
                        out_root / "normalized" / f"attn_in_L{layer_idx:02d}_{name}.npy"
                    )
                    manifest["artifacts"].append(
                        save_bf16_capture(
                            out_root,
                            norm_path,
                            normalized_holder.pop("value"),
                            layer=layer_idx,
                            text=name,
                        )
                    )

                    rvec = T._CAPTURE["rvec"]
                    if rvec is None:
                        raise RuntimeError(f"missing r-vector at L{layer_idx}, {name}")
                    rvec_array = rvec.numpy()
                    rvec_path = out_root / "replay" / f"rvec_L{layer_idx:02d}_{name}.npy"
                    manifest["artifacts"].append(
                        save_replay_array(
                            out_root,
                            rvec_path,
                            rvec_array,
                            kind="rvec_replay",
                            layer=layer_idx,
                            text=name,
                            original=OLD_CAPTURE / f"rvec_L{layer_idx:02d}_{name}.npy",
                            verify=verify_replay,
                        )
                    )

                    massive_path = (
                        out_root / "massive" / f"massive_L{layer_idx:02d}_{name}.npz"
                    )
                    massive_record, massive_summary = save_massive_census(
                        out_root,
                        massive_path,
                        hidden[name],
                        layer=layer_idx,
                        text=name,
                    )
                    manifest["artifacts"].append(massive_record)
                    manifest["massive_summary"][f"L{layer_idx:02d}"][name] = massive_summary

                    if qpos:
                        captured_rows = T._ACTIVE["needle_rows"]
                        if not captured_rows:
                            raise RuntimeError(f"missing needle rows at L{layer_idx}")
                        qs = sorted(captured_rows)
                        rows = np.stack([captured_rows[q].numpy() for q in qs])
                        rows_path = out_root / "replay" / f"needlerows_L{layer_idx:02d}.npz"
                        manifest["artifacts"].append(
                            save_replay_rows(
                                out_root,
                                rows_path,
                                np.asarray(qs, dtype=np.int64),
                                rows,
                                layer=layer_idx,
                                original=OLD_CAPTURE / f"needlerows_L{layer_idx:02d}.npz",
                                verify=verify_replay,
                            )
                        )
                    T._ACTIVE["needle_rows"] = None
                    T._ACTIVE["meter"] = None
                    T._CAPTURE["rvec"] = None
                    del meter
            finally:
                hook.remove()
                T._ACTIVE["meter"] = None
                T._CAPTURE["rvec"] = None

            del layer
            gc.collect()
            torch.cuda.empty_cache()
            manifest["last_completed_layer"] = layer_idx
            manifest["elapsed_seconds"] = round(time.time() - start_time, 3)
            atomic_json(manifest_path, manifest)
            print(
                f"layer {layer_idx:02d} {'G' if layer_idx in GLOBAL_LAYERS else '.'} "
                f"{time.time() - layer_start:.1f}s",
                flush=True,
            )

        T._CAPTURE["enabled"] = False

        if not args.skip_nll:
            final_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
            final_norm.weight = torch.nn.Parameter(
                rd.get("model.llm.norm.weight", "cuda").to(torch.bfloat16),
                requires_grad=False,
            )
            final_norm.eval()
            unembed = rd.get("model.llm.unembed.weight", "cuda").to(torch.bfloat16)
            for name in args.texts:
                nll = compute_nll(
                    hidden[name],
                    ids_by_text[name],
                    final_norm,
                    unembed,
                    mup_multiplier=float(config.logits_mup_width_multiplier),
                    unpadded_vocab=int(config.unpadded_vocab_size),
                    token_chunk=args.nll_token_chunk,
                )
                nll_path = out_root / "nll" / f"nll_{name}.npz"
                atomic_npz(
                    nll_path,
                    position=np.arange(len(nll), dtype=np.int32),
                    target_id=ids_by_text[name][1:].astype(np.int32),
                    nll=nll,
                )
                manifest["artifacts"].append(
                    artifact_record(
                        out_root,
                        nll_path,
                        kind="next_token_nll",
                        logical_dtype="position:int32, target_id:int32, nll:float32",
                        shape=nll.shape,
                        extra={"text": name},
                    )
                )
                manifest["nll_summary"][name] = {
                    "count": int(len(nll)),
                    "mean": float(np.mean(nll, dtype=np.float64)),
                    "median": float(np.median(nll)),
                    "min": float(np.min(nll)),
                    "max": float(np.max(nll)),
                    "finite": bool(np.isfinite(nll).all()),
                }
                print(f"NLL {name}: mean={manifest['nll_summary'][name]['mean']:.6f}", flush=True)
            del unembed, final_norm

            if full_run:
                uniform = math.log(int(config.unpadded_vocab_size))
                prose = manifest["nll_summary"]["01_prose_en"]["mean"]
                random = manifest["nll_summary"]["06_random"]["mean"]
                nll_gate = {
                    "uniform_nll": uniform,
                    "all_finite": all(x["finite"] for x in manifest["nll_summary"].values()),
                    "prose_positive_below_uniform": 0.0 < prose < uniform,
                    "random_above_prose": random > prose,
                }
                nll_gate["passed"] = all(
                    [
                        nll_gate["all_finite"],
                        nll_gate["prose_positive_below_uniform"],
                        nll_gate["random_above_prose"],
                    ]
                )
                manifest["nll_gate"] = nll_gate
                if not nll_gate["passed"]:
                    raise RuntimeError(f"NLL integrity gate failed: {nll_gate}")

        manifest["complete"] = True
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_seconds"] = round(time.time() - start_time, 3)
        manifest["artifact_count"] = len(manifest["artifacts"])
        atomic_json(manifest_path, manifest)
        print(
            f"DONE complete={manifest['complete']} artifacts={manifest['artifact_count']} "
            f"wall={manifest['wall_seconds'] / 60:.2f} min",
            flush=True,
        )
    except Exception as exc:
        manifest["complete"] = False
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = repr(exc)
        manifest["traceback"] = traceback.format_exc()
        manifest["wall_seconds"] = round(time.time() - start_time, 3)
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
