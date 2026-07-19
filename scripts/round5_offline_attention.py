"""LF5 offline pair-level attention instrument.

Two numerical modes are provided:

* ``replay``: CUDA/BF16, full Q projection and original 512-query chunk shapes;
  this mode must reproduce captured FP16 needle rows bit for bit.
* ``cpu``: CPU/FP32 convenience path.  Its registered production gate failed
  at L0 and it is not a verified backend.  Amendment A5 promotes ``replay`` to
  production and requires the CPU failure characterization whenever CPU is
  used.

The module never executes V/O projections, the MLP, or downstream layers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from tier2_stream import ShardReader  # noqa: E402


Backend = Literal["cpu", "replay"]
NVFP4_DIR = ROOT / "nvfp4"
DEFAULT_INPUTS = ROOT / "dumps" / "round5" / "attention_inputs"
DEFAULT_CAPTURE = ROOT / "dumps" / "tier2" / "capture"
REGISTRATION_COMMIT = "34278b4"
PLAN_COMMIT = "d4e2579"
GLOBAL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}
QCHUNK = 512
HEAD_DIM = 128
N_HEADS = 64
RMS_EPS = 1e-6
AMENDMENT_A5_COMMIT = "7bf608d9971997a655a4f9cd46e3bc921ffb74b8"
CPU_FAILURE_NOTICE = (
    "A5 NOTICE: CPU/FP32 is an unverified convenience backend. Its registered "
    "L0 gate failed (max|delta p|=0.00549644>0.001; argmax=98.9583%<100%; "
    "KL=7.50385e-5>1e-6). Production LF5 uses CUDA/BF16 replay."
)
_CPU_NOTICE_EMITTED = False


def emit_cpu_failure_notice() -> None:
    global _CPU_NOTICE_EMITTED
    if not _CPU_NOTICE_EMITTED:
        print(CPU_FAILURE_NOTICE, file=sys.stderr, flush=True)
        _CPU_NOTICE_EMITTED = True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def load_bf16_bits(path: Path, device: torch.device) -> torch.Tensor:
    bits = np.load(path, mmap_mode="r")
    if bits.dtype != np.uint16 or bits.ndim != 2:
        raise TypeError(f"invalid BF16-bit capture {path}: {bits.dtype}, {bits.shape}")
    # Copy because the read-only memmap cannot safely back a writable torch view.
    tensor = torch.from_numpy(np.array(bits, dtype=np.uint16, copy=True)).view(torch.bfloat16)
    return tensor.to(device)


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    input_dtype = hidden.dtype
    work = hidden.float()
    variance = work.pow(2).mean(-1, keepdim=True)
    work = work * torch.rsqrt(variance + eps)
    return weight * work.to(input_dtype)


def residual_short_convolution(projected: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Exact prefill branch of InklingShortConvolution for K states."""
    input_dtype = projected.dtype
    work = projected.float()
    residual = work
    channels = work.shape[-1]
    convolved = torch.nn.functional.conv1d(
        work.transpose(1, 2),
        weight=weight.float(),
        bias=None,
        padding=weight.shape[-1] - 1,
        groups=channels,
    )[:, :, : work.shape[1]]
    return (convolved.transpose(1, 2) + residual).to(input_dtype)


@dataclass
class OfflineRow:
    layer: int
    text: str
    query_position: int
    key_positions: np.ndarray
    content_logits: np.ndarray
    positional_bias: np.ndarray
    total_logits: np.ndarray
    attention_with_bias: np.ndarray
    attention_without_bias: np.ndarray
    support_start: int
    support_stop: int
    compact: bool

    def attention_fp16_bits(self) -> np.ndarray:
        return self.attention_with_bias.astype(np.float16).view(np.uint16)


class OfflineAttention:
    def __init__(
        self,
        layer: int,
        *,
        backend: Backend = "cpu",
        input_root: Path = DEFAULT_INPUTS,
        capture_root: Path = DEFAULT_CAPTURE,
        checkpoint_root: Path = NVFP4_DIR,
        qchunk: int = QCHUNK,
    ) -> None:
        if not 0 <= layer < 66:
            raise ValueError(layer)
        if backend not in ("cpu", "replay"):
            raise ValueError(backend)
        if backend == "replay" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for replay parity")
        if backend == "cpu":
            emit_cpu_failure_notice()

        self.layer = layer
        self.backend = backend
        self.input_root = Path(input_root)
        self.capture_root = Path(capture_root)
        self.checkpoint_root = Path(checkpoint_root)
        self.qchunk = qchunk
        self.is_global = layer in GLOBAL_LAYERS
        self.is_sliding = not self.is_global
        self.extent = 1024 if self.is_global else 512
        self.n_kv_heads = 8 if self.is_global else 16
        self.groups = N_HEADS // self.n_kv_heads
        self.device = torch.device("cuda" if backend == "replay" else "cpu")
        self.compute_dtype = torch.bfloat16 if backend == "replay" else torch.float32
        self.reader = ShardReader(str(self.checkpoint_root))
        self.prefix = f"model.llm.layers.{layer}.attn."
        self._load_weights()
        self._clear_text_cache()

    def _weight(self, suffix: str, *, force_float: bool = False) -> torch.Tensor:
        tensor = self.reader.get(self.prefix + suffix, str(self.device))
        dtype = torch.float32 if force_float or self.backend == "cpu" else torch.bfloat16
        return tensor.to(dtype)

    def _load_weights(self) -> None:
        self.wq = self._weight("wq_du.weight")
        self.wk = self._weight("wk_dv.weight")
        self.k_conv = self._weight("k_sconv.weight", force_float=True)
        self.q_norm_weight = self._weight("q_norm.weight")
        self.k_norm_weight = self._weight("k_norm.weight")
        self.rel_proj = self._weight("rel_logits_proj.proj")
        if self.wq.shape != (N_HEADS * HEAD_DIM, 6144):
            raise RuntimeError(f"unexpected Q shape: {self.wq.shape}")
        if self.wk.shape != (self.n_kv_heads * HEAD_DIM, 6144):
            raise RuntimeError(f"unexpected K shape: {self.wk.shape}")
        if self.rel_proj.shape != (16, self.extent):
            raise RuntimeError(f"unexpected relative projection shape: {self.rel_proj.shape}")

    def _clear_text_cache(self) -> None:
        self.text: str | None = None
        self.seq = 0
        self.hidden: torch.Tensor | None = None
        self.key_states: torch.Tensor | None = None
        self.query_states: torch.Tensor | None = None
        self.relative_logits: torch.Tensor | None = None
        self.rvec: torch.Tensor | None = None

    @torch.no_grad()
    def prepare_text(self, text: str) -> None:
        if self.text == text:
            return
        self._clear_text_cache()
        input_path = self.input_root / "normalized" / f"attn_in_L{self.layer:02d}_{text}.npy"
        hidden = load_bf16_bits(input_path, self.device)
        if self.backend == "cpu":
            hidden = hidden.float()
        self.seq = int(hidden.shape[0])
        if hidden.shape[1] != 6144:
            raise RuntimeError(f"unexpected normalized input shape: {hidden.shape}")
        hidden3 = hidden.unsqueeze(0)

        key_projected = torch.nn.functional.linear(hidden3, self.wk)
        key_projected = residual_short_convolution(key_projected, self.k_conv)
        key_states = key_projected.view(1, self.seq, self.n_kv_heads, HEAD_DIM)
        key_states = rms_norm(key_states, self.k_norm_weight).transpose(1, 2).contiguous()
        self.key_states = key_states

        rvec_np = np.load(
            self.capture_root / f"rvec_L{self.layer:02d}_{text}.npy", mmap_mode="r"
        )
        if rvec_np.shape != (self.seq, N_HEADS, 16) or rvec_np.dtype != np.float16:
            raise RuntimeError(f"unexpected r-vector: {rvec_np.shape}, {rvec_np.dtype}")
        rvec = torch.from_numpy(np.array(rvec_np, copy=True)).to(self.device)
        self.rvec = rvec.to(self.compute_dtype)

        if self.backend == "replay":
            query_projected = torch.nn.functional.linear(hidden3, self.wq)
            query_states = query_projected.view(1, self.seq, N_HEADS, HEAD_DIM)
            query_states = rms_norm(query_states, self.q_norm_weight).transpose(1, 2).contiguous()
            relative_states = self.rvec.unsqueeze(0)
            relative_logits = (relative_states @ self.rel_proj).transpose(1, 2).contiguous()
            if self.is_global:
                positions = torch.arange(self.seq, device=self.device)
                effective_n = (positions + 1).float()
                tau = 1.0 + 0.1 * torch.log((effective_n / 128000.0).clamp(min=1.0))
                tau = tau.view(1, 1, -1, 1)
                query_states = (query_states.float() * tau).to(query_states.dtype)
                relative_logits = (relative_logits.float() * tau).to(relative_logits.dtype)
            self.query_states = query_states
            self.relative_logits = relative_logits
            self.hidden = None
        else:
            self.hidden = hidden
        self.text = text

    def _support(self, q: int) -> tuple[int, int]:
        return (0 if self.is_global else max(0, q - 511), q + 1)

    @staticmethod
    def _row_object(
        *,
        layer: int,
        text: str,
        q: int,
        content: torch.Tensor,
        bias: torch.Tensor,
        with_bias: torch.Tensor,
        without_bias: torch.Tensor,
        support_start: int,
        support_stop: int,
        compact: bool,
    ) -> OfflineRow:
        seq = content.shape[-1]
        total = content + bias
        invalid = torch.ones(seq, dtype=torch.bool, device=content.device)
        invalid[support_start:support_stop] = False
        total = total.masked_fill(invalid.unsqueeze(0), float("-inf"))
        if compact:
            sl = slice(support_start, support_stop)
            key_positions = np.arange(support_start, support_stop, dtype=np.int32)
        else:
            sl = slice(None)
            key_positions = np.arange(seq, dtype=np.int32)

        def cpu32(tensor: torch.Tensor) -> np.ndarray:
            return tensor[:, sl].float().to("cpu").numpy()

        return OfflineRow(
            layer=layer,
            text=text,
            query_position=q,
            key_positions=key_positions,
            content_logits=cpu32(content),
            positional_bias=cpu32(bias),
            total_logits=cpu32(total),
            attention_with_bias=cpu32(with_bias),
            attention_without_bias=cpu32(without_bias),
            support_start=support_start,
            support_stop=support_stop,
            compact=compact,
        )

    @torch.no_grad()
    def rows(self, text: str, query_positions: Iterable[int], *, compact: bool = False) -> list[OfflineRow]:
        self.prepare_text(text)
        positions = sorted(set(int(q) for q in query_positions))
        if not positions or positions[0] < 0 or positions[-1] >= self.seq:
            raise ValueError(positions)
        if self.backend == "replay":
            return self._rows_replay(text, positions, compact=compact)
        return self._rows_cpu(text, positions, compact=compact)

    @torch.no_grad()
    def row(self, text: str, q: int, *, compact: bool = False) -> OfflineRow:
        return self.rows(text, [q], compact=compact)[0]

    def _rows_replay(self, text: str, positions: list[int], *, compact: bool) -> list[OfflineRow]:
        assert self.query_states is not None
        assert self.key_states is not None
        assert self.relative_logits is not None
        kx = self.key_states.repeat_interleave(self.groups, dim=1)
        key_pos = torch.arange(self.seq, device=self.device)
        requested = set(positions)
        rows: dict[int, OfflineRow] = {}
        chunks = sorted({q // self.qchunk for q in positions})
        neg = torch.finfo(torch.float32).min

        for chunk in chunks:
            start = chunk * self.qchunk
            end = min(start + self.qchunk, self.seq)
            qpos = torch.arange(start, end, device=self.device)
            distance = qpos[:, None] - key_pos[None, :]
            causal = distance >= 0
            if self.is_sliding:
                causal = causal & (distance < 512)
            content = (
                torch.matmul(
                    self.query_states[:, :, start:end],
                    kx.transpose(2, 3),
                )
                * (1.0 / HEAD_DIM)
            )[0]
            rel_chunk = self.relative_logits[0, :, start:end, :]
            in_extent = (distance >= 0) & (distance < self.extent)
            gather_index = (
                distance.clamp(0, self.extent - 1)
                .unsqueeze(0)
                .expand(N_HEADS, -1, -1)
            )
            bias = torch.gather(rel_chunk, 2, gather_index).masked_fill(
                ~in_extent.unsqueeze(0), 0.0
            )
            content32 = content.float()
            bias32 = bias.float()
            invalid = ~causal
            # A6 dtype boundary: stock adds content+bias in the native (BF16) dtype,
            # then softmaxes in FP32. The add must precede the upcast.
            with_bias = torch.softmax(
                (content + bias).float().masked_fill(invalid.unsqueeze(0), neg), dim=-1
            )
            without_bias = torch.softmax(
                content32.masked_fill(invalid.unsqueeze(0), neg), dim=-1
            )
            for q in range(start, end):
                if q not in requested:
                    continue
                index = q - start
                support_start, support_stop = self._support(q)
                rows[q] = self._row_object(
                    layer=self.layer,
                    text=text,
                    q=q,
                    content=content32[:, index],
                    bias=bias32[:, index],
                    with_bias=with_bias[:, index],
                    without_bias=without_bias[:, index],
                    support_start=support_start,
                    support_stop=support_stop,
                    compact=compact,
                )
            del content, bias, content32, bias32, with_bias, without_bias
        return [rows[q] for q in positions]

    def _rows_cpu(self, text: str, positions: list[int], *, compact: bool) -> list[OfflineRow]:
        assert self.hidden is not None
        assert self.key_states is not None
        assert self.rvec is not None
        query_input = self.hidden[positions]
        query = torch.nn.functional.linear(query_input, self.wq)
        query = query.view(len(positions), N_HEADS, HEAD_DIM)
        query = rms_norm(query, self.q_norm_weight)
        kx = self.key_states[0].repeat_interleave(self.groups, dim=0)
        content = torch.einsum("nhd,hsd->nhs", query, kx) * (1.0 / HEAD_DIM)
        rel_curves = self.rvec[positions] @ self.rel_proj
        key_pos = torch.arange(self.seq, device=self.device)
        neg = torch.finfo(torch.float32).min
        rows: list[OfflineRow] = []

        for index, q in enumerate(positions):
            distance = q - key_pos
            causal = distance >= 0
            if self.is_sliding:
                causal = causal & (distance < 512)
            in_extent = (distance >= 0) & (distance < self.extent)
            gather_index = distance.clamp(0, self.extent - 1).unsqueeze(0).expand(N_HEADS, -1)
            bias = torch.gather(rel_curves[index], 1, gather_index).masked_fill(
                ~in_extent.unsqueeze(0), 0.0
            )
            content_row = content[index].float()
            bias_row = bias.float()
            # A6 dtype boundary: add in native dtype before upcast (no-op when the CPU
            # path already runs FP32; exact when tensors are BF16). Keeps parity with
            # the corrected GPU/stock arithmetic.
            with_bias = torch.softmax(
                (content[index] + bias).float().masked_fill(~causal.unsqueeze(0), neg), dim=-1
            )
            without_bias = torch.softmax(
                content_row.masked_fill(~causal.unsqueeze(0), neg), dim=-1
            )
            support_start, support_stop = self._support(q)
            rows.append(
                self._row_object(
                    layer=self.layer,
                    text=text,
                    q=q,
                    content=content_row,
                    bias=bias_row,
                    with_bias=with_bias,
                    without_bias=without_bias,
                    support_start=support_start,
                    support_stop=support_stop,
                    compact=compact,
                )
            )
        return rows

    def close(self) -> None:
        self._clear_text_cache()
        for name in ("wq", "wk", "k_conv", "q_norm_weight", "k_norm_weight", "rel_proj"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self) -> "OfflineAttention":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


def offline_row(
    layer: int,
    text: str,
    q: int,
    *,
    compact: bool = False,
    backend: Backend = "replay",
    input_root: Path = DEFAULT_INPUTS,
    capture_root: Path = DEFAULT_CAPTURE,
) -> OfflineRow:
    with OfflineAttention(
        layer,
        backend=backend,
        input_root=input_root,
        capture_root=capture_root,
    ) as instrument:
        return instrument.row(text, q, compact=compact)


def safe_kl(original: np.ndarray, offline: np.ndarray) -> np.ndarray:
    """KL(original || offline) per head after float64 renormalization."""
    p = original.astype(np.float64)
    q = offline.astype(np.float64)
    p_sum = p.sum(axis=-1, keepdims=True)
    q_sum = q.sum(axis=-1, keepdims=True)
    if np.any(p_sum <= 0) or np.any(q_sum <= 0):
        raise RuntimeError("non-positive attention row sum")
    p /= p_sum
    q /= q_sum
    tiny = np.finfo(np.float64).tiny
    mask = p > 0
    terms = np.zeros_like(p)
    terms[mask] = p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], tiny)))
    return terms.sum(axis=-1)


def compare_layer(
    layer: int,
    backend: Backend,
    *,
    input_root: Path,
    capture_root: Path,
) -> tuple[dict[str, Any], bool]:
    with np.load(capture_root / f"needlerows_L{layer:02d}.npz") as captured:
        qpos = captured["qpos"].astype(np.int64)
        original = captured["rows"].astype(np.float16)
    started = time.time()
    with OfflineAttention(
        layer,
        backend=backend,
        input_root=input_root,
        capture_root=capture_root,
    ) as instrument:
        rows = instrument.rows("05_needles", qpos.tolist(), compact=False)
    offline = np.stack([row.attention_with_bias for row in rows]).astype(np.float32)
    if offline.shape != original.shape:
        raise RuntimeError((offline.shape, original.shape))

    report: dict[str, Any] = {
        "layer": layer,
        "backend": backend,
        "queries": int(len(qpos)),
        "heads": N_HEADS,
        "elements": int(offline.size),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if backend == "replay":
        actual_bits = offline.astype(np.float16).view(np.uint16)
        expected_bits = original.view(np.uint16)
        mismatch = actual_bits != expected_bits
        report.update(
            mismatch_words=int(np.count_nonzero(mismatch)),
            mismatch_fraction=float(np.mean(mismatch)),
            bitwise_equal=bool(not np.any(mismatch)),
        )
        passed = report["bitwise_equal"]
    else:
        original32 = original.astype(np.float32)
        delta = np.abs(offline - original32)
        row_sum_error = np.abs(offline.sum(axis=-1, dtype=np.float64) - 1.0)
        row_sum_vs_capture = np.abs(
            offline.sum(axis=-1, dtype=np.float64)
            - original32.sum(axis=-1, dtype=np.float64)
        )
        argmax_equal = np.argmax(offline, axis=-1) == np.argmax(original32, axis=-1)
        kl = safe_kl(original32, offline)
        report.update(
            max_abs_delta=float(np.max(delta)),
            p99_abs_delta=float(np.quantile(delta, 0.99)),
            max_row_sum_error=float(np.max(row_sum_error)),
            max_row_sum_delta_vs_capture=float(np.max(row_sum_vs_capture)),
            argmax_agreement=float(np.mean(argmax_equal)),
            argmax_mismatches=int(np.count_nonzero(~argmax_equal)),
            max_kl=float(np.max(kl)),
            p99_kl=float(np.quantile(kl, 0.99)),
        )
        thresholds = {
            "max_abs_delta": 1e-3,
            "max_row_sum_error": 1e-3,
            "argmax_agreement": 1.0,
            "max_kl": 1e-6,
        }
        report["thresholds"] = thresholds
        report["checks"] = {
            "max_abs_delta": report["max_abs_delta"] <= thresholds["max_abs_delta"],
            "max_row_sum_error": report["max_row_sum_error"] <= thresholds["max_row_sum_error"],
            "argmax_agreement": report["argmax_agreement"] == thresholds["argmax_agreement"],
            "max_kl": report["max_kl"] <= thresholds["max_kl"],
        }
        passed = all(report["checks"].values())
    report["passed"] = bool(passed)
    return report, bool(passed)


def parity_command(args: argparse.Namespace) -> None:
    layers = list(range(66)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round5_lf5_parity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_commit": REGISTRATION_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "backend": args.backend,
        "layers": layers,
        "input_manifest_sha256": sha256_file(args.input_root / "manifest.json"),
        "source_sha256": sha256_file(Path(__file__)),
        "results": [],
        "passed": True,
    }
    for layer in layers:
        result, passed = compare_layer(
            layer,
            args.backend,
            input_root=args.input_root,
            capture_root=args.capture_root,
        )
        report["results"].append(result)
        report["passed"] = bool(report["passed"] and passed)
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.stop_on_fail and not passed:
            break
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.report, report)
    print(f"wrote {args.report}; passed={report['passed']}")
    if not report["passed"]:
        raise SystemExit(1)


def row_command(args: argparse.Namespace) -> None:
    row = offline_row(
        args.layer,
        args.text,
        args.q,
        compact=args.compact,
        backend=args.backend,
        input_root=args.input_root,
        capture_root=args.capture_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        layer=np.int16(row.layer),
        text=np.array(row.text),
        query_position=np.int32(row.query_position),
        key_positions=row.key_positions,
        content_logits=row.content_logits.astype(np.float32),
        positional_bias=row.positional_bias.astype(np.float32),
        total_logits=row.total_logits.astype(np.float32),
        attention_with_bias=row.attention_with_bias.astype(np.float32),
        attention_without_bias=row.attention_without_bias.astype(np.float32),
        support_start=np.int32(row.support_start),
        support_stop=np.int32(row.support_stop),
    )
    print(f"wrote {args.out}")


def self_test() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 7, 8, dtype=torch.float32)
    weight = torch.randn(8, 1, 4, dtype=torch.float32)
    actual = residual_short_convolution(x, weight)
    expected = (
        torch.nn.functional.conv1d(
            x.transpose(1, 2), weight, padding=3, groups=8
        )[:, :, :7].transpose(1, 2)
        + x
    )
    if not torch.equal(actual, expected):
        raise AssertionError("short-convolution self-test failed")
    p = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
    if not np.array_equal(safe_kl(p, p), np.zeros(1)):
        raise AssertionError("KL self-test failed")
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")

    parity = sub.add_parser("parity")
    parity.add_argument("--backend", choices=["cpu", "replay"], required=True)
    parity.add_argument("--layers", default="all")
    parity.add_argument("--input-root", type=Path, default=DEFAULT_INPUTS)
    parity.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE)
    parity.add_argument("--report", type=Path, required=True)
    parity.add_argument("--stop-on-fail", action="store_true")

    row = sub.add_parser("row")
    row.add_argument("--backend", choices=["cpu", "replay"], default="replay")
    row.add_argument("--layer", type=int, required=True)
    row.add_argument("--text", required=True)
    row.add_argument("--q", type=int, required=True)
    row.add_argument("--compact", action="store_true")
    row.add_argument("--input-root", type=Path, default=DEFAULT_INPUTS)
    row.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE)
    row.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    elif args.command == "parity":
        parity_command(args)
    elif args.command == "row":
        row_command(args)
    else:
        raise SystemExit("choose parity or row, or pass --self-test")


if __name__ == "__main__":
    main()
