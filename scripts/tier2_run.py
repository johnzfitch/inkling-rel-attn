"""
D-TIER2 runner -- the streaming forward pass. See TIER2_SPEC.md.

Streams the 592GB Inkling-NVFP4 checkpoint layer-by-layer through the 4090.
Each decoder layer is materialized on GPU from mmap'd shards (experts kept
PACKED, dequantized per-expert JIT), applied to every text's activations, then
freed. One pass over the weights, both measurements, zero generation.

Correctness posture: the ONLY custom forward code is (1) the MoE expert loop
with JIT dequant, which mirrors the library InklingExperts.forward line-for-line
except for the weight source (validated: our dequant sits at the FP4 floor and
matches transformers' own Interleave), and (2) the attention measurement, which
returns the true with-bias output. Everything else -- decoder wiring, sconv,
RMSNorm, MoE routing, shared experts, masks -- is the library's own code,
unchanged. Per-layer loaded weights are pure renames of checkpoint tensors, so
if a shape is wrong load_state_dict raises rather than silently mis-wiring.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier2_stream as T
from tier2_stream import (GLOBAL_LAYERS, RENAME, Meter, PackedExperts, ShardReader,
                          deinterleave, measuring_attention, patched_experts_forward)

from transformers import AutoConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.inkling.modeling_inkling import InklingDecoderLayer

NVFP4_DIR = r"R:\inkling\nvfp4"
CORPUS = r"R:\inkling\corpus"
DUMP = r"R:\inkling\dumps\tier2"
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]


def verify_ids(name):
    p = os.path.join(CORPUS, f"{name}.ids.npy")
    want = json.load(open(os.path.join(CORPUS, "manifest.json")))["texts"][name]["ids_sha256"]
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert got == want, f"{name} sha mismatch: {got} != {want}"
    return np.load(p)


def build_layer(config, i, rd, device="cuda"):
    """Materialize decoder layer i on `device` from shards. Experts stay packed."""
    prefix = f"model.llm.layers.{i}."
    is_moe = config.mlp_layer_types[i] == "sparse"
    with torch.device("meta"):
        layer = InklingDecoderLayer(config, i)
    layer.eval()

    sd = {}
    fp32 = ("sconv",)  # _keep_in_fp32_modules_strict

    def load(name, dt=torch.bfloat16):
        t = rd.get(name, device)
        return t.to(dt)

    for k in rd.layer_keys(i):
        rel = k[len(prefix):]
        if rel in RENAME:
            tgt = RENAME[rel]
            dt = torch.float32 if "sconv" in tgt else torch.bfloat16
            sd[tgt] = load(k, dt)
        elif rel == "mlp.w13_dn.weight":                    # dense: interleave0 + chunk0
            g, u = deinterleave(load(k), 0).chunk(2, 0)
            sd["mlp.gate_proj.weight"] = g.contiguous()
            sd["mlp.up_proj.weight"] = u.contiguous()
        elif rel == "mlp.w2_md.weight":
            sd["mlp.down_proj.weight"] = load(k)
        elif rel == "mlp.shared_experts.shared_w13_weight":  # interleave1 + chunk1
            g, u = deinterleave(load(k), 1).chunk(2, 1)
            sd["mlp.shared_experts.gate_proj"] = g.contiguous()
            sd["mlp.shared_experts.up_proj"] = u.contiguous()
        elif rel == "mlp.shared_experts.shared_w2_weight":
            sd["mlp.shared_experts.down_proj"] = load(k)
        elif rel.startswith("mlp.experts."):
            pass                                             # -> PackedExperts below
        else:
            raise KeyError(f"unmapped checkpoint key: {rel}")

    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    # only the routed-expert params may be missing (handled via packing)
    bad = [m for m in missing if not m.startswith("mlp.experts.")]
    assert not bad, f"layer {i} missing non-expert params: {bad}"
    assert not unexpected, f"layer {i} unexpected: {unexpected}"

    if is_moe:
        pe = f"{prefix}mlp.experts."
        w13 = pe + "w13_weight"
        w2 = pe + "w2_weight"
        if rd.has(w13 + ".scale"):                          # quantized (layers 3-65)
            packed = PackedExperts(
                deinterleave(rd.get(w13, device), 1), deinterleave(rd.get(w13 + ".scale", device), 1),
                rd.get(w13 + ".scale2", device),
                rd.get(w2, device), rd.get(w2 + ".scale", device), rd.get(w2 + ".scale2", device))
        else:                                               # layer 2: BF16 experts (22GB) -> CPU
            gu = deinterleave(rd.get(w13, "cpu"), 1).to(torch.bfloat16)
            dn = rd.get(w2, "cpu").to(torch.bfloat16)
            packed = PackedExperts(None, None, None, None, None, None, bf16_gu=gu, bf16_down=dn)
        layer.mlp.experts._packed = packed
        layer.mlp.experts.forward = types.MethodType(patched_experts_forward, layer.mlp.experts)
        del layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj

    # guard: nothing left on meta
    for n, p in list(layer.named_parameters()) + list(layer.named_buffers()):
        assert p.device.type != "meta", f"layer {i} param still meta: {n}"
    return layer


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_savez(path, **arrs):
    # np.savez appends ".npz" if the name lacks it; use a tmp name that already
    # ends in .npz so no suffix is added, then rename onto the target.
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def provenance(config, args, layer_ids, texts, rd):
    import torch as _t
    import transformers
    scr = os.path.dirname(os.path.abspath(__file__))
    return dict(
        checkpoint_index_sha=sha256_file(os.path.join(NVFP4_DIR, "model.safetensors.index.json")),
        config_sha=sha256_file(os.path.join(NVFP4_DIR, "config.json")),
        tokenizer_sha=sha256_file(os.path.join(CORPUS, "tokenizer.json")),
        transformers=transformers.__version__, torch=_t.__version__,
        cuda=_t.version.cuda, device=_t.cuda.get_device_name(0),
        script_sha={f: sha256_file(os.path.join(scr, f)) for f in
                    ["tier2_run.py", "tier2_stream.py", "tier2_nvfp4.py"]},
        seq=args.seq, qchunk=args.qchunk, layer_ids=layer_ids, texts=texts,
        attn_impl="tier2_measure (compact rel_logits, twice-softmax)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--texts", default=",".join(TEXTS))
    ap.add_argument("--qchunk", type=int, default=512)   # matches registered 512-query plan
    ap.add_argument("--out", default=DUMP)
    ap.add_argument("--smoke", action="store_true", help="tiny self-check: layers 0-2, seq 128, text 06 only")
    ap.add_argument("--capture", action="store_true",
                    help="also dump per-layer hidden states + r-vectors + needle attention rows "
                         "(activation follow-up for C1/C2/C3). ~46GB.")
    args = ap.parse_args()

    if args.smoke:
        args.layers, args.seq, args.texts, args.qchunk = "0,1,2", 128, "06_random", 128
        args.out = os.path.join(DUMP, "_smoke")   # #3: never clobber production dumps
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    config = AutoConfig.from_pretrained(NVFP4_DIR).text_config
    ALL_ATTENTION_FUNCTIONS.register("tier2_measure", measuring_attention)
    config._attn_implementation = "tier2_measure"
    # #2: stop Inkling materializing the full [1,H,Q,K] position bias; return compact rel_logits
    from transformers.models.inkling.modeling_inkling import InklingRelativeLogits
    InklingRelativeLogits.forward = T.compact_relative_logits_forward
    n_layers = config.num_hidden_layers
    layer_ids = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    texts = args.texts.split(",")

    # #4: hidden states start from embeddings, so the run MUST begin at layer 0 and be contiguous.
    assert layer_ids[0] == 0 and layer_ids == list(range(layer_ids[0], layer_ids[-1] + 1)), \
        f"layers must start at 0 and be contiguous (no resume support): got {layer_ids}"
    # #10: never silently truncate/pad; corpus is exactly 8192 tokens.
    n_tok = json.load(open(os.path.join(CORPUS, "manifest.json")))["texts"][texts[0]]["n_tokens"]
    assert args.seq <= n_tok, f"--seq {args.seq} exceeds corpus length {n_tok}"

    print(f"seq={args.seq} layers={layer_ids[0]}..{layer_ids[-1]} texts={texts} qchunk={args.qchunk}", flush=True)
    print("tau == 1 for all pos < 128000: this pass says NOTHING about log scaling.", flush=True)

    rd = ShardReader(NVFP4_DIR)

    # #11: use the library's own RMSNorm for the embedding norm (exact parity with Inkling's
    # cast order), instead of a hand-rolled version. Decoder layers already use library norms.
    from transformers.models.inkling.modeling_inkling import InklingRMSNorm
    embed_w = rd.get("model.llm.embed.weight", "cuda").to(torch.bfloat16)
    embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
    embed_norm.weight = torch.nn.Parameter(rd.get("model.llm.embed_norm.weight", "cuda").to(torch.bfloat16),
                                           requires_grad=False)
    embed_norm.eval()

    # --capture setup: activation follow-up (hidden states + r-vectors + needle rows)
    capdir = os.path.join(args.out, "capture")
    needle_qpos = None
    if args.capture:
        os.makedirs(capdir, exist_ok=True)
        T._CAPTURE["enabled"] = True
        # recall-query positions of the needle text (from the sidecar), for matched seam rows
        sc = json.load(open(os.path.join(CORPUS, "05_needles.sidecar.json")))
        needle_qpos = sorted({e["token_positions"][1] for e in sc["entities"]
                              if e["token_positions"][1] < args.seq})
        json.dump(sc, open(os.path.join(capdir, "needles_sidecar.json"), "w"))

    hidden = {}
    for name in texts:
        ids = torch.from_numpy(verify_ids(name)[: args.seq].astype(np.int64)).cuda()
        h = torch.nn.functional.embedding(ids, embed_w).unsqueeze(0)   # [1,S,H]
        hidden[name] = embed_norm(h)
        if args.capture:   # residual stream entering layer 0
            np.save(os.path.join(capdir, f"hidden_embed_{name}.npy"),
                    hidden[name][0].to("cpu", torch.float16).numpy())
    print(f"embedded {len(texts)} texts @ seq {args.seq}{'  [+capture]' if args.capture else ''}", flush=True)

    written = []
    t0 = time.time()
    for li in layer_ids:
        lt = time.time()
        layer = build_layer(config, li, rd, "cuda")
        is_sliding = config.layer_types[li] == "hybrid_sliding"
        rel_extent = config.sliding_window_size if is_sliding else config.rel_extent
        n_heads = layer.self_attn.num_heads
        # local layers only ever populate d < window (512); only global layers need full range
        dmax = config.sliding_window_size if is_sliding else args.seq

        for name in texts:
            meter = Meter(n_heads, dmax, "cuda")
            # needle rows only for the needle text; fresh dict each (layer,text)
            nq = needle_qpos if (args.capture and name == "05_needles") else None
            T._ACTIVE.update(meter=meter, sliding=is_sliding,
                             window=config.sliding_window_size, qchunk=args.qchunk,
                             needle_qpos=nq, needle_rows=({} if nq else None))
            T._CAPTURE["rvec"] = None
            hidden[name] = layer(hidden[name], attention_mask=None, conv_mask=None, past_key_values=None)
            fn = f"layer{li:02d}_{name}_s{args.seq}.npz"   # #3: seq in filename, no cross-seq collision
            atomic_savez(os.path.join(args.out, fn), **meter.to_npz(), meta=json.dumps(dict(
                layer=li, text=name, is_global=li in GLOBAL_LAYERS, is_sliding=is_sliding,
                rel_extent=rel_extent, seq=args.seq, qchunk=args.qchunk, n_heads=n_heads)))
            written.append(fn)
            if args.capture:
                np.save(os.path.join(capdir, f"hidden_L{li:02d}_{name}.npy"),
                        hidden[name][0].to("cpu", torch.float16).numpy())      # [S,H] output of layer li
                if T._CAPTURE["rvec"] is not None:
                    np.save(os.path.join(capdir, f"rvec_L{li:02d}_{name}.npy"),
                            T._CAPTURE["rvec"].numpy())                        # [S,H_heads,d_rel]
                if nq and T._ACTIVE["needle_rows"]:
                    rows = T._ACTIVE["needle_rows"]
                    qs = sorted(rows)
                    arr = np.stack([rows[q].numpy() for q in qs])             # [n_needle, H, K]
                    np.savez(os.path.join(capdir, f"needlerows_L{li:02d}.npz"),
                             qpos=np.array(qs), rows=arr)
            del meter
        T._ACTIVE["meter"] = None
        # #1: actually release the layer BEFORE the next build_layer allocates.
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  layer {li:02d} {'G' if li in GLOBAL_LAYERS else '.'} "
              f"{time.time()-lt:5.1f}s  peakVRAM={mem:.1f}GB", flush=True)
        torch.cuda.reset_peak_memory_stats()

    # #9: run-level manifest proving all files belong to one consistent completed run
    manifest = provenance(config, args, layer_ids, texts, rd)
    manifest["files"] = sorted(written)
    manifest["complete"] = (layer_ids == list(range(n_layers)) and set(texts) == set(TEXTS))
    manifest["wall_min"] = round((time.time() - t0) / 60, 2)
    tmp = os.path.join(args.out, "manifest.json.tmp")
    json.dump(manifest, open(tmp, "w"), indent=2)
    os.replace(tmp, os.path.join(args.out, "manifest.json"))

    # smoke: sanity assertions
    if args.smoke:
        for name in texts:
            h = hidden[name]
            assert torch.isfinite(h).all(), f"{name} produced non-finite hidden"
            print(f"  [smoke] {name}: hidden finite, |h|={h.float().norm():.1f}")
        d = np.load(os.path.join(args.out, f"layer00_{texts[0]}_s{args.seq}.npz"), allow_pickle=True)
        # invariant: each query's probs sum to 1, so sum_d mass_with[h,d] == n_queries per head
        tot = d["mass_with"].sum(1)   # [H]
        exp = args.seq
        print(f"  [smoke] layer00 sum_d mass_with per head = {tot.mean():.2f} (expect {exp}); "
              f"max dev {np.abs(tot-exp).max():.3f}")
        assert np.abs(tot - exp).max() < 0.5, "attention mass not conserved"
        # without-bias must also conserve
        assert np.abs(d["mass_without"].sum(1) - exp).max() < 0.5, "without-bias mass not conserved"
        print("  [smoke] mass conserved for both with/without; PASS")
    print(f"DONE {len(layer_ids)} layers x {len(texts)} texts in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
