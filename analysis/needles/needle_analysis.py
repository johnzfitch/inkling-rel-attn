"""Needle retrieval analysis from captured attention rows (dump-first: reads only dumps).

For each global layer and each of 24 planted entities (12 below seam d~900-1000,
12 above d~1050-1150):
  - with-bias attention mass from the recall query onto the intro-mention window
    (captured rows, post-softmax)
  - without-bias mass reconstructed exactly: w_wo = renorm(w_with * exp(-b)),
    b(k) = rvec[q] . proj[:, q-k]  (0 outside extent)
  - distance-matched baseline: mean per-key mass in [d-128, d+128] excluding the
    intro window, times window length -> retrieval ratio = intro/baseline
Aggregate below vs above the seam, with vs without bias.
"""
import json, os
import numpy as np

CAP = r"R:\inkling\dumps\tier2\capture"
W = r"R:\inkling\weights"
CORPUS = r"R:\inkling\corpus"
GLOBAL = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]

sc = json.load(open(os.path.join(CORPUS, "05_needles.sidecar.json")))
ents = [e for e in sc["entities"] if len(e["token_positions"]) >= 2]

# measure codeword token length with the actual tokenizer
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(os.path.join(CORPUS, "tokenizer.json"))
wlens = {}
for e in ents:
    n = len(TOK.encode(" " + e["codeword"]).ids)
    wlens[e["codeword"]] = n
print("codeword token lengths:", sorted(set(wlens.values())))

ids = np.load(os.path.join(CORPUS, "05_needles.ids.npy"))

results = {}
for L in GLOBAL:
    z = np.load(os.path.join(CAP, f"needlerows_L{L:02d}.npz"))
    qpos, rows = z["qpos"], z["rows"].astype(np.float64)   # [24], [24,64,8192]
    proj = np.load(os.path.join(W, f"layer{L:02d}_rel_logits_proj.npy")).astype(np.float64)  # [16,1024]
    ext = proj.shape[1]
    rvec = np.load(os.path.join(CAP, f"rvec_L{L:02d}_05_needles.npy"), mmap_mode="r")
    per = []
    for e in ents:
        p0, p1 = e["token_positions"][:2]
        if p1 not in set(qpos.tolist()):
            continue
        i = int(np.where(qpos == p1)[0][0])
        q = p1
        wlen = wlens[e["codeword"]]
        win = slice(p0, p0 + wlen)
        row = rows[i]                                   # [64, 8192] with-bias
        # reconstruct bias per key for this query
        r = np.asarray(rvec[q], dtype=np.float64)       # [64,16]
        b_dist = r @ proj                               # [64, ext] bias at distance d
        k = np.arange(8192)
        d = q - k
        bias_k = np.zeros((64, 8192))
        m = (d >= 0) & (d < ext)
        bias_k[:, m] = b_dist[:, d[m]]
        causal = (k <= q)
        w_wo = row * np.exp(-bias_k)
        w_wo[:, ~causal] = 0.0
        w_wo /= w_wo.sum(-1, keepdims=True) + 1e-300
        dist = q - p0
        # distance-matched baseline band (keys at distance d+-128, excluding window)
        lo, hi = max(0, p0 - 128), min(q, p0 + 128 + wlen)
        band = np.zeros(8192, bool); band[lo:hi] = True
        band[win] = False; band[~causal] = False
        def metrics(wmat):
            intro = wmat[:, win].sum(-1)                          # [64]
            base = wmat[:, band].mean(-1) * wlen + 1e-12          # [64]
            return intro, intro / base
        iw, rw = metrics(row)
        io, ro = metrics(w_wo)
        per.append(dict(cw=e["codeword"], dist=int(dist), side=e["side_of_seam"],
                        with_mean=float(iw.mean()), with_max=float(iw.max()),
                        wo_mean=float(io.mean()), wo_max=float(io.max()),
                        with_ratio_max=float(rw.max()), wo_ratio_max=float(ro.max()),
                        argmax_head_with=int(iw.argmax()), argmax_head_wo=int(io.argmax()),
                        # fraction of total row mass on the intro window, best head
                        ))
    results[L] = per
    below = [p for p in per if p["side"] == "below"]
    above = [p for p in per if p["side"] == "above"]
    def agg(ps, key):
        return float(np.median([p[key] for p in ps])) if ps else float("nan")
    print(f"L{L:02d}  n={len(per)}  "
          f"WITH  below med(maxhead mass)={agg(below,'with_max'):.4f} above={agg(above,'with_max'):.4f} | "
          f"WITHOUT below={agg(below,'wo_max'):.4f} above={agg(above,'wo_max'):.4f}")

json.dump({str(k): v for k, v in results.items()},
          open(os.path.join(os.path.dirname(__file__), "needle_results.json"), "w"), indent=2)
print("wrote needle_results.json")
