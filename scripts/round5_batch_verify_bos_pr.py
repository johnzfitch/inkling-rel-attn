"""Second-analyst re-derivation: LF3 BOS-localization + R5-C PR flip-band.

BOS percentiles here are inclusive-rank; the producer reports midrank
(identical ranks, e.g. random 128/128 -> 1.0 here vs 127.5/128 = 0.996094
there). Verdicts agree under both conventions.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture"
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
out = {}

# ---- LF3 BOS localization (random + prose)
def bos_percentile(text):
    pcts = []
    for L in range(66):
        r = np.load(CAP / "replay" / f"rvec_L{L:02d}_{text}.npy").astype(np.float32).reshape(8192, 1024).astype(float)
        blocks = r.reshape(128, 64, 1024).mean(1)
        disp = np.zeros(128)
        total_sum = r.sum(0)
        for b in range(128):
            others = (total_sum - r[b * 64:(b + 1) * 64].sum(0)) / (8192 - 64)
            disp[b] = np.linalg.norm(blocks[b] - others)
        pcts.append(float((disp <= disp[0]).mean()))
    return float(np.median(pcts))

out["lf3_bos_random"] = bos_percentile("06_random")
out["lf3_bos_prose"] = bos_percentile("01_prose_en")

# ---- R5-C PR flip-band: fixed 128-token stratified sample, coordinate-centered PR
offs = np.concatenate([np.arange(32) * 256 + o for o in (31, 95, 159, 223)])
offs.sort()

def bf16_rows(path, rows):
    bits = np.load(path, mmap_mode="r")
    sel = np.asarray(bits[rows])
    return (sel.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

def pr_of(path):
    x = bf16_rows(path, offs)
    xc = x - x.mean(0)
    g = xc @ xc.T
    lam = np.linalg.eigvalsh(g)
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam * lam).sum())

pr = np.zeros((67, 6))  # embed + 66
for ti, t in enumerate(TEXTS):
    pr[0, ti] = pr_of(CAP / "states" / f"hidden_embed_{t}.npy")
    for L in range(66):
        pr[L + 1, ti] = pr_of(CAP / "states" / f"hidden_L{L:02d}_{t}.npy")

med = np.median(pr, axis=1)                      # 67 values: input-to-L0 ... output-of-L65
adj = np.abs(np.diff(med))                       # change into destination layer L = med[L+1]-med[L]
# scope boundaries: exclude local/global scope changes (destination layer scope != source layer scope)
GLOB = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}
def scope(L):
    return "g" if L in GLOB else "l"
same_scope = [L for L in range(66) if L == 0 or scope(L) == scope(L - 1)]
# destination layers 0..65; change into L uses adj[L]; exclude scope-change destinations
elig = [L for L in range(1, 66) if scope(L) == scope(L - 1)]
vals = {L: adj[L] for L in elig}
Lmax = max(vals, key=vals.get)
out["r5c_pr_flipband"] = {"argmax_destination_layer": int(Lmax),
                          "in_L13_28": bool(13 <= Lmax <= 28),
                          "top3": sorted(((float(v), int(L)) for L, v in vals.items()), reverse=True)[:3]}
print(json.dumps(out, indent=1))
dest = ROOT / "analysis" / "round5" / "dump_science_batch" / "verification_bos_pr.json"
dest.write_text(json.dumps(out, indent=1))
