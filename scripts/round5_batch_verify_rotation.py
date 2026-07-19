"""Second-analyst re-derivation: R5-C globals-rotate-more.

Median tokenwise rotation 1-cos(input,output), single pass per text over
embed + 66 layer states; exact 64-way one-sided sign-flip test.
Also dumps the full 66x6 median-rotation table for the flip-band record.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "dumps" / "round5" / "widened_corrected_capture" / "states"
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]
GLOBALS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}

def bf16(name):
    bits = np.load(CAP / name)
    return (bits.astype(np.uint32) << 16).view(np.float32)

rot = np.zeros((66, 6))
for ti, t in enumerate(TEXTS):
    prev = bf16(f"hidden_embed_{t}.npy").astype(np.float64)
    pn = np.linalg.norm(prev, axis=1)
    for L in range(66):
        cur = bf16(f"hidden_L{L:02d}_{t}.npy").astype(np.float64)
        cn = np.linalg.norm(cur, axis=1)
        c = (prev * cur).sum(1) / (pn * cn)
        rot[L, ti] = float(np.median(1.0 - c))
        prev, pn = cur, cn
    print(t, "done", flush=True)

diffs = []
for ti in range(6):
    g = np.median([rot[L, ti] for L in sorted(GLOBALS)])
    l = np.median([rot[L, ti] for L in range(66) if L not in GLOBALS])
    diffs.append(g - l)
diffs = np.array(diffs)
mean_diff = float(diffs.mean())
# exact one-sided sign-flip test over 64 assignments
flips = np.array([[1 if (m >> i) & 1 else -1 for i in range(6)] for m in range(64)])
null = (flips * np.abs(diffs)).mean(1)
p = float((null >= mean_diff).sum() / 64)
# rotation flip-band record: adjacent change of six-text median into same-scope destinations
med = np.median(rot, axis=1)
adj = np.abs(np.diff(med))                       # change into destination layer L (from L-1)
elig = [L for L in range(1, 66)
        if (L in GLOBALS) == ((L - 1) in GLOBALS)]
Lmax = max(elig, key=lambda L: adj[L - 1])
out = {"mean_paired_contrast": mean_diff, "p_one_sided_exact": p,
       "per_text_diffs": diffs.tolist(),
       "rot_L41_prose": rot[41, 0], "rot_L65_prose": rot[65, 0],
       "rotation_flipband_argmax_destination": int(Lmax),
       "rotation_flipband_in_L13_28": bool(13 <= Lmax <= 28),
       "median_rotation_table": np.round(rot, 6).tolist()}
print(json.dumps(out, indent=1))
dest = ROOT / "analysis" / "round5" / "dump_science_batch" / "verification_rotation.json"
dest.write_text(json.dumps(out, indent=1))
