"""LF2 — linguistic scales: sentence and paragraph knees in the kernel.

Registered prediction (ROUND5_LEFTFIELD_SPEC.md, blind): a sentence-scale knee
in >= 6/11 global layers; NO paragraph knee. Protocol: corpus scale
measurement FIRST and frozen, then segmented-regression knee detection on
log|mode-0| per global layer, null = knee detection re-run on smooth 2-exp
surrogates + residual jitter.

Method frozen before outcomes:
  - Scales stage (`scales`, must run first; output immutable): sentence and
    paragraph token-length distributions of 01_prose_en and 04_multilingual.
    Sentences split on [.!?] followed by whitespace; paragraphs on blank
    lines; token lengths counted with the model tokenizer via char offsets.
    The frozen "sentence range" ("paragraph range") = [min of the two texts'
    q25, max of the two texts' q75].
  - Knee stage: per global layer, y = log10(max(|mode-0(d)|, 1e-6 * peak)) on
    the search window d in [8, 400] — chosen to cover both candidate scales
    while excluding the KNOWN d=512 echo and extent-seam structure, which
    would otherwise dominate a breakpoint detector (disclosure: both are
    known findings, not new nulls).
  - Detector: continuous two-segment least squares over breakpoints
    d in [16, 392]; statistic = relative SSE improvement over one segment.
  - Null: fit a 2-exponential to |mode-0| on the window, build 200 surrogates
    (smooth fit + iid-resampled log-residuals, seeded), run the detector on
    each; p = fraction of surrogate improvements >= observed. Holm across the
    11 global layers. A significant knee "matches a scale" if its breakpoint
    lies inside that frozen range.

Dump-first: `dump` writes curves, fits, surrogate statistics before
`analyze` reads only the dump and the frozen scales file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
CORPUS = ROOT / "corpus"
DUMP_DIR = ROOT / "dumps" / "round5" / "lf2"
OUT_DIR = ROOT / "analysis" / "round5" / "lf2"
SCALES = OUT_DIR / "corpus_scales.json"
GLOBALS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
D_LO, D_HI = 8, 400
BREAK_LO, BREAK_HI = 16, 392
# 200 surrogates gave a p floor of 1/201, whose Holm-corrected minimum across
# 11 layers is 0.0547 — structurally incapable of reaching 0.05. Raised to
# 2000 BEFORE any verdict was recorded (resolution fix, not a threshold move).
SURROGATES = 2000
SEED = 0x1F2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def token_lengths(text: str, spans: list[tuple[int, int]]) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(CORPUS / "tokenizer.json"))
    enc = tok.encode(text)
    starts = np.array([a for a, _ in enc.offsets])
    lengths = []
    for a, b in spans:
        n = int(np.searchsorted(starts, b) - np.searchsorted(starts, a))
        if n > 0:
            lengths.append(n)
    return lengths


def scales_command() -> None:
    if SCALES.exists():
        raise SystemExit(f"REFUSING to overwrite frozen scales: {SCALES}")
    record: dict = {"kind": "round5_lf2_frozen_scales", "schema_version": 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "definitions": {
                        "sentence": r"split on [.!?]+ followed by whitespace",
                        "paragraph": "split on blank lines",
                        "range": "[min(q25 of both texts), max(q75 of both texts)]"},
                    "texts": {}}
    ranges = {"sentence": [[], []], "paragraph": [[], []]}
    for name in ("01_prose_en", "04_multilingual"):
        text = (CORPUS / f"{name}.txt").read_text(encoding="utf-8")
        sent_spans, pos = [], 0
        for match in re.finditer(r"[.!?]+(?=\s)|\Z", text):
            end = match.end()
            if end > pos:
                sent_spans.append((pos, end))
            pos = end
        para_spans = []
        pos = 0
        for match in re.finditer(r"\n\s*\n|\Z", text):
            if match.start() > pos:
                para_spans.append((pos, match.start()))
            pos = match.end()
        entry = {}
        for unit, spans in (("sentence", sent_spans), ("paragraph", para_spans)):
            lengths = token_lengths(text, spans)
            q25, q50, q75 = (float(np.percentile(lengths, q)) for q in (25, 50, 75))
            entry[unit] = {"n": len(lengths), "q25": q25, "median": q50, "q75": q75}
            ranges[unit][0].append(q25)
            ranges[unit][1].append(q75)
        record["texts"][name] = entry
        record["input_sha256"] = record.get("input_sha256", {})
        record["input_sha256"][f"{name}.txt"] = sha256_file(CORPUS / f"{name}.txt")
    record["frozen_ranges"] = {
        unit: [min(lo), max(hi)] for unit, (lo, hi) in ranges.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCALES.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record["frozen_ranges"], indent=2))


def mode0(proj: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(proj.astype(np.float64), full_matrices=False)
    curve = vt[0]
    if np.mean(curve[:32]) < 0:
        curve = -curve
    return curve


def detect(y: np.ndarray, d: np.ndarray) -> tuple[float, int]:
    ones = np.ones_like(d, dtype=np.float64)
    base = np.linalg.lstsq(np.stack([d, ones], 1), y, rcond=None)[1]
    sse1 = float(base[0]) if len(base) else float(np.sum((y - y.mean()) ** 2))
    best, best_b = np.inf, -1
    for b in range(BREAK_LO, BREAK_HI + 1, 2):
        left = d <= b
        hinge = np.where(left, 0.0, d - b)
        design = np.stack([d, hinge, np.ones_like(d, dtype=np.float64)], 1)
        res = np.linalg.lstsq(design, y, rcond=None)
        sse = float(res[1][0]) if len(res[1]) else float(
            np.sum((design @ res[0] - y) ** 2))
        if sse < best:
            best, best_b = sse, b
    return (sse1 - best) / max(sse1, 1e-300), best_b


def two_exp(d, a1, r1, a2, r2):
    return a1 * np.exp(-r1 * d) + a2 * np.exp(-r2 * d)


def dump_command() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = np.arange(D_LO, D_HI + 1, dtype=np.float64)
    out: dict[str, np.ndarray] = {"d": d}
    sources = {}
    stats = {}
    for layer in GLOBALS:
        path = WEIGHTS / f"layer{layer:02d}_rel_logits_proj.npy"
        sources[path.name] = sha256_file(path)
        curve = np.abs(mode0(np.load(path)))[D_LO:D_HI + 1]
        floor = 1e-6 * curve.max()
        y = np.log10(np.maximum(curve, floor))
        improvement, breakpoint = detect(y, d)
        p0 = (curve[0], 0.05, curve.max() * 0.01, 0.003)
        try:
            params, _ = curve_fit(two_exp, d, curve, p0=p0, maxfev=20000,
                                  bounds=(0, np.inf))
        except Exception:
            params = np.array(p0)
        smooth = np.log10(np.maximum(two_exp(d, *params), floor))
        residuals = y - smooth
        null_imp = np.empty(SURROGATES)
        null_break = np.empty(SURROGATES, dtype=int)
        for k in range(SURROGATES):
            surrogate = smooth + rng.choice(residuals, size=len(residuals))
            null_imp[k], null_break[k] = detect(surrogate, d)
        out[f"y_L{layer:02d}"] = y
        out[f"smooth_L{layer:02d}"] = smooth
        out[f"null_imp_L{layer:02d}"] = null_imp
        out[f"null_break_L{layer:02d}"] = null_break
        stats[f"L{layer:02d}"] = {"improvement": improvement,
                                  "breakpoint": breakpoint,
                                  "two_exp_params": [float(x) for x in params]}
        print(f"L{layer:02d}: imp={improvement:.4f} at d={breakpoint}", flush=True)
    np.savez(DUMP_DIR / "lf2_knees.npz", **out)
    manifest = {"kind": "round5_lf2_knee_dump", "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "observed": stats, "surrogates": SURROGATES, "seed": SEED,
                "window": [D_LO, D_HI], "break_range": [BREAK_LO, BREAK_HI],
                "source_sha256": sha256_file(Path(__file__)),
                "input_sha256": sources,
                "dump_sha256": sha256_file(DUMP_DIR / "lf2_knees.npz")}
    (DUMP_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("dump complete")


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def analyze_command() -> None:
    manifest = json.loads((DUMP_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dump_sha256"] != sha256_file(DUMP_DIR / "lf2_knees.npz"):
        raise RuntimeError("dump hash mismatch")
    scales = json.loads(SCALES.read_text(encoding="utf-8"))
    sent_lo, sent_hi = scales["frozen_ranges"]["sentence"]
    para_lo, para_hi = scales["frozen_ranges"]["paragraph"]
    z = np.load(DUMP_DIR / "lf2_knees.npz", allow_pickle=False)
    rows = []
    for layer in GLOBALS:
        observed = manifest["observed"][f"L{layer:02d}"]
        null = z[f"null_imp_L{layer:02d}"]
        p = float((1 + np.sum(null >= observed["improvement"])) / (len(null) + 1))
        rows.append({"layer": layer, **observed, "p": p})
    for row, adj in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adj
        row["significant"] = bool(adj < 0.05)
        row["sentence_scale"] = bool(row["significant"]
                                     and sent_lo <= row["breakpoint"] <= sent_hi)
        row["paragraph_scale"] = bool(row["significant"]
                                      and para_lo <= row["breakpoint"] <= para_hi)
    n_sentence = sum(r["sentence_scale"] for r in rows)
    n_paragraph = sum(r["paragraph_scale"] for r in rows)
    report = {"kind": "round5_lf2_knees", "schema_version": 1,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dump_sha256": manifest["dump_sha256"],
              "scales_sha256": sha256_file(SCALES),
              "source_sha256": sha256_file(Path(__file__)),
              "frozen_ranges": scales["frozen_ranges"],
              "layers": rows,
              "n_sentence_knees": n_sentence,
              "n_paragraph_knees": n_paragraph,
              "prediction": {
                  "registered": "sentence knee in >=6/11 globals; no paragraph knee",
                  "sentence_clause": bool(n_sentence >= 6),
                  "paragraph_clause": bool(n_paragraph == 0),
                  "passed": bool(n_sentence >= 6 and n_paragraph == 0)}}
    (OUT_DIR / "lf2_knees.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for r in rows:
        print(f"L{r['layer']:02d}: imp={r['improvement']:.4f} d={r['breakpoint']} "
              f"holm={r['p_holm']:.4f} sig={r['significant']} "
              f"sent={r['sentence_scale']} para={r['paragraph_scale']}")
    print("sentence knees:", n_sentence, "| paragraph knees:", n_paragraph,
          "| prediction passed:", report["prediction"]["passed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["scales", "dump", "analyze"])
    args = parser.parse_args()
    {"scales": scales_command, "dump": dump_command,
     "analyze": analyze_command}[args.stage]()


if __name__ == "__main__":
    main()
