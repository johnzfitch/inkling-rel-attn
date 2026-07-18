# Round 5 Amendment A5 — LF5 production backend (2026-07-17)

Public amendment required by ROUND5_LF5_LF4_PLAN.md §7 after the registered
Tier-2 gate failure (analysis/round5/lf5/PARITY_GATE_REPORT.md).

**Record preserved:** the registered CPU/FP32 production gate FAILED at its
first sentinel (max |Δp| 5.5e-3 vs 1e-3; argmax 98.96% vs 100%; KL 7.5e-5 vs
1e-6) and that failure remains the primary record. No threshold is relaxed.

**Diagnosis accepted:** every CPU variant (FP32, BF16-boundary emulation,
native BF16) fails identically while the exact same-GPU replay passes all four
numeric thresholds even before the FP16 storage cast — the discrepancy is
CPU-vs-CUDA reduction order at BF16 operator boundaries, not wiring (Tier-1:
all 830,472,192 stored FP16 words bit-identical across 66 layers × 24 queries).

**Amendment:** the production LF5 instrument is the **GPU replay backend** —
same device class, dtypes, and chunk boundaries as the original capture, which
passed Tier-1 bitwise. The CPU path is demoted to a non-registered convenience
tool and must print its characterization (documented failure vs registered
thresholds) whenever used. All registered LF5/LF4 readouts (loci rows, bracket
statistic, aperture contrasts) run on the GPU backend. Predictions, loci,
classes, statistics, and decision rules are unchanged from the plan + §8.

**Corpus caveat registered while unblocking (from the A2a NLL capture):** the
model has the public-domain corpus texts substantially memorized (mean next-
token NLL: prose 0.11, multilingual 0.10, code 0.71; median prose 1e-4) while
random text sits at chance (12.54 ≈ ln 200k ≈ 12.2, which also certifies the
end-to-end pipeline). Consequence: content-matching readouts on 01/02/04 ride
partly on weights-memorized text, not in-context retrieval. Needle codewords
are immune by construction (nonsense tokens). Future corpus passes add one
post-cutoff (unmemorizable) text arm; registered as a standing amendment to
the corpus protocol.
