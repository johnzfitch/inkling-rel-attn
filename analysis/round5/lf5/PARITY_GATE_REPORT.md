# Round 5 LF5 parity-gate report

Status: **failed as preregistered; semantic LF5 computation stopped**.

The full six-text, 66-layer normalized-BF16 recapture completed in 18.65
minutes. Its independent, non-fast validator checked all 1,260 artifacts and
reported no errors. The capture includes 396 normalized attention inputs, 396
massive-activation censuses, 396 bitwise-verified r-vector replays, 66
bitwise-verified needle-row replays, and six true next-token NLL records.

Tier 1 (same-GPU BF16 replay) passed all 66 layers and all 24 registered needle
queries per layer. All 830,472,192 stored FP16 probability words matched the
original capture bit for bit, including signed zero.

Tier 2 (production CPU/FP32) stopped at the first sentinel, L0, as required by
the frozen `--stop-on-fail` rule. Across 24 queries x 64 heads:

| Registered check | Threshold | L0 result | Status |
|---|---:|---:|---|
| maximum elementwise `abs(delta p)` | `<= 1e-3` | `0.0054964423` | fail |
| maximum row-sum error | `<= 1e-3` | `2.6403e-7` | pass |
| per-head argmax agreement | `100%` | `98.9583%` (16 mismatches) | fail |
| maximum `KL(original || offline)` | `<= 1e-6` | `7.50385e-5` | fail |

The diagnosis preserved the failed report and did not alter any threshold.
Restoring every original BF16 operator boundary on CPU improved the tail but
still failed (`max abs(delta p)=0.00152568`, six argmax mismatches, and
`max KL=6.92443e-5`). Native CPU BF16 produced the same numbers. Conversely,
the exact CUDA replay control passed all four numerical thresholds even before
its mandated FP16 storage cast. This localizes the discrepancy to
CPU-versus-CUDA reduction order at BF16 boundaries; tensor wiring, grouping,
masks, capture integrity, and the comparator are independently controlled.

Consequences under amendment A1:

- LF5 is numerically characterized but is not the registered verified offline
  deliverable.
- No production semantic-row dump or registered bracket readout was run.
- The LF4 executable's pre-outcome guard requires a passed LF5 confirmation, so
  no LF4 aperture or zoom-lens outcome was observed.
- Continuing either result-bearing path requires a new, public amendment; the
  failed registered result remains the primary record.

Machine-readable evidence is in `capture_validation.json`,
`parity_replay.json`, `parity_cpu_sentinels.json`,
`parity_backend_diagnostic_L00.json`, and `confirmation.json`.
