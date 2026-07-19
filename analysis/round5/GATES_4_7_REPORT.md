# Round 5 gates 4–7 — LF5 production and LF4 zoom lens

Executed 2026-07-17 under registration commit `34278b4`, implementation plan
commit `d4e2579`, and Amendment A5 commit
`7bf608d9971997a655a4f9cd46e3bc921ffb74b8`. No registered threshold or
decision rule was changed.

## Gate 4 — production LF5 rows: pass

The A5 GPU/BF16 replay backend sealed all 396 layer/text groups. The independent
full-dump validator checked 63,162 ragged attention rows and 24,236,544,640
stored FP16 values across 2,376 hashed files. It reported no errors; the worst
FP16 row-sum error was `0.0004202723503112793`.

- production manifest SHA-256:
  `27221e523fb6cd2ec8816af19e84015f7937a5278a983c17b32fef94ca258706`
- independent validation: `analysis/round5/lf5/row_dump_validation.json`
- the earlier Windows memmap-seal failure remains in
  `analysis/round5/lf5/production_attempts.json`
- the registered CPU/FP32 failure remains preserved and is not a production
  criterion under A5

## Gate 5 — LF5 bracket readout: methodology pass, prediction fail

The registered bracket statistic ran over all 11 global layers. The frozen
eligibility rule left one usable matched pair at every layer. Raw matched-opener
ratios were positive and reached `133.6089x` at L47, but no layer survived the
registered test: the smallest unadjusted one-sided p-value was `0.0820918`, and
zero of 11 layers were positive and Holm-significant. The prediction requiring
at least three significant layers therefore failed.

Independent confirmation passed: six CUDA/BF16 sentinel rows were FP16-bitwise
identical to capture, the independently coded bracket statistic reproduced the
main result exactly, the full row-dump validator was bound by manifest hash, and
the CPU failure remained explicit.

## Gate 6 — LF4 inputs: pass

All 462 frozen inputs (396 r-vector captures and 66 projection tables) passed
the registered shape, dtype, finiteness, and hash checks. The exact manifest was
committed publicly before LF4 computation:

- `analysis/round5/lf4/input_validation.json`
- SHA-256:
  `20dc754105823bc31aaabbba6f72f1589ed2c1c0389f49653494c5b6a1595c9b`

## Gate 7 — LF4 zoom lens: prediction fail; confirmation gate unclosed

The float64 aperture computation sealed all 396 layer/text artifacts in 703
seconds. The registered five-mid-global-layer contrasts were:

| Primary class | N | Registered effect | Holm p | Prediction |
|---|---:|---:|---:|---|
| code closers | 302 | -0.004296875 | 1.000000 | fail |
| prose pronouns | 413 | -0.050390625 | 1.000000 | fail (opposite sign) |
| prose function words | 1,843 | -0.021484375 | 0.00239976 | pass |

The three random-position-mask controls all remained null after Holm correction.
Because only one of three primary predictions passed, the LF4 flagship prediction
failed.

The independent scalar path recomputed raw aperture values from r-vectors and
projection weights to at most `4.440892098500626e-16` absolute error versus the
blocked production dump. The remaining gate failed at the frozen effect
reference:

| Class | Main full-bin-median effect | Frozen matched-sample effect | Independent p |
|---|---:|---:|---:|
| closers | -0.004296875 | +0.031222944 | 0.100190 |
| pronouns | -0.050390625 | -0.068067227 | 1.000000 |
| function words | -0.021484375 | -0.019482561 | 0.014299 |

For closers, the full-bin-median and frozen matched-control estimands have
opposite signs. The frozen verifier also defines its `effect_sign_agreement`
check against the registered prediction direction; consequently it marks the
pronoun arm false even though the main and independent effects are both
negative. That outcome-coupling is recorded here rather than silently repaired
after seeing the data. The closer sign disagreement alone is sufficient to
leave the independent confirmation gate false.

Therefore LF4 `methodology_passed` is false under the frozen Gate 7 verifier.
No LF4 claim is promoted. Secondary observations in `zoom_lens.json` remain
descriptive only until a separately registered follow-up resolves the effect
reference.

## Outcome

- LF5 is a validated A5 production instrument.
- The LF5 bracket prediction failed.
- The LF4 zoom-lens flagship prediction failed.
- LF4 raw aperture computation is numerically reproduced, but Gate 7 did not
  certify the class-effect readout.
- No threshold was relaxed, no failed attempt was removed, and no result was
  clipped, imputed, or reclassified.
