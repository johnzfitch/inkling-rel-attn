# Round 5 R5-D causal ablations — results

Status: **ANSWERED_PENDING_INDEPENDENT_REDERIVATION**. All 72 arms were sealed before verdict computation.

| Registered clause | Verdict | Key value |
|---|---:|---:|
| Bias-off depth | FAIL | see JSON layer table |
| Carrier equivalence | FAIL | 16/16 required |
| Near dominates far | PASS | ratio 81.12260205133592 |
| Wall incidental at 8k | PASS | |ΔNLL| 9.78453e-05 |
| CK1 clock mechanism | PASS | L53 0.0000; L59 0.0000; sham 0.6343 |
| CK2 log-extremes cost | FAIL | rho 0.0542; lower -0.1579 |
| CK3 L65 exemption | PASS | |ΔNLL| 6.74975e-05 |

The two head-class arms, needle splits, all per-text costs, bootstrap intervals, and full locus-meter summaries are in `r5d_results.json`. No threshold was invented for descriptive readouts.

Promotion state: **certified** — independent second-analyst re-derivation
below, with one structural disclosure on CK1.

## Independent second-analyst re-derivation (verification.json)

No runner/analyzer imports (`scripts/round5_r5d_verify.py`). All 1,062 sealed
artifacts re-hash cleanly; the baseline readouts are bitwise-equal to the
certified capture NLL; all 72 pooled costs and per-text means reproduce to
1e-9; every 5,000-draw bootstrap interval regenerates exactly from its
registered seed rule; all seven registered verdicts reproduce, including the
near/far ratio to full precision (81.12260205133592) and CK2's Spearman and
bootstrap bound to 1e-9. Physical spot-checks: the wall-heal L65 meter shows
genuine attention mass beyond d=1024, and the near-off L29 meter records
exactly zero bias at d<4.

**CK1 structural disclosure (second analyst).** For the single-layer clock
arms, the pre-intervention r-vectors are bitwise the certified capture data
from which `G_L` was estimated as the OLS slope on the CK1 regressor.
Projecting the OLS-slope direction out of the fitting data annihilates the
regressor correlation in every coordinate identically, so the real-arm CK1
statistic was structurally incapable of failing (float64 demonstration: the
projected block means' correlation numerator is ≤3e-13; the reported 2e-8 is
fp16 quantization of the saved vectors). CK1's verdict stands as registered,
but its evidentiary content is re-scoped: it certifies the freeze operator's
algebra and the sham contrast (0.6343, unchanged drift), not an empirical
flattening. The empirical content — that the random-arm clock direction is
THE drift direction on other texts too — is registered as a cross-text
transfer test (`ROUND5_R5D_CLOCK_TRANSFER_PREREG.md`) on cells that remain
uninspected in the sealed dumps. Design error acknowledged by the second
analyst, who authored the clock amendment.
