# Round 5 seven-experiment follow-up results

Status: **ANSWERED — INDEPENDENTLY VERIFIED**

- F7-1 signed stencil: **PASS**; all-head d0..3-only rescue `0.990`.
- F7-2 shoulder backups: **PASS**; interactions adjacent_23_29 `+0.602748`, adjacent_29_35 `+0.348319`, triple `+1.038303`.
- F7-3 r decomposition: **PASS**; mean-removal ratio `0.781`, non-carrier-mean ratio `0.796`.
- F7-4 head localization: **PASS**; top-16 stencil rescue `0.934`.
- F7-5 query patch: **PASS**; query rescue `0.541`, sham `-0.032`.
- F7-6 clock subspaces: **FAIL**; LOTO maximum kernel correlation `0.470`.
- F7-7 full vocabulary + fresh classes: **FAIL**; rank delta `+0.031144`, accuracy delta `-0.018271`, ECE20 delta `+0.004310`.

The independent verifier rehashed 563 artifacts and reproduced every reported
statistic and all seven verdicts from raw dumps with zero discrepancies. See
`verification.json`. The pre-A3 report and its five rejected float32-denominator
comparisons remain preserved under the `*_pre_a3` filenames.

## Second-analyst verification

The second analyst re-derived the campaign with fully separate code
(`scripts/round5_followup7_analyst2_verify.py`; neither the analyzer nor the
first verifier is imported): all 563 sealed artifacts rehashed against their
manifests, all 35 pooled arm costs recomputed in float64, every registered
bootstrap/permutation record regenerated from the registration-hash seeds
(effects, CIs, and p-values match to <2e-10), all 60 clock kernel-correlation
cells re-derived from the raw `rvec_pre` dumps, the matched-class contrasts
re-run on the fresh texts, the `bias_off_L29_fullvocab` exact-copy gate
confirmed bitwise against the certified R5-D parent, and all seven family
verdicts re-evaluated from the registered clauses. Zero errors; verdicts
agree: F7-1..F7-5 PASS, F7-6 FAIL, F7-7 FAIL. See
`analyst2_verification.json`.

Exploratory raw-dump patterns (non-registered) are recorded in
`exploration/patterns.py` / `patterns.out.json`. Highlights: d0-off damage is
tokenwise collinear with full bias-off (r 0.78–0.94, slope ~0.8); the mean
full-vocabulary rank shift (+168) is a `06_random` tail artifact (natural
texts ±2, median exactly 0; entropy *falls* on random); the fresh Slack text
shows the largest single-layer effect measured anywhere (+0.450 nats, top-1
accuracy −4.6 pp), so the overall L29 effect generalizes even though the
class geography does not; triple-knockout damage is a distinct failure mode
(correlation with single-L29 damage 0.17–0.70, half of prose tokens >1 nat)
that spares templated/needle texts; needle-recall damage tracks the
token-varying d=0 signal more than the static mean (d0-off reproduces 73% of
recall damage vs 21% for static-mean removal), unlike pooled NLL; per-text
clock directions share a dominant common mode (62–68% of six-direction
energy) with held-out directions only 46–95% inside the other-five span,
explaining the LOTO transfer failure; and five L29 heads carry *negative*
(anti-mute) stencil scores.
