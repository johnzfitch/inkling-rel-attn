# Question ledger

The purpose of this project, stated once: **map Inkling's learned positional
transport end to end — what structure SGD actually built, how it behaves in
operation, and what happens when pieces are removed — with every question given
an explicit registered answer.** Everything else (instruments, corpora,
verifiers, errata) exists to serve a row in this ledger.

## Working rules

1. **Every unit of work names its row before it starts.** Instrument, corpus,
   or validity work must name the question row it unblocks. Work that cannot
   name a row does not run.
2. **A verdict is written the moment it is computed.** Wrong-and-registered is
   a recoverable failure; unanswered is not. Retractions are edits with
   provenance stamps, never deletions.
3. **This file is updated in the same commit as any artifact that changes a
   row's status.** If a result lands and this file doesn't change, the commit
   is incomplete.
4. **New registrations must outrank idle rows.** A new spec or prereg states,
   in its own text, why it takes precedence over every OPEN row currently
   marked runnable-now. Otherwise the runnable-now rows go first.

Status vocabulary: **OPEN** (no verdict), **ANSWERED** (verdict on record),
**PROVISIONAL** (verdict on record; awaiting A6 re-certification),
**FORECAST** (registered prediction testable only outside our hardware),
**RETIRED** (question dissolved with reason).

## OPEN — runnable now (no dependence on the corrected recapture)

*(empty — the runnable queue is cleared)*

## OPEN — gated

| Row | Question | Gate | Next action |
|---|---|---|---|
| LF5-b | Bracket matching above baseline (the science LF5 was built for) | BPE starved the corpus (n=1) | engineered long-range bracket corpus, then rerun |
| R5-D | Ablation campaign (bias-off, carrier-out, near/far, heal-the-wall, head-class) | execution methods frozen; raw-row tail rejected pre-GPU and corrected by `ROUND5_R5D_TAIL_AMENDMENT_A.md`; runner not yet built | build/authenticate amended tail and runner, then execute 67 arms |

## FORECAST

| Row | Prediction on record |
|---|---|
| LF10 | At 1M context, tau=1.2056 steepens the L65 cliff ~8×→~13×; seam ratios 1.48–1.65 → 1.60–1.83. Testable by anyone with the hardware. |

## Decision queue (blocking human/agent calls, not compute)

- **D1 — scope RESOLVED; execution COMPLETE and CERTIFIED with D4**
  (`registrations/ROUND5_CAPTURE_SCOPE_D1.md`; production handoff:
  `analysis/round5/widened_capture/README.md`). The widened runner, A8
  Git-blob/parity/shard/package gate, paired builder/freeze, independent
  validator, normalized replay inputs, and all 66 × 24 LF5 needle rows are
  implemented, including the registered D4 extension. The 2,324-artifact
  production pass and independent validation passed at capture head `e291453`;
  LF5 replay is bitwise at all 66 layers (`analysis/round5/widened_capture/`).
- **D4 — scope RESOLVED, COMPLETE, and CERTIFIED: full-depth lossless states**
  (`registrations/ROUND5_CAPTURE_SCOPE_D4.md`). Capture BF16 embedding state
  plus all 66 layer outputs for all eight arms; six v1 texts are confirmatory,
  paired arms descriptive only. All 536 states passed independent hashes,
  shape/dtype checks, and BF16 finiteness with zero non-finite words.
- **D3 — engineered bracket corpus** for LF5-b (spec: pairs at d ∈ 64–2048,
  no BPE-merged pairs, distance-matched controls).

## ANSWERED — pending independent confirmation (promotion rule)

Verdicts on record; the LEFTFIELD promotion rule (control pass + independent
re-derivation from raw dumps) is not yet satisfied.

- **LF6: prediction failed — no MI mimicry** (robust to window choice per
  audit). "Crest → single-exp tail" and "content matching does long range"
  are now explicitly labeled interpretations, not LF6 results.
- **R4-W / R4-N:** as reported; same confirmation gap.
- **LF11:** bundle frozen; the naive-tool demonstration is itself the
  specimen — an independent rerun of the demo would complete the loop.
- **R5-A (Atlas): BUILT** (`analysis/round5/atlas.json` + poster, commit
  2792639; corrected refresh in this result batch). Descriptive glue with no
  claims of its own, so the promotion rule does not apply. All 66 layers now
  cite the A6-corrected LF4 and in-situ artifacts; no provisional cells remain.
- **LF4 corrected re-certification: no verdict flips.** Function words remain
  narrow and pass (effect −0.0215, Holm p=0.00360); pronouns remain a
  direction-reversed failure (−0.0527), closers remain null (+0.00039), and
  all random-mask controls remain null. Sentence starts remain strongly wide
  (+0.1223, BH p=0.000167).
- **Corrected in-situ findings: one registered flip, the rest retained.** The
  d=1024 seam remains positive and bias-attributable at all 11 globals with
  exact zero bias outside extent; the 512 echo remains 33/33; L65's wall is
  8.351× with bias versus 1.023× content-only; and every global retains the
  beyond-horizon heartbeat (minimum head-q97.5 ratio 1.684). The registered
  A6 no-flip expectation fails because the needle seam test is now significant
  only at L5 (L65 p=0.0999), not at both L5/L65.
- **P-e: the primary dose law passed; both directional consequences failed.**
  Deep boundary score rises with retired-context dose (rho +0.1041,
  structure-block p=0.0322, block-bootstrap 95% [+0.0058,+0.2022]); shallow,
  crossed-math, and frozen-random controls are null. Single-thread does not
  exceed multi-conversation (+0.0801 vs +0.0840), and seven higher-scope
  boundaries are not wider than ordinary starts (raw contrast −0.0375; only
  2/7 have exact dose-bin controls).
- **P-f: P-f1/P-f2 failed; P-f3 passed.** URLs (+0.244) and proper-noun
  proxies (+0.184) pass their anchor subtests, but gratitude (+0.0957) misses
  significance (p=0.0656), so the all-three P-f1 conjunction fails. The
  boundary transient itself is large and precise (deep−shallow +0.2574;
  bootstrap lower +0.2398), but its registered offsets-6..11 median is
  +0.000976 rather than <=0, so P-f2 fails at the frozen zero boundary. Label
  colons narrow in both shallow (−0.334) and deep (−0.244), so P-f3 passes.

## ANSWERED — corrected-capture certified (2026-07-17)

Full chain on the A6/A7/A8-corrected four-arm capture: attempt-2 validation
(git-blob authenticated) → novelty → compute → analyze → independent
confirmation (`analysis/round5/corpus_v2_corrected/verification.json`,
passed, zero errors). All true-null controls passed in both families.

- **P-v2-1..4 re-certified, every verdict identical** to the provisional run —
  A6's registered no-flip expectation confirmed. Novelty: slack 2.557 /
  math-LLM 1.617 nats (v2.1 arm 2.572, also novel; prose 0.114, memorized,
  outside the gate). Aperture–surprisal law holds: median ρ +0.148 (16/16
  bins) slack, +0.120 (13/16) math. Message-starts widen (+0.051, Holm
  0.0016). P-v2-4 still fails jointly: pronouns narrow (−0.040) but function
  words are null (−0.004).
- **P-d1 CONFIRMED — the delimiter depth flip is real.** On the new
  multi-conversation arm: speaker labels shallow −0.082, deep +0.133 (Holm
  4e-4). Replicated on different conversations under corrected arithmetic.
- **P-d2 FAILED on its terminal clause.** Pronoun shallow narrowing is strong
  (−0.094, p 1e-4) but the effect does NOT vanish at L53/L59 (+0.040 vs the
  registered |·|<0.03) — the two-band story was right, the vanishing wrong.
- **P-d3 FAILED with a sign reversal.** First-content deep effect +0.084
  (predicted ≤ −0.05). The v2.0 single-thread narrowing did not transfer.
- **P-d4 CONFIRMED.** Prose sentence-starts widen deep (+0.174, terminal
  +0.193). Combined with P-d3's reversal, the revised picture: content-onset
  widening in deep globals is the general phenomenon; the v2.0 single-thread
  arm's first-content narrowing was the outlier, not the rule.
- **LF5 production replay re-certified on the widened capture.** All 66 × 24
  registered needle rows remain bitwise equal across 830,472,192 FP16
  attention values after the independently validated corrected capture.

## ANSWERED — certified (corrected-capture dump science, 2026-07-18)

Six rows from the certified D1+D4 capture, run under the outcome-blind
execution plan at `88dd002` (estimators frozen pre-outcome), answered by the
first analyst (`d3cca8f`) and independently re-derived from raw dumps by the
second analyst with no producer imports
(`analysis/round5/dump_science_batch/verification_*.json`). An
execution-order deviation (these ran before the mechanical re-certifications)
is on record in the batch RESULTS.

- **LF3: prediction failed into the registered surprise branch — there IS an
  absolute-position clock in the r-channel.** Log-shaped tail signal, max
  |r| = 0.975 (L59, random text), search-wide circular-shift p = 1/127; BOS
  transient present. The prose control clocks too (0.939 at L11, same p
  floor). Exploratory, unregistered: 74 coordinates exceed |r| 0.9 at L59,
  concentrated on relative dims 5/9/10 across heads; every global layer from
  L11 on carries a ≥0.96 clock; the clock is NOT communal carrier component 1
  read out (0/74 clock coordinates track the carrier above 0.9; the carrier
  scalar itself drifts only at r = −0.72).
- **LF8: all three registered clauses failed.** Minimum cross-text mean-r
  cosine 0.385 at L53 (prose vs needles) — the disclosed anomaly is real
  under corrected arithmetic; no global/local depth-family split (p ≈ 0.75);
  47 chirality candidates under the frozen census. Exploratory: L53 mean-r
  splits the texts into camps — {prose, multilingual} ≥0.97 and
  {templated, needles} ≥0.93, cross-camp down to 0.385.
- **LF9: inverted.** The learned bias REDUCES far-field (d > 256) attention
  share at every registered mid-depth global (−0.036 to −0.086) and
  catastrophically at L65 (−0.410); the far-share peak sits at L11, not
  mid-depth (though L65 is the unique global minimum as predicted). The bias
  is a near-field concentrator everywhere tested.
- **R5-B: depth profile passed; text centrality failed.** Realized transport
  is most prompt-dependent at L41 (unique maximum; mid-band median 0.365 vs
  early 0.255) and collapses at L65 (0.164). Ordering inverted: code is the
  most central text (0.133), needles the farthest (0.325); random is not
  nearest.
- **R5-C lifecycle: half the registered pattern.** Channel 4786 has a clean
  sustained broadcast onset at L26 (inside the flip band; census and raw
  states agree bitwise); channel 3290 never crosses the 30k threshold in the
  window (no onset); the L39/40 handoff is gradual/mixed (0/6 texts sharp).
- **R5-C geometry: carrier prediction failed, one clause passed.** Communal
  component 1 carries up to 10.8% of hidden variance (L20/needles) against
  the registered <1% ceiling, with median positional read-energy share 64.1%;
  the flip-band discontinuity failed both metrics (PR max at L64, rotation
  max at L1); globals-rotate-more passed (+0.0235, exact sign-flip p = 1/64,
  positive in all six texts, driven by templated/needles).

## ANSWERED — certified (weight-level; unaffected by A5/A6)

- **LF1: no power-of-two pips, independently certified.** A standalone
  verifier rebuilt all mode-0 and raw-row pip curves from the 66 original
  projection banks (no producer imports): 0/341 published-family survivors
  and 0/341 max-row diagnostic survivors. The d=16 addition remains disclosed
  as unregistered. `analysis/round5/lf1/verification.json`.
- **LF2: inverted paragraph-scale result, independently certified.** The
  verifier rebuilt the log curves from weights, regenerated the registered
  IID null and both fixed-seed circular-block nulls, and reproduced the exact
  corrected hierarchy: 8 IID paragraph hinges, 7 block-robust hinges, 6
  rise-to-decay crests, 0 sentence knees. Prediction failed.
  `analysis/round5/lf2/verification.json`.
- **LF7: hidden-side no-fork verdict, independently certified.** From the
  self-contained K_STORE=1024 dump, the verifier rederived every primary and
  fixed-k distance via the projector-overlap identity (rather than the
  producer's SVD formula), rebuilt the sketch metric, and reproduced the
  clean null, 0/8 parent agreement, and no-fork decision. Certification does
  **not** promote a parent identity or the curve-side L47/L51 observation.
  `analysis/round5/lf7/verification.json`.
- **Rounds 1–3:** mechanism ground truth from source; transport is low-rank
  (~1.5–3 of 16), decay-dominated; positional energy concentrates early; no
  learned signal beyond d=1024 anywhere.
- **Round 4 battery + fingerprints:** exp-decay shape without any named
  scheme's ladder (cross-head rates ~2.7× vs RetNet ~128×); no genuine
  oscillation (fractional-cycle wins audited out; P3 retracted);
  C1–C7 answered with corrections stamped in place (C4 redo → 0.32 marginal;
  C5 survives at 1.93×; C2 weight-side metric retired as uninformative).

---

*Round-level history and full verdict provenance live in the round specs,
amendments, and `analysis/`; this ledger is the index, not the archive.*
