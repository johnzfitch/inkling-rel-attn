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

| Row | Question | Inputs | Next action |
|---|---|---|---|
| R5-A | The Atlas: one navigable end-to-end map | existing dumps | build atlas.json + poster; mark in-situ cells provisional |

## OPEN — gated

| Row | Question | Gate | Next action |
|---|---|---|---|
| R5-C | Activation-space geometry (intrinsic dim, carrier share, flip-band discontinuity) | corrected rvec now exists (4 arms); hidden-state side still reads the provisional round5 capture | run rvec-side now; hidden-state side per decision D1 |
| LF3 | Absolute-position counter hiding in the r-channel? | needs rvec on 06_random — **not in recapture scope** | decision D1 below |
| LF8 | Fiber orientation: content-stable anchor? chirality? | needs rvec across the 6 v1 texts — **not in recapture scope** | decision D1 below |
| LF9 | Long-range bandwidth budget per layer | reads tier2 meters — **not in recapture scope** | decision D1 below |
| R5-B | Does realized transport change with input? (depth profile) | reads tier2 meters — **not in recapture scope** | decision D1 below |
| LF5-b | Bracket matching above baseline (the science LF5 was built for) | BPE starved the corpus (n=1) | engineered long-range bracket corpus, then rerun |
| R5-D | Ablation campaign (bias-off, carrier-out, near/far, heal-the-wall, head-class) | own GPU campaign on corrected code | schedule after recapture |
| P-e | Does deep-global boundary widening scale with the amount of prior context a boundary retires? | REGISTERED (`ROUND5_APERTURE_CONTEXT_DOSE_PREREG.md`); paired group-channel arms not yet built | build paired renders from the largest group channel, freeze boundaries/dose, capture, evaluate |
| P-f | Are the deep-aperture extremes referential anchors (URLs, names, gratitude)? Is the depth flip a boundary-locked transient? Does delimiter punctuation narrow everywhere? | REGISTERED (`ROUND5_APERTURE_ANCHOR_PREREG.md`); rides the P-e capture | freeze classes on the fresh arms, evaluate after the P-e capture |

## FORECAST

| Row | Prediction on record |
|---|---|
| LF10 | At 1M context, tau=1.2056 steepens the L65 cliff ~8×→~13×; seam ratios 1.48–1.65 → 1.60–1.83. Testable by anyone with the hardware. |

## Decision queue (blocking human/agent calls, not compute)

- **D1 — RESOLVED: widen** (`registrations/ROUND5_CAPTURE_SCOPE_D1.md`). The
  P-e capture pass also takes v1-text rvec + corrected tier2 meters; every
  provisional stamp re-certifies and LF3/LF8/LF9/R5-B unblock when it lands.
- **D3 — engineered bracket corpus** for LF5-b (spec: pairs at d ∈ 64–2048,
  no BPE-merged pairs, distance-matched controls).

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

## PROVISIONAL — awaiting re-certification on v1-text captures

The corpus-v2 re-certification above confirms the A6 no-flip expectation where
tested; these rows still cite dumps from the uncorrected capture (Δp ≤ 0.025).

- **LF4 (flagship): ANSWERED, prediction failed informatively.** The aperture
  is real and content-responsive but tracks segment-integration scope, not
  referential need: sentence starts widen (+0.117, 62/64 heads), pronouns and
  function words narrow, closers null.
- **LF5 (instrument): ANSWERED.** Offline rows validated bitwise on the A5 GPU
  replay backend (24.24B values). CPU backend demoted (Amendment A5).
- **In-situ findings:** d=1024 seam real and bias-caused; needle retrieval
  seam-robust except L5/L65; d=512 echo (33/33); L65 terminal wall; heartbeat
  induction beyond the bias horizon.

## ANSWERED — certified (weight-level; unaffected by A5/A6)

- **LF11 (2026-07-17): prediction PASSED — fourth tool caught, first caught
  prospectively.** The damped-sinusoid BIC race, run naively, "detects
  oscillation" in 8/11 globals (ΔBIC up to 584); all are fractional-cycle
  phantoms (0.004–0.54 cycles in window); the cycle-audit rule kills all 8.
  Bundle frozen: 148 hashed channels + absent/present certifications +
  calibration rules. `analysis/round5/lf11/RESULTS.md`.
- **R4-W (2026-07-17): oversampled but not band-limited.** Mode-0 f90 median
  0.0195 cycles/token (~4% of Nyquist), yet 2×/4× decimation costs 7.3/9.1%
  RMS — a real broadband residue rides the smooth kernel. L5/L17 mode-0 are
  broadband outliers. `analysis/round4/RESULTS_BANDLIMIT_NEARFIELD.md`.
- **R4-N (2026-07-17): P5 supported — near field is its own regime.** Handoff
  discontinuity median 0.37 far-RMS (q90 1.22, max 3.1 at L48/L64); exactly
  two motifs (silhouette 0.514), depth-split: shallow "self-suppress" (d=0
  pinned to zero) vs deep "self-inclusive" (mild d=1 peak). Same file.
- **LF2 (2026-07-17): prediction FAILED inverted — paragraph crest, no
  sentence scale.** 8/11 globals have Holm-significant breakpoints in the
  frozen paragraph range [21,160] (median d≈116), 0/11 in the sentence range;
  slope is slightly POSITIVE before each break — the kernel rises to a
  paragraph-scale crest, then decays. `analysis/round5/lf2/RESULTS.md`.
- **LF6 (2026-07-17): prediction FAILED — no MI mimicry.** Prose rank-corr
  0.33–0.66 (needed >0.9, 0/11); far field d≥32 is SINGLE-exponential in
  9/11 globals (2-exp adds nothing; power law never wins). SGD didn't copy
  the corpus's power law — long range is delegated to content matching.
  `analysis/round5/lf6/RESULTS.md`.
- **LF7 (2026-07-17): prediction FAILED → the registered surprise: no fork.**
  MTP nearest-parent distances sit at the 49th–73rd percentile of the
  trunk-trunk null under two independent metrics that disagree on argmin
  (0/8) — the drafters re-derived transport on read directions unrelated to
  any trunk layer's. Convergent shape, not inheritance: a mini universality
  result inside one checkpoint. Curve-side L47/L51 bank convergence recorded
  as an unpromoted secondary (re-derivation discriminator saturated).
  `analysis/round5/lf7/RESULTS.md`.
- **LF1 (2026-07-17): clean null, prediction PASSED.** No power-of-two pip
  survives Holm in 341 tests (globals + locals); the pre-flagged d=128 is
  flat (best raw p 0.040, Holm 1.0); the d=512 echo does not register as a
  pip — band step and pip statistics dissociate correctly.
  `analysis/round5/lf1/RESULTS.md`.
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
