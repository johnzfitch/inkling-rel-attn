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
| LF7 | Are MTP drafter layers forks of specific trunk layers or fresh solutions? | weights only | run Procrustes/nearest-parent vs trunk-pair null |
| LF1 | Does the table treat power-of-two distances specially (pips)? | weights | z-score pips vs all-d null, multiple-comparison controlled |
| LF2 | Does the kernel have sentence/paragraph knees? | weights + corpus stats | freeze corpus scale distributions FIRST, then knee detection |
| LF6 | Is the 2-exp kernel a quadrature of corpus MI(d)? | weights + corpus | measure MI(d) frozen estimator; k-exp vs power-law BIC |
| LF11 | Which of our own tools false-positives on the certified null? | curation | freeze the null-benchmark bundle + certification statement |
| R5-A | The Atlas: one navigable end-to-end map | existing dumps | build atlas.json + poster; mark in-situ cells provisional |
| R4-W | Whittaker band-limit section (ROUND4_SPEC, never run) | weights | run as specified |
| R4-N | Near-field battery section (ROUND4_SPEC, never run) | weights | run as specified |

## OPEN — gated

| Row | Question | Gate | Next action |
|---|---|---|---|
| P-d1..4 | Delimiter depth flip; pronoun band; first-content narrowing; prose transfer | corrected capture + missing corrected verifier | agent lands verifier; run GPU pass; evaluate |
| P-v2-1..4 re-eval | Do the corpus-v2 verdicts survive corrected arithmetic? | same | re-run unchanged definitions on corrected capture |
| R5-C | Activation-space geometry (intrinsic dim, carrier share, flip-band discontinuity) | corrected capture (rvec + hidden) | run after capture; channel-lifecycle prereg already registered |
| LF3 | Absolute-position counter hiding in the r-channel? | needs rvec on 06_random — **not in recapture scope** | decision D1 below |
| LF8 | Fiber orientation: content-stable anchor? chirality? | needs rvec across the 6 v1 texts — **not in recapture scope** | decision D1 below |
| LF9 | Long-range bandwidth budget per layer | reads tier2 meters — **not in recapture scope** | decision D1 below |
| R5-B | Does realized transport change with input? (depth profile) | reads tier2 meters — **not in recapture scope** | decision D1 below |
| LF5-b | Bracket matching above baseline (the science LF5 was built for) | BPE starved the corpus (n=1) | engineered long-range bracket corpus, then rerun |
| R5-D | Ablation campaign (bias-off, carrier-out, near/far, heal-the-wall, head-class) | own GPU campaign on corrected code | schedule after recapture |

## FORECAST

| Row | Prediction on record |
|---|---|
| LF10 | At 1M context, tau=1.2056 steepens the L65 cliff ~8×→~13×; seam ratios 1.48–1.65 → 1.60–1.83. Testable by anyone with the hardware. |

## Decision queue (blocking human/agent calls, not compute)

- **D1 — recapture scope.** The corrected GPU pass covers rvec+NLL for
  07/08/07b/01_prose only. LF3, LF8, LF9, R5-B consume v1-text rvec and tier2
  meters that stay provisional (≤0.025 Δp) unless the pass is widened. Choose:
  widen the pass (add v1 rvec + meter re-run) or run those rows on provisional
  dumps with the deviation disclosed.
- **D2 — corrected verifier.** Nothing yet produces
  `corpus_v2_a6_capture_independent_validation`; the corrected pipeline
  hard-requires it. In flight with the agent.
- **D3 — engineered bracket corpus** for LF5-b (spec: pairs at d ∈ 64–2048,
  no BPE-merged pairs, distance-matched controls).

## PROVISIONAL — answered, awaiting A6 re-certification

Registered expectation (A6): effects 0.04–0.35 vs worst-case Δp 0.025 — no
verdict is expected to flip. Re-certification = corrected capture + re-run.

- **LF4 (flagship): ANSWERED, prediction failed informatively.** The aperture
  is real and content-responsive but tracks segment-integration scope, not
  referential need: sentence starts widen (+0.117, 62/64 heads), pronouns and
  function words narrow, closers null. Aperture–surprisal law: Spearman
  ~0.11–0.15 within 512-token bins, registered and replicated on corpus v2.
- **LF5 (instrument): ANSWERED.** Offline rows validated bitwise on the A5 GPU
  replay backend (24.24B values). CPU backend demoted (Amendment A5).
- **P-v2-1..3 pass, P-v2-4 fail jointly** (commit 2ffbc30).
- **In-situ findings:** d=1024 seam real and bias-caused; needle retrieval
  seam-robust except L5/L65; d=512 echo (33/33); L65 terminal wall; heartbeat
  induction beyond the bias horizon; delimiter deep-vs-shallow sign flip
  (peeked — now registered as P-d1..3).

## ANSWERED — certified (weight-level; unaffected by A5/A6)

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
