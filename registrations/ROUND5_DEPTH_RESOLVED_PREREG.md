# Round 5 — depth-resolved aperture registration (pre-registered)

Registered after (and because of) the full 66-layer aperture sweep of the v2.0
slack arm. DISCLOSURE: predictions P-d1–P-d3 are informed by that sweep (a
peeked, dtype-uncorrected, single-thread arm); they are tested on the NEW
`07b_slack_multi` arm under corrected arithmetic — different conversations,
different speakers, corrected dtype path. P-d4 extends to a text never
depth-swept. All evaluation exclusively on ROUND5_AMENDMENT_A6 corrected
captures. Same statistic machinery as LF4 (position-binned percentiles,
10,000 deterministic permutations, Holm across the P-d family).

Bands (fixed): shallow-global {L17, L23, L29}; deep-global {L35, L41, L47,
L53, L59}. Band statistic: median of per-layer effects.

- **P-d1 (delimiter depth flip):** on 07b_slack_multi, speaker-label tokens
  have band-median effect ≥ +0.10 in deep-global AND ≤ 0 in shallow-global.
  The v2.0 five-layer averaged "+0.043 pass" straddled this flip; this registers
  the flip itself. Confidence: medium-high.
- **P-d2 (pronoun band):** on 07b_slack_multi, pronouns have band-median
  ≤ −0.05 in shallow-global AND |band-median| < 0.03 in {L53, L59}. The two
  phenomena live in different depth bands. Confidence: medium.
- **P-d3 (first-content narrowing):** on 07b_slack_multi, the first content
  token of each message (delimiter position + 2) has deep-global band-median
  ≤ −0.05. Together with P-d1: the boundary MARKER opens the lens, the content
  that follows does not. Confidence: medium-high.
- **P-d4 (prose transfer, semi-blind):** on 01_prose_en (corrected recapture;
  memorization caveat stands), sentence-start tokens — which ARE first-content
  there — have deep-global band-median ≥ +0.05, i.e. OPPOSITE to slack
  first-content (P-d3). Registered rationale: prose sentence-openers are
  unpredictable content; chat message-openers are formulaic. If P-d4 fails with
  prose first-content also narrow deep, then the v1 sentence-start result was a
  shallow/mid-band phenomenon and the "anchoring" story needs revision.
  Confidence: low-medium (genuinely uncertain — that is the point).

Controls: each class mask crossed onto 08_math_llm (expected null), plus the
random-position mask per arm. Failure handling: report per-band effects and
verdicts exactly as computed; no band redefinition after outcome access.
