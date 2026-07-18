# Round 5 P-f preregistration — referential anchors, boundary transient, delimiter punctuation

Companion to `ROUND5_APERTURE_CONTEXT_DOSE_PREREG.md` (P-e). Both families
evaluate on the same fresh paired capture; this file must be committed before
that capture. The commit timestamp is the registration event.

## Priority statement (ledger rule 4)

P-f rides the P-e capture pass — no additional GPU cost — and freezes now to
stop further unregistered peeking at 07b. The runnable-now CPU rows are not
displaced; they precede any new analysis work.

## Prior disclosure (hypothesis-generating peeks, all on 07b)

- The 25 tokens extreme-wide across all five deep layers simultaneously
  decoded to: person names at first mention or in greetings, a product
  identifier, URL fragments, "Thank you"-type closers, and personal-news
  announcements. The 14 extreme-narrow tokens decoded to no coherent class.
- Deep−shallow band difference at message-boundary offset 0 was +0.182
  (paired message bootstrap 95% [0.151, 0.211]); the difference was ≤ 0 at
  pre-boundary offsets −3/−2 and drifted ≤ 0 over body offsets 6–11.
- The ':' after speaker labels: shallow −0.218, deep −0.105. The
  end-of-message newline narrowed both bands (≈ −0.13).

07 and 07b are excluded from P-f evaluation. Evaluation is exclusively on the
fresh P-e arms (primary: the multi-conversation render; the single-thread
render is descriptive replication), under the corrected instrument, after an
A8-style validated capture.

## Frozen classes (from ids/sidecars/text alone, before any readout)

- **URL tokens:** tokens whose character span lies inside a `https?://\S+`
  match of the built arm text.
- **Proper-noun proxy:** decoded fragment matching `^ ?[A-Z][a-z]{2,}$`,
  excluding message offsets 0–2 and positions whose previous fragment ends in
  sentence-terminal punctuation or a newline.
- **Gratitude tokens:** stripped, casefolded fragment in
  {thank, thanks, thx, ty} or beginning with "thank".
- **Label colon:** the token at message-start offset +1 when it decodes to ':'.

Class freezes use the existing tooling (deterministic seeds, stratified
position-matched random masks, overwrite guards). Empty classes (< 8 members)
are reported and that sub-test is voided, not redefined.

## Registered predictions

Statistic chain: within-256-token midrank percentiles; shallow {L17,L23,L29}
and deep {L35,L41,L47,L53,L59} bands; band effect = median of per-layer class
medians − 0.5; 10,000 deterministic permutations; paired message bootstrap
5,000 resamples for P-f2.

- **P-f1 (referential anchors).** URL, proper-noun-proxy, and gratitude
  classes each show deep-band effect ≥ +0.05, Holm-adjusted p < 0.05 across
  the three; crossed `08_math_llm` masks and random masks read null.
  P-f1 passes only if all three sub-classes pass. Confidence: medium-high
  (URL), medium (proper noun), low-medium (gratitude).
- **P-f2 (boundary-locked transient).** At ordinary message boundaries in the
  multi-conversation render: the deep−shallow difference at offset 0 has a
  paired-bootstrap 95% interval entirely above +0.10, AND its median over
  within-message offsets 6–11 is ≤ 0. The depth flip is an event at the
  boundary, not a tonic band property. Confidence: medium-high.
- **P-f3 (delimiter punctuation narrows everywhere).** The label-colon class
  shows band effect ≤ −0.05 in BOTH bands. Confidence: medium-high.

Holm is applied within P-f across the three top-level predictions (P-f1
already jointly gated internally). Failure handling: verdicts reported as
computed; no class, band, offset window, or threshold changes after outcome
access. An informative failure — e.g., anchors null on fresh data → the 07b
spike inventory was selection noise — is a result.
