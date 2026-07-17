# Corpus v2 Amendment A1 — public-registration and provider-budget correction

This amendment is written before any corpus-v2 capture, NLL measurement,
aperture computation, or registered readout. The four predictions P-v2-1
through P-v2-4, their thresholds, their directions, and their failure rules are
unchanged.

## Discovery record

During the agent's pre-capture audit on 2026-07-17, two registration defects
were found:

1. `CORPUS_V2_SPEC.md` at local commit `7fb84ab` stated math-provider budgets
   of claude/chatgpt/gemini = `2731/2731/2730`, but the private sidecar actually
   contained `2730/2730/2732` tokens.
2. The committed builder produced that allocation indirectly: claude and
   chatgpt were capped at 2,730, Gemini was allowed 3,244, and the final
   8,192-token corpus truncation retained 2,732 Gemini tokens.

The private artifacts otherwise passed preflight: both ID arrays were
`int32[8192]`, re-tokenizing each saved text reproduced its ID array exactly,
all manifest hashes matched, Slack had 311 unique in-range message starts, and
the scrub-residual checks were zero.

## Correction

The frozen provider allocation is explicitly:

```text
claude   2730
chatgpt  2730
gemini   2732
total    8192
```

The builder now gives each provider exactly that budget and asserts exact
completion per provider. The private corpus is rebuilt once under the corrected
builder; capture remains blocked unless the six data/sidecar/text artifacts are
byte-identical to the pre-amendment build and all preflight checks still pass.

## Public-registration clock

The same audit ran `git ls-remote` and found the public branch still at
`9bd68f2`; local commits `d02b190` and `7fb84ab` were not yet reachable from the
remote branch despite having local commit timestamps. They therefore do not
receive an earlier public-registration claim. The registration becomes public
when the branch containing `7fb84ab` and this amendment is pushed. No
result-bearing corpus-v2 computation may run before that push succeeds.

Raw text, token IDs, sidecars, and their private manifest remain gitignored.
