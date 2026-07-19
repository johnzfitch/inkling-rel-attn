# Round 5 follow-up seven: certified interpretation

Status: **post-verdict interpretation of independently verified results**

The compact answer is that L29 is not implementing a broad phrase-window
decay. Its causal object is a mostly static, non-carrier **d=0 stencil**,
concentrated in a small ranked head set and protected by redundant shoulders
at L23/L35. Retrieval depends on that object because it helps construct the
query representation; the eventual long-range match still happens elsewhere.
The late clock remains geometrically real but behaviorally free at 8k. The L29
bias acts primarily through token ranking, while the previously suggested
boundary/pronoun class geography does not replicate on fresh Slack text.

## What each experiment adds

1. **F7-1 localizes the distance term.** Full L29 bias-off costs +0.12079 NLL.
   Removing d=0 alone costs +0.09392 (77.8%); d=1..3 jointly costs only
   +0.000689. Conversely, restoring d=0 to an otherwise bias-off layer rescues
   98.1%, while restoring only d=1..3 rescues 6.6%. A d=0..3-only stencil
   rescues 99.0%. Thus the earlier “near field” result resolves almost entirely
   to the diagonal/self-distance term, not a four-token decay profile.

2. **F7-2 identifies the backups.** Every joint removal containing L29 is
   strongly super-additive: +0.60275 for L23+L29, +0.34832 for L29+L35, and
   +1.03830 for all three beyond their singleton sum. L23+L35 without L29 has
   only +0.02358 interaction. The shoulders therefore look like conditional
   backups for the same computation, not three independent causal stages.

3. **F7-3 identifies the r-space object.** Removing the static mean reproduces
   78.1% of full bias-off, and removing its non-carrier part reproduces 79.6%.
   Removing centered token variation costs only 1.9%. Keeping the non-carrier
   channel while removing the carrier is essentially free; leaving only the
   carrier reproduces 94.9% of the damage. The causal signal is a static
   non-carrier offset, not the communal scalar carrier and not a token-varying
   aperture dial.

4. **F7-4 localizes the heads.** The frozen top stencil-score quartile costs
   +0.03834 to remove; the other quartiles cost only +0.00057 to +0.00127.
   Keeping the top 8/16/32 d=0..3 stencils rescues 85.7%/93.4%/97.5% of full
   bias-off damage. Full bias on the top 16 rescues 95.1%, so almost all of
   their useful contribution is already in the four-entry stencil.

5. **F7-5 supplies the causal retrieval path.** Under L29 bias-off, restoring
   only the 24 registered needle-query residuals after L29 rescues 54.1% of
   recall damage (absolute rescue +0.1268; 95% interval +0.0374 to +0.2317).
   Patching matched sham positions rescues -3.2%, with an interval spanning
   zero. L29 does not need to carry the long-range retrieval edge itself: it
   makes a better query state for downstream content matching. The incomplete
   54% rescue leaves room for damage to other positions or later composition.

6. **F7-6 separates clock geometry from function.** Own-text and union
   projections flatten their fitted kernel drift almost completely. A basis
   learned from the other five texts does not transfer: only 1/12 held-out
   cells falls below 0.20, with median/max residual correlations 0.322/0.470;
   the sham median remains 0.589. Nevertheless, the union, per-text, and LOTO
   joint interventions cost only +0.000107 to +0.000585 NLL. The best current
   description is multiple text-conditioned clock directions with no measured
   behavioral role at 8k—not a single universal clock and not evidence that
   the clock becomes functional at longer context.

7. **F7-7 separates ranking, calibration, and class geography.** Bias-off
   increases mean log1p target rank by +0.03114 (95% interval +0.02413 to
   +0.03865) and reduces top-1 accuracy by 1.827 percentage points (interval
   -2.350 to -1.353), so the ranking prediction passes. ECE also worsens by a
   statistically positive +0.00431, but this misses the registered +0.01
   magnitude gate: calibration changes a little, not enough to promote a
   calibration mechanism. On fresh Slack, query-aligned first-content and
   pronoun contrasts are -0.090 and -0.084 with intervals spanning zero; the
   target-aligned pronoun secondary is significantly negative (-0.206). Math
   starts/pronouns are null. L29 helps broadly, but the old “integration-scope
   classes are especially dependent” story fails to generalize.

## Revised mechanism

At 8k context, the most economical model is:

`static non-carrier mean -> d=0 stencil in ~top 16 L29 heads -> improved local
state/ranking -> better downstream query composition`,

with L23/L35 providing redundant fallback paths. Far-field positional bias,
the communal carrier, centered r-variation, and the late log-position clock
are all measurable but not load-bearing in these interventions. This is an
8k-context conclusion; it does not settle LF10-style 128k–1M behavior.

## Evidence and audit trail

- Machine results: `results.json`
- Independent raw-dump verification: `verification.json` (563 artifacts
  rehashed; zero errors; producer not imported)
- Frozen registration and A1/A2/A3: `registrations/ROUND5_FOLLOWUP7_PREREG.md`
- Original A3-rejected output and five-error verifier record: `*_pre_a3.*`

The interpretations above were written after the registered verdicts and are
descriptive. They do not retroactively turn failed predictions into passes.
