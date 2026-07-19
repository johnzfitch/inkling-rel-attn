# The Story of Inkling: How a Trillion-Parameter Model Knows Where It Is

*A synthesis of the inkling-rel-attn research campaign — every claim below is
backed by a registered, independently re-derived result in this repository
(see `QUESTIONS.md` for the ledger and `analysis/` for the artifacts), or is
explicitly labeled as interpretation.*

---

## 1. The problem: attention has no sense of place

A transformer's attention mechanism is, at its heart, a matching machine.
Every token asks a question ("I'm a verb looking for my subject"), every
token offers an answer, and attention connects the best matches. But the
mechanism itself is a *bag* — if you shuffled the entire context like a deck
of cards, the matching scores wouldn't change. Word order, distance,
recency: invisible.

So every transformer ever built has had to smuggle "where" into a machine
that only understands "what." The history of how models do this is short
enough to tell in four moves, and Inkling is the fifth.

**Move one — stamp the cards (2017).** The original transformer added a
positional pattern directly onto each token's representation: coordinates
written on every card. Geometrically, each position gets a unique point on a
family of sine waves, and the model learns to read the stamps. It works, but
the model must learn arithmetic on coordinates it never chose, and positions
beyond the training range are stamps it has never seen.

**Move two — turn position into rotation (RoPE).** Rotary embeddings rotate
each token's query and key by an angle proportional to its position — think
of every token carrying a clock hand. When two tokens compare themselves,
the score depends only on the *angle between their hands*, which is to say,
on their distance. Elegant, relative by construction — but the rotation and
its ladder of frequencies are hard-coded geometry the model must live inside.

**Move three — just tilt the table (ALiBi).** Forget encoding position at
all; subtract a penalty proportional to distance from every attention score.
The table of candidates is tilted: nearby is uphill and bright, far away is
downhill and dim, with one fixed slope forever. Brutally simple, and it
revealed something important: a lot of what position encoding does for a
language model is simply *recency*.

**Move four — learn the tilt (Shaw, T5).** Replace the fixed slope with a
learned lookup table: "at distance d, add b(d)." The model chooses its own
distance curve. But the curve is the same for every token — a table policy,
not a per-token judgment.

**Move five — Inkling: let every token set its own dial.** Inkling (Thinking
Machines' ~975B-parameter mixture-of-experts model, 41B active, native 1M
context, no RoPE anywhere) computes, for every token and every attention
head, a small vector — 16 numbers per head, produced from the token's own
hidden state. That vector is multiplied against a learned per-layer codebook
of distance curves (the "projection bank"). The result: a distance-bias
curve b(d) that is *written fresh by each token*. Position handling becomes
a decision the content makes, sixty-four times per layer, at every position.

The rest of the architecture divides the labor. Fifty-five of the 66 layers
attend only within a 512-token sliding window; every sixth layer is
*global*, seeing everything, with learned bias out to distance 1024. A short
convolution (SConv) handles pure neighbor-shuffling so attention doesn't
waste itself on local bookkeeping — in Thinking Machines' words, to "free
the attention and MoE modules from local representations." Eight extra
"drafter" layers speculate future tokens for fast inference.

This design is the reason the campaign exists: Inkling is the first
frontier-scale model whose positional system is *entirely learned* — no
rotation, no fixed slope, no stamps. Whatever structure exists in its
distance handling, gradient descent put it there by choice. So: what did it
choose?

## 2. The method: X-ray, field observation, surgery

We studied Inkling the way you'd study an organism, in three passes, with a
rule that every question gets a registered prediction before its data is
opened, and every answer is re-derived from raw dumps by a second analyst
with independent code before it counts.

1. **X-ray (weights):** read the learned distance curves directly out of
   the checkpoint. No inputs, no behavior — just what SGD wrote down.
2. **Field observation (captures):** run real text through the model with
   bit-exact instrumented attention, recording every token's dial setting,
   attention distribution, and prediction quality.
3. **Surgery (causal ablation):** re-run the model with pieces surgically
   removed — bias zeroed at one layer, one direction pinned, the far field
   silenced — and measure exactly what breaks. 72 such operations, all
   preregistered.

A large fraction of our registered predictions failed. That was the plan
working: in this ledger, *wrong beats unanswered*, and the inversions were
where the model taught us something.

## 3. What gradient descent bought with total freedom

**It bought recency — with a paragraph-sized hump.** Given an unconstrained
b(d), Inkling's learned curves are smooth decays: no oscillations, no
resonances, no special treatment of any architectural number (341 tests for
power-of-two artifacts: zero survivors; the curves are certifiably smooth at
single-distance resolution). But they are not pure decay: in 6 of 11 global
layers the curve *rises* to a crest around 60–140 tokens — roughly a
paragraph — before beginning its single clean exponential fall. The model's
attention prior, drawn as terrain: a gentle ridge at "a paragraph ago,"
then a long fade. It chose ALiBi's *idea*, tuned to the statistics of text —
and it chose it independently of corpus word-statistics (the curves do not
mimic mutual information decay; that prediction failed cleanly).

**The same shape, rediscovered.** Inkling's eight drafter layers were
trained separately from the trunk, and they rebuilt qualitatively the same
decay-dominated kernel on completely unrelated internal coordinates —
nearest-parent analysis finds no fork, no inheritance (every drafter's
closest trunk layer sits in the statistical middle of unrelated-pair
distances). The kernel shape appears to be an attractor: any part of this
system that needs positional transport converges on the same curve.

## 4. The dial: attention as a camera aperture

The per-token vector is not decoration — it is a working control surface.
Pooling each token's realized curve into a single number (what fraction of
its positional weight lies beyond ~129 tokens) gives an *aperture*: how wide
this token opens its attention.

The aperture obeys a law we've now confirmed on held-out data under
corrected arithmetic: **it widens with surprise.** Tokens the model finds
unpredictable open up (median rank-correlation with surprisal ≈ +0.15,
consistent across 16 of 16 position bins on novel conversational text).
Tokens that begin new units — sentences, messages — open wide (+0.12 to
+0.17 class effects); grammatical filler (function words, label colons)
stays narrow; and at conversation boundaries in deep layers there is a
sharp, precisely-localized widening transient (+0.26 at the boundary token)
whose size scales with how much context the boundary just retired.

One of us proposed the right metaphor: **a television antenna's gain knob.**
The attention system is a receiver; content matching is the tuner that locks
onto stations; the positional bias is the squelch that dims the static
between them. When the picture is clear (predictable token), keep the gain
low and local. When it goes snowy (surprising token, fresh topic, new
speaker), open the gain and sweep. Within a distance ring the bias cannot
pick *which* token wins — that is always content's job — but it decides how
much each ring of distance is allowed to speak at all.

## 5. The clock: a model with no absolute positions knows the time anyway

Inkling has no absolute position encoding. Yet the per-token dial vectors
carry a clean signal of *absolute* position: a logarithmic "clock" that
drifts as context accumulates (correlation 0.975 with log-position on
structureless text; present in every global layer from the eleventh on;
survives a search-wide permutation control at its resolution floor). Through
the codebook, this clock acts as a slow gain schedule: in the deepest
global layers the whole positional curve starts strong — about 1.7× its
average at the start of a document — and anneals to ~0.9× by 8,000 tokens.

Two careful facts keep this honest. First, the clock is not one universal
direction: when we froze the direction estimated from random text and
tested whether it silenced the drift on other texts (a preregistered
transfer test), it failed in all four cells — each text drifts along a
partially different direction with a shared ingredient. Second, and more
striking: **pinning the clock changes essentially nothing** at 8k. Model
predictions shift by ~0.0001 nats; retrieval is, if anything, a hair
better. The clock is real, readable, geometrically clean — and causally
idle at this context length. Remember that phrase; it becomes a theme.

## 6. The ghost in the residual stream: big structure, deleted by the exit door

Inkling's internal states carry a huge communal positional component — one
shared direction holding up to 10.8% of hidden-state variance, through
which roughly 64% of the positional read energy flows. Enormous, by any
geometric measure. Causally? Removing it from the dial's input changes
almost nothing at 14 of 16 tested layers.

The resolution is a piece of mathematics every transformer inherits:
**softmax only sees differences.** Add the same number to every attention
score a token issues, and the attention doesn't move — such components are
"gauge," bookkeeping the exit door deletes. Measured directly, the
carrier's contribution to the distance curves is ~2.4× flatter across
distance than the true kernel: mostly a per-token constant. The model
maintains a large, tidy positional signal that its own readout mechanism is
built to ignore. (The same effect explains the wall, below.)

This connects Inkling to a broader literature: large transformers famously
develop "attention sinks" and massive activation channels — giant, stable
internal quantities whose job seems to be plumbing rather than computation.
Inkling has those too: we tracked one specific channel (4786) switching on
in a sharp broadcast at layer 26 and saturating by the layer-40s. It sits
suggestively adjacent to the model's causal hot spot — but carries no
measurable share of the dial's token-to-token variation, so on current
evidence it is infrastructure, not mechanism.

## 7. The surgery: where position actually lives

Then we cut things, 72 ways, and the map redrew itself.

**Nearly everything is one layer.** Zero the positional bias at any of the
first six layers: the model barely notices (≤0.002 nats; the five early
layers back each other up — removing all five together costs four times the
sum of removing them singly). Zero it at the last layer, the one with the
most dramatic bias in the entire network: *nothing* — in fact random text
gets slightly better. Zero it at **layer 29**, an unremarkable-looking
global layer at mid-depth: +0.121 nats pooled, the largest causal effect in
the campaign by a factor of six, with small shoulders at its neighbor
globals L23 and L35.

**And within that layer, nearly everything is close range.** Silencing only
distances closer than 4 tokens reproduces 84% of the entire effect — the
same tokens, token by token (correlation 0.91). Silencing everything
*beyond* 128 tokens, across all sixteen tested layers at once, costs
one-eightieth as much as silencing under-4. The learned far field — all
that careful curve out to distance 1024 — is causally almost free at 8k.

What the bias actually writes at L29 turned out to be sharper than a band
of attention: a **contrastive stencil** on the immediate neighborhood —
suppress the token itself (the realized curve is *negative* at distance 0
on every text) and boost distances 1–3. How strongly a text's stencil
contrasts predicts how much that text suffers (correlation 0.77 across the
six texts; exploratory). Full bias removal does collapse the
phrase-and-sentence band (4–128 tokens, ~26% → ~8% of attention on prose),
but that collapse is a side effect, not the mechanism: silencing *only*
d<4 leaves the band intact — even slightly fuller — and still does 84% of
the damage. The injury is broad (two-thirds of all tokens hurt; the worst
1% of tokens carry only 28% of the gross damage), essentially uncorrelated
with how difficult a token already was, and — after matching for position
and difficulty — concentrated at sentence starts (+0.14) and pronouns
(+0.08), the integration tokens. Texts that can be predicted by copying
from far away (templates, needle documents) sail through unharmed; texts
whose structure lives in the local neighborhood (code, prose) pay full
price. Even long-range *retrieval* breaks through the near field: recall
queries suffer 487× more than ordinary tokens when L29's bias goes, yet
silencing the far field does nothing to them — the far-field match still
happens; what fails is the locally-assembled representation doing the
asking.

**The engine and the guardrails.** Here is the picture the surgery paints.
Inkling's positional system at 8k is a small, hot **engine** — a
don't-look-at-yourself, look-just-behind-you stencil, sharpest at one
mid-depth layer — surrounded by an elaborate array of **quiet structure**:
the terminal layer's near-field vise (near-gauge at the readout: it shifts
target and competitors by a nearly identical +2.1 nats, mean-canceling
though not pointwise-silent), the 512-boundary echo, the distance-1024
wall (healing it: ±5-nat churn on individual tokens, net 0.0001), the log
clock, the giant carrier. All real, all replicated, all certified — all
mean-silent at 8,192 tokens.

## 8. Why keep guardrails you never touch? The 1M question

One fact reframes the entire quiet half of the map: **Inkling's native
context is one million tokens, and every experiment here ran at 8,192** —
less than 1% of the model's operating range. An 8k document does not
meaningfully dilute attention, crowd the far field, or stress the seam
between local and global scopes. The structures we found idling are
precisely the ones whose predicted purpose is regulation at scale: the wall
that steepens as context grows (our one standing forecast: at 1M, the
terminal cliff should sharpen from ~8× to ~13× — testable by anyone with
the hardware), the clock whose log-schedule has barely two decades to act
in 8k but five at 1M, the far field whose causal moment may only come when
there are hundreds of thousands of candidates to hold at bay.

So the honest reading is not "Inkling carries dead machinery," but:
**at short range we caught the machine at idle.** The engine runs at every
scale; the guardrails are for the highway. Whether that interpretation
survives contact with 128k–1M-token experiments is, deliberately, the next
falsifiable question on the ledger.

## 9. What this adds up to

Strip away the detail and the story is three sentences long.

**Told what to do about position, earlier models obeyed; left free, Inkling
built a short-range recency engine tuned to the shape of text — a
paragraph-crested decay, almost all of whose causal force lives within a
few tokens and one mid-depth sentence-window layer.** **It handed every
token a dial — open wide when surprised or starting fresh, stay narrow when
the next word is cheap — so that position policy became a per-token,
per-head decision made by content.** **And around that engine it grew
large, quiet regulatory structures — a wall, a clock, an echo, a carrier —
that do nothing measurable at the context lengths we probed and everything,
perhaps, at the million-token scale it was built for.**

The deeper lesson is about method as much as model. Most of what we now
know arrived through predictions that *failed* under preregistration:
the sentence-scale knees that turned out to be paragraph crests, the
"long-range bandwidth budget" that inverted into a near-field concentrator,
the depth map that pointed everywhere except layer 29, the universal clock
that dissolved into a family of text-specific schedules. A frontier model's
positional system turned out to be at once simpler than predicted (one
engine, close range) and stranger (elaborate structure held in reserve,
invisible to every 8k measurement except the geometry itself).

---

*Provenance: verdicts and effect sizes are drawn from the certified rows of
`QUESTIONS.md` — weight-level (LF1, LF2, LF6, LF7, Rounds 1–4), captures
(LF4, P-v2, P-d, P-e, P-f, LF3, LF8, LF9, R5-B, R5-C), and the R5-D causal
campaign with its post-hoc anatomy (`analysis/round5/r5d/`). Interpretive
passages — the antenna metaphor, the engine/guardrails picture, and all
claims about 1M-context purpose — are labeled interpretations, not
certified results. Architecture facts follow the public Inkling model card.*
