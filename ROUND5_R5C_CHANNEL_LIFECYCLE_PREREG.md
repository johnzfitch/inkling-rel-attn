# Round 5 R5-C targeted preregistration — channel 4786/3290 lifecycle

Registered on 2026-07-17 before any R5-C activation-geometry outcome run.
This is the first targeted R5-C analysis; the broader R5-C metrics and
predictions in `ROUND5_LEFTFIELD_SPEC.md` remain unchanged.

## Prior disclosure

The directing session supplied the channel identities and approximate lifecycle
peek: channel 4786/3290, a possible broadcast onset in L23–28, and a possible
handoff at L39/40. These facts are therefore not blind discoveries. The tests
below are confirmatory targets; all additional channel searches are secondary
and multiplicity-controlled.

## Primary target

For hidden channels 4786 and 3290 (zero-indexed), report their layer-by-layer
token distribution on every one of the six frozen texts: signed mean, median,
RMS, MAD, upper absolute quantiles, share of hidden-state variance, fraction of
tokens with `abs(h)>30,000`, and cross-text dispersion. Retain per-text curves;
no pooled curve may conceal a text-specific transition.

The first two confirmatory questions are:

1. **Broadcast onset:** does the first sustained layerwise expansion of the
   channel-4786/3290 token coverage in L23–28 align, layer by layer, with the
   independently registered flip band rather than merely landing somewhere in
   the same broad depth region?
2. **L39/40 handoff:** is the transfer between the two channels concentrated at
   the single L39→L40 boundary, or spread gradually across neighboring layers?

“Sustained” means the change has the same direction for at least the next two
layers and appears in at least four of six texts. A handoff is classified as
**sharp** only if the L39→L40 adjacent-layer change is the largest absolute
change for both channels over L35–44, the channels move in opposite directions,
and a 256-token-block bootstrap gives the same classification in at least five
of six texts. Otherwise it is classified as **gradual/mixed**. Report the full
trajectories and bootstrap intervals regardless of classification.

The activation readout must use the lossless BF16 normalized-input recapture or
a new explicitly registered lossless state capture. Overflowed FP16 residual
files remain inadmissible; no clipping, imputation, or finite-only deletion is
allowed.
