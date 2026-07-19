# Round 4 — pre-registered family battery: which classical PE did Inkling learn?

Author: research-assistant session, 2026-07-16.
Builder: the user's agent. Dump-first per ROUND3 conventions. **No new dumps
needed** — every analysis reads only `dumps/round3/` (mode_curves, perhead_svd).
Outputs -> `analysis/round4/`, figures -> `analysis/round4/figs/`.

## Motivation

Inkling's positional transport b(d) is a free-form learned table, so every
classical positional-encoding scheme is a *subspace* of its hypothesis space.
This round runs a formal, falsifiable competition: which named scheme (if any)
does the learned transport realize? Two levels:
1. **Curve-shape level**: which functional family best describes b(d)?
2. **Fingerprint level**: named schemes also fix *cross-head parameter
   structure* (geometric ladders). Matching the shape but not the ladder means
   "same family, different organization" — report both verdicts separately.

## Pre-registration disclosure (what we already know — stated so we can't
retro-fit the narrative)

Prior exposure from Rounds 1–3: dominant modes are exp/exp2-shaped; fitted rho
~ 0 on dominant modes; transport is low-rank (~1.5–3 of 16); early local
layers have rising heads; curves truncate at the extent with a cliff.
**Genuinely unexamined before this round**: modes 1–15 individually (only
mode-0 was ever headline-tested), all cross-head parameter-ladder structure,
the near-field d<8 (always excluded from fits), and held-out generalization
of any fit.

Predictions registered now (Claude, 2026-07-16, before running):
- P1: exp-family (F9) wins far-field dominant modes everywhere. HIGH confidence.
- P2: ALiBi (F2) is rejected on curvature. HIGH confidence.
- P3: RoPE/sinusoidal (F6/F7) rejected for dominant modes; **OPEN** for minor
  modes — no prior look. If oscillation exists anywhere, it is here. NO
  confident prediction.
- P4: per-head decay-rate ladder (F9 fingerprint): **OPEN**, never computed.
  If present, Inkling re-derived a RetNet-style retention ladder by SGD.
- P5: near-field d<8 is NOT any smooth family — expect a discrete/spiky
  structure (previous-token-style). MEDIUM confidence.

## The family battery

Fit each family to far-field (d in [8, extent)) of every mode curve
(layer x 16 modes, trunk + MTP, from D1) AND every per-head top-mode profile
(layer x 64 heads, S[h,0]*U[h,:,0], from D0). All fits multi-start as in
Round 2 E1; all parameters identifiable (no free phase next to free amplitude
— reuse the phase-free forms).

| id | family | form b(d) | named scheme it realizes |
|----|--------|-----------|--------------------------|
| F1 | zero/const | c | NoPE |
| F2 | linear | -m*d + c | ALiBi |
| F3 | log | -r*log1p(c2*d) + c (c2>0) | KERPLE-log |
| F4 | power | -r*d^p + c (p in (0,2]) | KERPLE-power |
| F5 | bucket | piecewise-const, 8 log-spaced knots over [8, extent) | T5 relative bias |
| F6 | fourier-K | sum_{k=1..3} a_k*cos(w_k*d) + c | sinusoidal APE kernel (Whittaker/band-limited) |
| F7 | rope-kernel | F6 constrained: w_k = w_0*g^(-k), fit w_0, g>1, a_k | RoPE-induced kernel |
| F8 | gaussian | -a*d^2 + c | window/Gaussian kernel |
| F9 | exp2 | a1*exp(-d1*d)+a2*exp(-d2*d)+c | retention/DA-decay (RetNet-like) |
| F10| dsin | a*exp(-delta*d)*cos(rho*d)+c | damped rotation (the (rho,delta,sigma) form) |

Parameter counts: F1:1 F2:2 F3:3 F4:3 F5:8 F6:7 F7:5 F8:2 F9:5 F10:4.

### Selection protocol (registered, no discretion later)
1. Fit all 10 families on EVEN distances only (d=8,10,12,...).
2. Score on ODD distances (held-out R^2) — functional-form generalization,
   not curve-memorization.
3. Winner = lowest BIC on the full far-field AND held-out R^2 within 0.02 of
   the best held-out R^2. If those disagree, verdict is "contested" and both
   are reported. Ties (ΔBIC<10) are ties — say so.
4. Report per (layer, mode) and per (layer, head): winner, ΔBIC to runner-up,
   held-out R^2 table for all 10.

### Fingerprint tests (the part that distinguishes *named schemes* from shapes)
Run on per-head far-field fits within each layer:
- FP-ALiBi: fit F2 per head; test whether slopes m_h, sorted, form a geometric
  sequence (regress log m_h on rank; report R^2 vs a permutation null of 1000
  shuffles). ALiBi's signature: m_h = 2^(-8h/H).
- FP-RetNet: same ladder test on F9's slow rate d1_h (and on d_half_h).
  RetNet's signature: gamma_h = 1 - 2^(-5-h).
- FP-RoPE: if F6/F7 ever wins a mode/head, test fitted frequencies for
  geometric spacing and for cross-head/cross-layer sharing (RoPE frequencies
  are global constants, not per-head free parameters).
- FP-T5: if F5 competitive, test whether the fitted knot values are shared
  across heads (T5 buckets are per-head scalars on SHARED bucket boundaries).

### Whittaker/band-limitedness test (separate from the family race)
For every mode curve: rFFT; compute effective bandwidth (fraction of spectral
energy below frequency f, report f90); then decimate the curve 2x and 4x,
reconstruct by sinc (Whittaker) interpolation, report reconstruction error.
Question: is the 512/1024-entry table just a band-limited function sampled at
integers (i.e. massively oversampled), and what is its true bandwidth? This
also bounds how much of the table's capacity the model actually uses —
complements the rank-~2 finding with a smoothness number.

### Near-field battery (d in [0,8) — first time under the microscope)
Per layer and per head: the 8 raw values (already dumped). Tests:
- distance of near-field vector from the far-field family fit extrapolated
  back to d<8 (how discontinuous is the near/far handoff?)
- is near-field structure discrete? Cluster the 4224 near-field vectors
  (k-means, k selected by silhouette over 2..8, seed 0) — do heads share a
  small set of near-field motifs (e.g. "d=1 spike", "d=0 suppress")?

## Outputs
- `analysis/round4/family_battery.json` — per (layer, mode) and (layer, head):
  all-family BIC/held-out-R^2, winner, verdict class.
- `analysis/round4/fingerprints.json` — ladder tests with permutation p-values.
- `analysis/round4/bandlimit.json` — f90, decimation-reconstruction errors.
- `analysis/round4/nearfield.json` — handoff discontinuity, motif clusters.
- Figures: winner-by-depth stack; ΔBIC(F9 vs F2) heatmap; per-head slope/rate
  ladders vs the ALiBi/RetNet reference ladders; f90 vs depth; near-field
  motif gallery.
- Verdict block at the top of family_battery.json: one of
  {"named scheme X realized (shape+fingerprint)",
   "family X shape without the named ladder",
   "novel transport — no classical family adequate"},
  decided strictly by the registered criteria above.

## Curiosity register (Claude's own questions — pre-registered 2026-07-16)

These are not family tests; they are questions the anatomy itself raised.
Each has a concrete weight-level test on existing dumps. Implement as
`analysis/round4/curiosity/` outputs, one JSON per item.

### C1 — Which way does the seam step? (the gauge question)
Softmax is shift-invariant per query row, so b(d) and b(d)+c are normally the
same mechanism — the constant is pure gauge. **The cliff breaks the gauge**:
past d=1024 the bias is pinned to exactly 0, so at global layers the value of
b just inside the boundary is physically meaningful relative to that 0. Test:
per global layer and per head, report b(d) averaged over d in [1008, 1024)
(from D0/D1, no fit — raw curve values, sign included). If it is POSITIVE,
in-window tokens are boosted and crossing the seam is a drop (attention falls
off the cliff). If NEGATIVE, in-window tokens are *suppressed* relative to the
far field and tokens beyond 1024 get MORE attention than tokens just inside —
which would be bizarre and important. Registered prediction: positive (drop),
LOW confidence — genuinely unsure. **Tier-2 hook**: this sign is a hard
prediction for the in-situ seam direction in mass_with(d); write it into the
JSON as `predicted_seam_direction` so the Tier-2 analysis can test it blind.

### C2 — Is the 16-dim interface earning its keep?
proj uses ~2 of its 16 modes. Why 16? Test whether wr_du even feeds the dead
modes: for each layer, take proj's left singular vectors U_k (in d_rel space,
from D1); for each head compute E_k = ||U_k^T Wr_h||_F^2 (energy the head
injects into table-mode k, from D0's V or raw wr_du). Compare the E_k profile
against proj's singular values S_k. Outcomes: (a) matched decay — interface
co-adapted, extra dims vestigial on BOTH sides; (b) heads inject energy into
modes the table ignores — wasted/vestigial r_proj capacity; (c) table has
live modes no head feeds — dormant table capacity. Report the energy-vs-mode
alignment (cosine of the two profiles) per layer and the vestigial fraction.

### C3 — One shared circuit or 64 private solutions?
Do heads in a layer read their positional recipe from a SHARED hidden
subspace, or did each head learn its own? Test: per layer, pairwise principal
angles between heads' top-2 right-singular subspaces (V from D0, the
hidden-space read directions); cluster; compare within-taxonomy-class vs
across-class similarity (join A3 classes). Sharpest version: are the rising
heads of layers 0–13 reading from ONE common subspace (a "look far" circuit
reused 64 times), or 64 independent implementations of the same idea? Also
run across adjacent layers (do layer 6's rising heads share subspace with
layer 7's?). Report effective number of distinct read-subspaces per layer
(participation ratio over cluster sizes).

### C4 — How much curve wanted to exist past the wall?
The cliff says training pushed amplitude INTO the boundary. Quantify the
counterfactual: per global layer, extrapolate the winning F9 fit beyond the
extent and report truncated_fraction = integral of |extrapolated b| over
[extent, 4*extent] divided by realized integral over [8, extent). A
truncated_fraction near 0 = the wall barely binds; >0.5 = the learned kernel
is mostly amputated. Registered prediction: substantial (>0.3) at global
layers, from d_half ~ 1500 > extent. MEDIUM confidence. Caveat in output:
extrapolation of a fit, not data — label it as such.

### C5 — Is the rising→decay flip a smooth rotation or a phase transition?
The taxonomy flips between layers ~13 and ~28. Is the underlying transport
morphing continuously or switching regimes? Test: embed all 74 layer
transports in curve space (mode-0 curves, resampled to common length 512,
L2-normalized; also the full rank-16 object via subspace principal angles);
report the trajectory: distance between adjacent layers vs depth. A smooth
arc = gradual handoff (fits the SConv division-of-labor story evolving);
a spike at the transition band = discrete regime switch. Also mark where
global layers sit relative to their local neighbors on this trajectory.

### C6 — Within-head anatomy: far-field engine + near-field corrector?
Per-head top-1 concentration is ~0.69, so mode 2 of each head carries real
weight. Hypothesis: mode 1 = smooth far-field kernel, mode 2 = near-field
correction (spike-like). Test: per head, near-field energy fraction
(d<8 share of L2 mass) of U[:,0] vs U[:,1]; report the distribution of the
ratio. If mode-2 profiles systematically concentrate near-field, each head is
a two-part machine and the near-field motifs (A4 clusters) should align with
mode-2, not mode-1 — check that alignment.

### C7 — Does draft depth shift the MTP near-field?
MTP layer k predicts token n+k+1: its effective "current position" is
displaced. Do MTP transports show a correspondingly shifted near-field
(spike at d=k+1 rather than d<=2)? Test: near-field argmax and motif class of
each MTP layer's modes/heads vs its draft depth. Registered prediction: no
shift (the drafter chains hidden states rather than re-indexing distance),
LOW confidence — but if the spike DOES track draft depth, that is a lovely
mechanistic signature of the chaining structure.

## Ground rules
Reuse fit conventions from `fit_transport_models.py` (sign canonicalization,
multi-start, safe_exp). Analysis-only: no network, no GPU, no new dumps.
[CONTRADICTION] protocol applies — especially against predictions P1–P5 and
the C1/C4/C7 predictions, which are recorded above precisely so they can lose.
NOTE for C1: sign canonicalization must NOT be applied — C1 is about the true
sign of the raw curves; read Vt/U from the dumps but recover original sign by
checking against the raw proj tensor (weights/layerNN_rel_logits_proj.npy).
