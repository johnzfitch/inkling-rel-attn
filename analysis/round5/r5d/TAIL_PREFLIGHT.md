# R5-D wall-tail preflight

The first, raw-row two-exponential implementation passed its narrow registered
finiteness/continuity/overshoot gate but failed scientific suitability before
any GPU or causal outcome: 9/11 layers had median row-fit R2 below 0.10 and
126/176 slow rates hit the lower bound. Its complete report is preserved in
`tail_fit.json`; the bound dump is immutable and will not be used.

`ROUND5_R5D_TAIL_AMENDMENT_A.md` replaces that ill-conditioned coordinate-wise
fit with the already-frozen LF6 mode-0 two-exponential envelope, normalized at
the learned endpoint and applied to all 16 endpoint coordinates per layer.

The amended build passed every replacement gate. The first new cell retains
99.80%--99.97% of the continuous endpoint envelope, and the d=8191 envelope
ranges from 6.35e-7 to 0.146 across layers. All projection/LF6 hashes reproduce;
full parameters and per-layer diagnostics are in `tail_fit_v2.json`.
