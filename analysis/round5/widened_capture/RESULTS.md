# D1+D4 widened corrected capture — execution and certification

**Status: CERTIFIED CAPTURE. No registered science outcome was evaluated in
this execution step.**

The D4 scope was registered publicly at `2264ae4` before implementation. The
runner/validator extension was committed at `e291453`, which is the capture's
recorded Git head.

## Production execution

- Full A8 preflight passed, including content SHA-256 for all 33 indexed trunk
  checkpoint shards, historical Git blobs, all eight input arms, installed
  package sources, and bitwise stock-attention parity for global and sliding
  cases.
- The fresh capture completed in 1,821.25 seconds (30.35 minutes) and occupies
  100.623 GiB locally.
- Exact inventory: 2,324 artifacts. The gitignored capture manifest SHA-256 is
  `2fc2ec9dd4d6b684a473847f6fbd4541d4326b49d7d6456b463ee5722399a75f`.
- D4 inventory: 536 complete `[8192, 6144]` BF16 state payloads (embedding plus
  66 layer outputs for eight arms), totaling 53,955,526,656 raw payload bytes.

## Independent certification

The independent validator rehashed all checkpoint shards and all 2,324
artifacts and returned zero errors:

- D4 satisfied: **true**;
- non-finite D4 BF16 words: **0**;
- non-finite normalized-input BF16 words: **0**;
- maximum Tier-2 meter mass error: `0.0002977301337523386`;
- NLL integrity gate: passed (aggregate integrity only; prediction readouts
  remain unopened).

After certification, the LF5 production CUDA/BF16 replay compared 66 layers ×
24 queries × 64 heads × 8,192 keys = 830,472,192 FP16 attention words against
the in-situ rows. All 66 layers were bitwise equal; mismatch count was **0**.

## Public evidence

- `preflight.json` — SHA-256
  `903e26ae4f8d3f6ea8846d04883f888cc10b4a0e5554c6e94825feb4c8f7f8fc`
- `capture_validation.json` — SHA-256
  `b101bcc181a2526f1b76e34ede6e3c337f17910ef3e450929529b1f62c0f7f2e`
- `lf5_replay_parity.json` — SHA-256
  `dc3b797a87e4dc93288ae4c7e3449be22192b0ce75089ae8770882c6b6e2e0ff`

The local dump remains gitignored. The newly runnable evaluation order is:
mechanical re-certifications; R5-C channel-4786/3290 lifecycle first; LF3,
LF8 (starting at L53), LF9, and R5-B; then P-e/P-f; then R5-D.
