# D1 widened corrected capture — production handoff

Status: **runner build complete; paired inputs frozen; GPU capture not fired**.

The production pass writes a fresh `dumps/round5/widened_corrected_capture/`
tree and refuses to overwrite a nonempty output. It captures all 66 layers for
the six v1 texts plus the fresh paired P-e/P-f arms under the A6 arithmetic
boundary. Old dumps remain immutable.

## Frozen scope

- all eight arms: `[8192, 64, 16]` fp16 r-vectors;
- all six v1 texts: lossless BF16 normalized attention inputs, corrected
  Tier-2 with/without-bias meters, massive-coordinate censuses, and NLL;
- `05_needles`: 24 full production attention rows at every layer for the LF5
  CUDA/BF16 replay handoff;
- paired arms: P-e boundaries/dose, message pairing, P-f classes, and random
  controls are frozen privately and bound by
  `analysis/round5/pe/corpus_freeze.json`;
- residual hidden states are excluded. This pass does **not** satisfy D4 or
  the registered hidden-state clauses of R5-C.

The expected production inventory is 1,788 artifacts. The runner records a
failed manifest on any exception and never treats a partial tree as reusable.

## Required environment

Use the pinned Tier-2 virtual environment. The system Python does not provide
this checkout's Inkling Transformers implementation.

```powershell
.\.venv-tier2\Scripts\python.exe scripts\round5_pe_paired_build.py check
.\.venv-tier2\Scripts\python.exe scripts\round5_widened_validate.py check-inputs
```

Both checks must pass before GPU work. The builder refuses to overwrite the
existing private freeze.

## Production sequence

The critical runner, builder, registration documents, and public freeze must
first be committed at `HEAD`; the A8 gate deliberately rejects working-tree
versions of any critical public dependency.

```powershell
.\.venv-tier2\Scripts\python.exe scripts\round5_widened_capture.py preflight
.\.venv-tier2\Scripts\python.exe scripts\round5_widened_capture.py capture
.\.venv-tier2\Scripts\python.exe scripts\round5_widened_validate.py validate
```

Production preflight hashes every indexed trunk checkpoint shard. Do not use
`--skip-shard-hashes` for the production decision. Likewise, do not use the
validator's `--skip-shard-rehash` when certifying the capture. Those switches
exist only for local development diagnostics.

After independent validation passes, run the LF5 CUDA/BF16 replay parity gate:

```powershell
.\.venv-tier2\Scripts\python.exe scripts\round5_offline_attention.py parity --backend replay --input-root dumps\round5\widened_corrected_capture --capture-root dumps\round5\widened_corrected_capture\replay --layers all --report analysis\round5\widened_capture\lf5_replay_parity.json
```

The production order after validation is: mechanical re-certifications first;
LF3, LF8 (beginning at the L53 anomaly), LF9, and R5-B; then P-e/P-f; then the
separate R5-D ablation campaign.
