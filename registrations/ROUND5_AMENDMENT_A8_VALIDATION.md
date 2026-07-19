# Round 5 Amendment A8 — historical-source validation after concurrent commit

Registered after the corrected four-arm capture completed, but before any NLL
mean, aperture value, or registered outcome was opened.

## What happened

The corrected capture started at Git commit `7dc4460` and recorded that commit
plus SHA-256 hashes of its four source dependencies. While it was running, two
separate documentation commits advanced the shared checkout; `76eb533` moved
all registration documents byte-for-byte into `registrations/` and updated
their path references in three corrected-run scripts. The capture manifest
finished at 16:58:38; the working `corpus_v2_capture.py` was rewritten for the
new paths at 16:59:02. No model-forward code changed.

Independent validation attempt 1 rehashed all 33 trunk shards and all 268
capture artifacts successfully, verified shapes/finiteness/NLL alignment, then
failed solely because it compared the historical capture source hash with the
new checkout's path-updated source. That failed report is preserved at
`analysis/round5/corpus_v2_corrected/capture_validation.json`.

## Corrected validation rule

Attempt 2 writes a new report, `capture_validation_v2.json`. Capture source and
registration hashes must equal the exact Git blobs at the manifest's recorded
`capture_git_head`; the validator obtains those blobs directly from Git and
does not trust the current working tree. Current-checkout drift is listed as a
diagnostic, not silently ignored. Private inputs, installed package/source
paths, checkpoint bytes, and artifact bytes are still checked against current
files because those are not recovered from the public Git tree.

The attempt-2 report records the SHA-256 of attempt 1. The corrected pipeline
accepts only the passed attempt-2 report. No threshold, class, band, seed,
statistic, prediction, control, or captured byte changes under this amendment;
no recapture is authorized or needed when the recorded Git blobs authenticate.
