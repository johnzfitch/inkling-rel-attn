# Registered documents

Every spec, pre-registration, execution plan, and amendment in the project.
The **git commit timestamp of each file is its registration** — predictions in
these documents were committed before the analyses that test them ran.

Two rules govern this directory:

1. **Contents are immutable after registration.** Several capture manifests and
   verification reports record SHA-256 hashes of these files; editing one
   breaks the frozen-provenance chain. Corrections and scope changes are made
   by adding a new amendment, never by editing a registered document. (Files
   were moved here from the repo root in one rename-only commit; contents are
   byte-identical, so all recorded hashes remain valid. Path references inside
   older documents refer to the original root-level locations.)
2. **A new registration must justify its priority.** Per the working rules in
   [`../QUESTIONS.md`](../QUESTIONS.md), a new spec states why it outranks the
   open runnable-now rows of the question ledger.

Reading order for newcomers: `TIER2_SPEC.md` → `ROUND4_SPEC.md` →
`ROUND5_LEFTFIELD_SPEC.md` (the question register) → amendments in order
(A5 → A6 → A7). The ledger at the repo root tracks which questions each
document poses and their current status.
