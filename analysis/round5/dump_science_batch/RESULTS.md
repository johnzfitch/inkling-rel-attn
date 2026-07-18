# Round 5 certified-capture dump-science batch

**Status: six rows answered by the first analyst; independent raw-dump
re-derivations remain required for certification.** All results use capture
manifest `2fc2ec9dd4d6b684a473847f6fbd4541d4326b49d7d6456b463ee5722399a75f`
and the publicly timestamped execution plan at `88dd002`.

| Row | Registered outcome |
|---|---|
| R5-C lifecycle | 4786 sustained onset at L26; 3290 has no registered positive-coverage onset; L39/40 handoff gradual/mixed (0/6 sharp) |
| R5-C geometry | carrier <1% failed (maximum cell 10.82%); flip-band discontinuity failed; globals-rotate-more passed (p=0.015625) |
| LF3 | failed into the surprise branch: random-text tail clock max abs(r)=0.9750 at L59, search-wide p=1/127; BOS transient also present |
| LF8 | all three failed: minimum cross-text cosine 0.3852 at L53; no global/local family split; 47 chirality candidates |
| LF9 | failed: capacity peaks at L11, though L65 is the unique minimum; bias decreases far share at all registered mid layers |
| R5-B | depth profile passed with unique peak L41 and L65 collapse; random-nearest/code-farthest failed (code nearest, needles farthest) |

Each row has a `RESULTS.md`, machine-readable JSON, compressed numerical dump,
input hashes, source hash, and an overwrite guard. Artifact hashes and array
finiteness were checked after execution. The surprising R5-C carrier maximum
was independently spot-checked from the full L20/needles state and communal
basis (`0.1081893562` direct versus `0.1081892867` reported).

## Execution-order deviation

The execution plan says the unchanged mechanical A6 re-certifications precede
these newly unblocked rows. The primary agent mistakenly ran these six rows
first. This is a procedural deviation, not an outcome-dependent method change:
all six estimators and verdict mappings were committed publicly before their
outcomes, none consumes a mechanical re-certification result, and no threshold,
band, input, control, or failure disposition changed. The mechanical
re-certifications remain next and this deviation is not repaired by rewriting
timestamps or rerunning outcomes.
