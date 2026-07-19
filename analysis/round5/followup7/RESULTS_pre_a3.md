# Round 5 seven-experiment follow-up results

Status: **ANSWERED_PENDING_INDEPENDENT_VERIFICATION**

- F7-1 signed stencil: **PASS**; all-head d0..3-only rescue `0.990`.
- F7-2 shoulder backups: **PASS**; interactions adjacent_23_29 `+0.602748`, adjacent_29_35 `+0.348319`, triple `+1.038303`.
- F7-3 r decomposition: **PASS**; mean-removal ratio `0.781`, non-carrier-mean ratio `0.796`.
- F7-4 head localization: **PASS**; top-16 stencil rescue `0.934`.
- F7-5 query patch: **PASS**; query rescue `0.541`, sham `-0.032`.
- F7-6 clock subspaces: **FAIL**; LOTO maximum kernel correlation `0.470`.
- F7-7 full vocabulary + fresh classes: **FAIL**; rank delta `+0.031144`, accuracy delta `-0.018271`, ECE20 delta `+0.004310`.

Every number above is post-preregistration but remains uncertified until the independent verifier agrees from raw dumps.
