# Interior spring-crossing audit

Audited all10 original/optimized Reid pairs shown by the Adam cohort notebook, configuration `lr0.08_b0.25_p50_mask_blocking_nodes`, from their saved physical design graphs before LAMMPS. All10 originals have zero detected crossings; all10 optimized designs have16–66 proper interior crossings between nonincident physical springs. These fail the user's planar no-crossing requirement regardless of recorded p-ratio. No claim is made here about the unexamined mixed-temperature designs or other runs.

| Reid ID | Original crossing pairs | Optimized crossing pairs |
|---|---:|---:|
| 2 | 0 | 40 |
| 52 | 0 | 32 |
| 53 | 0 | 39 |
| 57 | 0 | 16 |
| 59 | 0 | 38 |
| 74 | 0 | 30 |
| 84 | 0 | 66 |
| 93 | 0 | 53 |
| 96 | 0 | 30 |
| 97 | 0 | 33 |

This conservative audit counts proper intersections of unique undirected spring segments only when neither segment wraps across a periodic boundary. Shared-node joints are excluded. It therefore demonstrates actual internal crossings, independently of periodic rendering. It does not enumerate collinear overlaps, near contacts, or crossings involving wrapping springs; counts are not a complete collision audit. Exact node-index edge pairs and graph hashes are in `details.json`; summary in `reid_showcase_crossings.csv`. Four analytic checks cover true crossings, legitimate shared joints, duplicate orientations and disjoint segments (`validation.json`).

Current `valid_geometry` and `blocking_nodes` in `src/lss/engineering/geometry.py` constrain spring lengths and pairwise node separation; `valid_geometry` also preserves boundary memberships. They do not test spring–spring intersections. The optimizer has not yet been changed by this audit. Changed constraints require a new frozen engineering recipe and must operate without LAMMPS feedback.

Read-only local analysis: no optimization, simulation, notebook execution or image inspection. Historical graphs and p-ratio records were preserved. The cohort notebook now explicitly labels the planar geometry failure.
