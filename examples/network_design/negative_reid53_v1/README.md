# Negative-response Reid53 design

Original p-ratio: +0.09049822. Previous design: +0.04956001.
New first/last physical p-ratio: -0.00601217539, confirmed with60 and120
compression increments at the same matched x compression0.9700107.

`engineered.pt` and `engineered_network.lmp` contain the edited initial network.
`original.pt` contains the original initial graph. Positions and topology are
unchanged; positive spring stiffness multipliers range0.546115–2.013753 relative
to original. Original registry-response metadata is cleared on the edited graph;
new physical measurements are in summary.json and evaluations.csv.
`confirmed_initial.dump` and `confirmed_final.dump` are the120-step verification
endpoints. The AE remained frozen; physical measurements selected downstream
edits. This is one targeted validation-graph engineering result.

Exact recipe, hashes, seed, accepted steps and all22 physical checks are retained.
