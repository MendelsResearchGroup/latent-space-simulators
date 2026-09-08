# Verified network edits

Each folder contains original.pt, engineered.pt, original_network.lmp,
engineered_network.lmp and provenance.json. Positions, topology and zero springs
are unchanged. Positive stiffness edits obey |log(k/k0)| <= 0.25.
The response is measured from first/last unwrapped states with60 compression
increments. Registry response metadata on edited graphs is cleared because it
belongs to the original graph. See provenance.json for newly measured values.

These are the best designs found among seven methods/random seeds per network.
This selection is useful for engineering; it is not an unbiased estimate of AE
performance. Use the paired source-wise experiment tables for method comparisons.
All six graphs came from the validation design subdivision, not final test.
