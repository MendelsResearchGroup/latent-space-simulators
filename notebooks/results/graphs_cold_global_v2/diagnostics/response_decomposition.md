# Frozen v2 validation response decomposition

Scope: seed-123 `fractional_bond` and `fractional_contact` checkpoints only, replayed autonomously on the frozen 12-trajectory validation split. No training, checkpoint selection, or final-test loading occurred. Position p-ratios use the frozen nearest-reference periodic alignment and independent affine fits. Box-only p-ratios use `-((L_y,z/L_y,z0)-1)/((L_x/L_x0)-1)`.

At horizon 399, both variants have weak global box response: y/x box R2 is -0.065 (bond) / -0.072 (contact), with predicted/GT SD 0.290 / 0.296. This is already sufficient to preclude a reliable y/x response claim. Position-derived y/x is slightly worse (-0.156 / -0.321) and becomes weakly coupled to its box-only counterpart (correlation 0.291 / 0.151), so fractional/nonaffine evolution adds a late error but does not rescue the global box defect.

For z/x, fractional positions are the larger late degradation: box-only R2 is -0.189 / -0.241 and MAE 0.077 / 0.079, while position-derived R2 is -1.640 / -0.687 and MAE 0.153 / 0.120. The position-minus-box response gap is 0.066 / 0.048. Thus the evidence is component-specific: deficient global box response is a necessary failure mode for y/x, whereas nonaffine fractional-position drift materially compounds the z/x failure.

At horizon 100, the same direction is visible: box y/x R2 is 0.016 / -0.014 versus position 0.317 / 0.199, while box z/x R2 is -0.352 / -0.380 versus position -1.318 / -0.989. Position/box z/x correlations remain high (0.933 / 0.970), so the divergence is predominantly magnitude/affine-fit sensitivity rather than unrelated response rankings.

Exact per-network values, response SDs, MSEs, correlations, and frozen hashes are in `per_network_response_decomposition.csv`, `horizon_response_summary.csv`, and `response_decomposition.json`.
