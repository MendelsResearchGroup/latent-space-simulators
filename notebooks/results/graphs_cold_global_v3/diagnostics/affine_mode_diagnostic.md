# Frozen v3 validation affine-mode diagnostic

Scope: autonomous replay of the completed seed-123 `anchored_contact` and `anchored_contact_recurrent` checkpoints on their frozen 12 validation trajectories only. No final-test trajectory was loaded; no model, checkpoint, or selection state changed.

The deterministic counterfactual subtracts, separately for each coordinate, the least-squares constant plus all three reference-fractional-coordinate affine terms from the frozen predicted fractional residual, then retains the frozen predicted box. Its response R2 is numerically the box-only R2 at every horizon (roundoff-scale differences). Therefore removal of those node residual affine modes would make the position-derived p-ratio equal the predicted-box p-ratio; it cannot make response accurate when the box-only response is inaccurate.

At horizon 399, box-only y/x and z/x R2 are respectively -5.576/-9.982 for anchored contact and -8.171/-12.528 for recurrent. Position R2 is almost the same (-5.597/-9.949 and -8.326/-12.425), and the mean position-versus-box p gaps are small (y/z: .00345/.00341 and .00658/.00249). Hence late failure is overwhelmingly global affine/box response, not a residual nonaffine fractional-position affine mode.

Mean fitted fractional-residual diagonal affine slopes are tiny at horizon 399: anchored xx/yy/zz = -8.13e-6/-3.95e-7/1.77e-5; recurrent = 3.60e-5/-2.45e-5/5.24e-6. Mean x-strain bias is also small (-1.87e-5 and 2.46e-5), whereas transverse strain biases are about -1.75e-3/-2.40e-3 and -2.11e-3/-2.76e-3 (y/z). This identifies transverse/global affine bias as the larger strain component, although the unstable x-box denominator makes all early p-ratio R2 values extremely negative.

Exact per-network strains, residual affine coefficients, counterfactual p-ratios, horizon R2, and hashes are in the adjacent CSV and JSON files.
