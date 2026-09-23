# Frozen AE early-frame audit

Validation only:20 networks per dataset, seed123, frozen2D AEs. No training or final-test use. Original decoder, reference subtraction and translation-only control; fixed original directional sides and existing response estimator.

| Dataset | Frame | Decoding | p-ratio R² | Valid/total | Physical MSE |
| --- | --- | --- | --- | --- | --- |
| mini_node | 1 | original | -29.359 | 20/20 | 1.05e-09 |
| mini_node | 1 | reference_subtracted | -46.595 | 20/20 | 6.16e-10 |
| mini_node | 5 | original | -22.422 | 20/20 | 1.57e-08 |
| mini_node | 5 | reference_subtracted | -24.542 | 20/20 | 7.49e-09 |
| mini_node | 19 | original | 0.327 | 20/20 | 7.42e-08 |
| mini_node | 19 | reference_subtracted | 0.345 | 20/20 | 7.86e-08 |
| mini_stiffness | 1 | original | -1.852 | 20/20 | 4.25e-09 |
| mini_stiffness | 1 | reference_subtracted | -4.745 | 20/20 | 9.97e-10 |
| mini_stiffness | 5 | original | -3.947 | 20/20 | 4.08e-09 |
| mini_stiffness | 5 | reference_subtracted | -5.370 | 20/20 | 6.31e-09 |
| mini_stiffness | 19 | original | 0.548 | 20/20 | 1.24e-07 |
| mini_stiffness | 19 | reference_subtracted | 0.597 | 20/20 | 1.7e-07 |

Original prediction parity against saved validation CSV: maximum absolute p-ratio difference0 for both datasets. Maximum reference physical-inverse error 3.18e-07. Translation-only removal preserves p-ratio; no dominant-axis switches for original predictions at1/5/19. Reference subtraction gives exactly zero displacement at0, but worsens early response scores.

At frame5, mean predicted/true boundary change: stiffness transverse y0.628 and axial compression x1.154; node transverse y0.487 and compression x1.415. Thus the relative deformation components are biased, not merely a shared rigid shift. These component errors matter despite small coordinate MSE.

The reference error is not constant across time: stiffness median error alignment with referenceoffset .997 at1,.612 at5,−.691 at19. Post-hoc anchoring is not supported as an early-response fix. This does not test training with a reference-consistent decoder, and does not establish a unique causal explanation for learned deformation bias. Training objectives emphasize coordinate reconstruction rather than relative small-frame motion; investigating a position-only frame-balanced objective remains a hypothesis, not a result.
