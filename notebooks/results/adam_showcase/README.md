# Adam presentation provenance

The notebook presents one common joint-Adam recipe per dataset, with positions and stiffnesses edited through the frozen 2D AE and historical response-calibrated z1 readout. The model was trained on positions/graphs, not p-ratio targets. LAMMPS verified the frozen final designs only.

- **Reid:** all ten positive development originals; recipe `lr0.08_b0.25_p50_mask_blocking_nodes`, chosen retrospectively for its conversion count in the completed sweep. One common recipe, not a different physical winner for each network. Nine of ten become negative; Reid93 remains positive and Reid52 is close to zero.
- **Mixed-T:** all thirty development networks, five at each temperature; recipe A (lr0.02, log-stiffness bound0.25, patience50). Statistics use three velocity seeds per network. Seven of fifteen positive originals become negative; the other fifteen originals were already negative.
- **GIFs:** first five Reid cohort entries (93,84,96,97,52), including the failure to cross zero. Two illustrative thermal conversions at20 K/ID234 and30 K/ID262 use the fixed velocity seed786; their labels describe that displayed trajectory, whereas the statistics summarize all three seeds. GIFs run for4.000 seconds, display actual physical frames and do not magnify displacement.

Reid endpoint runs lacked intermediate dumps. The same frozen graphs were replayed under the identical120-increment protocol solely to save121 frames; endpoint p-ratio must match within1e-6. Mixed-T GIFs reuse existing200-frame trajectories. Periodic bonds and positions are drawn in the simulation box with fixed paired axes.

The full Reid and thermal sweeps remain in `../reid_joint_sweep/study_v1/` and `../thermal_adam30/study_v1/`. All sources, inputs and code hashes are in `provenance.json`; job identities are in `submission_status.json`; replay checks and GIF hashes/durations are in `assets/` and `index.json`. The initially rejected compute-host submission is preserved separately; actual submissions use the login server.

These are development examples, not an unseen-network benchmark. Mixed-T includes20 validation and10 training-partition networks; four match AE-training topology/stiffnesses and five match historical readout calibration. The physical thermal protocol is explicitly declared and is not exact replay of the unavailable GT generator. Full research evidence remains in [the engineering log](../../latent_space/network_engineering_log.md).
