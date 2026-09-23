# Thermal engineering protocol audit

Read-only audit on 2026-09-08. Consulted CONTRIBUTING.md, the experiment
results index, 06b log, frozen recipes, donor code, and MetaForge. No raw
reserved-test rows were loaded; no simulation or training ran.

## Safe frozen mixed-T cohort

Data artifact: /rg/mendels_prj/alexander.z/DL-course-project/data/depablo-10k-mix-temp.pt.
The normalization audit records 300 trajectories x 200 frames with physical
reference/box inversion checks passed:
notebooks/results/in_memory_normalization_audit/depablo_mixed_temp.json.

The safe 20-ID validation-only mixed-T cohort is frozen in:

- notebooks/results/06b_ae_mixed_ablation/exclude_mixed_d2_s786/recipe.json,
  mixed_evaluation.dataset_mixture[0].split_indices.val. It loads mixed-T only
  after AE/propagator fitting and selection.
- notebooks/results/06b_4d_reconstruction/d2_s786/recipe.json, the same
  validation IDs but with mixed-T included in AE exposure.

IDs: 268,161,47,191,242,208,221,140,118,158,250,136,68,96,262,70,78,44,36,39.
Five frozen seed bundles exist: 123,456,786,2026,3456456. Both recipes declare
test: [] for their used mixed-T spec. Do not load remaining dataset indices.

Root's existing temporal-linearity recipe gives the safe one-ID-per-temperature,
response-blind existing-validation order: T=0.1 ID47, T=1 ID68, T=5 ID140,
T=10 ID161, T=20 ID242, T=30 ID268. The mixed split builder,
/rg/mendels_prj/alexander.z/DL-course-project/scripts/make_latent_study_manifest.py,
also records six T strata and validation quotas 4/4/3/3/3/3.

## Linear p-ratio across all frames

The existing repository temperature estimator is
src/lss/latent/simulation.py:trajectory_p_ratio_sides_strain_gated. It fixes
quantile-0.10 reference-frame boundary groups, forms width/height strains,
drops nonfinite and low-strain frames, requires >=8 frames and strain range
>=1e-3, and performs two ordinary numpy.polyfit(frame_index, strain, deg=1)
fits. It selects the negative transverse/driven slope of smaller absolute
magnitude. Dispatch is src/lss/latent/experiment.py:ground_truth_p_ratio with
cfg p_ratio_estimator=strain_gated_trajectory.

That function is not the root-selected estimator. The selected definition uses
two OLS fits with intercept over all 200 physical unwrapped frames, against the
saved frame index: epsilon_x(t)=a_x+b_x*t and epsilon_y(t)=a_y+b_y*t, then
p=-b_y/b_x for known x-driven compression. The implementation and six raw
validation baselines are thermal_metric.py and raw_linear_fit.csv in this
folder. Use fixed initial quantile-0.10 side groups. Do not use the
endpoint-only graph_utils package, the robust Theil-Sen estimator,
reciprocal/minimum branch, or physics-selected frame window. Return NaN and
report valid/total below the predeclared fitted axial-total-strain threshold.
Physical time/frame remains undocumented; graph.time metadata is retained but
not substituted for saved frame index.

For model frames, invert predicted normalized positions with the original
reference_box_half_extent and reference_box_center before the fit. The exact
physical inverse pattern is hpc/experiments/compact_lj_response_rollout/evaluate.py:physical.
Its current p-ratio call is endpoint-only, so it is an inversion reference,
not a thermal estimator implementation.

## Thermal LAMMPS final verification: no reproducible protocol found

No checked-in repository, donor, or MetaForge source specifies the generation
protocol for depablo-10k-mix-temp.pt: units, T-to-ID source mapping beyond the
existing validation recipe, thermostat/integrator, damping, timestep,
deformation cadence, dump cadence, random seeds, or physical time/frame.
The data file is derived data, not a recoverable LAMMPS recipe. Therefore no
exact thermal verification can be claimed until this provenance is recovered.

Existing physics paths are athermal and cannot substitute:

- donor notebooks/network_design/endpoint_physics.py:validate_endpoints uses
  2D quasistatic compression, box relaxation/minimization, and a first/last
  endpoint ratio.
- MetaForge src/auxetic/simulation.py:run_elastic_simulation and
  src/auxetic/scripts.py:ElasticScript calculate elastic constants by
  minimization/finite deformation. ElasticScript has dt=0.01, strain=1e-4,
  srate=1e-5 and nominal 1000 steps, but no thermostat or thermal trajectory.

They can check graph export only. A future verifier must recover and freeze the
thermal protocol, then predeclare T, three seeds, integrator/thermostat,
timestep, deformation and dump schedules, the all-frame OLS estimator, and
all final-design hashes before any simulation. Preserve per-T and pooled
uncertainty over networks/seeds with valid/total and failures.

## Boundary

The frozen low-T z1 readout can be described only as a zero-shot direction for
mixed-T if it is transferred without mixed-T response fitting. Existing
mixed-T reconstruction/rollout bundles do not establish joint
position-plus-stiffness intervention reliability. ML-only optimization must
freeze every positions and joint final before any recovered-protocol thermal
verification.
