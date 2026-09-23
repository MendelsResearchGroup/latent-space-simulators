
### Temporal linearity quick check — 2026-09-08

User asked when Reid/dePablo trajectories become linear. Definition: fit each
node coordinate against saved frame index, pool temporally centered SSE/SST;
20-frame sliding windows,20 existing validation trajectories per source. No
model training, response selection, or final-test analysis. All60 trajectories
and every window finite; exact IDs/code hashes/manifest in
notebooks/results/temporal_linearity_check/recipe.json, raw fits and summaries
alongside it. These are coordinate-linearity R², not p-ratio scores.

For all20 trajectories to pass in every subsequent window: threshold0.99 is
met from window start0 for Reid and low-T; stricter0.999 is met from Reid6
and low-T2. Mixed-T never meets either criterion through the last window180–199.
Whole0–199 straight-line fit median R²: Reid0.999939,low-T0.999833,mixed-T0.766289.
Mixed-T early/late20-frame window medians remain about0.18–0.23; no clear
noise-free linear regime. Minimum-image fractional-coordinate unwrapping using
each evolving box found zero crossings in these60 trajectories and preserved
results; this does not establish absence of wrapping elsewhere in the datasets.
Saved frame index is the reported unit; physical time per frame is undocumented.
Threshold-dependent empirical characterization, not a universal onset or proof
that the smaller motion components relevant to p-ratio are easy to predict.
