### Matched message-passing dimension and autonomous response study — 2026-09-08

User requested matched 2D message passing, autonomous propagation, and inline
source-wise p-ratio R² versus step plots comparing reconstruction with rollout.
LJ R² >= 0.4 at the declared nonzero horizons is the desired outcome, not a
checkpoint-selection rule or a promised result. Retain all negative/undefined
scores. AE reconstruction is an observed-frame reference, not a mathematical
upper bound on an observable score.

New AE: mp2_r16_d2, five seeds 3456456/123/456/786/2026, matching completed
mp2_r16_d4/code_v2 except latent dimension and output paths. Two encoder message
passing rounds, 16D static reference/node, hidden96, 32 decoder tokens. Full
200 frames (0–199), not just 100: 30 Reid + 30 low-T + 60 LJ trajectories,
24,000 frame samples/epoch, validation20/source, mixed-T20 post-fit only.
Required graph-distance-two/three LJ augmentation and exact existing splits
remain unchanged. Existing MP4D/8D checkpoints are reused. AE selection remains
worst-source coordinate validation reconstruction; no response callbacks.

Propagation design: frozen MP2 2D and 4D AEs, five seeds each, shared deltaMLP
hidden64 and mean static graph context projected16, with one-step delta loss
versus eight-step autonomous latent-state supervision (horizons1–8). Both learn
from the three retained sources only, full199 available transitions. State-only
validation loss selects propagator checkpoints; explicitly disable historical
default p-ratio selection. No expert inputs, physics losses, temperature/source
metadata, or final-test use. Reference initial state is frame0; no future
observations are supplied during autonomous rollout.

Report physical-coordinate endpoint directional-side p-ratio at25/50/100/150/199,
using exactly the same physical inverse, original reference side groups, and
20 validation networks/source as the audited AE backfill. Mixed-T remains
post-fit transfer. Save per-network true/predicted values, valid/total counts,
seed means/SDs, source-wise curves, checkpoint/code/data hashes and PBS recipes.
The completed MP4D reconstruction is 0.633±0.095 at100 but0.127±0.284 at199;
late representation error and propagation error must be distinguished.

Execution artifacts: compact_lj_mp2d/ and compact_lj_response_rollout/ under
notebooks/results; runners under matching hpc/experiments subdirectories.
Integration smokes gate full submission. Shared mendels_q, place=free:shared,
no host pins; full training8CPU16GB. Submission and completion status will be
recorded separately from this prospective recipe.
