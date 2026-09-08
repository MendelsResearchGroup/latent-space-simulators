# Repository instructions

Follow the research and notebook conventions in `CONTRIBUTING.md`. Before proposing or launching experiments, read `notebooks/latent_space/experiment_results_index.md` and the relevant logs. Preserve exact recipes, splits, seeds, code hashes, job identities, failures, and source-wise metrics with valid/total counts. Update the index after collecting results. For notebook 07b, also update `notebooks/latent_space/07b_experiment_log.md`.

Prefer broad, hypothesis-driven matrices and multiple seeds when they resolve an open question. Reuse existing evidence and avoid needless duplication. Treat latent size as an empirical question: matched 2D/4D and, for noisy-LJ, 6D/8D comparisons. Do not introduce expert observable supervision or selection. Keep LJ graph-distance-two/three augmentation consistent in all stages.

Use `mendels_q`, `place=free:shared`, no host pins, and measured CPU/memory requests. Ordinary attention-AE runs default to 8 CPUs and 16 GB. Do not reserve whole nodes exclusively.

Keep notebook Markdown minimal, plots inline, and use the Editorial theme. Do not save figure files unless requested. Implement requested features directly rather than introducing optional switches without a reason.

## Agent roles

The requested primary role is GPT-6 Astra, medium reasoning (`gpt-6-astra`, `medium`): architecture, scientific hypotheses, interpretation, and overall research direction. These instructions do not change an already-running session's model.

Delegate bounded operational tasks to GPT-5.6 Terra, medium (`gpt-5.6-terra`, `medium`) while the primary continues independent work. Terra may inspect configurations, submit and monitor PBS jobs, analyze logs/results, and make small experiment-specific changes. It must not redesign the architecture or change scientific objectives. Supply exact scope, owned files, relevant prior evidence, expected outputs, and stopping criteria. Use explicit model/effort overrides with limited or no history when supported. Reuse existing agents and assign one writer to shared logs.
