# Dense Reid with angles: completed-matrix review

Reviewed 2026-09-09 from the cached collector tables and run provenance. No notebooks were executed and no result, checkpoint, or training code was changed during this review.

## Completion and protocol checks

All predeclared full-matrix jobs completed successfully: `4677439`–`4677444` are PBS state `F` with `Exit_status = 0`; collector `4677445` is also `F` with `Exit_status = 0`. `collection_status.json` records all six cells complete. The six cells are 2D/4D by seeds 123/456/786; each has both `one_step` and `multistep8` frozen-AE propagator fits and validation/final-test outputs.

Each cell contains 101 frames (0–100) for every AE/strategy and split. Validation has 10 networks and final test has 20. At all requested final-test milestones (1, 5, 10, 25, 50, 100), all three metrics are valid for all 20/20 networks in each seed. The metrics below are seed mean +/- sample SD (`n=3`). `D` is displacement-from-reference R2 pooled over *x/y* coordinates; `I` is consecutive-frame displacement R2 pooled over *x/y* coordinates. They are coordinate diagnostics, not a y-only score. `P` is the post-fit p-ratio transverse-response R2.

No failure, traceback, or collector exception appears in the seven job logs. Every propagator completion record reports `ae_unchanged: true` (12/12), and each propagator log explicitly reloads its cell's pretrained AE. Thus there is no AE-change or unfrozen-AE anomaly.

The strongly negative step-1 pooled displacement scores coexist with positive AE step-1 increment scores, whereas the rollout step-1 increment scores are negative. One possible explanation is an initial-frame reconstruction offset: AE reconstruction at frame 0 is decoded, while an autonomous rollout is anchored to the ground-truth initial frame. This is a hypothesis about the evaluation convention, not a demonstrated cause, and does not by itself show that a propagator degrades small increments at later steps.

The frozen training code manifest is `2c126a8a25c071ca614899aa2bd435a35fbc7e9a3ee2f6de2f4cc65eab0c8f8e`; the data manifest is `75b3d61adb3ef0c1f1b7b72fc583a99db23928217ba30c8ec54da834ff08feb4`. The collection-script hash is `99a5ea4121c8dc7a004e4e5539b8739ce8ce2ca5859c8ed499a81f4342b9eb05`.

## Final-test scores

| latent | model | step | D (x/y pooled) | I (x/y pooled) | P (transverse response) |
|---:|---|---:|---:|---:|---:|
| 2 | AE | 1 | -29.061233 +/- 21.868736 | 0.439644 +/- 0.321905 | -1.544372 +/- 3.313098 |
| 2 | AE | 5 | 0.613118 +/- 0.157443 | 0.605048 +/- 0.147011 | 0.415447 +/- 0.264789 |
| 2 | AE | 10 | 0.920173 +/- 0.029652 | 0.820366 +/- 0.078041 | 0.743232 +/- 0.131660 |
| 2 | AE | 25 | 0.930262 +/- 0.000633 | 0.904054 +/- 0.001182 | 0.946392 +/- 0.017481 |
| 2 | AE | 50 | 0.923720 +/- 0.001023 | 0.900442 +/- 0.009651 | 0.965313 +/- 0.009027 |
| 2 | AE | 100 | 0.918537 +/- 0.000692 | 0.830400 +/- 0.043647 | 0.933301 +/- 0.022105 |
| 2 | one_step | 1 | -28.914418 +/- 22.013033 | -28.914418 +/- 22.013033 | -0.996010 +/- 2.344723 |
| 2 | one_step | 5 | 0.621504 +/- 0.196263 | 0.622811 +/- 0.163580 | 0.390512 +/- 0.276265 |
| 2 | one_step | 10 | 0.919302 +/- 0.024861 | 0.795951 +/- 0.080141 | 0.593814 +/- 0.163193 |
| 2 | one_step | 25 | 0.923552 +/- 0.002795 | 0.909897 +/- 0.005271 | 0.775485 +/- 0.062213 |
| 2 | one_step | 50 | 0.922625 +/- 0.000867 | 0.903099 +/- 0.008965 | 0.815100 +/- 0.046758 |
| 2 | one_step | 100 | 0.918347 +/- 0.001809 | 0.859778 +/- 0.032175 | 0.821303 +/- 0.040173 |
| 2 | multistep8 | 1 | -28.926266 +/- 22.092687 | -28.926266 +/- 22.092687 | -0.562754 +/- 1.593251 |
| 2 | multistep8 | 5 | 0.618754 +/- 0.220245 | 0.649445 +/- 0.170618 | 0.443946 +/- 0.303362 |
| 2 | multistep8 | 10 | 0.924665 +/- 0.013631 | 0.813709 +/- 0.075570 | 0.628359 +/- 0.159827 |
| 2 | multistep8 | 25 | 0.926127 +/- 0.003080 | 0.911399 +/- 0.003150 | 0.784476 +/- 0.063265 |
| 2 | multistep8 | 50 | 0.922655 +/- 0.001447 | 0.905989 +/- 0.007154 | 0.813332 +/- 0.050123 |
| 2 | multistep8 | 100 | 0.918821 +/- 0.001750 | 0.861021 +/- 0.028288 | 0.816260 +/- 0.045016 |
| 4 | AE | 1 | -22.490456 +/- 21.223366 | 0.360712 +/- 0.345303 | -0.056508 +/- 0.552053 |
| 4 | AE | 5 | 0.757445 +/- 0.179150 | 0.520072 +/- 0.328232 | 0.269479 +/- 0.374877 |
| 4 | AE | 10 | 0.912705 +/- 0.036100 | 0.811496 +/- 0.133414 | 0.678230 +/- 0.186053 |
| 4 | AE | 25 | 0.923479 +/- 0.012365 | 0.909622 +/- 0.004525 | 0.932200 +/- 0.025717 |
| 4 | AE | 50 | 0.923623 +/- 0.005941 | 0.906390 +/- 0.006897 | 0.957869 +/- 0.016539 |
| 4 | AE | 100 | 0.914588 +/- 0.008256 | 0.828090 +/- 0.026886 | 0.944904 +/- 0.010001 |
| 4 | one_step | 1 | -21.970057 +/- 20.626675 | -21.970057 +/- 20.626675 | -0.524784 +/- 1.444658 |
| 4 | one_step | 5 | 0.769035 +/- 0.183404 | 0.572146 +/- 0.271720 | 0.373588 +/- 0.323449 |
| 4 | one_step | 10 | 0.917565 +/- 0.022573 | 0.783702 +/- 0.138345 | 0.596662 +/- 0.257074 |
| 4 | one_step | 25 | 0.917135 +/- 0.015052 | 0.908944 +/- 0.004387 | 0.776886 +/- 0.102353 |
| 4 | one_step | 50 | 0.921956 +/- 0.004335 | 0.912526 +/- 0.004252 | 0.806576 +/- 0.046836 |
| 4 | one_step | 100 | 0.914155 +/- 0.007564 | 0.867862 +/- 0.049477 | 0.816709 +/- 0.029053 |
| 4 | multistep8 | 1 | -22.107979 +/- 20.510251 | -22.107979 +/- 20.510251 | -0.743239 +/- 1.832126 |
| 4 | multistep8 | 5 | 0.749062 +/- 0.176640 | 0.649957 +/- 0.244249 | 0.396762 +/- 0.246997 |
| 4 | multistep8 | 10 | 0.929634 +/- 0.006511 | 0.829884 +/- 0.123945 | 0.606669 +/- 0.186165 |
| 4 | multistep8 | 25 | 0.925238 +/- 0.011376 | 0.913208 +/- 0.005300 | 0.775098 +/- 0.076458 |
| 4 | multistep8 | 50 | 0.924064 +/- 0.004652 | 0.913029 +/- 0.000719 | 0.804249 +/- 0.053727 |
| 4 | multistep8 | 100 | 0.915601 +/- 0.007155 | 0.871383 +/- 0.039407 | 0.816356 +/- 0.034519 |

## Run identity and timing

| cell | job | AE state SHA-256 | AE epochs | AE seconds | full-run seconds |
|---|---|---|---:|---:|---:|
| d2, s123 | 4677439 | `a26a80c4ab16ff938f2efcdb859c3c1716f57354c3a99b63835c16135ea83fd7` | 27 | 3388.7 | 7400.3 |
| d2, s456 | 4677440 | `4f78051e2e39f3c3c7863e8f2934f112dc1457949079b4459d718d46a201ae0b` | 35 | 5130.2 | 6594.9 |
| d2, s786 | 4677441 | `4b596729f3f74a7d6ac316dcd50f97d285a5944823c8d10f1e385592564dc7db` | 60 | 7636.3 | 12273.5 |
| d4, s123 | 4677442 | `4c7e2d2faefacacba6f096e6f422461db59379a315b14790431f7616ce703d01` | 28 | 3669.0 | 7279.9 |
| d4, s456 | 4677443 | `303b4f6eb02f014128d2a82b48f0a64cdf5f6654aa7b58f853d83dea12add8e1` | 60 | 7951.9 | 12903.5 |
| d4, s786 | 4677444 | `f4aa085f17d4bff8a4eaf361df6432fef40003bdb459b601fb7ee3c22a290721` | 60 | 8330.6 | 13151.5 |

All full jobs requested 8 CPUs and 16 GB in `mendels_q` with `place=free:shared`; their PBS peak resident memory was 1.75–1.84 GB. Collector `4677445` completed in 18 seconds with 25.7 MB resident memory.

Checkpoint bundle hashes, per-seed steps, source-wise counts, and the collector output remain in the run directories and cached CSVs; this review does not select a preferred latent size or propagation strategy.
