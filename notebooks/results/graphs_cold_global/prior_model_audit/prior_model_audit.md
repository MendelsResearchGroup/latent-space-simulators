# Prior NN_Simulator audit

Audit date: 2026-09-17. Scope was read-only inspection of `/rg/mendels_prj/s.sergey/prj/NN_Simulator`, plus the minimum current cold-graph registry/loader context. This is provenance, not a performance claim.

## Bottom line

The saved historical model is a **2D, bond-topology-only, one-step acceleration/velocity-change GNN**. The most clearly evidenced saved run used the `data_aux_opt_lowT_448sims_noang.pt` dataset (448 trajectories by filename), random trajectory split 90%/10% with `random_state=42`, 15 epochs, skip 1, and four message-passing blocks. It predicts a per-node 2D velocity change, integrates it with two position frames, and recomputes bond geometry; it neither predicts nor evolves the `box` object. No evidence links that model/data/checkpoint to the current 3D `graphs_cold/rubber_*.pt` registry.

## Historical state and graph convention

* Frames are PyG `Data` objects with `x = [x,y]` coordinates in the active velocity training path, `edge_index`, four-channel edge attributes `[periodic dx, periodic dy, bond length, bond stiffness]`, and a `box` object. `convert.py` can also create `[x,y,vx,vy]`, but the historical trainer builds its own velocity input from coordinate differences.
* The launch script selects `node_features=velocity`; `build_graph(prev, cur, "velocity")` produces `cur.x - prev.x`, hence **two 2D velocity channels**. Target acceleration/velocity change is `(next - cur) - (cur - prev)`.
* Encoders map nodes and edges to hidden size 128. The decoder outputs two channels. The production launch uses `n_layers=4`; each layer does additive message aggregation on the supplied edges, followed by residual-style concatenation of aggregation and node embedding. The DDP script has a default of 2 layers, but the saved `e15...nl4` checkpoint and its launch recipe establish 4 for the most recent identified run.
* The update is explicit second-order integration: `v_t = x_t-x_{t-1}`, `v_{t+1}=v_t+predicted_delta_v`, and `x_{t+1}=x_t+v_{t+1}`. On every predicted frame it recomputes edge vectors/lengths from predicted positions while retaining the same edge index and final edge attribute (bond stiffness).

## Box deformation

`box` is carried unchanged from the current graph in both model output and rollout (`box=inputs.cur_graph.box`). It is not a decoded target or model input feature (beyond use in periodic edge-vector reconstruction). Therefore box deformation is neither prescribed to each predicted frame nor predicted by this model. Notebook p-ratio diagnostics infer an effective deformation from the predicted position spread and the **initial** box; that is a post hoc calculation, not a box rollout.

## Training evidence and artifacts

The strongest executable recipe is `run_ddp_script.sh`:

```text
python training_ddp.py data/data_aux_opt_lowT_448sims_noang.pt \\
  -e 15 -s 1 -ts 0.1 -trs 0.9 -nl 4 -n test.pt -f velocity
```

`training_ddp.py` performs `train_test_split(..., shuffle=True, random_state=42)` at the trajectory level. On 448 items, its 0.9/0.1 request corresponds to 403 train and 45 validation trajectories under scikit-learn's usual rounding; the exact serialized length was not loaded because the 4.36-GB tensor dataset is not required to establish the recipe. Unlike conventional distributed data parallel training, the script does not shard data by rank, so the one-GPU launch is the relevant operational configuration.

Saved run artifacts:

* `model/model_velocity_e15_s1_nl4_test.pt` SHA-256 `5ee7cb03a8f094ab0c82823851562511c9f599dceb4e525ffe95a22a98f3282e` (4,391,622 bytes).
* `model/model_velocity_e1_s1_nl4_test.pt` and the epoch checkpoints support that this 15-epoch, four-layer training produced persistent artifacts. `model/test.log` has no scalar metrics.
* `no_bias.log` records a distinct 5-epoch, 2-layer, six-GPU attempted run on the same 448-simulation input: loss 0.9093 -> 0.8620 and 204 min 58 s. Those values are training loss only.
* `model/training_stats.pt` exists but no plain-text extraction or cold/rubber metric is present in the saved logs.

## Rollout and prior metrics

`utils.get_vel_rollout` is autonomous after two ground-truth seed frames: it repeatedly feeds its prior two predicted states through `build_graph`, calls the model, integrates, and rebuilds bond-edge geometry. `test.ipynb` contains executed low-temperature notebook cells that evaluate coordinate MSE and a custom `calc_p_ratio_rollout` at 250 steps across its validation trajectories, with `dummy_edges=False`. It references `data/data_aux_opt_lowT_full_noang.pt` and a checkpoint name `model_velocity_e253_nl3_s1_nobias.pt`, neither of which is present in the inspected saved data/model list. The notebook's relevant loop stores only plotted arrays; it does not preserve tabulated aggregate values, R2, valid/total counts, reproducible trajectory IDs, or a checkpoint hash for these diagnostics.

There are **no references to `rubber`, `graphs_cold`, or the current registry** anywhere in the inspected NN_Simulator non-binary source/logs/notebooks. Thus: historical low-temperature rollout/p-ratio diagnostics exist, but **no prior cold-rubber rollout metric exists that can be validly assigned to the current `graphs_cold` cohort**.

## Spatial/contact edges

`graph_utils.add_dummy_edges` implements optional radius-like spatial augmentation: all periodic images within `max(existing bond length)+0.5`; added edges receive `[dx,dy,distance,0]` (zero stiffness). This is not a general contact/nonbonded potential representation.

It was not included in the identified trained/checkpointed models: both calls in `training.py`/`training_ddp.py` are commented out, and identified notebook rollouts explicitly set `dummy_edges=False`. The helper's presence means spatial edges were prototyped as code, but there is no evidence of a saved training run or reported metric using them. No `contact` or `nonbonded` implementation/reference was found in NN_Simulator source/logs.

## Current cold-registry contrast (inspection only)

`/rg/mendels_prj/s.sergey/graphs_cold/registry.csv` has 2,300 data rows (2,301 including header). `clean_400.json` declares 1,706 complete **400-frame** trajectory IDs, not 400 trajectories. Current loader validation requires every frame to contain `x`, `pos`, `edge_index`, `edge_attr`, `time`, `box`, `box_tensor`, `atom_ids`, `atom_types`, `molecule_ids`, and `bond_types`; positions are float32 `[nodes,3]`, edge attributes float32 `[edges,5]`, and boxes float32 `[3]`. This is incompatible by dimensionality and schema with the historical 2D dataset without a new conversion/modeling decision.

## Source hashes

| Source | SHA-256 |
| --- | --- |
| `/rg/mendels_prj/s.sergey/prj/NN_Simulator/training_ddp.py` | `d438a51c49ed6c7d1e9c2d452e05b6a16a651b7b3990dc64e857135b9a0ec673` |
| `/rg/mendels_prj/s.sergey/prj/NN_Simulator/simulator_ddp.py` | `819cd532ae9bef069caad867a93787fa0eb89fa96b38ed9dba0529405e853368` |
| `/rg/mendels_prj/s.sergey/prj/NN_Simulator/utils.py` | `4996bbce753977c9e6fa432315a26402f9907e1804e6338a145523ec71c4e932` |
| `/rg/mendels_prj/s.sergey/prj/NN_Simulator/convert.py` | `40333651292f2bda597e51c52f202652c88e4a65ffcf965e8cfa6952ccc98ef6` |
| `/rg/mendels_prj/s.sergey/prj/NN_Simulator/graph_utils.py` | `753b6425642951bfbd9cdf0cccf0ca55cf6fb67c7d4afda63b0f35174c95aca0` |
| `/rg/mendels_prj/s.sergey/prj/NN_Simulator/run_ddp_script.sh` | `83ce121d520cba89abc92a51e32715f31f2726f6fac7859a8b688f0253574475` |
| `/rg/mendels_prj/s.sergey/prj/NN_Simulator/test.ipynb` | `93c13a6bf33e5ae2456d8047616749c4e4feb3405e192d1d140e207f97f90e7a` |
| `/rg/mendels_prj/s.sergey/prj/NN_Simulator/data/data_aux_opt_lowT_448sims_noang.pt` | `19a6e9e5827a35b273b36c403ea56e8bbdedf5db9445db0588acd04f8c4d9538` |
| `src/lss/dynamics/cold_graphs.py` | `1648ff8d6610700185d889c1fc137805cdc208800ceb533e44b8ab89f06e9894` |
| `/rg/mendels_prj/s.sergey/graphs_cold/registry.csv` | `c5f67015d105d7e6d2f56af7a77bf5ddf00ba381f31a4925c123cafa6d89ff11` |
| `/rg/mendels_prj/s.sergey/graphs_cold/clean_400.json` | `8b4364cf83674089a9d22e0849742d16b24605abf2df191cd67f760853753681` |

Machine-readable findings are in `prior_model_audit.json` beside this report.
