# Data

The local project includes the six available PyTorch trajectory datasets and their available companion metadata. `manifest.json` records their sizes and SHA-256 hashes. The loaders in `lss.data` handle graph canonicalization and in-memory reference-box normalization; separate pre-normalized copies are unnecessary.

| File | Ensemble |
| --- | --- |
| `reid_200_frames.pt` | Reid trajectories used in the main shared-AE studies |
| `real_reid_200_frames.pt` | Additional Reid trajectories; companion manifest retained |
| `depablo-near-zero-temp.pt` | dePablo low-temperature trajectories |
| `depablo-10k-mix-temp.pt` | dePablo mixed-temperature trajectories |
| `2340_dePablo_networks_OOL_undirected.pt` | Earlier dePablo collection |
| `lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt` | Noisy-LJ trajectories; companion CSV retained |

PyTorch datasets contain graph objects and require `torch_geometric` and `graph_utils`. Large data files are excluded from Git. The older baseline/archive notebooks additionally reference two datasets that are not included; see the notebook map.

For LJ, keep the original spring edges and add missing node pairs at graph distance two or three in the reference spring graph. Added edges have zero raw spring stiffness, an LJ indicator, and geometric features. This same five-channel representation is used in fitting and inference.

`data_mini.tar.gz` contains separate node-position-optimized and stiffness-optimized collections. `tools/data/convert_data_mini.py` converts their first 50 consecutive frames into canonical graph datasets and records archive provenance, original frame counts, and trajectory identities in `mini_data_converted.manifest.json`. The stiffness study additionally uses a 120-trajectory cohort with distinct fitting (first 20 frames) and evaluation (first 50 frames) files. See `notebooks/results/mini_stiffness/prepared.json` for exact 20/20/80 split indices once conversion completes. The original archive retains all 1,500 frames per sampled trajectory.

The canonical mini-data target is **200 stiffness trajectories × 1,500 frames and 200 node-optimized trajectories × 500 frames**, saved as `mini_node_optimized_200_500.pt` and `mini_stiff_optimized_200_1500.pt`. The stiffness selection includes the existing 120-trajectory study cohort. Smaller first-20/first-50 files are experiment inputs, not the canonical full trajectories; the remaining 80 trajectories are outside this study’s 20/20/80 split. Full-conversion provenance records the exact selections.
