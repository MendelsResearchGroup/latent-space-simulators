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
