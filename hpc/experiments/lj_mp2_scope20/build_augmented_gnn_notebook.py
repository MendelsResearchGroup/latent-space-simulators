#!/usr/bin/env python3
"""Build the compact cached-results notebook for the scope-20 GNN comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "notebooks/lj_mp2_scope20/01_augmented_gnn_comparison.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                              "language_info": {"name": "python", "pygments_lexer": "ipython3"}})
    notebook.cells = [
        nbf.v4.new_markdown_cell("# Augmented GNN comparison\n\nCached post-fit validation results only. The GNN was fit on positions, trajectories, and graph structure; p-ratio is evaluated only after fitting."),
        nbf.v4.new_code_cell("""from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / 'notebooks').is_dir():
    raise RuntimeError('Run from the repository root.')
sys.path.insert(0, str(ROOT / 'src'))
from lss.plotting import apply_editorial_style
apply_editorial_style('light')
EVAL = ROOT / 'notebooks/results/lj_mp2_scope20/downstream/evaluation'
required = ['postfit_seed_aggregate.csv', 'postfit_per_network.csv', 'added_edge_count_summary.csv', 'evaluation.json']
missing = [name for name in required if not (EVAL / name).is_file()]
if missing:
    raise FileNotFoundError(f'Run downstream post-fit evaluation first; missing: {missing}')
aggregate = pd.read_csv(EVAL / 'postfit_seed_aggregate.csv')
per_network = pd.read_csv(EVAL / 'postfit_per_network.csv')
edge_summary = pd.read_csv(EVAL / 'added_edge_count_summary.csv')
metadata = json.loads((EVAL / 'evaluation.json').read_text())
assert metadata['final_test_used'] is False
COLORS = {'base': '#4C78A8', 'random': '#F58518', 'latent': '#54A24B'}
LABELS = {'ljonly60': 'LJ-only AE (60 LJ)', 'shared3x20': 'Shared AE (20 each)'}"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (scope, group) in zip(axes, aggregate.groupby('scope', sort=False)):
    for arm in ['base', 'random', 'latent']:
        values = group[group.arm.eq(arm)].sort_values('frame')
        ax.plot(values.frame, values.postfit_p_ratio_r2_mean, marker='o', ms=3, color=COLORS[arm], label=arm)
        ax.fill_between(values.frame, values.postfit_p_ratio_r2_mean - values.postfit_p_ratio_r2_sd, values.postfit_p_ratio_r2_mean + values.postfit_p_ratio_r2_sd, color=COLORS[arm], alpha=.16)
    ax.axhline(0, color='.45', lw=.8)
    ax.set(title=LABELS[scope], xlabel='Horizon', xlim=(10, 100))
axes[0].set_ylabel('Post-fit p-ratio R²')
axes[1].legend(title='GNN graph')
fig.suptitle('Autonomous GNN response by AE scope', y=1.03)
fig.tight_layout()"""),
        nbf.v4.new_code_cell("""final = per_network[per_network.frame.eq(100)]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
for ax, ((scope, arm), group) in zip(axes.flat, final.groupby(['scope', 'arm'], sort=False)):
    valid = group[group.finite_pair]
    ax.scatter(valid.true_p_ratio, valid.pred_p_ratio, s=22, color=COLORS[arm], alpha=.78)
    lo = min(valid.true_p_ratio.min(), valid.pred_p_ratio.min()); hi = max(valid.true_p_ratio.max(), valid.pred_p_ratio.max())
    ax.plot([lo, hi], [lo, hi], color='.35', lw=.8)
    row = aggregate[(aggregate.scope == scope) & (aggregate.arm == arm) & (aggregate.frame == 100)].iloc[0]
    ax.set(title=f\"{LABELS[scope]} · {arm}\\nR²={row.postfit_p_ratio_r2_mean:.2f} ± {row.postfit_p_ratio_r2_sd:.2f}; n={int(row.p_ratio_valid)}/{int(row.p_ratio_total)}\")
for ax in axes[-1]: ax.set_xlabel('True fitted p-ratio')
for ax in axes[:, 0]: ax.set_ylabel('Predicted fitted p-ratio')
fig.suptitle('Frame 100 fitted response', y=1.01)
fig.tight_layout()"""),
        nbf.v4.new_code_cell("""import networkx as nx
import numpy as np
import torch
from matplotlib.collections import LineCollection
from graph_utils import draw_graph

example = torch.load(EVAL / 'edge_example.pt', map_location='cpu', weights_only=True)
xy = example['node_xy_physical'].numpy()
spring_edges = example['spring_edges'].T.numpy()
network = nx.Graph()
network.add_nodes_from((i, {'pos': xy[i]}) for i in range(len(xy)))
network.add_edges_from(map(tuple, spring_edges))
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
for ax, scope in zip(axes, ['ljonly60', 'shared3x20']):
    draw_graph(network, periodic_edges=True, node_color='#334155', node_size=4,
               show_labels=False, ax=ax, show=False)
    values = example['scope_edges'][scope]
    edges = values['added_edges'].T.numpy()
    strongest = values['scores'].argsort(descending=True)[:60].numpy()
    segments = np.stack((xy[edges[strongest, 0]], xy[edges[strongest, 1]]), axis=1)
    ax.add_collection(LineCollection(segments, colors='#e76f51', linewidths=.8, alpha=.75))
    ax.set(title=f\"{LABELS[scope]} · 60 strongest of {len(edges)} added\", aspect='equal')
fig.suptitle(f\"Held-out LJ network {example['original_index']}: springs and learned edges\", y=1.02)"""),
        nbf.v4.new_markdown_cell("The graph shows physical springs and the 60 strongest learned edges for one held-out network. The GNN also retains all graph-distance-two/three relations; every learned edge excludes those existing relations."),
        nbf.v4.new_code_cell("""edge_summary.sort_values(['scope', 'arm']).style.format({'added_edges_mean': '{:.1f}', 'added_edges_sd': '{:.1f}'})"""),
        nbf.v4.new_markdown_cell("The table reports unique undirected added edges per validation network, aggregated across seeds. The base graph carries zero added edges; random and latent arms use the same edge-count rule."),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
