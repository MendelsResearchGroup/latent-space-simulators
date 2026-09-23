#!/usr/bin/env python3
"""Physical frame-zero network descriptors for post-fit latent interpretation."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
import pandas as pd
import torch
from lss.data import normalize_dataset_edges

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'notebooks/results/mp2_latent_analysis/structure_descriptors'
RECIPE = ROOT / 'notebooks/results/compact_lj_mp2d/mp2_r16_d2_s786_code_v4/recipe.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(g):
    x = g.x[:, :2].numpy().astype(float)
    edges = g.edge_index.numpy()
    attr = g.edge_attr.numpy().astype(float)
    assert attr.shape[1] == 4 and g.edge_multiplicity == 1
    assert len(set(map(tuple, np.sort(edges, axis=0).T))) == edges.shape[1]
    vectors, length, k = attr[:, :2], attr[:, 2], attr[:, 3]
    assert np.allclose(np.linalg.norm(vectors, axis=1), length, atol=2e-5)
    direction = vectors / length[:, None]
    degree = np.bincount(edges.ravel(), minlength=len(x))
    lx, ly = g.box.x2-g.box.x1, g.box.y2-g.box.y1
    normalized = (x-np.array([g.box.x1, g.box.y1])) / np.array([lx, ly])
    centered = x-x.mean(0)
    var = np.square(centered).mean(0)
    nx, ny = direction.T
    row = dict(node_count=len(x), edge_count=edges.shape[1],
               mean_degree=degree.mean(), std_degree=degree.std(),
               mean_stiffness=k.mean(), std_stiffness=k.std(),
               mean_edge_length=length.mean(), std_edge_length=length.std(),
               box_width=lx, box_height=ly, box_area=lx*ly, box_aspect=lx/ly,
               node_density=len(x)/(lx*ly), edge_density=len(k)/(lx*ly),
               mean_x=x[:, 0].mean(), mean_y=x[:, 1].mean(),
               gyration_radius=np.sqrt(var.sum()), position_anisotropy=(var[0]-var[1])/var.sum(),
               position_xy=np.mean(centered[:, 0]*centered[:, 1])/var.sum(),
               edge_anisotropy=np.mean(nx**2-ny**2), edge_xy=np.mean(nx*ny),
               stiffness_anisotropy=np.sum(k*(nx**2-ny**2))/k.sum(),
               stiffness_xy=np.sum(k*nx*ny)/k.sum(),
               degree_stiffness_covariance=np.mean((degree[edges[0]]+degree[edges[1]])*k)/2-degree.mean()*k.mean(),
               boundary_fraction=np.mean(np.any((normalized < .1) | (normalized > .9), axis=1)),
               boundary_x_fraction=np.mean((normalized[:, 0]<.1)|(normalized[:, 0]>.9)),
               boundary_y_fraction=np.mean((normalized[:, 1]<.1)|(normalized[:, 1]>.9)),
               stiffness_x_moment=np.sum(k*vectors[:, 0]**2)/(lx*ly),
               stiffness_y_moment=np.sum(k*vectors[:, 1]**2)/(lx*ly))
    if 'registry_poisson_ratio' in g:
        row['registry_poisson_ratio'] = float(g.registry_poisson_ratio)
    return row


def main():
    start = time.monotonic(); torch.set_num_threads(1)
    recipe = json.loads(RECIPE.read_text())
    rows, sources = [], {}
    for spec in recipe['source']['dataset_mixture']:
        print('Loading', spec['name'], flush=True)
        raw = torch.load(spec['path'], map_location='cpu', weights_only=False)
        ids = spec['split_indices']['val']
        normalize_dataset_edges([[raw[ident][0]] for ident in ids], edge_multiplicity=1, edge_vector_dim=2)
        for ident in ids:
            rows.append(dict(source=spec['name'], simulation_id=ident, **describe(raw[ident][0])))
        sources[spec['name']] = dict(path=spec['path'], sha256=sha(spec['path']), validation_ids=ids,
                                     frame0_keys=sorted(raw[ids[0]][0].keys()))
        assert sources[spec['name']]['sha256'] == recipe['dataset_sha256'][spec['name']]
        del raw
    table = pd.DataFrame(rows)
    assert len(table)==60
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT/'descriptor_rows.csv', index=False)
    metadata = dict(sources=sources, rows=len(table), code_sha256=sha(__file__), recipe_sha256=sha(RECIPE),
                    definition='Original physical frame0; unique undirected spring edges. Raw vector norm equals stored length. No rest-length, spring-extension, or energy claim. Population standard deviations. Orientation moments use unit stored edge vectors; stiffness moments are sum(k*vector_component^2)/box_area. Boundary fractions use outer10% of physical box. Registry p-ratio is source metadata, distinct from measured LJ response.',
                    new_training=False, new_model_inference=False, final_test_used=False,
                    seconds=time.monotonic()-start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved', len(table), 'networks', flush=True)


if __name__ == '__main__':
    main()
