#!/usr/bin/env python3
"""Read-only audit of unambiguous interior spring crossings in saved designs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'notebooks/results/engineering_geometry_audit'


def crossing_pairs(graph):
    pos=graph.x[:, :2].cpu().numpy().astype(float)
    edges=np.unique(np.sort(graph.edge_index.cpu().numpy(),axis=0),axis=1).T
    box=np.array([graph.box.x,graph.box.y])
    vector=pos[edges[:,1]]-pos[edges[:,0]]
    # Strictly interior, non-wrapping edges only: this cannot be a periodic drawing artifact.
    keep=np.all(np.abs(vector)<box/2,axis=1)
    edges=edges[keep]
    p=pos[edges[:,0]];v=pos[edges[:,1]]-p
    i,j=np.triu_indices(len(edges),1)
    independent=np.all(edges[i,:,None]!=edges[j,None,:],axis=(1,2))
    i,j=i[independent],j[independent]
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    denom=cross(v[i],v[j]);q=p[j]-p[i]
    nonparallel=np.abs(denom)>1e-12*np.linalg.norm(v[i],axis=1)*np.linalg.norm(v[j],axis=1)
    i,j,q,denom=i[nonparallel],j[nonparallel],q[nonparallel],denom[nonparallel]
    t=cross(q,v[j])/denom;u=cross(q,v[i])/denom
    hit=(t>1e-8)&(t<1-1e-8)&(u>1e-8)&(u<1-1e-8)
    return [(tuple(map(int,edges[a])),tuple(map(int,edges[b]))) for a,b in zip(i[hit],j[hit])]


def main():
    results=pd.read_csv(ROOT/'notebooks/results/reid_joint_sweep/study_v1/results.csv')
    selected=results[results.config_id.eq('lr0.08_b0.25_p50_mask_blocking_nodes')&results.kind.eq('ml')]
    original=results[results.kind.eq('original')].set_index('index')
    assert len(selected)==10
    rows=[];details=[]
    for row in selected.itertuples():
        original_path=Path(original.loc[row.index,'physicalpath']);optimized_path=Path(row.physicalpath)
        pairs={}
        for kind,path in [('original',original_path),('optimized',optimized_path)]:
            graph=torch.load(path,map_location='cpu',weights_only=False)
            pairs[kind]=set(crossing_pairs(graph))
            details.append(dict(source='reid',index=int(row.index),kind=kind,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crossing_edge_pairs=sorted(pairs[kind])))
        rows.append(dict(source='reid',index=int(row.index),original_crossings=len(pairs['original']),optimized_crossings=len(pairs['optimized']),new_crossing_pairs=len(pairs['optimized']-pairs['original']),physical_p_ratio=row.physical_p_ratio))
    table=pd.DataFrame(rows).sort_values('index')
    table.to_csv(OUT/'reid_showcase_crossings.csv',index=False)
    (OUT/'details.json').write_text(json.dumps(dict(method='Proper intersections of nonincident, non-wrapping straight spring segments. Common-node joints excluded. This is a conservative interior-only crossing count; does not count collinear overlaps, near contacts or wrapping edges.',runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),designs=details,simulation_performed=False,optimization_performed=False),indent=2)+'\n')
    print(table.to_string(index=False))


if __name__=='__main__':main()
