#!/usr/bin/env python3
"""Independent read-only periodic spring-contact audit for planar study_v1."""
from __future__ import annotations
import argparse, csv, hashlib, json, multiprocessing as mp
from itertools import product
from pathlib import Path
from time import time
import numpy as np
import torch

ROOT = Path('notebooks/results/planar_engineering/study_v1')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def cross(a,b): return a[:,0]*b[:,1]-a[:,1]*b[:,0]
def edges(index): return np.unique(np.sort(np.asarray(index,dtype=np.int64),axis=0).T,axis=0)
def point_distance(point,start,vector):
    q=np.clip(np.einsum('ij,ij->i',point-start,vector)/np.einsum('ij,ij->i',vector,vector),0.,1.)
    return np.linalg.norm(point-start-q[:,None]*vector,axis=1)

def candidates(low,high,shift,tol):
    """Sweep AABBs; emit only verifier-order i<j candidate pairs."""
    lo,hi=low+shift,high+shift; order=np.argsort(lo[:,0],kind='stable'); sx=lo[order,0]; aa=[];bb=[]
    for i in range(len(low)):
        js=order[:np.searchsorted(sx,high[i,0]+tol,side='right')]
        keep=(js>i)&(hi[js,0]+tol>=low[i,0])&(lo[js,1]<=high[i,1]+tol)&(hi[js,1]+tol>=low[i,1])
        if keep.any(): aa.append(np.full(keep.sum(),i,dtype=np.int64));bb.append(js[keep])
    return (np.concatenate(aa),np.concatenate(bb)) if aa else (np.empty(0,dtype=np.int64),np.empty(0,dtype=np.int64))

def narrow(p,v,e,a,b,shift,tol):
    """Signed-orientation narrow phase; no PlanarConstraints helpers used."""
    if not len(a): return []
    q=p[b]+shift; va,vb=v[a],v[b]; la,lb=np.linalg.norm(va,axis=1),np.linalg.norm(vb,axis=1); den=cross(va,vb); d=q-p[a]
    par=np.abs(den)<=tol*np.maximum(la,lb); t=np.zeros(len(a));u=np.zeros(len(a)); non=~par
    t[non]=cross(d[non],vb[non])/den[non];u[non]=cross(d[non],va[non])/den[non]; et,eu=tol/la,tol/lb
    hit=non&(t>=-et)&(t<=1+et)&(u>=-eu)&(u<=1+eu)
    proper=hit&(t>et)&(t<1-et)&(u>eu)&(u<1-eu)
    interior=hit&(((t>et)&(t<1-et))|((u>eu)&(u<1-eu)))
    col=par&(np.abs(cross(d,va))<=tol*la); start=np.einsum('ij,ij->i',d,va)/la; end=start+np.einsum('ij,ij->i',vb,va)/la
    overlap=np.minimum(la,np.maximum(start,end))-np.maximum(0.,np.minimum(start,end)); colover=col&(overlap>tol)
    dist=np.full(len(a),np.inf); unresolved=~hit&~colover
    if unresolved.any():
        k=np.flatnonzero(unresolved);dist[k]=np.minimum.reduce((point_distance(p[a[k]],q[k],vb[k]),point_distance(p[a[k]]+va[k],q[k],vb[k]),point_distance(q[k],p[a[k]],va[k]),point_distance(q[k]+vb[k],p[a[k]],va[k])))
    shared=np.any(e[a,:,None]==e[b,None,:],axis=(1,2)); flagged=np.where(shared,interior|colover,hit|colover|(dist<=tol))
    return [(int(a[i]),int(b[i]),bool(shared[i]),bool(proper[i]),bool(colover[i]),bool(np.any(shift!=0))) for i in np.flatnonzero(flagged)]

def audit_frame(position,bounds,e):
    bounds=np.asarray(bounds)[:2,:2]; box=bounds[:,1]-bounds[:,0]; x=bounds[:,0]+np.mod(np.asarray(position)[:,:2]-bounds[:,0],box)
    wind=np.round((x[e[:,1]]-x[e[:,0]])/box);p=x[e[:,0]];v=x[e[:,1]]-p-wind*box; tol=1e-8*float(np.median(np.linalg.norm(v,axis=1))); lo,hi=np.minimum(p,p+v),np.maximum(p,p+v); events=[]
    for image in product((-1,0,1),repeat=2):
        shift=np.asarray(image,dtype=float)*box;a,b=candidates(lo,hi,shift,tol);events.extend(narrow(p,v,e,a,b,shift,tol))
    pairs={}
    for event in events:pairs.setdefault(event[:2],[]).append(event)
    values=list(pairs.values())
    return dict(pair_count=len(values),proper_nonshared=sum(any(not z[2] and z[3] for z in y) for y in values),endpoint_nonshared=sum(any(not z[2] and not z[3] and not z[4] for z in y) for y in values),collinear_overlap=sum(any(z[4] for z in y) for y in values),shared_joint=sum(any(z[2] for z in y) for y in values),periodic_nonshared=sum(any(not z[2] and z[5] for z in y) for y in values),periodic_image_only=sum(all(z[5] for z in y) for y in values),tolerance=tol)

def audit(task):
    case,row=task; track=np.load(row['trajectory_path']);graph=torch.load(row['graph_path'],weights_only=False,map_location='cpu'); e=edges(graph.edge_index.numpy()); frames=[audit_frame(x,b,e) for x,b in zip(track['positions'],track['boxes'])]; old=json.loads(row['contact_pairs_by_frame']); counts=[x['pair_count'] for x in frames]; hit=[i for i,x in enumerate(counts) if x]
    categories=('proper_nonshared','endpoint_nonshared','collinear_overlap','shared_joint','periodic_nonshared','periodic_image_only')
    return dict(case=case,source=row['source'],index=int(row['index']),kind=row['kind'],thermal_seed=int(row['thermal_seed']),graph_path=row['graph_path'],graph_sha256=sha256(row['graph_path']),trajectory_path=row['trajectory_path'],trajectory_sha256=sha256(row['trajectory_path']),edge_count=len(e),frames=len(frames),old_contact_pairs_by_frame=old,independent_contact_pairs_by_frame=counts,old_count_parity=(old==counts),initial_pairs=counts[0],first_flagged_frame=(hit[0] if hit else None),flagged_frames=len(hit),max_pairs=max(counts),category_frames={k:sum(x[k]>0 for x in frames) for k in categories},category_pair_occurrences={k:int(sum(x[k] for x in frames)) for k in categories})

def main(root,output,workers):
    start=time();tasks=[]
    for path in sorted((root/'cases').glob('*/final_verification/results.csv')):
        with path.open() as f:
            tasks += [(path.parents[1].name,row) for row in csv.DictReader(f) if row['status']=='completed']
    with mp.Pool(workers) as pool: rows=list(pool.imap_unordered(audit,tasks))
    rows.sort(key=lambda x:(x['case'],x['kind'],x['thermal_seed'])); output.mkdir(parents=True,exist_ok=True); elapsed=time()-start
    cases={case:{kind:[x for x in rows if x['case']==case and x['kind']==kind] for kind in ('original','optimized')} for case in sorted({x['case'] for x in rows})}
    provenance=dict(script_sha256=sha256(__file__),verifier_sha256=sha256(root/'verification_code/verify.py'),planarity_sha256=sha256('src/lss/engineering/planarity.py'),source='saved final-verification trajectories; no physics rerun',workers=workers,expected_cases=40,completed_cases=38,total_frames=sum(x['frames'] for x in rows),started_at=start,completed_at=time(),elapsed_seconds=elapsed)
    (output/'full_trajectory_audit.json').write_text(json.dumps(dict(provenance=provenance,trajectories=rows,cases=cases),indent=2)+'\n')
    summary=[dict(case=x['case'],kind=x['kind'],seed=x['thermal_seed'],frames=x['frames'],initial_pairs=x['initial_pairs'],flagged_frames=x['flagged_frames'],first_flagged_frame=x['first_flagged_frame'],max_pairs=x['max_pairs'],old_count_parity=x['old_count_parity'],**x['category_frames']) for x in rows]
    with (output/'full_trajectory_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    print(json.dumps(dict(trajectories=len(rows),frames=provenance['total_frames'],parity=sum(x['old_count_parity'] for x in rows),elapsed_seconds=elapsed),indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ROOT/'flag_audit/full');p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.root,a.output,a.workers)
