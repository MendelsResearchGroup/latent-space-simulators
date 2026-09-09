#!/usr/bin/env python3
"""Frozen AE gradient engineering; optimization and final physics are separate jobs."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

DONOR = Path('/rg/mendels_prj/alexander.z/DL-course-project')
CODE = DONOR / 'notebooks/results/network_design/code_endpoint_v1'
PREP = Path(__file__).resolve().parent.parent / 'prepared'
sys.path[:0] = [str(DONOR.parent / 'MetaForge/src'), str(CODE / 'src'), str(CODE / 'notebooks/network_design')]
import auxetic
import numpy as np
import pandas as pd
import torch
from design import encode_initial, readout_value, physical_candidate

NETWORKS = {'reid': [57, 15], 'depablo_low_temp': [105, 152]}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def load(source):
    data = torch.load(PREP / source / 'prepare/prepared.pt', weights_only=False, map_location='cpu')
    data['ae'].eval()
    for p in data['ae'].parameters(): p.requires_grad_(False)
    return data

def optimize(out):
    # This process cannot spawn a simulator (or any other child process).
    def prohibit(event, args):
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec', 'os.fork'):
            raise RuntimeError('External process forbidden during ML optimization: ' + event)
    sys.addaudithook(prohibit)
    torch.set_num_threads(4)
    torch.manual_seed(786)
    np.random.seed(786)
    recipe = dict(stage='optimize', status='running', job_id=os.environ.get('PBS_JOBID'),
        script_sha256=sha(__file__), frozen_source_manifest_sha256=sha(CODE/'hashes.json'),
        prepared_sha256={s: sha(PREP/s/'prepare/prepared.pt') for s in NETWORKS},
        preparation={s: json.loads((PREP/s/'prepare/preparation.json').read_text()) for s in NETWORKS},
        networks=NETWORKS, model_seed=786, policies=['z1', 'ridge2d'], bounds=[2.0],
        gradient_steps=150, learning_rate=0.05, target=None,
        objective='minimize signed frozen latent readout; centered max-normalized projected gradient; no predicted-response stopping or latent-distance penalty',
        selection='minimum ML objective over initial and 150 iterates; no physics or target-network response consulted',
        calibration='existing source affine readouts fitted on original calibration responses; do not load simulator-selected policy/sign',
        starts='pristine originals from prepared bundle; no engineered starts',
        split='four previously unengineered AE validation networks, disjoint from calibration and earlier designs; not final test',
        counts=dict(ml_designs=8, random_controls=4, originals=4),
        controls='one permutation of signed centered edits per source/index/bound, based on ridge2d edit; preserves norm, mean, bound; seed123+index',
        verification='separate job only after entire design manifest; 120 quasistatic increments per frozen graph; all finals reported',
        external_processes_allowed=False, centered_log_edits=True,
        hypothesis='transfer the exact frozen v3 optimization protocol to predeclared previously unengineered validation networks without tuning')
    dump(out/'recipe.json', recipe)
    records=[]
    for source, indices in NETWORKS.items():
        data=load(source)
        for index in indices:
            ref=data['references'][index]; raw=data['physical'][index]
            k0=ref.edge_attr[:,3].detach().clone(); active=k0>0
            assert active.any() and not (k0<0).any()
            z0=encode_initial(data['ae'],ref,k0,data['normalizers']).detach()
            folder=out/f'{source}_{index}'; folder.mkdir(exist_ok=True)
            def save(label, u, extra):
                graph=physical_candidate(raw,ref,k0*u.exp())
                if hasattr(graph,'registry_poisson_ratio'): graph.registry_poisson_ratio=None
                assert torch.equal(graph.x,raw.x) and torch.equal(graph.edge_index,raw.edge_index)
                assert torch.equal(graph.edge_attr[:,3]==0,raw.edge_attr[:,3]==0)
                assert (graph.edge_attr[raw.edge_attr[:,3]>0,3]>0).all()
                path=folder/f'{label}.pt'; torch.save(graph,path)
                editpath=folder/f'{label}_edit.pt'; torch.save(u,editpath)
                records.append(dict(source=source,index=index,variant=label,file=str(path.resolve()),
                    sha256=sha(path),edit_file=str(editpath.resolve()),edit_sha256=sha(editpath),
                    scale_x=float(data['scales'][index]),max_abs_log_edit=float(u.abs().max()),**extra))
            save('original',torch.zeros_like(k0),dict(kind='original'))
            for bound in (2.0,):
                for policy in ('z1','ridge2d'):
                    r=data['policies'][policy]
                    probe=torch.zeros_like(k0,requires_grad=True)
                    initial=readout_value(encode_initial(data['ae'],ref,k0*probe.exp(),data['normalizers']),r)
                    gradient=torch.autograd.grad(initial,probe)[0]
                    direction=gradient.detach()*active
                    direction=direction/direction.abs().max().clamp_min(1e-12)
                    with torch.no_grad():
                        plus=readout_value(encode_initial(data['ae'],ref,k0*(.001*direction).exp(),data['normalizers']),r)
                        minus=readout_value(encode_initial(data['ae'],ref,k0*(-.001*direction).exp(),data['normalizers']),r)
                    fd=float((plus-minus)/.002); exact=float((gradient*direction).sum())
                    relative=abs(fd-exact)/max(abs(fd),abs(exact),1e-8)
                    dump(folder/f'{policy}_b{bound:g}_gradient_check.json',dict(autograd=exact,finite_difference=fd,relative_error=relative))
                    if relative>.1: raise RuntimeError('Initial gradient check failed')
                    u=torch.nn.Parameter(torch.zeros_like(k0)); opt=torch.optim.SGD([u],lr=.05)
                    best=float('inf'); best_u=None; best_info=None; history=[]
                    for step in range(151):
                        z=encode_initial(data['ae'],ref,k0*u.exp(),data['normalizers'])
                        pred=readout_value(z,r)
                        distance=((z-z0)/r['scale']).norm()
                        loss=pred
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite ML objective')
                        row=dict(step=step,loss=float(loss.detach()),predicted_p_ratio=float(pred.detach()),latent_distance=float(distance.detach()))
                        history.append(row)
                        if row['loss']<best:
                            best=row['loss']; best_u=u.detach().clone(); best_info=row.copy()
                        if step==150: break
                        opt.zero_grad();loss.backward()
                        if not torch.isfinite(u.grad).all(): raise RuntimeError('Nonfinite gradient')
                        # Remove global stiffness scaling from the gradient, then project
                        # the update onto sum(u_active)=0 intersect the box.
                        u.grad[active]-=u.grad[active].mean(); u.grad[~active]=0
                        u.grad.div_(u.grad.abs().max().clamp_min(1e-12))
                        opt.step()
                        with torch.no_grad():
                            values=u[active].clone(); lower=float(values.min())-bound; upper=float(values.max())+bound
                            for _ in range(50):
                                shift=(lower+upper)/2
                                if float((values-shift).clamp(-bound,bound).mean())>0: lower=shift
                                else: upper=shift
                            u[active]=(values-(lower+upper)/2).clamp(-bound,bound); u[~active]=0
                            if abs(float(u[active].mean()))>1e-6: raise RuntimeError('Centered projection failed')
                    label=f'{policy}_b{bound:g}'
                    pd.DataFrame(history).to_csv(folder/f'{label}_history.csv',index=False)
                    save(label,best_u,dict(kind='ml',policy=policy,bound=bound,**best_info))
                    print(source,index,label,best_info,flush=True)
                    if policy=='ridge2d':
                        ids=active.nonzero().flatten(); generator=torch.Generator().manual_seed(123+index)
                        random=torch.zeros_like(best_u)
                        random[ids]=best_u[ids][torch.randperm(len(ids),generator=generator)]
                        save(f'random_b{bound:g}',random,dict(kind='random',bound=bound,seed=123+index))
    assert len(records)==16 and sum(r['kind']=='ml' for r in records)==8
    recipe.update(status='completed',max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(out/'recipe.json',recipe)
    dump(out/'frozen_designs.json',dict(recipe_sha256=sha(out/'recipe.json'),frozen_at_unix=time.time(),records=records))

def verify(out):
    torch.set_num_threads(1)
    # Check ALL files before allowing the first physics call.
    manifest=json.loads((out/'frozen_designs.json').read_text())
    assert manifest['recipe_sha256']==sha(out/'recipe.json')
    for r in manifest['records']:
        assert sha(r['file'])==r['sha256'] and sha(r['edit_file'])==r['edit_sha256']
    import importlib.util
    helper = DONOR/'notebooks/network_design/endpoint_physics.py'
    spec=importlib.util.spec_from_file_location('verified_endpoint_physics',helper)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    validate_endpoints=module.validate_endpoints
    lammps=DONOR/'.runtime/lammps/bin/lmp'
    dump(out/'verification_recipe.json',dict(job_id=os.environ.get('PBS_JOBID'),manifest_sha256=sha(out/'frozen_designs.json'),
        script_sha256=sha(__file__),endpoint_helper_sha256=sha(helper),
        lammps_sha256=sha(lammps),steps=120,feedback_to_optimizer=False))
    rows=[]
    for r in manifest['records']:
        graph=torch.load(r['file'],weights_only=False,map_location='cpu')
        result=validate_endpoints({(r['index'],r['variant']):graph},out/'physics'/r['source'],lammps,
            {r['index']:r['scale_x']},compression_steps=120).iloc[0].to_dict()
        rows.append({**r,**result})
        pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    table=pd.DataFrame(rows)
    baseline=table[table.kind=='original'].set_index(['source','index']).physical_p_ratio
    table['original_p_ratio']=[baseline.loc[(r.source,r['index'])] for _,r in table.iterrows()]
    table['change']=table.physical_p_ratio-table.original_p_ratio
    table.to_csv(out/'results.csv',index=False)
    summary=[]
    for keys,group in table[table.kind=='ml'].groupby(['source','policy','bound']):
        valid=group.status.eq('completed') & np.isfinite(group.physical_p_ratio) & np.isfinite(group.original_p_ratio)
        g=group[valid]
        summary.append(dict(source=keys[0],policy=keys[1],bound=float(keys[2]),valid=int(valid.sum()),total=len(group),
            improved=int((g.change < -1e-5).sum()),negative=int((g.physical_p_ratio<0).sum()),target=int((g.physical_p_ratio<=-.1).sum()),
            mean_change=float(g.change.mean()),mean_physical_p=float(g.physical_p_ratio.mean())))
    dump(out/'summary.json',clean(summary))
    dump(out/'verification_completed.json',dict(valid=int(table.status.eq('completed').sum()),total=len(table),
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,manifest_sha256=sha(out/'frozen_designs.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['optimize','verify'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='optimize':
        if (args.output/'recipe.json').exists(): raise RuntimeError('Refuse to overwrite optimization study')
        optimize(args.output)
    else: verify(args.output)
