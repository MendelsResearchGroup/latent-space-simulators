#!/usr/bin/env python3
"""Frozen-2D-AE high-p design study; optimization and physics are isolated."""
import argparse, copy, hashlib, importlib.util, json, os, resource, sys, time
from pathlib import Path
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project'); CODE=DONOR/'notebooks/results/network_design/code_endpoint_v1'; HELPER=DONOR/'notebooks/network_design/endpoint_physics.py'
sys.path[:0]=[str(DONOR.parent/'MetaForge/src'),str(CODE/'src'),str(CODE/'notebooks/network_design')]
import auxetic # noqa
import numpy as np, pandas as pd, torch
from graph_utils import directional_side_indices_from_box
from lss.dynamics.training import encode_frame_latent
from design import readout_value, encode_initial

EPS=1e-8
RADIUS=2.0
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list): return [clean(v) for v in x]
    return None if isinstance(x,float) and not np.isfinite(x) else x
def project_u_(u,a,b):
    v=u[a].clone(); lo=float(v.min())-b; hi=float(v.max())+b
    for _ in range(60):
        c=(lo+hi)/2
        if float((v-c).clamp(-b,b).mean())>0: lo=c
        else: hi=c
    u[a]=(v-(lo+hi)/2).clamp(-b,b);u[~a]=0
def geometry(reference,raw,u,q):
    """Build differentiable model graph and physical export graph from zero-based edits."""
    e=reference.edge_index.long(); k0=reference.edge_attr[:,3]; x0=reference.reference_context_positions[:,:2]
    L=reference.reference_context_edge_attr[:,2]; local=torch.stack([L[(e==i).any(0)].min() for i in range(len(x0))])
    box=torch.tensor([raw.box.x,raw.box.y],dtype=x0.dtype); center=torch.tensor([(raw.box.x1+raw.box.x2)/2,(raw.box.y1+raw.box.y2)/2],dtype=x0.dtype)
    x=x0+q*(.25*local[:,None]); k=k0*u.exp();
    def edges(pos,bb):
        v=pos[e[1]]-pos[e[0]];v=v-torch.round(v/bb)*bb
        return torch.cat([v,v.norm(dim=1,keepdim=True),k[:,None]],1)
    g=reference.clone(); n=(x-center)/(box/2); g.x=torch.cat([n,reference.x[:,2:]],1);g.pos=n;g.edge_attr=edges(n,torch.tensor([2.,2.]));g.reference_context_positions=x;g.reference_context_edge_attr=edges(x,box)
    return g,x,k,g.reference_context_edge_attr[:,2],box
def valid(reference,raw,u,q,sides):
    g,x,k,L,box=geometry(reference,raw,u,q); e=reference.edge_index.long(); old=reference.reference_context_edge_attr[:,2]
    if not torch.isfinite(x).all() or (L<.25*old).any() or (L>1.75*old).any(): return False
    pair=torch.triu_indices(len(x),len(x),1);d=x[pair[1]]-x[pair[0]];d-=torch.round(d/box)*box
    d0=reference.reference_context_positions[:,:2];d0=d0[pair[1]]-d0[pair[0]];d0-=torch.round(d0/box)*box
    if d.norm(dim=1).min()<.1*d0.norm(dim=1).min(): return False
    probe=copy.deepcopy(raw);probe.x=x.detach()
    now=directional_side_indices_from_box(probe,quantile=.1)
    if any(not np.array_equal(np.sort(now[k]),np.sort(sides[k])) for k in sides): return False
    return True
def blocking_nodes(ref,raw,u,q):
    _,x,_,L,box=geometry(ref,raw,u,q); old=ref.reference_context_edge_attr[:,2];e=ref.edge_index.long()
    blocked=torch.zeros(len(x),dtype=torch.bool);bad=(L<.25*old)|(L>1.75*old);blocked[e[:,bad].flatten()]=True
    pair=torch.triu_indices(len(x),len(x),1);d=x[pair[1]]-x[pair[0]];d-=torch.round(d/box)*box
    x0=ref.reference_context_positions[:,:2];d0=x0[pair[1]]-x0[pair[0]];d0-=torch.round(d0/box)*box
    badpair=d.norm(dim=1)<.1*d0.norm(dim=1).min();blocked[pair[:,badpair].flatten()]=True
    return blocked
def score(data,ref,raw,u,q):
    g,_,_,_,_=geometry(ref,raw,u,q)
    z=encode_frame_latent(data['ae'],[g],0,pos_dim=2,node_feature_mode='normalized_delta',normalizers=data['normalizers'],device='cpu')
    return readout_value(z,data['policies']['z1'])
def constraints(ref,raw):
    x=ref.reference_context_positions[:,:2]; sides=directional_side_indices_from_box(raw,quantile=.1); fixed=torch.zeros(len(x),dtype=torch.bool)
    for v in sides.values(): fixed[torch.as_tensor(v)]=True
    eps=1e-5*float(ref.reference_context_edge_attr[:,2].median()); lo=torch.stack([x[sides['left'],0].max()+eps,x[sides['bottom'],1].max()+eps]);hi=torch.stack([x[sides['right'],0].min()-eps,x[sides['top'],1].min()-eps])
    return sides,fixed,lo,hi
def project_q_(q,x,scale,fixed,lo,hi):
    q.mul_((RADIUS/q.norm(dim=1,keepdim=True).clamp_min(EPS)).clamp_max(1)); xx=(x+q*scale[:,None]).maximum(lo).minimum(hi);q[:]=(xx-x)/scale[:,None];q[fixed]=0
def check(data,ref,raw,u,q,active,movable,sides,index):
    """Probe each editable gradient block without random-direction cancellation."""
    checks=[]
    v=score(data,ref,raw,u.requires_grad_(True),q.requires_grad_(True))
    gu,gq=torch.autograd.grad(v,(u,q))
    for block in ('spring','position'):
        du=torch.zeros_like(u);dq=torch.zeros_like(q)
        if block=='spring':
            du=gu.detach().clone();du[~active]=0;du[active]-=du[active].mean();du/=du.abs().max().clamp_min(EPS)
        else:
            dq=gq.detach().clone();dq[~movable]=0;dq/=dq.abs().max().clamp_min(EPS)
        exact=float((gu*du).sum()+(gq*dq).sum())
        with torch.no_grad():
            fd=float((score(data,ref,raw,u.detach()+.01*du,q.detach()+.01*dq)-score(data,ref,raw,u.detach()-.01*du,q.detach()-.01*dq))/.02)
        rel=abs(fd-exact)/max(abs(fd),abs(exact),EPS)
        checks.append(dict(index=index,block=block,autograd=exact,finite_difference=fd,relative_error=rel,probe='projected block gradient direction',epsilon=.01))
    return checks
def provenance_record(manifest, index, variant):
    rows=[r for r in json.loads(Path(manifest).read_text())['records'] if r['index']==index and r['variant']==variant]
    if len(rows)!=1: raise RuntimeError(f'missing unique prior record {index} {variant}')
    return rows[0]

def assert_reuse_reference(ref, raw, index):
    old=Path('notebooks/results/high_pratio_engineering/study_v1/prepared/reid/prepare/prepared.pt')
    prior=torch.load(old,weights_only=False,map_location='cpu')['references'][index]
    for key in ['x','edge_index','edge_attr','reference_context_positions','reference_context_edge_attr','reference_box_center','reference_box_half_extent']:
        a,b=getattr(ref,key),getattr(prior,key)
        if a.shape != b.shape or not torch.equal(a,b): raise RuntimeError(f'pristine reference parity failed for reused {index} {key}')
    old_raw=torch.load(DONOR/'data/reid_200_frames.pt',weights_only=False,map_location='cpu',mmap=True)[index][0]
    if not torch.equal(raw.x,old_raw.x) or not torch.equal(raw.edge_attr,old_raw.edge_attr):
        raise RuntimeError(f'raw pristine parity failed for reused {index}')

def optimize(out,source):
    global RADIUS
    sys.addaudithook(lambda e,a: (_ for _ in ()).throw(RuntimeError('external process forbidden')) if e in {'subprocess.Popen','os.system','os.posix_spawn','os.exec','os.fork'} else None)
    torch.set_num_threads(4);torch.manual_seed(786);np.random.seed(786)
    out.mkdir(parents=True,exist_ok=True)
    p=out/'prepared'/'reid'/'prepare'/'prepared.pt';data=torch.load(p,weights_only=False,map_location='cpu');data['ae'].eval()
    for z in data['ae'].parameters():z.requires_grad_(False)
    prep=json.loads((p.parent/'preparation.json').read_text()); ids=prep['cohort_ids_in_validation_order']
    if ids!=[93,84,96,97,52,59,57,53,2,74]: raise ValueError('cohort order mismatch')
    recipe=dict(stage='optimize',status='running',source=source,source_ids=ids,prepared_sha256=sha(p),script_sha256=sha(__file__),code_manifest_sha256=sha(CODE/'hashes.json'),cohort_sha256=prep['cohort_sha256'],cohort_path=prep['cohort_path'],model_seed=786,policy='z1',arms=['positions_r4_b0','joint_r4_b0.25'],steps=1200,step_sizes=[.05,.025,.0125,.00625],finite_difference_epsilon=.01,selection='lowest frozen z1 readout among valid ML-only iterates',counts=dict(ml=20,random=10,original=10,total=40),job_id=os.environ.get("PBS_JOBID"),split=prep['split'],geometry_constraints="fixed box/topology/boundary nodes; radius4 equals <=1.0 shortest original incident bond; lengths 0.25..1.75 original; pair separation >=0.1 original minimum",gradient_scaling="center stiffness; normalize spring and geometry blocks separately",preparation=prep,no_response_targets=True,no_external_processes=True,feasible_search='up to16 node-mask retries per iteration; mask endpoints of invalid bonds/close pairs from smallest invalid backtrack; reset mask next iteration',control='joint-only signed centered stiffness permutation and movable q permutation, seed123+ID; geometry-only q halving, no norm matching or score selection',reuse='93 positions_r4_b0 from high_pratio_engineering/study_v3 and joint_r4_b0.25 from joint_v1 are completed frozen outputs, never starts')
    dump(out/'recipe.json',recipe); records=[];checks=[]
    for index in ids:
        ref,raw=data['references'][index],data['physical'][index];k0=ref.edge_attr[:,3].detach();active=k0>0;sides,fixed,lo,hi=constraints(ref,raw);x0=ref.reference_context_positions[:,:2].detach(); L=ref.reference_context_edge_attr[:,2].detach();e=ref.edge_index.long();scale=.25*torch.stack([L[(e==i).any(0)].min() for i in range(len(x0))]);movable=(~fixed)[:,None].expand_as(x0)
        folder=out/f'{source}_{index}';folder.mkdir(exist_ok=True)
        def save(name,u,q,kind,mode,bound,pred,extra={}):
            g,x,k,_,box=geometry(ref,raw,u,q); physical=copy.deepcopy(raw);physical.x=x.detach().to(raw.x);physical.pos=physical.x.clone(); lookup={tuple(sorted(z)):i for i,z in enumerate(e.T.tolist())}; order=torch.tensor([lookup[tuple(sorted(z))] for z in raw.edge_index.T.tolist()]);v=physical.x[physical.edge_index[1]]-physical.x[physical.edge_index[0]];v-=torch.round(v/box)*box;physical.edge_attr=torch.cat([v,v.norm(dim=1,keepdim=True),k.detach()[order,None].to(v)],1)
            if hasattr(physical,'registry_poisson_ratio'): physical.registry_poisson_ratio=None
            assert valid(ref,raw,u,q,sides)
            assert torch.equal(physical.edge_index,raw.edge_index)
            assert torch.equal(physical.edge_attr[:,3]==0,raw.edge_attr[:,3]==0)
            assert (physical.edge_attr[raw.edge_attr[:,3]>0,3]>0).all()
            assert torch.equal(physical.x[fixed],raw.x[fixed])
            assert float((x-x0).norm(dim=1).div(scale).max())<=RADIUS+0.0001
            assert abs(float(u[active].mean()))<2e-6
            if mode in ('stiffness','original'): assert torch.equal(physical.x,raw.x)
            if mode=='positions': assert torch.equal(physical.edge_attr[:,3],raw.edge_attr[:,3])
            gp,ep=folder/f'{name}.pt',folder/f'{name}_edit.pt';torch.save(physical,gp);torch.save(dict(log_stiffness=u.detach(),position_q=q.detach()),ep);records.append(dict(source=source,index=index,variant=name,kind=kind,policy='z1',mode=mode,bound=bound,model_seed=786,predicted_p=float(pred),scale_x=float(data['scales'][index]),physicalpath=str(gp.resolve()),sha256=sha(gp),edit_file=str(ep.resolve()),edit_sha256=sha(ep),**extra))
        def reuse(name, prior_manifest, prior_folder, mode, bound):
            assert_reuse_reference(ref,raw,index)
            prior=provenance_record(prior_manifest,index,name)
            gp,ep=Path(prior['physicalpath']),Path(prior['edit_file'])
            if sha(gp)!=prior['sha256'] or sha(ep)!=prior['edit_sha256'] or json.loads(Path(prior_manifest).read_text())['recipe_sha256']!=sha(Path(prior_manifest).parent/'recipe.json'): raise RuntimeError(f'prior artifact or recipe hash mismatch {name}')
            old_graph=torch.load(gp,weights_only=False,map_location='cpu'); edit=torch.load(ep,weights_only=False,map_location='cpu')
            u,q=edit['log_stiffness'].detach(),edit['position_q'].detach()
            if not valid(ref,raw,u,q,sides): raise RuntimeError(f'reused edit invalid {name}')
            _,x,k,_,box=geometry(ref,raw,u,q); e=ref.edge_index.long(); lookup={tuple(sorted(z)):i for i,z in enumerate(e.T.tolist())}; order=torch.tensor([lookup[tuple(sorted(z))] for z in raw.edge_index.T.tolist()]); v=x[raw.edge_index[1]]-x[raw.edge_index[0]];v-=torch.round(v/box)*box
            expected_edge=torch.cat([v,v.norm(dim=1,keepdim=True),k[order,None].to(v)],1)
            if not torch.equal(old_graph.x, x.to(old_graph.x)) or not torch.equal(old_graph.edge_index,raw.edge_index) or not torch.equal(old_graph.edge_attr,expected_edge.to(old_graph.edge_attr)) or any(getattr(old_graph.box,k)!=getattr(raw.box,k) for k in ('x1','x2','y1','y2','z1','z2')):
                raise RuntimeError(f'reused design does not exactly match pristine reconstruction {name}')
            prior_history=Path(prior_folder)/f'{name}_history.csv'
            if name!='original' and not prior_history.is_file(): raise RuntimeError(f'missing prior history {name}')
            pred=float(score(data,ref,raw,u,q));
            if abs(pred-float(prior['predicted_p']))>1e-6: raise RuntimeError(f'reused model score mismatch {name}')
            records.append(dict(source=source,index=index,variant=name,kind=prior['kind'],policy='z1',mode=mode,bound=bound,model_seed=786,predicted_p=pred,scale_x=float(data['scales'][index]),physicalpath=str(gp.resolve()),sha256=sha(gp),edit_file=str(ep.resolve()),edit_sha256=sha(ep),position_radius=4.,reused_optimizer_output=True,reused_physical_measurement=True,prior_manifest=str(Path(prior_manifest).resolve()),prior_manifest_sha256=sha(prior_manifest),prior_recipe_sha256=sha(Path(prior_manifest).parent/'recipe.json'),prior_results_csv=str((Path(prior_manifest).parent.parent/'results.csv').resolve()),prior_results_csv_sha256=sha(Path(prior_manifest).parent.parent/'results.csv'),prior_history=None if name=='original' else str(prior_history.resolve()),prior_history_sha256=None if name=='original' else sha(prior_history)))
            return u,q
        u0=torch.zeros_like(k0);q0=torch.zeros_like(x0)
        if not valid(ref,raw,u0,q0,sides): raise RuntimeError('Pristine geometry invalid')
        base=readout_value(encode_initial(data['ae'],ref,k0,data['normalizers']),data['policies']['z1'])
        zero=score(data,ref,raw,u0,q0)
        dump(folder/'zero_edit_parity.json',dict(original_model_score=float(base.detach()),rebuilt_model_score=float(zero.detach()),error=abs(float(base.detach()-zero.detach()))))
        if abs(float(base.detach()-zero.detach()))>1e-4: raise RuntimeError('Zero-edit encoder parity failed')
        if index==93:
            reuse('original','notebooks/results/high_pratio_engineering/study_v3/reid/frozen_designs.json','notebooks/results/high_pratio_engineering/study_v3/reid/reid_93','original',0)
        else: save('original',u0,q0,'original','original',0,score(data,ref,raw,u0,q0))
        local_checks=check(data,ref,raw,u0,q0,active,movable,sides,index); checks+=local_checks
        dump(out/'gradient_checks.json',checks)
        if any(c['relative_error']>.1 or not np.isfinite(c['relative_error']) for c in local_checks): raise RuntimeError('Initial finite-difference check failed')
        for mode,bound,radius in [('positions',0.,4.),('joint',.25,4.)]:
            RADIUS=radius
            if index==93:
                previous=('notebooks/results/high_pratio_engineering/study_v3/reid/frozen_designs.json','notebooks/results/high_pratio_engineering/study_v3/reid/reid_93') if mode=='positions' else ('notebooks/results/high_pratio_engineering/joint_v1/reid/frozen_designs.json','notebooks/results/high_pratio_engineering/joint_v1/reid/reid_93')
                bu,bq=reuse(f'{mode}_r4_b{bound:g}',previous[0],previous[1],mode,bound)
                if mode=='joint':
                    gen=torch.Generator().manual_seed(123+index);rq=torch.zeros_like(bq);ii=(~fixed).nonzero().flatten();rq[ii]=bq[ii][torch.randperm(len(ii),generator=gen)]
                    ru=torch.zeros_like(bu);jj=active.nonzero().flatten();ru[jj]=bu[jj][torch.randperm(len(jj),generator=gen)]
                    factor=1.0
                    for attempt in range(25):
                        cq=rq*factor;project_q_(cq,x0,scale,fixed,lo,hi)
                        if valid(ref,raw,ru,cq,sides): break
                        factor*=.5
                    else: raise RuntimeError('No feasible geometry control')
                    save('joint_r4_b0.25_permuted',ru,cq,'random','joint',.25,score(data,ref,raw,ru,cq),dict(control_seed=123+index,position_radius=4.,control_shrink_factor=factor,control_q_norm_ratio=float(cq.norm()/bq.norm().clamp_min(EPS))))
                continue
            u=torch.zeros_like(k0);q=torch.zeros_like(x0);best=float(score(data,ref,raw,u,q));bu,bq=u.clone(),q.clone();hist=[]
            for step in range(1201):
                cur=score(data,ref,raw,u.requires_grad_(True),q.requires_grad_(True));hist.append(dict(step=step,predicted_p=float(cur.detach())))
                if float(cur)<best:best=float(cur);bu,bq=u.detach().clone(),q.detach().clone()
                if step==1200:break
                gu,gq=torch.autograd.grad(cur,(u,q))
                if not torch.isfinite(gu).all() or not torch.isfinite(gq).all() or not torch.isfinite(cur): raise RuntimeError('Nonfinite gradient/score')
                gu[active]-=gu[active].mean();gu[~active]=0;gq[~movable]=0
                if mode=='joint': gu/=gu.abs().max().clamp_min(EPS);gq/=gq.abs().max().clamp_min(EPS)
                elif mode=='stiffness':gu/=gu.abs().max().clamp_min(EPS);gq.zero_()
                elif mode=='positions':gu.zero_();gq/=gq.abs().max().clamp_min(EPS)
                accepted=False; masked=torch.zeros(len(q),dtype=torch.bool)
                for mask_pass in range(17):
                    offenders=None
                    for h in (.05,.025,.0125,.00625):
                        nu=(u.detach()-h*gu).clone();nq=(q.detach()-h*gq).clone();project_u_(nu,active,bound);project_q_(nq,x0,scale,fixed,lo,hi)
                        if valid(ref,raw,nu,nq,sides):
                            if float(score(data,ref,raw,nu,nq))<float(cur):
                                u,q=nu,nq;accepted=True;break
                        else: offenders=blocking_nodes(ref,raw,nu,nq)
                    if accepted: break
                    if offenders is None or not (offenders & ~masked & ~fixed).any(): break
                    masked |= offenders;gq[masked]=0
                hist[-1].update(mask_passes=mask_pass,masked_nodes=int(masked.sum()),accepted=accepted)
                if not accepted: break
            name=f'{mode}_r{radius:g}_b{bound:g}';pd.DataFrame(hist).to_csv(folder/f'{name}_history.csv',index=False);save(name,bu,bq,'ml',mode,bound,best,dict(position_radius=radius))
            if mode=='joint' and bound==.25:
                gen=torch.Generator().manual_seed(123+index);rq=torch.zeros_like(bq);ii=(~fixed).nonzero().flatten();rq[ii]=bq[ii][torch.randperm(len(ii),generator=gen)]
                # Only geometric feasibility can shrink a control; no score-based selection.
                ru=torch.zeros_like(bu);jj=active.nonzero().flatten();ru[jj]=bu[jj][torch.randperm(len(jj),generator=gen)]
                factor=1.0
                for attempt in range(25):
                    cq=rq*factor;project_q_(cq,x0,scale,fixed,lo,hi)
                    if valid(ref,raw,ru,cq,sides): break
                    factor*=.5
                else: raise RuntimeError('No feasible geometry control')
                save(name+'_permuted',ru,cq,'random','joint',bound,score(data,ref,raw,ru,cq),dict(control_seed=123+index,position_radius=radius,control_shrink_factor=factor,control_q_norm_ratio=float(cq.norm()/bq.norm().clamp_min(EPS))))
    dump(out/'gradient_checks.json',checks)
    if any(c['relative_error']>.1 or not np.isfinite(c['relative_error']) for c in checks):raise RuntimeError('finite-difference check failed')
    if len(records)!=40:raise RuntimeError('unexpected frozen-design count')
    recipe['status']='completed';recipe['max_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;dump(out/'recipe.json',recipe);dump(out/'frozen_designs.json',dict(recipe_sha256=sha(out/'recipe.json'),frozen_at_unix=time.time(),records=records))
def optimize_all(out):
    if (out/'recipe.json').exists(): raise RuntimeError('refuse overwrite')
    optimize(out,'reid')
    sub=json.loads((out/'frozen_designs.json').read_text())
    assert sub['recipe_sha256']==sha(out/'recipe.json') and len(sub['records'])==40
    # optimize() writes the global manifest directly because this study has one
    # source and all forty designs must freeze together before verification.
def verify(out):
    m=json.loads((out/'frozen_designs.json').read_text());assert m['recipe_sha256']==sha(out/'recipe.json')
    for r in m['records']:assert sha(r['physicalpath'])==r['sha256'] and sha(r['edit_file'])==r['edit_sha256']
    s=importlib.util.spec_from_file_location('endpoint',HELPER);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod);rows=[];lmp=DONOR/'.runtime/lammps/bin/lmp'
    dump(out/'verification_recipe.json',dict(job_id=os.environ.get('PBS_JOBID'),manifest_sha256=sha(out/'frozen_designs.json'),script_sha256=sha(__file__),endpoint_helper_sha256=sha(HELPER),lammps_sha256=sha(lmp),steps=120,feedback_to_optimizer=False))
    reused=0
    for r in m['records']:
        if r.get('reused_physical_measurement',False):
            prior_results=Path(r['prior_results_csv'])
            if sha(prior_results)!=r['prior_results_csv_sha256']: raise RuntimeError('prior results CSV hash changed')
            prior_manifest=Path(r['prior_manifest'])
            if sha(prior_manifest)!=r['prior_manifest_sha256'] or json.loads(prior_manifest.read_text())['recipe_sha256']!=r['prior_recipe_sha256']: raise RuntimeError('prior manifest/recipe provenance changed')
            prior_rows=pd.read_csv(prior_results)
            prior=prior_rows[(prior_rows.source==r['source']) & (prior_rows['index']==r['index']) & (prior_rows.variant==r['variant'])]
            if len(prior)!=1: raise RuntimeError(f"missing reused physical row {r['index']} {r['variant']}")
            z=prior.iloc[0]
            if z.sha256!=r['sha256'] or z.edit_sha256!=r['edit_sha256'] or z.status!='completed' or abs(float(z.scale_x)-float(r['scale_x']))>1e-12 or int(z.compression_steps)!=120: raise RuntimeError('reused prior result does not match frozen design/protocol')
            outcome={k:z[k] for k in ['simulation_id','compression_steps','physical_p_ratio','status','run_dir','protocol','box_p_ratio']}
            rows.append({**r,**outcome,'verification_provenance':'reused matched prior 120-step endpoint measurement'}); reused+=1
        else:
            g=torch.load(r['physicalpath'],weights_only=False,map_location='cpu');z=mod.validate_endpoints({(r['index'],r['variant']):g},out/'physics'/r['source'],lmp,{r['index']:r['scale_x']},compression_steps=120).iloc[0].to_dict();rows.append({**r,**z,'verification_provenance':'new 120-step endpoint measurement'});
        pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    table=pd.DataFrame(rows);original=table[table.kind=='original'].set_index(['source','index']).physical_p_ratio
    table['original_p_ratio']=[original.loc[(r.source,r['index'])] for _,r in table.iterrows()];table['change']=table.physical_p_ratio-table.original_p_ratio
    table.to_csv(out/'results.csv',index=False)
    if len(rows)!=40 or reused!=3: raise RuntimeError(f'unexpected verification counts rows={len(rows)} reused={reused}')
    dump(out/'verification_completed.json',dict(total=len(rows),valid=int(table.status.eq('completed').sum()),new_physical_evaluations=37,reused_physical_measurements=reused,feedback_to_optimizer=False,manifest_sha256=sha(out/'frozen_designs.json')))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['optimize','verify'],required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();x.output.mkdir(parents=True,exist_ok=True)
    if x.stage=='optimize':
        if (x.output/'recipe.json').exists():raise RuntimeError('refuse overwrite')
        optimize_all(x.output)
    else:verify(x.output)
