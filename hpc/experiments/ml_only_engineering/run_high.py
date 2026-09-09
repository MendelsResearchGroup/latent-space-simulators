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
    q.mul_((2/q.norm(dim=1,keepdim=True).clamp_min(EPS)).clamp_max(1)); xx=(x+q*scale[:,None]).maximum(lo).minimum(hi);q[:]=(xx-x)/scale[:,None];q[fixed]=0
def check(data,ref,raw,u,q,active,movable,sides,index):
    checks=[]
    for block in ('spring','position'):
        du=torch.zeros_like(u);dq=torch.zeros_like(q)
        if block=='spring': du[active]=torch.randn(int(active.sum()));du[active]-=du[active].mean();du/=du.abs().max().clamp_min(EPS)
        else:
            nodes=movable[:,0];values=torch.randn((int(nodes.sum()),2),dtype=q.dtype);values-=values.mean(dim=0,keepdim=True);dq[nodes]=values;dq/=dq.abs().max().clamp_min(EPS)
        v=score(data,ref,raw,u.requires_grad_(True),q.requires_grad_(True)); grad=torch.autograd.grad(v,(u,q),allow_unused=True); exact=float(((grad[0] if grad[0] is not None else torch.zeros_like(u))*du).sum()+((grad[1] if grad[1] is not None else torch.zeros_like(q))*dq).sum())
        with torch.no_grad(): fd=float((score(data,ref,raw,u.detach()+.001*du,q.detach()+.001*dq)-score(data,ref,raw,u.detach()-.001*du,q.detach()-.001*dq))/.002)
        rel=abs(fd-exact)/max(abs(fd),abs(exact),EPS); checks.append(dict(index=index,block=block,autograd=exact,finite_difference=fd,relative_error=rel))
    return checks
def optimize(out,source):
    sys.addaudithook(lambda e,a: (_ for _ in ()).throw(RuntimeError('external process forbidden')) if e in {'subprocess.Popen','os.system','os.posix_spawn','os.exec','os.fork'} else None)
    torch.set_num_threads(4);torch.manual_seed(786);np.random.seed(786)
    out.mkdir(parents=True,exist_ok=True)
    p=out.parent/'prepared'/source/'prepare'/'prepared.pt';data=torch.load(p,weights_only=False,map_location='cpu');data['ae'].eval()
    for z in data['ae'].parameters():z.requires_grad_(False)
    ids=sorted(data['references']);
    if len(ids)!=2: raise ValueError('Expected two predeclared high-response references per source.')
    recipe=dict(stage='optimize',status='running',source=source,source_ids=ids,prepared_sha256=sha(p),script_sha256=sha(__file__),code_manifest_sha256=sha(CODE/'hashes.json'),model_seed=786,policy='z1',arms=['stiffness_b2','stiffness_b4','positions_b0','joint_b4'],steps=150,step_sizes=[.05,.025,.0125,.00625],selection='lowest frozen z1 readout among valid ML-only iterates',counts=dict(ml=8,random=4,original=2,total=14),job_id=os.environ.get("PBS_JOBID"),split="highest original-response train/calibration/validation development cases; no final test",geometry_constraints="fixed box/topology/boundary nodes; movement <=0.5 shortest original incident bond; lengths 0.25..1.75 original; pair separation >=0.1 original minimum",gradient_scaling="center stiffness; normalize spring and geometry blocks separately",preparation=json.loads((p.parent/"preparation.json").read_text()),no_response_targets=True,no_external_processes=True)
    dump(out/'recipe.json',recipe); records=[];checks=[]
    for index in ids:
        ref,raw=data['references'][index],data['physical'][index];k0=ref.edge_attr[:,3].detach();active=k0>0;sides,fixed,lo,hi=constraints(ref,raw);x0=ref.reference_context_positions[:,:2].detach(); L=ref.reference_context_edge_attr[:,2].detach();e=ref.edge_index.long();scale=.25*torch.stack([L[(e==i).any(0)].min() for i in range(len(x0))]);movable=(~fixed)[:,None].expand_as(x0)
        folder=out/f'{source}_{index}';folder.mkdir(exist_ok=True)
        def save(name,u,q,kind,mode,bound,pred,extra={}):
            g,x,k,_,box=geometry(ref,raw,u,q); physical=copy.deepcopy(raw);physical.x=x.detach().to(raw.x);physical.pos=physical.x.clone(); lookup={tuple(sorted(z)):i for i,z in enumerate(e.T.tolist())}; order=torch.tensor([lookup[tuple(sorted(z))] for z in raw.edge_index.T.tolist()]);v=physical.x[physical.edge_index[1]]-physical.x[physical.edge_index[0]];v-=torch.round(v/box)*box;physical.edge_attr=torch.cat([v,v.norm(dim=1,keepdim=True),k.detach()[order,None].to(v)],1)
            if hasattr(physical,'registry_poisson_ratio'): physical.registry_poisson_ratio=None
            assert torch.equal(physical.edge_index,raw.edge_index)
            assert torch.equal(physical.edge_attr[:,3]==0,raw.edge_attr[:,3]==0)
            assert (physical.edge_attr[raw.edge_attr[:,3]>0,3]>0).all()
            assert torch.equal(physical.x[fixed],raw.x[fixed])
            assert float((x-x0).norm(dim=1).div(scale).max())<=2.0001
            assert abs(float(u[active].mean()))<2e-6
            if mode in ('stiffness','original'): assert torch.equal(physical.x,raw.x)
            if mode=='positions': assert torch.equal(physical.edge_attr[:,3],raw.edge_attr[:,3])
            gp,ep=folder/f'{name}.pt',folder/f'{name}_edit.pt';torch.save(physical,gp);torch.save(dict(log_stiffness=u.detach(),position_q=q.detach()),ep);records.append(dict(source=source,index=index,variant=name,kind=kind,policy='z1',mode=mode,bound=bound,model_seed=786,predicted_p=float(pred),scale_x=float(data['scales'][index]),physicalpath=str(gp.resolve()),sha256=sha(gp),edit_file=str(ep.resolve()),edit_sha256=sha(ep),**extra))
        u0=torch.zeros_like(k0);q0=torch.zeros_like(x0)
        if not valid(ref,raw,u0,q0,sides): raise RuntimeError('Pristine geometry invalid')
        base=readout_value(encode_initial(data['ae'],ref,k0,data['normalizers']),data['policies']['z1'])
        zero=score(data,ref,raw,u0,q0)
        dump(folder/'zero_edit_parity.json',dict(original_model_score=float(base.detach()),rebuilt_model_score=float(zero.detach()),error=abs(float(base.detach()-zero.detach()))))
        if abs(float(base.detach()-zero.detach()))>1e-4: raise RuntimeError('Zero-edit encoder parity failed')
        save('original',u0,q0,'original','original',0,score(data,ref,raw,u0,q0))
        local_checks=check(data,ref,raw,u0,q0,active,movable,sides,index); checks+=local_checks
        dump(out/'gradient_checks.json',checks)
        if any(c['relative_error']>.1 or not np.isfinite(c['relative_error']) for c in local_checks): raise RuntimeError('Initial finite-difference check failed')
        for mode,bound in [('stiffness',2.),('stiffness',4.),('positions',0.),('joint',4.)]:
            u=torch.zeros_like(k0);q=torch.zeros_like(x0);best=float(score(data,ref,raw,u,q));bu,bq=u.clone(),q.clone();hist=[]
            for step in range(151):
                cur=score(data,ref,raw,u.requires_grad_(True),q.requires_grad_(True));hist.append(dict(step=step,predicted_p=float(cur.detach())))
                if float(cur)<best:best=float(cur);bu,bq=u.detach().clone(),q.detach().clone()
                if step==150:break
                gu,gq=torch.autograd.grad(cur,(u,q))
                if not torch.isfinite(gu).all() or not torch.isfinite(gq).all() or not torch.isfinite(cur): raise RuntimeError('Nonfinite gradient/score')
                gu[active]-=gu[active].mean();gu[~active]=0;gq[~movable]=0
                if mode=='joint': gu/=gu.abs().max().clamp_min(EPS);gq/=gq.abs().max().clamp_min(EPS)
                elif mode=='stiffness':gu/=gu.abs().max().clamp_min(EPS);gq.zero_()
                elif mode=='positions':gu.zero_();gq/=gq.abs().max().clamp_min(EPS)
                accepted=False
                for h in (.05,.025,.0125,.00625):
                    nu=(u.detach()-h*gu).clone();nq=(q.detach()-h*gq).clone();project_u_(nu,active,bound);project_q_(nq,x0,scale,fixed,lo,hi)
                    if valid(ref,raw,nu,nq,sides) and float(score(data,ref,raw,nu,nq))<float(cur):u,q=nu,nq;accepted=True;break
                if not accepted: break
            name=f'{mode}_b{bound:g}';pd.DataFrame(hist).to_csv(folder/f'{name}_history.csv',index=False);save(name,bu,bq,'ml',mode,bound,best)
            if mode=='stiffness':
                gen=torch.Generator().manual_seed(123+index+int(bound));ru=torch.zeros_like(bu);ii=active.nonzero().flatten();ru[ii]=bu[ii][torch.randperm(len(ii),generator=gen)];save(name+'_permuted',ru,torch.zeros_like(bq),'random',mode,bound,score(data,ref,raw,ru,torch.zeros_like(bq)),dict(control_seed=123+index+int(bound)))
    dump(out/'gradient_checks.json',checks)
    if any(c['relative_error']>.1 or not np.isfinite(c['relative_error']) for c in checks):raise RuntimeError('finite-difference check failed')
    if len(records)!=14:raise RuntimeError('unexpected per-source frozen-design count')
    recipe['status']='completed';recipe['max_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;dump(out/'recipe.json',recipe);dump(out/'frozen_designs.json',dict(recipe_sha256=sha(out/'recipe.json'),frozen_at_unix=time.time(),records=records))
def optimize_all(out):
    if (out/'recipe.json').exists(): raise RuntimeError('refuse overwrite')
    records=[];source_recipes={}
    for source in ['reid','depablo_low_temp']:
        sub=out/source
        if (sub/'recipe.json').exists(): raise RuntimeError('refuse partial overwrite')
        optimize(sub,source)
        m=json.loads((sub/'frozen_designs.json').read_text())
        assert m['recipe_sha256']==sha(sub/'recipe.json')
        records.extend(m['records']);source_recipes[source]=json.loads((sub/'recipe.json').read_text())
    assert len(records)==28
    dump(out/'recipe.json',dict(status='completed',job_id=os.environ.get('PBS_JOBID'),source_recipes=source_recipes,script_sha256=sha(__file__),counts=dict(originals=4,ml=16,random=8,total=28),screening_recipe_sha256=sha(out.parent/'screening/recipe.json')))
    dump(out/'frozen_designs.json',dict(recipe_sha256=sha(out/'recipe.json'),frozen_at_unix=time.time(),records=records))
def verify(out):
    m=json.loads((out/'frozen_designs.json').read_text());assert m['recipe_sha256']==sha(out/'recipe.json')
    for r in m['records']:assert sha(r['physicalpath'])==r['sha256'] and sha(r['edit_file'])==r['edit_sha256']
    s=importlib.util.spec_from_file_location('endpoint',HELPER);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod);rows=[];lmp=DONOR/'.runtime/lammps/bin/lmp'
    dump(out/'verification_recipe.json',dict(job_id=os.environ.get('PBS_JOBID'),manifest_sha256=sha(out/'frozen_designs.json'),script_sha256=sha(__file__),endpoint_helper_sha256=sha(HELPER),lammps_sha256=sha(lmp),steps=120,feedback_to_optimizer=False))
    for r in m['records']:
        g=torch.load(r['physicalpath'],weights_only=False,map_location='cpu');z=mod.validate_endpoints({(r['index'],r['variant']):g},out/'physics'/r['source'],lmp,{r['index']:r['scale_x']},compression_steps=120).iloc[0].to_dict();rows.append({**r,**z});pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    table=pd.DataFrame(rows);original=table[table.kind=='original'].set_index(['source','index']).physical_p_ratio
    table['original_p_ratio']=[original.loc[(r.source,r['index'])] for _,r in table.iterrows()];table['change']=table.physical_p_ratio-table.original_p_ratio
    table.to_csv(out/'results.csv',index=False)
    dump(out/'verification_completed.json',dict(total=len(rows),valid=int(table.status.eq('completed').sum()),feedback_to_optimizer=False,manifest_sha256=sha(out/'frozen_designs.json')))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['optimize','verify'],required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();x.output.mkdir(parents=True,exist_ok=True)
    if x.stage=='optimize':
        if (x.output/'recipe.json').exists():raise RuntimeError('refuse overwrite')
        optimize_all(x.output)
    else:verify(x.output)
