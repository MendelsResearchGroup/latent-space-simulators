#!/usr/bin/env python3
"""Train one broad trajectory model using cached frozen-AE tensors."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ['LSS_PROJECT_ROOT']).resolve()
CODE = Path(os.environ['LSS_CODE_ROOT']).resolve()
sys.path.insert(0, str(CODE/'src'))
from lss.dynamics.trajectory_propagator import make_trajectory_propagator, FAMILIES
from lss.dynamics.experiment import _autoencoder_class
from evaluate import decode, evaluate
from prepare import sha, digest

OUT = ROOT/'notebooks/results/lj100_broad'
SUP = (1,2,3,5,10,25,50,75,100)
NAMES = ('reid','depablo_low_temp','lj_noisy')


def load_ae(data):
    bundle = torch.load(data['ae_checkpoint'], map_location='cpu', weights_only=False)
    p, n = bundle['params'], data['normalizers']
    ae = _autoencoder_class(p['autoencoder_model'])(pos_dim=2,
        node_feature_dim=n['node_feature_mean'].numel(), edge_dim=n['edge_mean'].numel(),
        hidden_size=p['hidden_size'], latent_dim=2, latent_tokens=p['latent_tokens'],
        reconstruction_dim=n['target_mean'].numel(), message_passing_steps=p['message_passing_steps'])
    ae.set_edge_normalization(n['edge_mean'],n['edge_std'],n['ref_edge_mean'],n['ref_edge_std'])
    ae.edge_mode = p['edge_mode']
    ae.load_state_dict(bundle['ae_state_dict'], strict=True)
    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    return ae


def losses(ae, model, row, stats):
    path = model(row['latent_norm'][0], row['reference_tokens'])
    latent = (path[list(SUP)]-row['latent_norm'][list(SUP)]).square().mean()
    decoded = torch.stack([decode(ae,row,path[frame],stats) for frame in SUP])
    position = (decoded-row['target_norm'][list(SUP)]).square().mean()
    return position, latent


@torch.no_grad()
def validate(ae, model, data):
    model.eval()
    result = {}
    for source in NAMES:
        scores = [losses(ae,model,row,data['latent_stats']) for row in data['validation'] if row['source']==source]
        result['val_position_'+source] = float(np.mean([float(v[0]) for v in scores]))
        result['val_latent_'+source] = float(np.mean([float(v[1]) for v in scores]))
    result['val_position'] = float(np.mean([result['val_position_'+s] for s in NAMES]))
    assert np.isfinite(list(result.values())).all()
    return result


def train_epoch(ae, model, data, optimizer, rng):
    model.train()
    groups = {s:[r for r in data['train'] if r['source']==s] for s in NAMES}
    for group in groups.values():
        rng.shuffle(group)
    logs = {s:[] for s in NAMES}
    steps = max(len(groups[NAMES[0]]),len(groups[NAMES[1]]),(len(groups[NAMES[2]])+1)//2)
    for step in range(steps):
        source_losses = []
        for source, size in zip(NAMES,(1,1,2)):
            batch = [groups[source][(size*step+j)%len(groups[source])] for j in range(size)]
            values = [losses(ae,model,row,data['latent_stats']) for row in batch]
            objective = torch.stack([pos+.1*lat for pos,lat in values]).mean()
            logs[source].append(float(objective.detach()))
            source_losses.append(objective)
        objective = torch.stack(source_losses).mean()
        assert torch.isfinite(objective)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    return {'train_'+s:float(np.mean(v)) for s,v in logs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(123,456,786),required=True)
    parser.add_argument('--family',choices=FAMILIES,required=True)
    parser.add_argument('--context',choices=('mean','attention'),required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest = json.loads((CODE/'manifest.json').read_text())
    assert {key:sha(CODE/key) for key in manifest}==manifest
    name = f'{args.family}_{args.context}_s{args.seed}'
    if args.smoke:
        name += '_smoke_'+CODE.name
    out = OUT/'runs'/name
    out.mkdir(parents=True,exist_ok=False)
    start = time.time()
    try:
        cache_done = json.loads(args.cache.with_suffix('.completed.json').read_text())
        assert cache_done['status']=='completed' and cache_done['cache_sha256']==sha(args.cache)
        data = torch.load(args.cache,map_location='cpu',weights_only=False)
        assert data['provenance']['seed']==args.seed and data['provenance']['smoke']==args.smoke
        assert sha(data['ae_checkpoint'])==data['provenance']['ae_sha256']
        protocol = json.loads((CODE/'protocol.json').read_text())
        assert data['provenance']['protocol_sha256']==sha(CODE/'protocol.json')
        planned = dict(seed=args.seed,family=args.family,context_mode=args.context,smoke=args.smoke,
            cache=str(args.cache),cache_sha256=sha(args.cache),protocol=protocol,
            protocol_sha256=sha(CODE/'protocol.json'),code_manifest_sha256=sha(CODE/'manifest.json'),
            cache_provenance=data['provenance'],job_id=os.environ['PBS_JOBID'],
            response_selection=False,final_test_used=False)
        (out/'recipe.json').write_text(json.dumps(planned,indent=2)+'\n')
        ae = load_ae(data)
        before = dict(ae=digest(ae.state_dict()),normalizers=digest(data['normalizers']),stats=digest(data['latent_stats']))
        model = make_trajectory_propagator(args.family,args.context)
        optimizer = torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5)
        history = [dict(epoch=0,**validate(ae,model,data))]
        best,best_epoch,bad = history[0]['val_position'],0,0
        state = copy.deepcopy(model.state_dict())
        rng = random.Random(args.seed)
        for epoch in range(1,2 if args.smoke else 61):
            tick = time.time()
            train = train_epoch(ae,model,data,optimizer,rng)
            val = validate(ae,model,data)
            history.append(dict(epoch=epoch,seconds=time.time()-tick,**train,**val))
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print(json.dumps(history[-1]),flush=True)
            if val['val_position']<best-1e-5:
                best,best_epoch,bad = val['val_position'],epoch,0
                state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if bad >= (1 if args.smoke else 12):
                break
        model.load_state_dict(state)
        checks = {}
        if args.smoke:
            optimizer.zero_grad(set_to_none=True)
            position,_ = losses(ae,model,data['train'][0],data['latent_stats'])
            position.backward()
            gradient = float(sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None))
            assert np.isfinite(gradient) and gradient>0
            checks['pure_position_gradient'] = gradient
            sample = data['validation'][0]
            with torch.no_grad():
                path = model(sample['latent_norm'][0],sample['reference_tokens'])
                assert torch.equal(path[0],sample['latent_norm'][0])
                restored = make_trajectory_propagator(args.family,args.context)
                restored.load_state_dict(state)
                torch.testing.assert_close(path,restored(sample['latent_norm'][0],sample['reference_tokens']),rtol=0,atol=0)
                checks['reload_max_abs'] = 0.0
        after = dict(ae=digest(ae.state_dict()),normalizers=digest(data['normalizers']),stats=digest(data['latent_stats']))
        assert before==after
        table = evaluate(ae,model,data,args.seed,args.family,args.context)
        assert len(table)==len(data['validation'])*200
        table.to_csv(out/'per_network_rows.csv',index=False)
        torch.save(dict(state_dict=state,family=args.family,context_mode=args.context),out/'model.pt')
        planned.update(best_epoch=best_epoch,best_val_position=best,epochs_trained=len(history)-1,
            frozen_before=before,frozen_after=after,checks=checks,seconds=time.time()-start,
            checkpoint_sha256=sha(out/'model.pt'))
        (out/'recipe.json').write_text(json.dumps(planned,indent=2)+'\n')
        (out/'completed.json').write_text(json.dumps(dict(status='completed',rows=len(table),
            seconds=time.time()-start,job_id=os.environ['PBS_JOBID']),indent=2)+'\n')
    except Exception as error:
        (out/'failed.json').write_text(json.dumps(dict(status='failed',error=f'{type(error).__name__}: {error}',
            job_id=os.environ['PBS_JOBID']),indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
