"""Evaluate frozen radical models on the previously excluded mixed-T cohort."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
ROOT=Path(os.environ['LSS_PROJECT_ROOT']).resolve()
CODE=Path(os.environ['LSS_CODE_ROOT']).resolve()
sys.path.insert(0,str(CODE/'src'))
from lss.dynamics.radical_propagator import make_radical_propagator
from run import load_ae
from evaluate import evaluate
from utilities import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--family',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    study=ROOT/'notebooks/results/lj100_radical'
    name=f'{args.family}_attention_s{args.seed}'+('_smoke_'+CODE.name if args.smoke else '')
    folder=study/'runs'/name
    assert json.loads((folder/'completed.json').read_text())['status']=='completed'
    checkpoint=folder/'model.pt'
    recipe=json.loads((folder/'recipe.json').read_text())
    assert sha(checkpoint)==recipe['checkpoint_sha256']
    cache=study/'cache'/(f'mixed_smoke_s{args.seed}_{CODE.name}.pt' if args.smoke else f'mixed_s{args.seed}.pt')
    assert sha(cache)==json.loads(cache.with_suffix('.completed.json').read_text())['cache_sha256']
    data=torch.load(cache,map_location='cpu',weights_only=False)
    assert not data['train'] and len(data['validation'])==20
    ae=load_ae(data)
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=make_radical_propagator(args.family,bundle['motion_scale'])
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    table=evaluate(ae,model,data,args.seed,args.family,'attention')
    table['split']='mixed_transfer'
    assert len(table)==4000
    table.to_csv(folder/'mixed_per_network_rows.csv',index=False)
    (folder/'mixed_completed.json').write_text(json.dumps(dict(status='completed',rows=len(table),
        cache_sha256=sha(cache),model_sha256=sha(checkpoint),job_id=os.environ['PBS_JOBID']),indent=2)+'\n')


if __name__=='__main__':main()
