#!/usr/bin/env python3
"""Independent raw-data reference p-ratios and input hashes; no model fitting."""
from pathlib import Path
import hashlib
import json
import gc
import os
import sys
ROOT = Path(os.environ.get('LSS_PROJECT_ROOT', Path.cwd())).resolve()
sys.path.insert(0, str(ROOT/'src'))
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides
from lss.data import load_dataset

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    recipe_file=ROOT/'notebooks/results/compact_lj_reconstruction/r16_d4_s123_code_v2/recipe.json'
    recipe=json.loads(recipe_file.read_text())
    specs=recipe['source']['dataset_mixture']+recipe['mixed_evaluation']['dataset_mixture']
    records=[]; fingerprints={}
    for spec in specs:
        path=ROOT/'data'/Path(spec['path']).name
        actual_hash=sha(path)
        assert actual_hash==recipe['dataset_sha256'][spec['name']]
        fingerprints[spec['name']]={'path':str(path),'sha256':actual_hash,'bytes':path.stat().st_size}
        raw=load_dataset(path,coordinate_normalization=None,pos_dim=2)
        for i in spec['split_indices']['val']:
            sim=raw[i]
            for f in [25,50,100,150,199]:
                pair=[sim[0].clone(),sim[f].clone()]
                for g in pair: g.x=g.x.double()
                records.append({'source':spec['name'],'original_index':i,'frame':f,
                                'raw_physical_true_p_ratio':float(calc_p_ratio_rollout_sides(pair,-1))})
        del raw; gc.collect()
        print('audited',spec['name'],flush=True)
    output=ROOT/'notebooks/results/compact_lj_pratio/raw_target_audit'
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(output/'raw_target_rows.csv',index=False)
    (output/'metadata.json').write_text(json.dumps({'datasets':fingerprints,'recipe_sha256':sha(recipe_file),
        'script_sha256':sha(Path(__file__)),'rows':len(records),'selection':'none','final_test_used':False},indent=2)+'\n')
if __name__=='__main__':main()
