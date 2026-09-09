#!/usr/bin/env python3
"""Global freeze and post-verification collection for the Reid joint Adam sweep."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def freeze(root):
    matrix=load(root/'matrix.json'); records=[]
    base=matrix['baseline']; bm=load(base['manifest_path']); br=pd.read_csv(base['results_path'])
    dependencies=load(root/'code/dependency_manifest.json')['dependencies']
    if dependencies.get(str(Path(base['results_path']))) != sha(base['results_path']) or dependencies.get(str(Path(base['manifest_path']))) != sha(base['manifest_path']): raise RuntimeError('baseline dependency hashes changed')
    if bm['recipe_sha256']!=sha(Path(base['manifest_path']).parent/'recipe.json'): raise RuntimeError('baseline recipe mismatch')
    keep={'original','joint_r4_b0.25','joint_r4_b0.25_permuted'}
    baseline=[r for r in bm['records'] if r['variant'] in keep]
    if len(baseline)!=30: raise RuntimeError(f'baseline has {len(baseline)}, expected 30')
    for r in baseline:
        row=br[(br.source==r['source'])&(br['index']==r['index'])&(br.variant==r['variant'])]
        if len(row)!=1 or row.iloc[0].sha256!=r['sha256'] or row.iloc[0].edit_sha256!=r['edit_sha256'] or row.iloc[0].status!='completed': raise RuntimeError(f"baseline result mismatch {r['index']} {r['variant']}")
        if sha(r['physicalpath'])!=r['sha256'] or sha(r['edit_file'])!=r['edit_sha256']: raise RuntimeError('baseline frozen artifact mismatch')
        records.append({**r,'config_id':'original' if r['kind']=='original' else base['id'],'learning_rate':.02,'log_stiffness_bound':.25,'patience':50,'feasibility_mode':'whole_proposal','reused_baseline_measurement':True,'source_manifest':base['manifest_path'],'source_manifest_sha256':sha(base['manifest_path'])})
    for setting in matrix['settings']:
        out=Path(setting['output']); manifest=load(out/'frozen_designs.json')
        if manifest['recipe_sha256']!=sha(out/'recipe.json') or len(manifest['records'])!=20: raise RuntimeError(f"setting incomplete {setting['id']}")
        for r in manifest['records']:
            if sha(r['physicalpath'])!=r['sha256'] or sha(r['edit_file'])!=r['edit_sha256']: raise RuntimeError(f"artifact mismatch {setting['id']} {r['index']} {r['variant']}")
            records.append({**r,'config_id':setting['id'],'learning_rate':setting['learning_rate'],'log_stiffness_bound':setting['log_stiffness_bound'],'patience':setting['patience'],'feasibility_mode':setting['feasibility_mode'],'reused_baseline_measurement':False,'source_manifest':str((out/'frozen_designs.json').resolve()),'source_manifest_sha256':sha(out/'frozen_designs.json')})
    if len(records)!=490: raise RuntimeError(f'global record count {len(records)}, expected 490')
    keys=[(r['config_id'],r['index'],r['variant']) for r in records]
    if len(set(keys))!=490: raise RuntimeError('global records not unique')
    payload={'matrix_sha256':sha(root/'matrix.json'),'records':records,'counts':{'ml':240,'control':240,'original':10,'total':490},'frozen_before_new_physics':True}
    (root/'global_frozen_designs.json').write_text(json.dumps(payload,indent=2)+'\n')
    (root/'global_freeze_completed.json').write_text(json.dumps({'total':490,'manifest_sha256':sha(root/'global_frozen_designs.json'),'new_physics_started':False},indent=2)+'\n')
def collect(root):
    global_manifest=load(root/'global_frozen_designs.json'); matrix=load(root/'matrix.json'); rows=[]
    if global_manifest['matrix_sha256']!=sha(root/'matrix.json'): raise RuntimeError('matrix changed after freeze')
    deps=load(root/'code/dependency_manifest.json')['dependencies']
    for name in ['manifest_path','results_path']:
        path=matrix['baseline'][name]
        if deps[str(Path(path))]!=sha(path): raise RuntimeError('baseline provenance changed after freeze')
    for record in global_manifest['records']:
        if sha(record['physicalpath'])!=record['sha256'] or sha(record['edit_file'])!=record['edit_sha256']: raise RuntimeError('frozen artifact changed')
        if sha(record['source_manifest'])!=record['source_manifest_sha256']: raise RuntimeError('source manifest changed')
    base=pd.read_csv(matrix['baseline']['results_path'])
    for r in global_manifest['records']:
        if r['reused_baseline_measurement']:
            src=base
        else:
            setting=next(s for s in matrix['settings'] if s['id']==r['config_id']); src=pd.read_csv(Path(setting['output'])/'results.csv')
            done=load(Path(setting['output'])/'verification_completed.json')
            verification_recipe=load(Path(setting['output'])/'verification_recipe.json')
            if done['manifest_sha256']!=r['source_manifest_sha256'] or verification_recipe['manifest_sha256']!=r['source_manifest_sha256'] or verification_recipe['steps']!=120: raise RuntimeError('verification provenance mismatch')
            if done['total']!=20: raise RuntimeError(f"setting verification incomplete {r['config_id']}")
        z=src[(src.source==r['source'])&(src['index']==r['index'])&(src.variant==r['variant'])]
        if len(z)!=1: raise RuntimeError(f"missing result {r['config_id']} {r['index']} {r['variant']}")
        row=z.iloc[0].to_dict()
        if row['sha256']!=r['sha256'] or row['edit_sha256']!=r['edit_sha256']: raise RuntimeError('result artifact mismatch')
        if row['status']=='completed' and int(row['compression_steps'])!=120: raise RuntimeError('completed row protocol mismatch')
        rows.append({**r,**{k:v for k,v in row.items() if k not in r}})
    table=pd.DataFrame(rows)
    if len(table)!=490: raise RuntimeError('collection count mismatch')
    table.to_csv(root/'results.csv',index=False)
    summary=[]
    for config_id,g in table[table.kind.eq('ml')].groupby('config_id'):
        v=g[g.status.eq('completed') & np.isfinite(g.physical_p_ratio)]
        summary.append(dict(config_id=config_id,learning_rate=float(g.learning_rate.iloc[0]),log_stiffness_bound=float(g.log_stiffness_bound.iloc[0]),patience=int(g.patience.iloc[0]),feasibility_mode=g.feasibility_mode.iloc[0],valid=len(v),total=len(g),mean_p=float(v.physical_p_ratio.mean()),median_p=float(v.physical_p_ratio.median()),negative=int((v.physical_p_ratio<0).sum()),mean_change=float(v.change.mean()),median_change=float(v.change.median()),mean_actual_steps=float(v.actual_steps.mean()) if 'actual_steps' in v else np.nan,median_actual_steps=float(v.actual_steps.median()) if 'actual_steps' in v else np.nan))
    pd.DataFrame(summary).to_csv(root/'summary.csv',index=False)
    (root/'collection_completed.json').write_text(json.dumps({'total':490,'valid':int(table.status.eq('completed').sum()),'retrospective_ranking_only':True,'global_manifest_sha256':sha(root/'global_frozen_designs.json')},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['freeze','collect'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try: freeze(a.output) if a.stage=='freeze' else collect(a.output)
    except Exception as exc:
        (a.output/'collection_incomplete.json').write_text(json.dumps({'stage':a.stage,'error':repr(exc),'status':'incomplete'},indent=2)+'\n')
        raise
