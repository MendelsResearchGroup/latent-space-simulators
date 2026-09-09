#!/usr/bin/env python3
"""Strict non-selective collector; preserves partial failures and missing keys."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out):
 out=Path(out);m=json.loads((out/'frozen_designs.json').read_text());p=json.loads((out/'thermal_protocol.json').read_text());assert len(m['records'])==210 and p['manifest_sha256']==sha(out/'frozen_designs.json')
 assert m['recipe_sha256']==sha(out/'recipe.json')
 for record in m['records']:
  assert sha(record['physicalpath'])==record['sha256'] and sha(record['edit_file'])==record['edit_sha256']
 expected={(int(r['index']),r['variant'],int(s)) for r in m['records'] for s in p['thermal_seeds']};tables=[];missing=[]
 for i in sorted({int(r['index']) for r in m['records']}):
  f=out/'thermal_physics'/f'id{i}'/'results.csv'
  if not f.exists():missing.append(i);continue
  recipe=json.loads((f.parent/'verification_recipe.json').read_text())
  assert recipe['manifest_sha256']==sha(out/'frozen_designs.json') and recipe['protocol_sha256']==sha(out/'thermal_protocol.json')
  t=pd.read_csv(f);assert len(t)<=21
  if (f.parent/'verification_completed.json').exists():
   completion=json.loads((f.parent/'verification_completed.json').read_text())
   assert recipe['manifest_sha256']==completion['manifest_sha256'] and recipe['protocol_sha256']==completion['protocol_sha256']
   assert completion['total']==len(t) and completion['valid']==int(t.status.eq('completed').sum())
  else:missing.append(i)
  tables.append(t)
 rows=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame();got=set(zip(rows.get('index',[]),rows.get('variant',[]),rows.get('thermal_seed',[])))
 if len(rows):
  if not got.issubset(expected) or rows.duplicated(['index','variant','thermal_seed']).any():raise RuntimeError('invalid thermal keys')
  rec={(int(r['index']),r['variant']):r for r in m['records']}
  for r in rows.to_dict('records'):
   d=rec[(int(r['index']),r['variant'])];assert r['sha256']==d['sha256'] and r['edit_sha256']==d['edit_sha256'] and r['temperature']==d['temperature']
  rows['physical_p_ratio']=rows.physical_p_ratio.where(rows.status.eq('completed'))
  base=rows[rows.kind.eq('original')].set_index(['index','thermal_seed']).physical_p_ratio;rows['original_p_ratio']=[base.get((r['index'],r['thermal_seed']),np.nan) for r in rows.to_dict('records')];rows['change']=rows.physical_p_ratio-rows.original_p_ratio
 rows.to_csv(out/'thermal_results.csv',index=False);summary=[]
 if len(rows):
  for k,g in rows.groupby(['temperature','index','variant','kind','arm'],dropna=False):
   ok=g.status.eq('completed')&np.isfinite(g.physical_p_ratio);paired=ok&np.isfinite(g.change);v=g.loc[ok,'physical_p_ratio'];d=g.loc[paired,'change']
   summary.append(dict(temperature=k[0],index=k[1],recipe_id=k[4] if pd.notna(k[4]) else 'original',variant=k[2],kind=k[3],thermal_valid=int(ok.sum()),thermal_total=3,thermal_observed=len(g),thermal_mean=float(v.mean()),thermal_sd=float(v.std(ddof=1)),paired_valid=int(paired.sum()),change_mean=float(d.mean()),change_sd=float(d.std(ddof=1)),positive_to_negative=int(((g.loc[paired,'original_p_ratio']>0)&(g.loc[paired,'physical_p_ratio']<0)).sum()),already_negative=int((g.loc[paired,'original_p_ratio']<0).sum()),final_negative=int((g.loc[ok,'physical_p_ratio']<0).sum())))
 pd.DataFrame(summary).to_csv(out/'thermal_summary.csv',index=False)
 # Network is the unit of replication; velocity replicates only form each network mean.
 net=pd.DataFrame(summary); network=[]
 if len(net):
  for (temperature,recipe_id,kind),g in net.groupby(['temperature','recipe_id','kind']):
   network.append(dict(temperature=temperature,recipe_id=recipe_id,kind=kind,networks_with_any_results=int(np.isfinite(g.thermal_mean).sum()),networks_with_all_three_seeds=int((g.thermal_valid==3).sum()),networks_expected=5,network_mean_of_thermal_means=float(g.thermal_mean.mean()),network_sd_of_thermal_means=float(g.thermal_mean.std(ddof=1)),thermal_seed_valid_total=int(g.thermal_valid.sum()),thermal_seed_expected_total=15))
 pd.DataFrame(network).to_csv(out/'thermal_network_summary.csv',index=False)
 status=dict(expected=630,observed=len(rows),valid=int((rows.status.eq('completed')&np.isfinite(rows.physical_p_ratio)).sum()) if len(rows) else 0,missing_ids=missing,missing_keys=len(expected-got),manifest_sha256=sha(out/'frozen_designs.json'),protocol_sha256=sha(out/'thermal_protocol.json'),feedback_to_optimizer=False);(out/'thermal_collect_status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);main(a.parse_args().output)
