#!/usr/bin/env python3
"""Hash and freeze the direct 1,000-MD-step mini-temperature inputs."""
import hashlib,json
from pathlib import Path
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators')
OUT=ROOT/'notebooks/results/mini_temperature1000_uniform'
BASE=Path('/rg/mendels_prj/alexander.z/MetaForge/results/mini_temperature_1000')
ITEMS={'node_low':('low','mini_node_optimized_200_500'),'node_mixed':('mixed','mini_node_optimized_200_500'),'stiffness_low':('low','mini_stiff_optimized_200_1500'),'stiffness_mixed':('mixed','mini_stiff_optimized_200_1500')}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=False); ds={}
 for name,(temp,kind) in ITEMS.items():
  d=BASE/temp/kind; m=json.loads((d/'manifest.json').read_text());r=json.loads((d/'recipe.json').read_text())
  assert m['status']=='complete' and m['source_trajectory_count']==200 and m['frames_per_trajectory']==200
  assert m['md_steps_between_frames']==1000 and m['last_frame_md_step']==199000
  assert r['config']['strain']==.0199 and r['strain_increment_per_frame']==.0001 and r['config']['sample_every']==1000
  assert all(q['source_index']==i for i,q in enumerate(m['records']))
  h=sha(d/'dataset.pt');ds[name]={'trainval_path':str(d/'dataset.pt'),'trainval_sha256':h,'test_path':str(d/'dataset.pt'),'test_sha256':h,'manifest_path':str(d/'manifest.json'),'manifest_sha256':sha(d/'manifest.json'),'recipe_path':str(d/'recipe.json'),'recipe_sha256':sha(d/'recipe.json'),'p_ratio_validation_path':str(d/'p_ratio_validation.json'),'p_ratio_validation_sha256':sha(d/'p_ratio_validation.json')}
 for a,b in [('node_low','node_mixed'),('stiffness_low','stiffness_mixed')]:
  ma=json.loads(Path(ds[a]['manifest_path']).read_text())['records'];mb=json.loads(Path(ds[b]['manifest_path']).read_text())['records'];assert all(x['source_index']==y['source_index'] and x['source_chunk']==y['source_chunk'] for x,y in zip(ma,mb))
 (OUT/'data_manifest.json').write_text(json.dumps({'status':'complete','study':'mini_temperature1000_uniform','datasets':ds,'split':{'train':[0,20],'validation':[20,40],'test':[40,120]},'frames':{'count':200,'times':'0..199000 step1000','compression':.0199},'pairing':'low/mixed paired source_index/source_chunk verified'},indent=2)+'\n')
if __name__=='__main__':main()
