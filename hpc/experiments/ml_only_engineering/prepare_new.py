"""Prepare only initial frames of predeclared new engineering-validation networks."""
from pathlib import Path
import argparse, sys, json, hashlib
from copy import deepcopy
import torch
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project')
CODE=DONOR/'notebooks/results/network_design/code_endpoint_v1'
sys.path[:0]=[str(DONOR.parent/'MetaForge/src'),str(CODE/'src')]
import auxetic
from lss.data import normalize_dataset_edges, normalize_trajectory_to_reference_box, tag_simulation_source
IDS={'reid':[57,15],'depablo_low_temp':[105,152]}
PATHS={'reid':'reid_200_frames.pt','depablo_low_temp':'depablo-near-zero-temp.pt'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def main(out):
 torch.set_num_threads(4)
 for source,ids in IDS.items():
  old=DONOR/'notebooks/results/network_design/calibrated_direction_v1'/source/'prepare'
  info=json.loads((old/'preparation.json').read_text())
  assert not set(ids)&set(info['calibration_ids']+info['design_ids'])
  data=torch.load(old/'prepared.pt',weights_only=False,map_location='cpu')
  path=DONOR/'data'/PATHS[source]
  raw=torch.load(path,weights_only=False,map_location='cpu',mmap=True)
  def normalize(index):
   sim=[deepcopy(raw[index][0])]
   normalize_dataset_edges([sim],edge_multiplicity=1,edge_vector_dim=2)
   normalize_trajectory_to_reference_box(sim,pos_dim=2)
   tag_simulation_source(sim,source)
   return sim[0]
  check=info['design_ids'][0]; control=normalize(check); previous=data['references'][check]
  errors={}
  for key in ['x','edge_index','edge_attr','reference_context_positions','reference_context_edge_attr','reference_box_center','reference_box_half_extent']:
   a,b=getattr(control,key),getattr(previous,key)
   assert a.shape==b.shape,(source,key,a.shape,b.shape)
   errors[key]=float((a.double()-b.double()).abs().max())
   assert errors[key]<2e-6,(source,key,errors[key])
  scale=list(data['scales'].values())[0]
  assert all(abs(float(v)-scale)<1e-8 for v in data['scales'].values())
  data['references']={i:normalize(i) for i in ids}
  data['physical']={i:deepcopy(raw[i][0]) for i in ids}
  data['scales']={i:scale for i in ids}
  folder=out/source/'prepare';folder.mkdir(parents=True,exist_ok=False)
  torch.save(data,folder/'prepared.pt')
  info.update(new_engineering_validation_ids=ids,preprocessing_parity=errors,
      original_dataset=str(path),original_dataset_sha256=sha(path),old_prepared_sha256=sha(old/'prepared.pt'),
      split='previously unengineered original AE validation networks, disjoint from readout calibration and previous design IDs; not final test',
      frame_access='only original frame0; compression scale fixed from prior protocol; no new response observations')
  (folder/'preparation.json').write_text(json.dumps(info,indent=2)+'\n')
  print(source,ids,'prepared',errors,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
