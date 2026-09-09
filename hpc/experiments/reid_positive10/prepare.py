"""Prepare pristine frame-0 graphs for the frozen positive-Reid cohort."""
from pathlib import Path
import argparse, sys, json, hashlib
from copy import deepcopy
import torch
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project')
CODE=DONOR/'notebooks/results/network_design/code_endpoint_v1'
sys.path[:0]=[str(DONOR.parent/'MetaForge/src'),str(CODE/'src')]
import auxetic
from lss.data import normalize_dataset_edges, normalize_trajectory_to_reference_box, tag_simulation_source
PATH='reid_200_frames.pt'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def main(out):
 torch.set_num_threads(4)
 cohort_path=out.parent/'cohort.json'
 cohort=json.loads(cohort_path.read_text()); ids=[int(r['index']) for r in cohort['cohort']]
 assert ids==[93,84,96,97,52,59,57,53,2,74]
 old=DONOR/'notebooks/results/network_design/calibrated_direction_v1/reid/prepare'
 info=json.loads((old/'preparation.json').read_text())
 data=torch.load(old/'prepared.pt',weights_only=False,map_location='cpu')
 path=DONOR/'data'/PATH
 raw=torch.load(path,weights_only=False,map_location='cpu',mmap=True)
 def normalize(index):
  sim=[deepcopy(raw[index][0])]
  normalize_dataset_edges([sim],edge_multiplicity=1,edge_vector_dim=2)
  normalize_trajectory_to_reference_box(sim,pos_dim=2)
  tag_simulation_source(sim,'reid')
  return sim[0]
 errors={}
 keys=['x','edge_index','edge_attr','reference_context_positions','reference_context_edge_attr','reference_box_center','reference_box_half_extent']
 for index in ids:
  control=normalize(index)
  errors[str(index)]={}
  # The calibration bundle retains only its historical design references.
  # Check every new raw->normalized frame against its physical reference, and
  # additionally compare all overlapping frozen bundle references exactly.
  if index in data['references']:
   previous=data['references'][index]
   for key in keys:
    a,b=getattr(control,key),getattr(previous,key)
    assert a.shape==b.shape,(index,key,a.shape,b.shape)
    errors[str(index)][f'bundle_{key}']=float((a.double()-b.double()).abs().max())
    assert errors[str(index)][f'bundle_{key}']<2e-6,(index,key,errors[str(index)][f'bundle_{key}'])
  physical=raw[index][0]
  for key in ['reference_context_positions','reference_context_edge_attr']:
   a=getattr(control,key); b=physical.x if key.endswith('positions') else None
   if b is not None:
    errors[str(index)][f'raw_inverse_{key}']=float((a[:,:2].double()-b[:,:2].double()).abs().max())
    assert errors[str(index)][f'raw_inverse_{key}']<2e-6,(index,key,errors[str(index)][f'raw_inverse_{key}'])
  # 93 is the reused case and must also agree with its previous high-p prepare.
  if index==93:
   old93=torch.load(Path('notebooks/results/high_pratio_engineering/study_v1/prepared/reid/prepare/prepared.pt'),weights_only=False,map_location='cpu')['references'][93]
   for key in keys:
    a,b=getattr(control,key),getattr(old93,key)
    errors[str(index)][f'prior93_{key}']=float((a.double()-b.double()).abs().max())
    assert errors[str(index)][f'prior93_{key}']<2e-6,(index,key,errors[str(index)][f'prior93_{key}'])
 scale=list(data['scales'].values())[0]
 assert all(abs(float(v)-scale)<1e-8 for v in data['scales'].values())
 data['references']={i:normalize(i) for i in ids}
 data['physical']={i:deepcopy(raw[i][0]) for i in ids}
 data['scales']={i:scale for i in ids}
 folder=out/'reid'/'prepare';folder.mkdir(parents=True,exist_ok=False)
 torch.save(data,folder/'prepared.pt')
 info.update(cohort_path=str(cohort_path.resolve()),cohort_sha256=sha(cohort_path),cohort_ids_in_validation_order=ids,
     preprocessing_parity=errors,original_dataset=str(path),original_dataset_sha256=sha(path),old_prepared_sha256=sha(old/'prepared.pt'),
     split='all and only positive-p existing Reid validation rows, fixed before preparation; validation/development cohort, not a reserved final test',
     frame_access='preparation uses only original frame0; no response fitting or new physics before the frozen final manifest')
 (folder/'preparation.json').write_text(json.dumps(info,indent=2)+'\n')
 print('reid',ids,'prepared',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
