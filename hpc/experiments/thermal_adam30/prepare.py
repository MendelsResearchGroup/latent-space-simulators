#!/usr/bin/env python3
"""Prepare only pristine direct mixed-T frame-0 graphs for thermal Adam30."""
from pathlib import Path
import argparse, copy, hashlib, json, sys
import torch
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project'); CODE=DONOR/'notebooks/results/network_design/code_endpoint_v1'
sys.path[:0]=[str(DONOR.parent/'MetaForge/src'),str(CODE/'src')]
import auxetic # noqa
from lss.data import normalize_dataset_edges,normalize_trajectory_to_reference_box,tag_simulation_source
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(out):
 torch.set_num_threads(4); out=Path(out); cohort_path=out/'cohort.json'; cohort=json.loads(cohort_path.read_text()); rows=cohort['rows']
 ids=[int(r['index']) for r in rows]; temps=[float(r['temperature']) for r in rows]
 assert len(ids)==30 and len(set(ids))==30 and all(r['split'] in ('train','val') for r in rows)
 assert {t:temps.count(t) for t in cohort['temperatures']}=={.1:5,1.:5,5.:5,10.:5,20.:5,30.:5}
 raw_path=Path(cohort['dataset']); raw=torch.load(raw_path,weights_only=False,map_location='cpu',mmap=True)
 prior=DONOR/'notebooks/results/network_design/calibrated_direction_v1/depablo_low_temp/prepare'; data=torch.load(prior/'prepared.pt',weights_only=False,map_location='cpu')
 refs={}; physical={}; errors={}
 for r in rows:
  i=int(r['index']); assert float(raw[i][0].temperature)==float(r['temperature'])
  sim=[copy.deepcopy(raw[i][0])]; normalize_dataset_edges([sim],edge_multiplicity=1,edge_vector_dim=2);normalize_trajectory_to_reference_box(sim,pos_dim=2);tag_simulation_source(sim,'depablo_mixed_temp'); refs[i]=sim[0]; physical[i]=copy.deepcopy(raw[i][0])
  inv=refs[i].x[:,:2]*refs[i].reference_box_half_extent[:2]+refs[i].reference_box_center[:2]; errors[str(i)]=float((inv-physical[i].x[:,:2]).abs().max());assert errors[str(i)]<2e-6
 data['references']=refs;data['physical']=physical;data['temperatures']={int(r['index']):float(r['temperature']) for r in rows}; data['scales']={i:float(next(iter(data['scales'].values()))) for i in ids}
 folder=out/'prepared'/'depablo_mixed_temp'/'prepare';folder.mkdir(parents=True,exist_ok=False);torch.save(data,folder/'prepared.pt')
 info=dict(cohort_path=str(cohort_path.resolve()),cohort_sha256=sha(cohort_path),cohort_ids_in_order=ids,temperatures=data['temperatures'],split='frozen mixed-T cohort: 20 validation and 10 train rows, no reserved test; response-blind original order',frame_access='only selected original frame0 via mmap; direct mixed-T geometry, never corresponding low-T geometry',dataset=str(raw_path),dataset_sha256=sha(raw_path),historical_lowT_prepared=str(prior/'prepared.pt'),historical_lowT_prepared_sha256=sha(prior/'prepared.pt'),normalization='normalize_dataset_edges then normalize_trajectory_to_reference_box, matching prepare_thermal.py',physical_inverse_max_errors=errors,model_seed=786,objective='frozen low-T-trained 2D AE786 plus historical low-T z1 response-calibrated readout; no thermal-response fitting or physics')
 (folder/'preparation.json').write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
