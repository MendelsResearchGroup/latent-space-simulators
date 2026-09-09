"""Prepare response-blind, temperature-stratified validation originals; no fitting."""
from pathlib import Path
import argparse,sys,json,hashlib,copy
import torch
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project');CODE=DONOR/'notebooks/results/network_design/code_endpoint_v1'
sys.path[:0]=[str(DONOR.parent/'MetaForge/src'),str(CODE/'src')]
import auxetic
from lss.data import normalize_dataset_edges,normalize_trajectory_to_reference_box,tag_simulation_source
SELECTED={47:.1,68:1.,140:5.,161:10.,242:20.,268:30.}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(out):
 torch.set_num_threads(4)
 prior=DONOR/'notebooks/results/network_design/calibrated_direction_v1/depablo_low_temp/prepare'
 data=torch.load(prior/'prepared.pt',weights_only=False,map_location='cpu')
 path=DONOR/'data/depablo-10k-mix-temp.pt';raw=torch.load(path,weights_only=False,map_location='cpu',mmap=True)
 refs={};physical={};temperatures={};errors={}
 for i,T in SELECTED.items():
  assert float(raw[i][0].temperature)==T
  physical[i]=copy.deepcopy(raw[i][0]);sim=[copy.deepcopy(raw[i][0])]
  normalize_dataset_edges([sim],edge_multiplicity=1,edge_vector_dim=2);normalize_trajectory_to_reference_box(sim,pos_dim=2);tag_simulation_source(sim,'depablo_mixed_temp');ref=sim[0]
  inverse=ref.x[:,:2]*ref.reference_box_half_extent[:2]+ref.reference_box_center[:2]
  err=float((inverse-physical[i].x[:,:2]).abs().max());assert err<2e-6
  refs[i]=ref;temperatures[i]=T;errors[i]=err
 data['references']=refs;data['physical']=physical;data['temperatures']=temperatures
 # Legacy runner field is not used to select the thermal verification protocol.
 data['scales']={i:float(next(iter(data['scales'].values()))) for i in refs}
 folder=out/'depablo_mixed_temp/prepare';folder.mkdir(parents=True,exist_ok=False);torch.save(data,folder/'prepared.pt')
 info=dict(source='depablo_mixed_temp',design_ids=list(SELECTED),temperatures=temperatures,selection='first validation ID per temperature in existing temporal_linearity_check recipe order; no response screening',split='existing mixed-T validation only; no final-test rows; source excluded from frozen AE and low-T readout fitting',frame_access='only original frame0 at selected validation IDs; metadata tensor containers restored by mmap',original_dataset=str(path),original_dataset_sha256=sha(path),old_prepared_sha256=sha(prior/'prepared.pt'),old_preparation=json.loads((prior/'preparation.json').read_text()),physical_inverse_max_errors=errors,model_seed=786,objective='unchanged historical low-T z1 response-calibrated affine readout; zero-shot directional transfer, not a calibrated thermal or position-rollout prediction',script_sha256=sha(__file__))
 (folder/'preparation.json').write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
