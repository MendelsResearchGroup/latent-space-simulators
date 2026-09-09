#!/usr/bin/env python3
"""Replay one frozen Reid original/design pair and retain every physical frame."""
import argparse, hashlib, json, os, subprocess, sys
from copy import deepcopy
from pathlib import Path
import numpy as np, torch
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project');sys.path[:0]=[str(DONOR.parent/'MetaForge/src'),str(DONOR/'notebooks/network_design')]
from auxetic.utils import write_network
from auxetic.scripts import ElasticScript
from endpoint_physics import read_endpoint
from graph_utils import calc_p_ratio_rollout_sides
from graph_utils.box import Box
SCALE=.9700106999999999; STEPS=120
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse(path,g):
 lines=Path(path).read_text().splitlines();i=0;out=[];boxes=[];graphs=[]
 while i<len(lines):
  if lines[i]!='ITEM: TIMESTEP':raise ValueError('bad dump')
  n=int(lines[i+3]);i+=4; bounds=[list(map(float,lines[i+j].split()[:2])) for j in range(1,4)];i+=4; names=lines[i].split()[2:];i+=1;a=np.array([list(map(float,lines[i+j].split())) for j in range(n)]);i+=n;a=a[np.argsort(a[:,names.index('id')])];pos=a[:,[names.index('xu'),names.index('yu')]];h=deepcopy(g);h.x=torch.tensor(pos,dtype=torch.float64);h.box=Box(*bounds[0],*bounds[1],*bounds[2]);out.append(pos);boxes.append(bounds[:2]);graphs.append(h)
 return np.asarray(out),np.asarray(boxes),graphs
def run(graph,path,expected,expected_sha):
 if sha(graph)!=expected_sha: raise RuntimeError('selected frozen graph hash mismatch before LAMMPS')
 g=torch.load(graph,weights_only=False,map_location='cpu');path.mkdir(parents=True,exist_ok=True);write_network(path,deepcopy(g),mass=1e6,angles=0.,box=g.box);ElasticScript('network.lmp').write_to_file(str(path));step=SCALE**(1/STEPS)
 script=f'''dimension 2
include init.mod
include potential.mod
fix planar all enforce2d
fix relax all box/relax y 0.0 vmax 0.001
minimize 0.0 1e-10 10000 100000
unfix relax
write_dump all custom frame_0.dump id xu yu modify sort id format float %.15g
variable increment loop {STEPS}
label compression
change_box all x scale {step:.17g} remap units box
fix relax all box/relax y 0.0 vmax 0.001
minimize 0.0 1e-10 10000 100000
unfix relax
write_dump all custom frame_${{increment}}.dump id xu yu modify sort id format float %.15g
next increment
jump SELF compression
write_dump all custom final.dump id x y xu yu modify sort id format float %.15g
''';(path/'in.replay').write_text(script);done=subprocess.run([str(DONOR/'.runtime/lammps/bin/lmp'),'-screen','none','-in','in.replay'],cwd=path,check=True,capture_output=True,text=True);(path/'stdout.txt').write_text(done.stdout);(path/'stderr.txt').write_text(done.stderr)
 z=[parse(path/f'frame_{i}.dump',g) for i in range(121)];x=np.concatenate([q[0] for q in z]);b=np.concatenate([q[1] for q in z]);frames=[q[2][0] for q in z];p=np.array([np.nan]+[float(calc_p_ratio_rollout_sides([frames[0],frames[i]],-1)) for i in range(1,121)]);endpoint=float(calc_p_ratio_rollout_sides([frames[0],read_endpoint(path/'final.dump',g)],-1));err=abs(endpoint-expected)
 if len(x)!=121 or abs(p[-1]-endpoint)>1e-12 or err>1e-6:raise RuntimeError(f'parity frames={len(x)} err={err}')
 return x,b,p,dict(graph_sha256=sha(graph),expected_p=expected,observed_p=endpoint,endpoint_parity_error=err,frame_count=121,scale_x=SCALE,steps=120,pbs_job_id=os.environ.get('PBS_JOBID'),runner_sha256=sha(Path(__file__)),endpoint_helper_sha256=sha(DONOR/'notebooks/network_design/endpoint_physics.py'),lammps_sha256=sha(DONOR/'.runtime/lammps/bin/lmp'),exporter_sha256=sha(DONOR.parent/'MetaForge/src/auxetic/utils.py'),stdout_path=str((path/'stdout.txt').resolve()),stderr_path=str((path/'stderr.txt').resolve()))
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--id',type=int,required=True);x=a.parse_args();case=json.loads((x.output/'inputs.json').read_text())['reid'][str(x.id)];results={}
 for label in ['original','optimized']:
  pos,boxes,p,meta=run(case[label]['physicalpath'],x.output/'replays'/f'{x.id}_{label}',case[label]['physical_p_ratio'],case[label]['sha256']);np.savez_compressed(x.output/'assets'/f'reid_{x.id}_{label}.npz',positions=pos,boxes=boxes,pratio=p);results[label]=meta
 (x.output/'assets'/f'reid_{x.id}_replay.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
