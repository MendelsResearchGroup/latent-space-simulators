"""Final-only finite-temperature verification under a separately declared protocol."""
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project')
sys.path[:0]=[str(DONOR.parent/'MetaForge/src')]
import auxetic
from auxetic.utils import write_network
from auxetic.scripts import ElasticScript
from graph_utils import directional_side_indices_from_box

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 return None if isinstance(x,float) and not np.isfinite(x) else x
def dump(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def read_frames(path,nodes):
 frames=[];boxes=[];steps=[]
 with Path(path).open() as f:
  while line:=f.readline():
   if line.strip()!='ITEM: TIMESTEP':raise ValueError('Unexpected dump header')
   steps.append(int(f.readline()));assert f.readline().strip()=='ITEM: NUMBER OF ATOMS';n=int(f.readline());assert n==nodes
   assert f.readline().startswith('ITEM: BOX BOUNDS');bounds=np.array([[float(v) for v in f.readline().split()[:2]] for _ in range(3)])
   names=f.readline().split()[2:];atoms=np.array([[float(v) for v in f.readline().split()] for _ in range(n)]);atoms=atoms[np.argsort(atoms[:,names.index('id')])]
   assert np.array_equal(atoms[:,names.index('id')],np.arange(1,n+1))
   frames.append(atoms[:,[names.index('xu'),names.index('yu')]]);boxes.append(bounds);
 return np.array(frames),np.array(boxes),np.array(steps)
def script(p,T,seed):
 return f'''dimension 2
include init.mod
include potential.mod
# init.mod uses an athermal dummy mass: explicitly replace it.
mass * {p['mass_amu']}
timestep {p['timestep_ps']}
thermo 1000
thermo_style custom step temp pe press pxx pyy lx ly ke etotal
fix planar all enforce2d
fix relax all box/relax y 0.0 vmax 0.001
minimize 0.0 1.0e-10 10000 100000
unfix relax
unfix planar
velocity all create {T} {seed} mom yes rot no dist gaussian
velocity all set NULL NULL 0.0
velocity all scale {T}
fix bath all npt temp {T} {T} {p['thermostat_damping_ps']} y 0.0 0.0 {p['barostat_damping_ps']} couple none tchain 3 pchain 3
fix planar all enforce2d
run {p['equilibration_steps']}
unfix planar
unfix bath
reset_timestep 0
fix bath all npt temp {T} {T} {p['thermostat_damping_ps']} y 0.0 0.0 {p['barostat_damping_ps']} couple none tchain 3 pchain 3
fix compress all deform 1 x scale {p['scale_x']} remap x units box
fix planar all enforce2d
dump trajectory all custom {p['dump_every']} trajectory.dump id xu yu
dump_modify trajectory sort id format float %.15g
run {p['compression_steps']}
'''
def verify(out,temperature=None,index=None):
 torch.set_num_threads(1);out=Path(out);m=json.loads((out/'frozen_designs.json').read_text());assert m['recipe_sha256']==sha(out/'recipe.json');assert len(m['records'])==210
 protocol_path=out/'thermal_protocol.json';p=json.loads(protocol_path.read_text());assert p['simulation_kind']=='new_declared_thermal_protocol_not_dataset_reproduction'
 for r in m['records']:assert sha(r['physicalpath'])==r['sha256'] and sha(r['edit_file'])==r['edit_sha256']
 metric_path=out/'code/thermal_metric.py';s=importlib.util.spec_from_file_location('metric',metric_path);metric=importlib.util.module_from_spec(s);s.loader.exec_module(metric)
 assert sha(metric_path)==p['metric_sha256'];assert sha(__file__)==p['verifier_sha256']
 lmp=DONOR/'.runtime/lammps/bin/lmp';assert sha(lmp)==p['lammps_sha256']
 assert sha(Path(sys.modules[ElasticScript.__module__].__file__))==p['elastic_script_sha256']
 assert sha(Path(sys.modules[write_network.__module__].__file__))==p['network_export_sha256']
 selected=[r for r in m['records'] if (temperature is None or float(r['temperature'])==float(temperature)) and (index is None or int(r['index'])==index)]
 assert len(selected)==(210 if index is None else 7)
 tag='all' if index is None else f'id{index}';folder=out/'thermal_physics'/tag;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'verification_recipe.json').exists():raise RuntimeError('Refuse overwrite of thermal verification attempt')
 dump(folder/'verification_recipe.json',dict(job_id=os.environ.get('PBS_JOBID'),manifest_sha256=sha(out/'frozen_designs.json'),protocol_sha256=sha(protocol_path),started_at_unix=time.time(),temperature=temperature,index=index,expected=len(selected)*len(p['thermal_seeds']),feedback_to_optimizer=False))
 rows=[]
 for r in selected:
  graph=torch.load(r['physicalpath'],weights_only=False,map_location='cpu');sides=directional_side_indices_from_box(graph,quantile=.1)
  for seed in p['thermal_seeds']:
   run=folder/f"{r['index']}_{r['variant']}_s{seed}";run.mkdir(exist_ok=False)
   row={**r,'thermal_seed':seed,'status':'failed','physical_p_ratio':float('nan'),'run_dir':str(run.resolve()),'valid_frames':0,'total_frames':200}
   try:
    write_network(run,copy.deepcopy(graph),mass=p['mass_amu'],angles=0.,box=graph.box);ElasticScript('network.lmp').write_to_file(str(run));(run/'in.thermal').write_text(script(p,r['temperature'],seed))
    with (run/'stdout.log').open('w') as stdout,(run/'stderr.log').open('w') as stderr:
     subprocess.run([str(lmp),'-screen','none','-in','in.thermal'],cwd=run,check=True,stdout=stdout,stderr=stderr,timeout=p['per_simulation_timeout_seconds'])
    pos,boxes,steps=read_frames(run/'trajectory.dump',len(graph.x));assert len(pos)==200;assert np.array_equal(steps,np.arange(200)*p['dump_every'])
    scale=(boxes[-1,0,1]-boxes[-1,0,0])/(boxes[0,0,1]-boxes[0,0,0]);assert abs(scale-p['scale_x'])<1e-7
    fit=metric.frame_time_ols_p_ratio(pos,sides);row.update(physical_p_ratio=fit['p_ratio'],valid_frames=fit['valid_frames'],total_frames=fit['total_frames'],fit_r2_x=fit['fit_r2_x'],fit_r2_y=fit['fit_r2_y'],slope_x=fit['slope_x'],slope_y=fit['slope_y'],fitted_axial_total_strain=fit['fitted_axial_total_strain'],actual_box_scale_x=scale)
    np.savez_compressed(run/'trajectory.npz',positions=pos,boxes=boxes,steps=steps,strain_x=fit['strain_x'],strain_y=fit['strain_y'])
    if not np.isfinite(fit['p_ratio']):raise ValueError('Undefined all-frame linear-fit p-ratio')
    row['status']='completed'
   except Exception as exc:
    row['error']=str(exc);row['log_tail']=(run/'log.lammps').read_text()[-2500:] if (run/'log.lammps').exists() else ''
   dump(run/'result.json',row);rows.append(row);pd.DataFrame(rows).to_csv(folder/'results.csv',index=False);print(r['index'],r['variant'],seed,row['status'],row['physical_p_ratio'],flush=True)
 dump(folder/'verification_completed.json',dict(total=len(rows),valid=sum(r['status']=='completed' for r in rows),manifest_sha256=sha(out/'frozen_designs.json'),protocol_sha256=sha(protocol_path),feedback_to_optimizer=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--temperature',type=float);a.add_argument('--index',type=int);args=a.parse_args();verify(args.output,args.temperature,args.index)
