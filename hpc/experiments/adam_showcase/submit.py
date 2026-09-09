#!/usr/bin/env python3
import argparse,hashlib,json,shutil,subprocess,re
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'notebooks/results/adam_showcase';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(stage,key,dep=None):
 e=f'SHOWCASE_OUT={OUT},SHOWCASE_STAGE={stage},SHOWCASE_KEY={key}'
 cmd=['qsub','-q','mendels_q','-l','select=1:ncpus=1:mem=2gb','-l','place=free:shared','-l','walltime=00:20:00','-v',e,'-N','adam_gif']
 if dep:cmd+=['-W',f'depend={dep}']
 result=subprocess.check_output(cmd+[str(OUT/'code/run.pbs')],text=True).strip()
 if not re.fullmatch(r'\d+(?:\.[A-Za-z0-9_-]+)?',result):raise RuntimeError(f'qsub did not return a job ID: {result}')
 return result
def main():
 a=argparse.ArgumentParser();a.add_argument('--submit',action='store_true');x=a.parse_args()
 if OUT.exists():raise RuntimeError('refuse overwrite')
 r=pd.read_csv(ROOT/'notebooks/results/reid_joint_sweep/study_v1/results.csv');cfg='lr0.08_b0.25_p50_mask_blocking_nodes'; ids=[93,84,96,97,52];cases={'reid':{},'render':{}}
 for i in ids:
  oo=r[(r['index']==i)&(r.config_id=='original')];zz=r[(r['index']==i)&(r.config_id==cfg)&(r.kind=='ml')]
  if len(oo)!=1 or len(zz)!=1 or oo.iloc[0].status!='completed' or zz.iloc[0].status!='completed':raise RuntimeError(f'nonunique/incomplete Reid selection {i}')
  o,z=oo.iloc[0],zz.iloc[0];cases['reid'][str(i)]={'original':o[['physicalpath','physical_p_ratio','sha256']].to_dict(),'optimized':z[['physicalpath','physical_p_ratio','sha256']].to_dict()};cases['render'][str(i)]={'source':'reid','index':i,'title':f'Reid {i}: retrospective development recipe','original_p':float(o.physical_p_ratio),'optimized_p':float(z.physical_p_ratio),'filename':f'reid_{i}.gif','original_graph':o.physicalpath,'optimized_graph':z.physicalpath}
 t=pd.read_csv(ROOT/'notebooks/results/thermal_adam30/study_v1/thermal_results.csv')
 for i,temp in [(234,20.),(262,30.)]:
  oo=t[(t['index']==i)&(t.kind=='original')&(t.thermal_seed==786)];zz=t[(t['index']==i)&(t.variant=='joint_A_r4_b0.25')&(t.thermal_seed==786)]
  if len(oo)!=1 or len(zz)!=1 or oo.iloc[0].status!='completed' or zz.iloc[0].status!='completed':raise RuntimeError(f'nonunique/incomplete thermal selection {i}')
  o,z=oo.iloc[0],zz.iloc[0];key=f'thermal_{i}';cases['render'][key]={'source':'depablo_mixed_temp','index':i,'temperature':temp,'thermal_seed':786,'title':f'T={temp:g}, seed 786 (actual replay)','original_p':float(o.physical_p_ratio),'optimized_p':float(z.physical_p_ratio),'filename':f'{key}.gif','original_trajectory':str(Path(o.run_dir)/'trajectory.npz'),'optimized_trajectory':str(Path(z.run_dir)/'trajectory.npz'),'original_graph':o.physicalpath,'optimized_graph':z.physicalpath}
 OUT.mkdir();(OUT/'assets').mkdir();(OUT/'inputs.json').write_text(json.dumps(cases,indent=2)+'\n');(OUT/'code').mkdir()
 for n in ['replay.py','render.py','run.pbs']:shutil.copy2(HERE/n,OUT/'code'/n)
 deps=[OUT/'inputs.json',OUT/'code/replay.py',OUT/'code/render.py',ROOT/'notebooks/results/reid_joint_sweep/study_v1/results.csv',ROOT/'notebooks/results/thermal_adam30/study_v1/thermal_results.csv',ROOT/'../DL-course-project/notebooks/network_design/endpoint_physics.py',ROOT/'../DL-course-project/.runtime/lammps/bin/lmp',ROOT/'../MetaForge/src/auxetic/utils.py']
 for key,c in cases['render'].items():
  deps += [Path(c['original_graph']),Path(c['optimized_graph'])]
  if c['source']!='reid':deps += [Path(c['original_trajectory']),Path(c['optimized_trajectory'])]
 (OUT/'provenance.json').write_text(json.dumps({'recipe':cfg,'reid_ids':ids,'thermal_ids':[234,262],'dependencies':{str(p):sha(p) for p in deps},'duration_seconds':4,'reid_frames':121,'thermal_frames':200},indent=2)+'\n')
 if not x.submit:return
 ledger={'jobs':{},'collection_job':None,'status':'submitting'};(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
 for i in ids:
  ledger['jobs'][str(i)]=q('reid',str(i));(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
 for i in [234,262]:
  key=f'thermal_{i}';ledger['jobs'][key]=q('thermal',key);(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
 ledger['collection_job']=q('collect','collect','afterok:'+':'.join(ledger['jobs'].values()));ledger['status']='submitted';(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
if __name__=='__main__':main()
