"""Save inline overnight plots and append complete, source-wise development summaries."""
from pathlib import Path
import hashlib,json,os
import nbformat
from nbclient import NotebookClient
import pandas as pd
import numpy as np
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators')
OUT=ROOT/'notebooks/results/overnight_engineering_report'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True,parents=True)
 notebook=ROOT/'notebooks/engineering/01_adam_engineering.ipynb'
 nb=nbformat.read(notebook,as_version=4)
 NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(notebook.parent)}}).execute()
 nbformat.write(nb,notebook)
 if not any('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[])): raise RuntimeError('Notebook missing inline plots')
 entry='\n\n### Overnight joint-Adam collection — '+pd.Timestamp.now(tz='UTC').isoformat()+'\n\n'
 status={'job_id':os.environ.get('PBS_JOBID'),'notebook':str(notebook),'notebook_sha256':sha(notebook),'inline_pngs':sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))}
 reid=ROOT/'notebooks/results/reid_joint_sweep/study_v1'
 if (reid/'collection_completed.json').exists():
  done=json.loads((reid/'collection_completed.json').read_text());status['reid']=done
  summary=pd.read_csv(reid/'summary.csv');entry+=f"Reid: {done['valid']}/{done['total']} valid finalrecords including explicitlyreused30baseline records; all24predeclaredrecipes reported. ResultsSHA `{sha(reid/'results.csv')}`.\n\n| Recipe | Valid/10 | Negative/10 | Mean physical p | Mean change | Mean steps |\n| --- | --- | --- | --- | --- | --- |\n"
  for r in summary.to_dict('records'):entry+=f"| {r['config_id']} | {r['valid']}/{r['total']} | {r['negative']}/10 | {r['mean_p']:.6f} | {r['mean_change']:.6f} | {r['mean_actual_steps']:.1f} |\n"
 else:status['reid']={'status':'collection pending/incomplete'};entry+='Reid globalcollection pending/incomplete; no missingcell treatedas success.\n'
 thermal=ROOT/'notebooks/results/thermal_adam30/study_v1'
 if (thermal/'thermal_collect_status.json').exists():
  done=json.loads((thermal/'thermal_collect_status.json').read_text());status['thermal']=done
  entry+=f"\nThermal: {done['valid']}/{done.get('expected',630)} valid finalchecks; observed {done.get('observed','unknown')}, missing {done.get('missing_keys','unknown')}. ResultsSHA `{sha(thermal/'thermal_results.csv')}`. Per-ID/per-recipe thermal-seedmeans/SD andvalid/total in thermal_summary.csv.\n"
  try: table=pd.read_csv(thermal/'thermal_summary.csv')
  except pd.errors.EmptyDataError:table=pd.DataFrame()
  if len(table):
   originals=table[table.kind.eq('original')].set_index('index').thermal_mean
   entry+='\n| T | Recipe | Networks with3/3valid | Mean of network p means | Negative network means | Positive→negative network means |\n| --- | --- | --- | --- | --- | --- |\n'
   for (T,recipe),g in table[table.kind.eq('ml')].groupby(['temperature','recipe_id']):
    valid=g[(g.thermal_valid==3)&np.isfinite(g.thermal_mean)];initial=valid['index'].map(originals)
    entry+=f"| {T:g} | {recipe} | {len(valid)}/5 | {valid.thermal_mean.mean():.6f} | {int((valid.thermal_mean<0).sum())}/5 | {int(((initial>0)&(valid.thermal_mean<0)).sum())}/{int((initial>0).sum())} |\n"
 else:status['thermal']={'status':'collection pending/incomplete'};entry+='\nThermal collection pending/incomplete.\n'
 entry+='\nThese are retrospective development comparisons on onefrozen2DAE/modelseed786 with historicalresponsecalibratedz1 readouts, not autonomouspositionp-ratioR² or heldoutreliability. Thermalspread isvelocity-seedSD, notmodelseedSD; identical separatelydeclarednewthermalprotocol, notGTgeneratorreplay. Thermalcohort20val+10train,4/30AEtrainingtopology/stiffnessmatches and5/30readoutcalibrationmatches; geometrydiffers. Allphysicalfailures/controls retained, no physicalfeedback duringoptimization orfinal-designselection. Current engineering notebook savedwithinlineEditorialplots.\n'
 key=hashlib.sha256(entry.encode()).hexdigest();status['report_sha256']=key
 # One scheduled reporter owns shared research logs after both collectors.
 for path in ['notebooks/latent_space/experiment_results_index.md','notebooks/latent_space/network_engineering_log.md']:
  with (ROOT/path).open('a') as f:f.write(entry)
 (OUT/'completed.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
if __name__=='__main__':main()
