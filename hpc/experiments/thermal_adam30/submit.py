#!/usr/bin/env python3
"""Snapshot recipe only; --submit is deliberately required for PBS writes."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'notebooks/results/thermal_adam30/study_v1'; DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project'); PROTO=ROOT/'notebooks/results/thermal_joint_engineering/study_v1_retry/thermal_protocol.json'; METRIC=ROOT/'hpc/experiments/ml_only_engineering/thermal_metric.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def qsub(stage,cpu,mem,wall,dep=None,temp=None):
 env=f'LSS_STAGE={stage},LSS_OUTPUT={OUT},LSS_RUNNER={OUT}/code/'+('prepare.py' if stage=='prepare' else ('verify.py' if stage=='verify' else ('collect.py' if stage=='collect' else 'run.py')))
 if temp is not None:env+=f',LSS_TEMPERATURE={temp[1]:g},LSS_INDEX={temp[0]}'
 cmd=['qsub','-q','mendels_q','-l',f'select=1:ncpus={cpu}:mem={mem}','-l','place=free:shared','-l',f'walltime={wall}','-v',env]
 if dep:cmd+=['-W',f'depend={dep}']
 return subprocess.check_output(cmd+['-N',f'tadam30_{stage}',str(OUT/'code/run.pbs')],text=True).strip()
def main():
 a=argparse.ArgumentParser();a.add_argument('--submit',action='store_true');x=a.parse_args()
 if not (OUT/'cohort.json').is_file():raise RuntimeError('root-owned cohort required')
 if (OUT/'code').exists():raise RuntimeError('refuse overwrite')
 code=OUT/'code';code.mkdir()
 for n in ['prepare.py','run.py','verify.py','collect.py','run.pbs']:shutil.copy2(HERE/n,code/n)
 shutil.copy2(METRIC,code/'thermal_metric.py');shutil.copy2(OUT/'cohort.json',code/'cohort.json'); protocol=json.loads(PROTO.read_text());protocol.update(expected_designs=210,expected_evaluations=630,manifest_sha256='set after global optimizer freeze',limitations='New kinetic/thermostat/barostat choices, not a recovered original data-generation protocol. Thirty frozen development networks, five per temperature; mixed-T training/calibration overlap audit is reported separately. Velocity seeds are within-network thermal replicates, not independent network/model replications. Preserve failures; no final-physics selection.');(OUT/'thermal_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 deps=[code/n for n in ['prepare.py','run.py','verify.py','collect.py','run.pbs','thermal_metric.py','cohort.json']]+[OUT/'thermal_protocol.json',DONOR/'notebooks/results/network_design/calibrated_direction_v1/depablo_low_temp/prepare/prepared.pt']
 manifest=dict(study='thermal_adam30/study_v1',dependencies={str(p):sha(p) for p in deps},resources=dict(prepare='4CPU 2GB 20m',optimize='4CPU 2GB 2h',verify='30x 1CPU 2GB 4h'),order='prepare -> optimize global 210-design freeze -> 30 per-ID verify jobs -> collect')
 (code/'dependency_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if not x.submit:print(json.dumps(dict(status='snapshotted_not_submitted',manifest=str(code/'dependency_manifest.json')),indent=2));return
 ledger=dict(verify={},status='submitting'); pre=qsub('prepare',4,'2gb','00:20:00');ledger['prepare']=pre;(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n');opt=qsub('optimize',4,'2gb','02:00:00',f'afterok:{pre}');ledger['optimize']=opt;(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
 for r in json.loads((OUT/'cohort.json').read_text())['rows']:
  i=int(r['index']);t=float(r['temperature']);ledger['verify'][str(i)]=qsub('verify',1,'2gb','04:00:00',f'afterok:{opt}',(i,t));(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
 dep='afterany:'+':'.join(ledger['verify'].values());ledger['collect']=qsub('collect',1,'2gb','00:20:00',dep);ledger['status']='collector_afterany_submitted';(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
 (OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
if __name__=='__main__':main()
