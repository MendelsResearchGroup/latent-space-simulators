#!/usr/bin/env python3
"""Collect exact accepted cells; never pool architectures or failed attempts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def read_json(path):
    try: return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): return None
def value(args, flag): return args[args.index(flag)+1] if flag in args else None
def run_directory(base, study, row, version):
    args=row["arguments"]; seed=value(args,"--seed"); smoke="_smoke" if row.get("smoke") else ""
    if study=="reference_simplification": name=f"r{value(args,'--reference-dim')}_d{value(args,'--latent-dim')}_s{seed}_{version}{smoke}"
    else: name=f"{value(args,'--variant')}_s{seed}_{version}{smoke}"
    return base/study/name
def terminal_state(out):
    completed=out/"completed.json"; failed=out/"failed.json"
    if completed.exists(): return "completed",read_json(completed)
    if failed.exists(): return "failed",read_json(failed)
    return "incomplete",None
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results-root",required=True);ap.add_argument("--study",required=True);ap.add_argument("--code-version",required=True);a=ap.parse_args()
    base=Path(a.results_root); ledger=base/a.study/"jobs.jsonl"; attempts=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []
    accepted=[x for x in attempts if x.get("kind")=="run" and x.get("job_id") and x.get("code_version")==a.code_version]
    records=[];frames=[];parse_failures=[]
    for row in accepted:
        args=row["arguments"]; out=run_directory(base,a.study,row,a.code_version)
        state_name,state=terminal_state(out)
        label={"run_directory":str(out),"job_id":row["job_id"],"seed":value(args,"--seed"),"variant":value(args,"--variant"),"reference_dim":value(args,"--reference-dim"),"latent_dim":value(args,"--latent-dim"),"code_version":a.code_version,"smoke":row.get("smoke")}
        records.append({**label,"state":state_name,"result":state})
        if a.study=="reference_simplification":
            candidates=((out/"ae_all_validation_rows.csv","ae"),*( (out/stage/"all_validation_rows.csv",stage) for stage in ("context16","context0") ))
        elif a.study in ("compact_lj_reconstruction","compact_lj_spatial_decoder"):
            candidates=((out/"source_frame_rows.csv","reconstruction"),)
        else:
            candidates=((out/"validation_rows.csv","ae"),)
        for path,stage in candidates:
            if not path.exists(): continue
            try:
                frame=pd.read_csv(path)
                for key,val in label.items(): frame[key]=val
                frame["stage"]=stage
                frames.append(frame)
            except Exception as error: parse_failures.append({**label,"file":str(path),"error":repr(error)})
    study_base=base/a.study
    (study_base/"collection_status.json").write_text(json.dumps({"study":a.study,"code_version":a.code_version,"accepted_runs":len(accepted),"qsub_failed_attempts":sum(x.get("status")=="qsub_failed" for x in attempts),"records":records,"csv_parse_failures":parse_failures},indent=2,default=str)+"\n")
    if frames: pd.concat(frames,ignore_index=True).to_csv(study_base/"accepted_run_rows.csv",index=False)
if __name__=="__main__":main()
