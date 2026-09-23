#!/usr/bin/env python3
"""Collect every predeclared outcome, including physical and geometry failures."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from lss.engineering.planarity import PlanarConstraints
from lss.engineering.evaluation import sha256


def contacts(path):
    graph=torch.load(path,weights_only=False,map_location='cpu')
    p=PlanarConstraints(graph.x[:,:2].numpy(),graph.edge_index.numpy(),[graph.box.x,graph.box.y],.05)
    return int(len(p.violating_pairs(graph.x[:,:2].numpy(),enforce_clearance=False)))


def main(output):
    output=Path(output);recipe=json.loads((output/'cohort_and_recipe.json').read_text())
    rows=[];gifs=[]
    for declared in recipe['rows']:
        case=output/'cases'/f"{declared['cohort']}_{declared['index']}"
        outcome=json.loads((case/'outcome.json').read_text())
        total_seeds=1 if declared['cohort']=='same10_reid' else 3
        row=dict(source=declared['source'],index=declared['index'],temperature=declared['temperature'],status=outcome['status'],
            total_seeds=total_seeds,original_p_ratio=np.nan,optimized_p_ratio=np.nan,original_p_sd=np.nan,optimized_p_sd=np.nan,
            valid_original_seeds=0,valid_optimized_seeds=0,initial_design_contacts=len(outcome['periodic_pristine_violating_pairs']),
            final_design_contacts=np.nan,trajectory_contact_frames_original=np.nan,trajectory_contact_frames_optimized=np.nan,
            trajectory_frames_total=total_seeds*(121 if total_seeds==1 else 200))
        result_file=case/'final_verification/results.csv'
        if outcome['status']=='frozen':
            designs={r['name']:r for r in outcome['designs']}
            for design in designs.values():assert sha256(Path(design['graph_path']))==design['graph_sha256']
            row['initial_design_contacts']=contacts(designs['original']['graph_path'])
            row['final_design_contacts']=contacts(designs['optimized']['graph_path'])
            row['status']='verification_failed'
            if result_file.is_file():
                result=pd.read_csv(result_file)
                for kind in ['original','optimized']:
                    g=result[result.kind.eq(kind)]
                    good=g[g.status.eq('completed')&np.isfinite(g.physical_p_ratio)]
                    row[f'valid_{kind}_seeds']=len(good)
                    if len(good)==total_seeds:
                        row[f'{kind}_p_ratio']=float(good.physical_p_ratio.mean())
                        row[f'{kind}_p_sd']=float(good.physical_p_ratio.std(ddof=1)) if total_seeds>1 else 0.
                        row[f'trajectory_contact_frames_{kind}']=int(good.trajectory_contact_frames.sum())
                if row['valid_original_seeds']==row['valid_optimized_seeds']==total_seeds:
                    row['status']='completed'
                    if row['final_design_contacts'] or row['trajectory_contact_frames_optimized'] or row['trajectory_contact_frames_original']:
                        row['status']='trajectory_crossing'
        rows.append(row)
        gif_file=case/'gif.json'
        if gif_file.is_file():gifs.append(json.loads(gif_file.read_text()))
        else:gifs.append(dict(source=declared['source'],index=declared['index'],temperature=declared['temperature'],
                              status='unavailable',reason=row['status']))
    table=pd.DataFrame(rows);assert len(table)==40 and not table.duplicated(['source','index']).any()
    table.to_csv(output/'summary.csv',index=False)
    (output/'gif_index.json').write_text(json.dumps(dict(gifs=gifs),indent=2)+'\n')
    report=dict(expected_networks=40,observed_networks=len(table),status_counts=table.groupby(['source','status']).size().to_string(),
        valid_networks=int(table.status.eq('completed').sum()),completed_gifs=sum(g['status']=='completed' for g in gifs),
        summary_sha256=sha256(output/'summary.csv'),feedback_to_optimizer=False)
    (output/'collection_completed.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.output)
