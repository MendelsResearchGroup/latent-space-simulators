#!/usr/bin/env python3
"""Final-only paired verification and contact audit of globally frozen designs."""
from pathlib import Path
from copy import deepcopy
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from auxetic.scripts import ElasticScript
from auxetic.utils import write_network
from graph_utils import directional_side_indices_from_box
from lss.engineering.evaluation import sha256, verify_frozen_designs
from lss.engineering.planarity import PlanarConstraints


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def contact_audit(positions, boxes, edge_index):
    counts=[]
    for position,bounds in zip(positions,boxes):
        bounds=bounds[:2,:2];length=bounds[:,1]-bounds[:,0]
        wrapped=bounds[:,0]+np.mod(position[:,:2]-bounds[:,0],length)
        planar=PlanarConstraints(wrapped,edge_index,length,.05)
        counts.append(len(planar.violating_pairs(wrapped,enforce_clearance=False)))
    return dict(trajectory_contact_frames=int(np.count_nonzero(counts)),trajectory_total_frames=len(counts),
                trajectory_max_contact_pairs=int(max(counts)),contact_pairs_by_frame=counts)


def main(output,cohort,index):
    torch.set_num_threads(1);output=Path(output)
    gate=json.loads((output/'global_freeze_completed.json').read_text())
    outcomes=json.loads((output/'global_frozen_outcomes.json').read_text())
    assert outcomes['observed']==outcomes['expected']==40
    assert sha256(output/'global_frozen_outcomes.json')==gate['outcomes_sha256']
    outcome=[r for r in outcomes['outcomes'] if r['cohort']==cohort and r['index']==index][0]
    assert outcome['status']=='frozen'
    frozen=Path(outcome['frozen_manifest']).parent
    assert sha256(frozen/'frozen_manifest.json')==outcome['frozen_manifest_sha256']
    manifest=json.loads((frozen/'frozen_manifest.json').read_text())
    assert sha256(Path(manifest['recipe_path']))==manifest['recipe_sha256']
    recipe=json.loads(Path(manifest['recipe_path']).read_text())
    for design in manifest['designs']:
        assert sha256(Path(design['graph_path']))==design['graph_sha256']
    lmp=Path('/rg/mendels_prj/alexander.z/DL-course-project/.runtime/lammps/bin/lmp')
    case=frozen.parent;folder=case/'final_verification';folder.mkdir(exist_ok=False)
    (folder/'recipe.json').write_text(json.dumps(dict(job_id=os.environ.get('PBS_JOBID'),
        global_manifest_sha256=sha256(output/'global_frozen_outcomes.json'),frozen_manifest_sha256=sha256(frozen/'frozen_manifest.json'),
        verification=recipe['verification'],runner_sha256=sha256(Path(__file__)),lammps_sha256=sha256(lmp),
        feedback_to_optimizer=False,started_at=time.time()),indent=2)+'\n')
    rows=[]
    if cohort=='same10_reid':
        results=verify_frozen_designs(frozen,lmp)
        for kind,item in results.items():
            graph=torch.load(item['graph_path'],weights_only=False,map_location='cpu')
            track=np.load(item['trajectory_path'])
            row=dict(source='reid',index=index,temperature=np.nan,kind=kind,thermal_seed=0,
                status='completed',physical_p_ratio=item['physical_p_ratio'],valid_frames=item['frame_count'],total_frames=item['frame_count'],
                graph_path=item['graph_path'],trajectory_path=item['trajectory_path'],**contact_audit(track['positions'],track['boxes'],graph.edge_index.numpy()))
            rows.append(row)
    else:
        protocol=recipe['verification']
        legacy=module(output/'verification_code/thermal_dynamics.py','planar_thermal_dynamics')
        metric=module(output/'verification_code/thermal_metric.py','planar_thermal_metric')
        assert sha256(lmp)==protocol['lammps_sha256']
        assert sha256(Path(sys.modules[ElasticScript.__module__].__file__))==protocol['elastic_script_sha256']
        assert sha256(Path(sys.modules[write_network.__module__].__file__))==protocol['network_export_sha256']
        assert sha256(output/'verification_code/thermal_metric.py')==protocol['metric_sha256']
        for design in manifest['designs']:
            graph=torch.load(design['graph_path'],weights_only=False,map_location='cpu')
            sides=directional_side_indices_from_box(graph,quantile=.1)
            for seed in protocol['thermal_seeds']:
                run=folder/f"{design['name']}_s{seed}";run.mkdir()
                row=dict(source='depablo_mixed_temp',index=index,temperature=float(outcome['temperature']),kind=design['name'],
                    thermal_seed=seed,status='failed',physical_p_ratio=np.nan,valid_frames=0,total_frames=200,
                    graph_path=design['graph_path'],trajectory_path=str((run/'trajectory.npz').resolve()))
                try:
                    write_network(run,deepcopy(graph),mass=protocol['mass_amu'],angles=0.,box=graph.box)
                    ElasticScript('network.lmp').write_to_file(str(run))
                    (run/'in.thermal').write_text(legacy.script(protocol,outcome['temperature'],seed))
                    with (run/'stdout.log').open('w') as stdout,(run/'stderr.log').open('w') as stderr:
                        subprocess.run([str(lmp),'-screen','none','-in','in.thermal'],cwd=run,check=True,stdout=stdout,stderr=stderr,
                                       timeout=protocol['per_simulation_timeout_seconds'])
                    positions,boxes,steps=legacy.read_frames(run/'trajectory.dump',len(graph.x))
                    assert len(positions)==200 and np.array_equal(steps,np.arange(200)*protocol['dump_every'])
                    actual_scale=(boxes[-1,0,1]-boxes[-1,0,0])/(boxes[0,0,1]-boxes[0,0,0])
                    assert abs(actual_scale-protocol['scale_x'])<1e-7
                    fit=metric.frame_time_ols_p_ratio(positions,sides)
                    np.savez_compressed(run/'trajectory.npz',positions=positions,boxes=boxes,steps=steps,strain_x=fit['strain_x'],strain_y=fit['strain_y'])
                    assert np.isfinite(fit['p_ratio'])
                    row.update(status='completed',physical_p_ratio=fit['p_ratio'],valid_frames=fit['valid_frames'],
                        fit_r2_x=fit['fit_r2_x'],fit_r2_y=fit['fit_r2_y'],**contact_audit(positions,boxes,graph.edge_index.numpy()))
                except Exception as exc:
                    row['error']=f'{type(exc).__name__}: {exc}'
                (run/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
                pd.DataFrame(rows).to_csv(folder/'results.csv',index=False)
                print(index,design['name'],seed,row['status'],row['physical_p_ratio'],flush=True)
    pd.DataFrame(rows).to_csv(folder/'results.csv',index=False)
    (folder/'completed.json').write_text(json.dumps(dict(rows=len(rows),valid=sum(r['status']=='completed' for r in rows),
        source=recipe['source'],index=index,job_id=os.environ.get('PBS_JOBID'),feedback_to_optimizer=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort',required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();main(a.output,a.cohort,a.index)
