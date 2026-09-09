"""Portable raw thermal plots and strict frozen-design collection."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
DONOR = Path("/rg/mendels_prj/alexander.z/DL-course-project")
sys.path[:0] = [str(DONOR.parent / "MetaForge/src"), str(ROOT / "hpc/experiments/ml_only_engineering")]
from legacy_graph_utils import directional_side_indices_from_box
from thermal_metric import DATA, SELECTED, frame_time_ols_p_ratio, sha, unwrap_positions

AUDIT = ROOT / "notebooks/results/thermal_engineering_audit"
STUDY = ROOT / "notebooks/results/thermal_joint_engineering/study_v1_retry"


def raw_plotdata():
    raw = torch.load(DATA, weights_only=False, map_location="cpu", mmap=True)
    arrays = {}
    for index, temperature in SELECTED.items():
        sim = raw[index]
        positions, _, _ = unwrap_positions(sim)
        sides = directional_side_indices_from_box(sim[0], quantile=.10)
        metric = frame_time_ols_p_ratio(positions, sides)
        arrays[f"{index}_time"] = np.arange(len(sim), dtype=float)
        arrays[f"{index}_strain_x"] = metric["strain_x"]
        arrays[f"{index}_strain_y"] = metric["strain_y"]
        arrays[f"{index}_fit_x"] = metric["intercept_x"] + metric["slope_x"] * arrays[f"{index}_time"]
        arrays[f"{index}_fit_y"] = metric["intercept_y"] + metric["slope_y"] * arrays[f"{index}_time"]
        arrays[f"{index}_temperature"] = np.asarray([temperature])
    np.savez_compressed(AUDIT / "raw_linear_fit_plotdata.npz", **arrays)


def collect_frozen():
    manifest_path = STUDY / "frozen_designs.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"]
    if len(records) != 24 or manifest["recipe_sha256"] != sha(STUDY / "recipe.json"):
        raise RuntimeError("Unexpected thermal frozen manifest")
    expected = {(idx, variant) for idx in SELECTED for variant in
                ("original", "positions_r4_b0", "joint_r4_b0.25", "joint_r4_b0.25_permuted")}
    got = {(int(r["index"]), r["variant"]) for r in records}
    if got != expected:
        raise RuntimeError("Thermal manifest does not contain the predeclared 24 designs")
    rows = []
    for record in records:
        if sha(record["physicalpath"]) != record["sha256"] or sha(record["edit_file"]) != record["edit_sha256"]:
            raise RuntimeError(f"Frozen hash mismatch: {record['index']} {record['variant']}")
        edit = torch.load(record["edit_file"], weights_only=False, map_location="cpu")
        rows.append(dict(source=record["source"], original_index=int(record["index"]),
                         temperature=float(record["temperature"]), variant=record["variant"],
                         kind=record["kind"], mode=record["mode"], bound=float(record["bound"]),
                         latent_proxy_score=float(record["predicted_p"]),
                         stiffness_edit_l2=float(edit["log_stiffness"].norm()),
                         position_q_l2=float(edit["position_q"].norm()),
                         physicalpath=record["physicalpath"], sha256=record["sha256"],
                         edit_file=record["edit_file"], edit_sha256=record["edit_sha256"]))
    pd.DataFrame(rows).sort_values(["temperature", "variant"]).to_csv(AUDIT / "frozen_design_table.csv", index=False)
    return True


def collect_physics():
    protocol_path=STUDY/'thermal_protocol.json'
    if not protocol_path.exists(): return dict(physical_complete=False)
    protocol=json.loads(protocol_path.read_text());manifest=json.loads((STUDY/'frozen_designs.json').read_text())
    assert protocol['manifest_sha256']==sha(STUDY/'frozen_designs.json')
    tables=[];jobs={}
    for temperature in protocol['temperatures_K']:
        folder=STUDY/'thermal_physics'/f'T{temperature:g}'
        done=folder/'verification_completed.json'
        if not done.exists(): return dict(physical_complete=False)
        completion=json.loads(done.read_text());recipe=json.loads((folder/'verification_recipe.json').read_text())
        for obj in [completion,recipe]:
            assert obj['manifest_sha256']==sha(STUDY/'frozen_designs.json')
            assert obj['protocol_sha256']==sha(protocol_path)
        table=pd.read_csv(folder/'results.csv');assert len(table)==12==completion['total']
        assert int(table.status.eq('completed').sum())==completion['valid']
        tables.append(table);jobs[str(temperature)]=recipe['job_id']
    rows=pd.concat(tables,ignore_index=True);records={(r['index'],r['variant']):r for r in manifest['records']}
    expected={(r['index'],r['variant'],seed) for r in manifest['records'] for seed in protocol['thermal_seeds']}
    got=set(zip(rows['index'],rows.variant,rows.thermal_seed));assert len(rows)==72 and got==expected
    for row in rows.to_dict('records'):
        record=records[row['index'],row['variant']]
        assert row['sha256']==record['sha256'] and row['edit_sha256']==record['edit_sha256']
        assert row['temperature']==record['temperature']
    diagnostics=[]
    for row in rows.to_dict('records'):
        names=[];values=[];log=Path(row['run_dir'])/'log.lammps'
        if log.exists():
            for line in log.read_text().splitlines():
                tokens=line.split()
                if tokens[:2]==['Step','Temp']:
                    names=tokens;values=[]
                elif names and len(tokens)==len(names):
                    try: values.append([float(v) for v in tokens])
                    except ValueError: pass
        d=dict(achieved_temperature_mean=float('nan'),achieved_temperature_sd=float('nan'),mean_pyy=float('nan'),compression_thermo_rows=0)
        if values:
            array=np.asarray(values);temps=array[:,names.index('Temp')];d.update(achieved_temperature_mean=float(temps.mean()),achieved_temperature_sd=float(temps.std(ddof=1)),compression_thermo_rows=len(temps))
            if 'Pyy' in names:d['mean_pyy']=float(array[:,names.index('Pyy')].mean())
        diagnostics.append(d)
    rows=pd.concat([rows,pd.DataFrame(diagnostics)],axis=1)
    original=rows[rows.kind.eq('original')].set_index(['index','thermal_seed']).physical_p_ratio
    rows['original_p_ratio']=[original.loc[(r['index'],r['thermal_seed'])] for r in rows.to_dict('records')]
    rows['change']=rows.physical_p_ratio-rows.original_p_ratio
    rows.to_csv(STUDY/'thermal_results.csv',index=False)
    summary=[]
    for (temperature,index,variant),group in rows.groupby(['temperature','index','variant']):
        valid=group.status.eq('completed') & np.isfinite(group.physical_p_ratio)
        paired=valid & np.isfinite(group.original_p_ratio)
        p=group.loc[valid,'physical_p_ratio'];d=group.loc[paired,'change']
        summary.append(dict(temperature=temperature,index=index,readout_calibration_topology_overlap=index in (47,161),variant=variant,kind=group.kind.iloc[0],model_seed=786,valid=int(valid.sum()),total=len(group),paired_valid=int(paired.sum()),mean_p=float(p.mean()),sd_p=float(p.std(ddof=1)),mean_change=float(d.mean()),sd_change=float(d.std(ddof=1)),improved=int((d<0).sum()),negative=int((p<0).sum()),converted_to_negative=int(((group.loc[paired,'original_p_ratio']>=0)&(group.loc[paired,'physical_p_ratio']<0)).sum())))
    pd.DataFrame(summary).to_csv(STUDY/'thermal_summary.csv',index=False)
    done=dict(total=len(rows),valid=int((rows.status.eq('completed')&np.isfinite(rows.physical_p_ratio)).sum()),manifest_sha256=sha(STUDY/'frozen_designs.json'),protocol_sha256=sha(protocol_path),jobs=jobs,feedback_to_optimizer=False)
    (STUDY/'verification_completed.json').write_text(json.dumps(done,indent=2)+'\n')
    return dict(physical_complete=True,**done)

def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    if not (AUDIT/"raw_linear_fit_plotdata.npz").exists(): raw_plotdata()
    frozen = collect_frozen()
    physical = collect_physics() if frozen else dict(physical_complete=False)
    (AUDIT / "thermal_collect_status.json").write_text(json.dumps(
        dict(study=str(STUDY), frozen_manifest_present=frozen, **physical,
             raw_plotdata="raw_linear_fit_plotdata.npz",
             note="Separate declared new thermal verification protocol; latent_proxy_score is a historical low-T readout, not position prediction."), indent=2) + "\n")
    print("frozen_manifest_present", frozen)


if __name__ == "__main__":
    main()
