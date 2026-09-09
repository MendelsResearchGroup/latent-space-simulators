#!/usr/bin/env python3
"""Collect every predeclared final design; never select or revise designs."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'notebooks/results/ml_only_engineering_review'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    all_rows=[]; provenance=[]
    for version, suffix, expected, original_count in [(1,'v1',42,6),(2,'v2',42,6),(3,'v3',24,6),(4,'v4',16,4),(5,'rollout_v1',15,3)]:
        folder=ROOT/f'notebooks/results/ml_only_engineering_{suffix}'
        completion=json.loads((folder/'verification_completed.json').read_text())
        manifest=json.loads((folder/'frozen_designs.json').read_text())
        assert completion['total']==expected and len(manifest['records'])==expected
        assert manifest['recipe_sha256']==sha(folder/'recipe.json')
        for record in manifest['records']:
            assert sha(record['file'])==record['sha256'] and sha(record['edit_file'])==record['edit_sha256']
        rows=pd.read_csv(folder/'results.csv')
        assert len(rows)==expected
        assert not rows.duplicated(['source','index','variant']).any()
        assert (rows.compression_steps==120).all()
        assert len(rows[rows.kind=='original'])==original_count
        if 'model_seed' not in rows: rows['model_seed']=786
        all_rows.append(rows.assign(study=version))
        provenance.append(dict(study=version,manifest_sha256=sha(folder/'frozen_designs.json'),
            recipe_sha256=sha(folder/'recipe.json'),results_sha256=sha(folder/'results.csv'),
            verification=json.loads((folder/'verification_recipe.json').read_text()),
            jobs=json.loads((folder/'submission_status.json').read_text())))
    table=pd.concat(all_rows,ignore_index=True)
    summaries=[]
    for (version,source,policy,bound,seed),group in table[table.kind=='ml'].groupby(['study','source','policy','bound','model_seed']):
        valid=group.status.eq('completed') & np.isfinite(group.physical_p_ratio) & np.isfinite(group.original_p_ratio)
        g=group[valid]
        control=table[(table.study==version)&(table.source==source)&(table.kind=='random')&np.isclose(table.bound,bound)&(table.model_seed==seed)].set_index('index')
        controls=control.reindex(g['index'])
        good_controls=controls.status.eq('completed').to_numpy() & np.isfinite(controls.physical_p_ratio.to_numpy())
        beaten=(g.physical_p_ratio.to_numpy()<controls.physical_p_ratio.to_numpy()-1e-5)&good_controls
        summaries.append(dict(study=version,source=source,policy=policy,bound=bound,model_seed=int(seed),valid=int(valid.sum()),total=len(group),
            improved=int((g.change < -1e-5).sum()),negative=int((g.physical_p_ratio<0).sum()),
            converted_to_negative=int(((g.original_p_ratio>=0)&(g.physical_p_ratio<0)).sum()),
            crossed_minus_point1=int(((g.original_p_ratio>-.1)&(g.physical_p_ratio<=-.1)).sum()),target_minus_point1=int((g.physical_p_ratio<=-.1).sum()),
            mean_change=float(g.change.mean()),random_beaten=int(beaten.sum()),random_valid=int(good_controls.sum())))
    summary=pd.DataFrame(summaries)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT/'all_final_results.csv',index=False)
    summary.to_csv(OUT/'source_summary.csv',index=False)
    rollout_metrics=[]
    for seed,g in table[(table.study==5)&(table.kind=='ml')].groupby('model_seed'):
        mask=g.status.eq('completed') & np.isfinite(g.physical_p_ratio) & np.isfinite(g.predicted_p_ratio)
        y=g.loc[mask,'physical_p_ratio'].to_numpy(); pred=g.loc[mask,'predicted_p_ratio'].to_numpy()
        denominator=float(((y-y.mean())**2).sum()) if len(y) else 0.
        rollout_metrics.append(dict(source='depablo_low_temp',model_seed=int(seed),frame=199,valid=int(mask.sum()),total=len(g),
            physical_endpoint_p_ratio_r2=float(1-((pred-y)**2).sum()/denominator) if denominator>1e-12 else float('nan'),
            p_ratio_mae=float(np.abs(pred-y).mean()) if len(y) else float('nan'),
            cohort='each model own three final engineered development networks; post-optimization positions-based prediction'))
    pd.DataFrame(rollout_metrics).to_csv(OUT/'rollout_engineered_prediction_metrics.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=['# ML-only gradient engineering results','',
        'Frozen AE gradients produced positive-to-negative responses in Reid53, Reid59 and Reid57, with LAMMPS used only for final verification. The unchanged method improved all four previously unengineered validation networks (2/2 Reid, 2/2 low-T). Strong edits remain unreliable on the positive-p low-T development networks; adding the tested propagator did not resolve that limitation.', '',
        'All designs start from original networks. Each study freezes every final graph before a separate LAMMPS verification job. No simulator result selects an optimizer iterate. Studies1–3 use six previously studied development networks and one frozen AE seed786. Study4 applies the exact study3 method to four previously unengineered AE-validation networks. Study5 uses two frozen AE+propagator seeds786/123 on the three low-T development networks. None uses the reserved final test.',
        '', 'Studies1–4 use frozen AEs with downstream source-calibrated affine latent readouts. Study5 instead differentiates physical positions from199 autonomous AE+propagator steps, without a response readout. The readout is a directional proxy; its numerical output is not a reliable physical-response prediction under edits.',
        '', 'Studies: v1 target-based ML optimization; v2 zero-mean log-stiffness projection; v3 fixed-budget centered normalized-gradient descent. Study4 is a new-network check of v3; study5 uses a physical-position objective through a frozen propagator. All initial geometry/topology is fixed, positive springs remain positive, zero springs remain zero. All physical checks use120 quasistatic increments and the physical first/last boundary-node ratio.', '',
        '| Study | Source | Policy | Seed | Bound | Valid/total | Improved | Negative | ≤−0.1 | Beats random |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in summary.to_dict('records'):
        lines.append(f"| {r['study']} | {r['source']} | {r['policy']} | {r['model_seed']} | {r['bound']:g} | {r['valid']}/{r['total']} | {r['improved']} | {r['negative']} | {r['target_minus_point1']} | {r['random_beaten']}/{r['random_valid']} |")
    lines+=['','Final fixed-budget and transfer results (every ML design in studies3–5):','',
        '| Study | Source | Network | Policy | Seed | Original | Final | Change | Status |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in table[(table.study>=3)&(table.kind=='ml')].to_dict('records'):
        lines.append(f"| {r['study']} | {r['source']} | {r['index']} | {r['policy']} | {int(r['model_seed'])} | {r['original_p_ratio']:.6f} | {r['physical_p_ratio']:.6f} | {r['change']:+.6f} | {r['status']} |")
    lines+=['','Negative and target-attainment columns include already-negative originals. Explicit positive-to-negative conversion counts and newly crossed targets are retained in source_summary.csv. Physical-position rollout p-ratio R² on each model own three optimized designs is in rollout_engineered_prediction_metrics.csv; these are post-optimization development results, not an independent final-test cohort.', '', 'The first verifier4675255 failed before physics because the frozen endpoint helper lacked the requested compression_steps argument. Retry4675268 changed only the verifier, preserving the complete v1 design manifest. All failures, original jobs, code hashes, per-step ML objectives, finite-difference gradient checks, final graphs, and physical dumps remain in the study directories.',
        '', 'Larger-bound v1/v2 arms sometimes produce identical designs because the surrogate target is reached before the bound. Do not count those as independent replications. Random controls use one seed per network, not a multi-seed confidence estimate. No p-ratio-based AE training or checkpoint selection occurred.']
    (OUT/'review.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False))

if __name__=='__main__': main()
