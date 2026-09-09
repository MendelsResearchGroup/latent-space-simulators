"""Collect frozen high-p engineering studies after every verification row exists."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DONOR = Path('/rg/mendels_prj/alexander.z/DL-course-project')
sys.path.insert(0, str(DONOR.parent / 'MetaForge/src'))
import auxetic  # noqa: F401
import numpy as np
import pandas as pd
import torch

EXPECTED_TOTALS = {'study_v1': 28, 'study_v2': 20, 'study_v3': 16, 'joint_v1': 16}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(study):
    out = ROOT / 'notebooks/results/high_pratio_engineering' / study
    manifest_path = out / 'frozen_designs.json'
    completed_path = out / 'verification_completed.json'
    if not completed_path.exists():
        raise RuntimeError(f'{study} is not ready: missing {completed_path.name}')
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads(completed_path.read_text())
    verification_recipe = json.loads((out / 'verification_recipe.json').read_text())
    total = len(manifest['records'])
    if total != EXPECTED_TOTALS[study]:
        raise RuntimeError(f'{study}: manifest has {total} records, expected {EXPECTED_TOTALS[study]}')
    if completion['total'] != total:
        raise RuntimeError(f'{study}: completion total does not match frozen manifest')
    manifest_sha = sha(manifest_path)
    if completion['manifest_sha256'] != manifest_sha or verification_recipe['manifest_sha256'] != manifest_sha:
        raise RuntimeError(f'{study}: verification manifest hash mismatch')
    if manifest['recipe_sha256'] != sha(out / 'recipe.json'):
        raise RuntimeError(f'{study}: recipe hash mismatch')
    rows = pd.read_csv(out / 'results.csv')
    if len(rows) != total or rows.duplicated(['source', 'index', 'variant']).any():
        raise RuntimeError(f'{study}: results do not form one row per frozen design')
    manifest_keys = {(r['source'], r['index'], r['variant']) for r in manifest['records']}
    result_keys = set(rows[['source', 'index', 'variant']].itertuples(index=False, name=None))
    if result_keys != manifest_keys:
        raise RuntimeError(f'{study}: result keys differ from frozen manifest')
    for record in manifest['records']:
        if sha(record['physicalpath']) != record['sha256'] or sha(record['edit_file']) != record['edit_sha256']:
            raise RuntimeError(f"{study}: frozen artifact hash mismatch for {record['source']} {record['index']} {record['variant']}")
        result = rows[(rows.source == record['source']) & (rows['index'] == record['index']) & (rows.variant == record['variant'])].iloc[0]
        if result.sha256 != record['sha256'] or result.edit_sha256 != record['edit_sha256']:
            raise RuntimeError(f"{study}: result hash differs from frozen manifest for {record['source']} {record['index']} {record['variant']}")
    if completion['valid'] != int(rows.status.eq('completed').sum()):
        raise RuntimeError(f'{study}: completion valid count differs from results status')

    table = rows[rows.kind.eq('ml')].copy()
    group_columns = ['source', 'mode', 'bound']
    if 'position_radius' in table:
        group_columns.append('position_radius')
    summary = []
    for keys, group in table.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        item = dict(zip(group_columns, keys))
        valid = group.status.eq('completed') & np.isfinite(group.physical_p_ratio) & np.isfinite(group.original_p_ratio)
        accepted = group[valid]
        item.update(model_seed=int(group.model_seed.iloc[0]), valid=int(valid.sum()), total=len(group),
                    improved=int((accepted.change < -1e-5).sum()), negative=int((accepted.physical_p_ratio < 0).sum()),
                    target=int((accepted.physical_p_ratio <= -.1).sum()), mean_change=float(accepted.change.mean()))
        summary.append(item)
    pd.DataFrame(summary).to_csv(out / 'source_summary.csv', index=False)

    arrays = {}
    for record in manifest['records']:
        if record['kind'] == 'original' or record['mode'] in ('positions', 'joint'):
            graph = torch.load(record['physicalpath'], weights_only=False, map_location='cpu')
            arrays[f"{record['source']}_{record['index']}_{record['variant']}_positions"] = graph.x.detach().numpy()[:, :2]
    np.savez_compressed(out / 'geometry.npz', **arrays)

    control_rows = rows[rows.kind.eq('random')]
    lines = [f'# High-initial-response gradient engineering: {study}', '',
             'The selected original networks come from existing train/validation trajectories and readout calibration; this is targeted development, not a held-out test. Each physical result verifies a frozen final graph and did not select an optimizer iterate.', '',
             '| Source | Network | Variant | Kind | q radius (physical displacement <=0.25 q-radius × shortest original bond) | Original physical p | Final physical p | Change | Status |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows.to_dict('records'):
        radius = row.get('position_radius', '')
        radius = '' if radius == '' or pd.isna(radius) else f'{float(radius):g}'
        lines.append(f"| {row['source']} | {row['index']} | {row['variant']} | {row['kind']} | {radius} | {row['original_p_ratio']:.6f} | {row['physical_p_ratio']:.6f} | {row['change']:+.6f} | {row['status']} |")
    if study in ('study_v2', 'study_v3', 'joint_v1'):
        control_note = ('The joint-permutation controls preserve the centered stiffness edit, but their position q-norm can shrink to meet geometry constraints. They are therefore not jointly norm matched and do not support a matched-control comparison.'
                        if study == 'joint_v1' else
                        'The position-permutation controls are geometrically feasible frozen controls only. They are not norm matched and do not support a matched-control comparison.')
        lines += ['', control_note, '',
                  '| Source | Network | Variant | Control q-norm / ML q-norm | Shrink factor |',
                  '| --- | --- | --- | --- | --- |']
        for row in control_rows.to_dict('records'):
            lines.append(f"| {row['source']} | {row['index']} | {row['variant']} | {row.get('control_q_norm_ratio', float('nan')):.6f} | {row.get('control_shrink_factor', float('nan')):.6f} |")
    if study == 'study_v2':
        lines += ['', 'Study v2 is a separate reused-development-budget comparison after the v1 diagnosis; it does not replace or merge with v1.']
    if study == 'study_v3':
        lines += ['', 'Study v3 was declared after partial v2 physics and is a separate pristine-start, position-only reused-development comparison. It does not merge with v1 or v2.']
    if study == 'joint_v1':
        lines += ['', 'Joint v1 is a separate pristine-start, 1,200-step joint stiffness-and-position development comparison. It reports both predeclared stiffness bounds for every network and does not select a winner after verification.']
    lines += ['', 'All rows, including controls and failures, remain in results.csv. Source-wise counts are in source_summary.csv; frozen provenance is in frozen_designs.json, recipe.json, and verification_recipe.json.']
    (out / 'review.md').write_text('\n'.join(lines) + '\n')
    selected = ['source', 'index', 'variant', 'mode', 'bound', 'position_radius', 'original_p_ratio', 'physical_p_ratio', 'change', 'status']
    print(table.reindex(columns=selected).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=sorted(EXPECTED_TOTALS), default='study_v1')
    collect(parser.parse_args().study)


if __name__ == '__main__':
    main()
