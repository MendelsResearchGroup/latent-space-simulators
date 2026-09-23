#!/usr/bin/env python3
"""Freeze the two original-only cohorts and import their historical bundles once."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DONOR = Path('/rg/mendels_prj/alexander.z/DL-course-project')
MIGRATOR = ROOT / 'tools/migrations/checkpoint_imports.py'
REID_IDS = [93, 84, 96, 97, 52, 59, 57, 53, 2, 74]
THERMAL = {0.1: [47, 44, 36, 39, 13], 1.0: [68, 96, 70, 78, 77],
           5.0: [140, 118, 136, 109, 145], 10.0: [161, 191, 158, 198, 151],
           20.0: [242, 208, 221, 229, 234], 30.0: [268, 250, 262, 265, 290]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(output: Path) -> None:
    output = Path(output)
    if output.exists():
        raise RuntimeError(f'refuse overwrite: {output}')
    output.mkdir(parents=True)
    bundles = {
        'reid': DONOR / 'notebooks/results/network_design/calibrated_direction_v1/reid/prepare/prepared.pt',
        'thermal_arm_a': DONOR / 'notebooks/results/network_design/calibrated_direction_v1/depablo_low_temp/prepare/prepared.pt',
    }
    imported = {}
    for name, source in bundles.items():
        destination = output / 'model_bundles' / f'{name}.pt'
        subprocess.run([sys.executable, str(MIGRATOR), str(source), str(destination)], check=True)
        imported[name] = json.loads(destination.with_suffix('.pt.migration.json').read_text())
    rows = ([{'cohort': 'same10_reid', 'source': 'reid', 'index': index, 'temperature': None}
             for index in REID_IDS] +
            [{'cohort': 'mixedT_thermal_armA', 'source': 'depablo_mixed_temp', 'index': index, 'temperature': temperature}
             for temperature, ids in THERMAL.items() for index in ids])
    recipe = {
        'study': 'planar_engineering/study_v1', 'outcomes_expected': 40,
        'selection': 'exact pre-existing cohorts, originals only; no physics screening or restart selection',
        'reid': {'ids': REID_IDS, 'dataset': str(DONOR / 'data/reid_200_frames.pt'),
                 'bundle': str((output / 'model_bundles/reid.pt').resolve()),
                 'optimization': {'learning_rate': .08, 'stiffness_bound': .25, 'max_steps': 1200,
                                  'patience': 50, 'min_delta': 1e-5, 'position_radius': 4., 'mask_blocking_nodes': True,
                                  'spring_clearance_fraction': .05}},
        'thermal_arm_a': {'temperatures_to_ids': THERMAL, 'dataset': str(DONOR / 'data/depablo-10k-mix-temp.pt'),
                          'bundle': str((output / 'model_bundles/thermal_arm_a.pt').resolve()),
                          'optimization': {'learning_rate': .02, 'stiffness_bound': .25, 'max_steps': 1200,
                                           'patience': 50, 'min_delta': 1e-5, 'position_radius': 4., 'mask_blocking_nodes': True,
                                           'spring_clearance_fraction': .05}},
        'thermal_final_protocol': {'velocity_seeds': [786, 123, 456], 'frames': 200,
                                   'estimator': 'two intercept OLS over frame indices; p=-slope_y/slope_x'},
        'bundle_imports': imported, 'rows': rows,
    }
    (output / 'cohort_and_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    (output / 'prepared.json').write_text(json.dumps({'status': 'prepared', 'recipe_sha256': sha256(output / 'cohort_and_recipe.json')}, indent=2) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
