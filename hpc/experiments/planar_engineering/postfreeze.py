#!/usr/bin/env python3
"""Create the immutable verification/render queue after all forty outcomes freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(output: Path) -> None:
    output = Path(output)
    gate = output / 'global_freeze_completed.json'
    if not gate.is_file():
        raise RuntimeError('all forty ML-only outcomes must freeze before verification planning')
    outcomes = json.loads((output / 'global_frozen_outcomes.json').read_text())
    if outcomes['observed'] != outcomes['expected'] != 40:
        raise RuntimeError('incomplete frozen outcome manifest')
    cases = []
    for outcome in outcomes['outcomes']:
        item = {key: outcome[key] for key in ('cohort', 'index', 'temperature', 'status')}
        if outcome['status'] == 'invalid_original':
            item['verification_status'] = 'not_run_invalid_original'
            item['original_graph'] = outcome['original_graph']
            item['original_graph_sha256'] = outcome['original_graph_sha256']
        else:
            manifest = json.loads(Path(outcome['frozen_manifest']).read_text())
            item['verification_status'] = 'pending_final_only'
            item['designs'] = manifest['designs']
            for design in manifest['designs']:
                if sha256(Path(design['graph_path'])) != design['graph_sha256']:
                    raise RuntimeError(f"graph hash mismatch: {design['graph_path']}")
        cases.append(item)
    plan = {
        'global_outcomes_sha256': sha256(output / 'global_frozen_outcomes.json'),
        'all_outcomes_predeclared': 40,
        'physics_can_never_select_or_restart_a_design': True,
        'reid_protocol': 'canonical final-only athermal endpoint verification',
        'thermal_protocol': {'velocity_seeds': [786, 123, 456], 'frames': 200,
                             'p_ratio': 'fixed original q=.1 side groups; two OLS fits with intercept over frame index'},
        'render': {'all_cases': 40, 'duration_seconds': 4,
                   'notebook_display_reid': [93, 84, 96, 97, 52],
                   'notebook_display_thermal': [47, 68, 140, 161, 242, 268],
                   'folder': str((output / 'gifs').resolve())},
        'cases': cases,
    }
    (output / 'final_verification_plan.json').write_text(json.dumps(plan, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
