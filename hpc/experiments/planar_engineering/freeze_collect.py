#!/usr/bin/env python3
"""Freeze the complete 40-network outcome manifest before any verification jobs exist."""
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
    recipe = json.loads((output / 'cohort_and_recipe.json').read_text())
    rows, missing = [], []
    for declared in recipe['rows']:
        path = output / 'cases' / f"{declared['cohort']}_{declared['index']}" / 'outcome.json'
        if not path.is_file():
            missing.append({'cohort': declared['cohort'], 'index': declared['index']})
            continue
        outcome = json.loads(path.read_text())
        if outcome['status'] == 'frozen':
            manifest = Path(outcome['frozen_manifest'])
            if sha256(manifest) != outcome['frozen_manifest_sha256']:
                raise RuntimeError(f'frozen manifest hash mismatch: {manifest}')
        rows.append(outcome)
    result = {'expected': 40, 'observed': len(rows), 'missing': missing, 'outcomes': rows,
              'invalid_original_count': sum(row['status'] == 'invalid_original' for row in rows),
              'physics_feedback_to_optimizer': False}
    (output / 'global_frozen_outcomes.json').write_text(json.dumps(result, indent=2) + '\n')
    if missing:
        raise RuntimeError(f'incomplete freeze: {len(missing)} missing outcomes')
    (output / 'global_freeze_completed.json').write_text(json.dumps({
        'expected': 40, 'observed': 40, 'outcomes_sha256': sha256(output / 'global_frozen_outcomes.json'),
        'verification_authorized_only_after_this_file': True}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
