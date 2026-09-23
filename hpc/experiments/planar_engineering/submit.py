#!/usr/bin/env python3
"""Snapshot first; PBS submission requires explicit --submit and a root readiness gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'notebooks/results/planar_engineering/study_v1'
THERMAL_PROTOCOL = ROOT / 'notebooks/results/thermal_adam30/study_v1/thermal_protocol.json'


def qsub(stage: str, *, dependency: str | None = None, cohort: str | None = None, index: int | None = None) -> str:
    env = [f'LSS_STAGE={stage}', f'LSS_OUTPUT={OUTPUT}', f'LSS_RUNNER={OUTPUT}/code']
    if cohort is not None:
        env.extend([f'LSS_COHORT={cohort}', f'LSS_INDEX={index}'])
    memory = '8gb' if stage == 'prepare' else '4gb'
    command = ['qsub', '-q', 'mendels_q', '-l', f'select=1:ncpus=1:mem={memory}', '-l', 'place=free:shared',
               '-l', 'walltime=02:00:00', '-v', ','.join(env)]
    if dependency:
        command.extend(['-W', f'depend={dependency}'])
    job_id = subprocess.check_output(command + ['-N', f'planar_{stage}', str(OUTPUT / 'code/run.pbs')], text=True).strip()
    if not re.fullmatch(r'\d+(?:\.[A-Za-z0-9_.-]+)?', job_id):
        raise RuntimeError(f'unexpected qsub job identifier: {job_id!r}')
    return job_id


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--constraints-ready', action='store_true')
    args = parser.parse_args()
    if OUTPUT.exists() and (OUTPUT / 'code').exists():
        raise RuntimeError(f'refuse overwrite: {OUTPUT / "code"}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'code').mkdir()
    for name in ('prepare.py', 'run.py', 'freeze_collect.py', 'postfreeze.py', 'run.pbs'):
        shutil.copy2(HERE / name, OUTPUT / 'code' / name)
    shutil.copytree(ROOT / 'src/lss', OUTPUT / 'code/src/lss')
    shutil.copy2(THERMAL_PROTOCOL, OUTPUT / 'thermal_protocol.json')
    dependencies = {str(path.resolve()): sha256(path) for path in (OUTPUT / 'code').rglob('*') if path.is_file()}
    (OUTPUT / 'code/dependency_manifest.json').write_text(json.dumps({'dependencies': dependencies}, indent=2) + '\n')
    if not args.submit:
        print(json.dumps({'status': 'snapshotted_not_submitted', 'output': str(OUTPUT)}, indent=2))
        return
    if not args.constraints_ready:
        raise RuntimeError('root constraints/tests-ready signal is required before optimization submission')
    prepare = None if (OUTPUT / 'prepared.json').is_file() else qsub('prepare')
    # The cohort is created by prepare, so IDs are deliberately fixed here as part of this frozen launcher.
    reid = [93, 84, 96, 97, 52, 59, 57, 53, 2, 74]
    thermal = [47, 44, 36, 39, 13, 68, 96, 70, 78, 77, 140, 118, 136, 109, 145,
               161, 191, 158, 198, 151, 242, 208, 221, 229, 234, 268, 250, 262, 265, 290]
    jobs = {'prepare': prepare, 'optimize': {}}
    for cohort, ids in (('same10_reid', reid), ('mixedT_thermal_armA', thermal)):
        for index in ids:
            jobs['optimize'][f'{cohort}_{index}'] = qsub('optimize', dependency=f'afterok:{prepare}' if prepare else None, cohort=cohort, index=index)
    jobs['freeze'] = qsub('freeze', dependency='afterany:' + ':'.join(jobs['optimize'].values()))
    jobs['status'] = 'optimization_and_global_freeze_submitted; final verification intentionally not submitted'
    (OUTPUT / 'submission_status.json').write_text(json.dumps(jobs, indent=2) + '\n')


if __name__ == '__main__':
    main()
