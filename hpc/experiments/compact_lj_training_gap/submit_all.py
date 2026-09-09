#!/usr/bin/env python3
"""Submit five frozen MP2-r16-d4 training-split p-ratio diagnostics."""
from __future__ import annotations
import hashlib, json, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT/'notebooks/results/compact_lj_training_gap'
RUNNER = Path(__file__).with_name('run_train_pratio.py')
PBS = Path(__file__).with_name('run_train_pratio.pbs')
PYTHON = '/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python'
SEEDS = (3456456, 123, 456, 786, 2026)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    snapshot = RESULTS/'snapshots'/('runner_'+sha(RUNNER)[:16]+'_v1')
    snapshot.mkdir(parents=True, exist_ok=True)
    frozen = snapshot/'run_train_pratio.py'
    if not frozen.exists():
        shutil.copy2(RUNNER, frozen)
        shutil.copytree(ROOT/'src', snapshot/'src')
    wrapper = snapshot/'run_train_pratio.pbs'
    if not wrapper.exists():
        shutil.copy2(PBS, wrapper)
    intended = []
    for seed in SEEDS:
        run = RESULTS/'runs'/f'mp2_r16_d4_s{seed}'
        checkpoint = ROOT/'notebooks/results/compact_lj_reconstruction'/f'mp2_r16_d4_s{seed}_code_v2'/'ae.pt'
        recipe = checkpoint.with_name('recipe.json')
        if (run/'completed.json').exists():
            continue
        run.mkdir(parents=True, exist_ok=True)
        env = ','.join((f'LSS_PROJECT_ROOT={ROOT}', f'LSS_EVAL_CODE_ROOT={snapshot}',
                        f'LSS_RUNNER={frozen}', f'LSS_PYTHON={PYTHON}',
                        f'CHECKPOINT={checkpoint}', f'RECIPE={recipe}', f'OUT={run}'))
        job = subprocess.check_output(['qsub', '-v', env, str(wrapper)], text=True).strip()
        entry = {'seed': seed, 'job_id': job, 'checkpoint': str(checkpoint),
                 'checkpoint_sha256': sha(checkpoint), 'training_recipe': str(recipe),
                 'training_recipe_sha256': sha(recipe), 'frozen_runner': str(frozen),
                 'frozen_runner_sha256': sha(frozen), 'pbs': str(wrapper), 'split': 'train'}
        (run/'submission.json').write_text(json.dumps(entry, indent=2)+'\n')
        intended.append(entry)
    (RESULTS/'jobs_submitted.json').write_text(json.dumps(intended, indent=2)+'\n')
    print(json.dumps(intended, indent=2))

if __name__ == '__main__':
    main()
