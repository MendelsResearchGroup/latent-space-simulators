"""Schedule frozen collection and inline notebook refresh after all study jobs."""
from pathlib import Path
import fcntl
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'notebooks/results/compact_lj_response_rollout'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    with (OUT / 'collection_submit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = OUT / 'collection_job.json'
        if record.exists() and json.loads(record.read_text()).get('job_id'):
            print(record.read_text())
            return
        ledger = [json.loads(line) for line in (OUT / 'jobs.jsonl').read_text().splitlines()]
        jobs = [r for r in ledger if r.get('kind') == 'full' and r.get('status') == 'submitted']
        if len(jobs) != 20 or len({(r['variant'], r['strategy'], r['seed']) for r in jobs}) != 20:
            raise ValueError('Expected exactly 20 distinct submitted propagator cells')
        d2ledger = ROOT / 'notebooks/results/compact_lj_mp2d/jobs.jsonl'
        d2 = [json.loads(line) for line in d2ledger.read_text().splitlines()]
        evals = [r['job_id'] for r in d2 if r.get('kind') == 'pratio' and r.get('status') == 'submitted']
        if len(evals) != 5:
            raise ValueError('Expected five scheduled independent MP2D p-ratio evaluations')
        dependencies = sorted({r['job_id'] for r in jobs} | set(evals) | {'4674660.zeus-master'})
        frozen = OUT / 'collection_code_v1'
        frozen.mkdir(exist_ok=True)
        for name in ('collect.py', 'collect.pbs'):
            target = frozen / name
            if target.exists() and sha(target) != sha(HERE / name):
                raise ValueError('Collection snapshot exists with different content')
            if not target.exists():
                shutil.copy2(HERE / name, target)
        recipe = dict(dependencies=dependencies, dependency_type='afterany',
            queue='mendels_q', placement='free:shared', ncpus=1, memory_gb=2,
            collector=str(frozen / 'collect.py'), collector_sha256=sha(frozen / 'collect.py'),
            wrapper_sha256=sha(frozen / 'collect.pbs'),
            notebook='notebooks/dynamics/diagnostics/compact_lj_response_rollout.ipynb')
        record.write_text(json.dumps(recipe, indent=2) + '\n')
        env = f'LSS_PROJECT_ROOT={ROOT},LSS_COLLECTOR={frozen / "collect.py"}'
        job = subprocess.check_output(['qsub', '-W', 'depend=afterany:' + ':'.join(dependencies),
            '-v', env, str(frozen / 'collect.pbs')], cwd=ROOT, text=True).strip()
        recipe['job_id'] = job
        record.write_text(json.dumps(recipe, indent=2) + '\n')
        print(json.dumps(recipe, indent=2))


if __name__ == '__main__':
    main()
