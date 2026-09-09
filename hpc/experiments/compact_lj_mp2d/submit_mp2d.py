#!/usr/bin/env python3
"""Freeze, smoke-check, and submit the matched 2D MP compact-LJ AE cells."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "notebooks/results/compact_lj_mp2d"
HERE = Path(__file__).resolve().parent
DONOR_CODE = ROOT / "notebooks/results/compact_lj_reconstruction/code_v2"
DONOR_RECIPE = ROOT / "notebooks/results/compact_lj_reconstruction/mp2_r16_d4_s123_code_v2/recipe.json"
SEEDS = (3456456, 123, 456, 786, 2026)
PYTHON = "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python"
RUN_VERSION = "code_v4"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(path: Path, item: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def freeze() -> Path:
    """Freeze orchestration while reusing the donor's already immutable source."""
    frozen = OUT / RUN_VERSION
    files = ("run_compact_lj_mp2d.py", "run_mp2d.pbs", "run_pratio_mp2d.py", "run_pratio_mp2d.pbs", "run_collect.pbs", "submit_mp2d.py")
    if frozen.exists():
        manifest = frozen / "hashes.json"
        if not manifest.exists():
            raise RuntimeError(f"Existing {RUN_VERSION} lacks its immutable manifest.")
        recorded = json.loads(manifest.read_text())
        actual = {str(p.relative_to(frozen)): sha(p) for p in sorted(frozen.rglob("*")) if p.is_file() and p.name != "hashes.json" and "pratio_snapshot" not in p.relative_to(frozen).parts}
        if actual != recorded["frozen_files"]:
            raise RuntimeError(f"Existing {RUN_VERSION} hash validation failed; do not overwrite it.")
        return frozen
    frozen.mkdir(parents=True)
    for name in files:
        shutil.copy2(HERE / name, frozen / name)
    provenance = {
        "training_source": str(DONOR_CODE),
        "training_source_hashes_sha256": sha(DONOR_CODE / "hashes.json"),
        "donor_recipe": str(DONOR_RECIPE),
        "donor_recipe_sha256": sha(DONOR_RECIPE),
        "declared_changes": ["variant mp2_r16_d4 to mp2_r16_d2", "output study compact_lj_mp2d", "p-ratio evaluator study allowlist adds compact_lj_mp2d"],
    }
    (frozen / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    frozen_files = {str(p.relative_to(frozen)): sha(p) for p in sorted(frozen.rglob("*")) if p.is_file()}
    (frozen / "hashes.json").write_text(json.dumps({"frozen_files": frozen_files}, indent=2, sort_keys=True) + "\n")
    return frozen


def check_donor() -> dict:
    recorded = json.loads((DONOR_CODE / "hashes.json").read_text())
    actual = {str(p.relative_to(DONOR_CODE)): sha(p) for p in sorted(DONOR_CODE.rglob("*")) if p.is_file() and p.name != "hashes.json" and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != recorded:
        raise RuntimeError("The immutable donor source hash manifest does not validate.")
    recipe = json.loads(DONOR_RECIPE.read_text())
    config = recipe["config"]["ae_config"]
    expected = {"model": "message_passing_reference16", "latent_dim": 4, "latent_tokens": 32, "hidden_size": 96, "edge_feature_dim": 5, "message_passing_steps": 2, "max_train_frames_per_sim": 200, "max_val_frames_per_sim": 200, "checkpoint_metric": "val_max_source_reconstruction", "pratio_eval_every": 0}
    if any(config[k] != value for k, value in expected.items()):
        raise RuntimeError("Donor MP2D4 configuration does not match the requested controlled recipe.")
    if recipe["response_selection"] or recipe["final_test_used"]:
        raise RuntimeError("Donor violates response-selection/final-test constraints.")
    return {"donor_code_hashes_sha256": sha(DONOR_CODE / "hashes.json"), "donor_recipe_sha256": sha(DONOR_RECIPE), "donor_job_id": recipe["job_id"], "checked_fields": expected}


def qsub(args: list[str]) -> str:
    value = subprocess.check_output(["qsub", *args], text=True).strip().splitlines()[-1]
    if not value or "." not in value:
        raise RuntimeError(f"Unexpected qsub response: {value!r}")
    return value


def snapshot_evaluator(frozen: Path) -> Path:
    snap = frozen / "pratio_snapshot"
    if snap.exists():
        return snap
    snap.mkdir()
    shutil.copy2(frozen / "run_pratio_mp2d.py", snap / "run_pratio_mp2d.py")
    shutil.copy2(frozen / "run_pratio_mp2d.pbs", snap / "run_pratio_mp2d.pbs")
    shutil.copytree(DONOR_CODE / "src", snap / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snap / "provenance.json").write_text(json.dumps({"source": str(DONOR_CODE / "src"), "source_hashes_sha256": sha(DONOR_CODE / "hashes.json"), "base_evaluator": str(ROOT / "hpc/experiments/compact_lj_pratio/run_pratio.py"), "base_evaluator_sha256": sha(ROOT / "hpc/experiments/compact_lj_pratio/run_pratio.py")}, indent=2, sort_keys=True) + "\n")
    return snap


def snapshot_collector() -> Path:
    source = ROOT / "hpc/experiments/compact_lj_response_rollout/collect.py"
    snap = OUT / "collector_snapshot"
    if snap.exists():
        manifest = snap / "manifest.json"
        if not manifest.exists() or json.loads(manifest.read_text())["collector_sha256"] != sha(snap / "collect.py"):
            raise RuntimeError("Existing collector snapshot is incomplete or modified.")
        return snap
    snap.mkdir()
    shutil.copy2(source, snap / "collect.py")
    (snap / "manifest.json").write_text(json.dumps({"source": str(source), "source_sha256": sha(source), "collector_sha256": sha(snap / "collect.py")}, indent=2, sort_keys=True) + "\n")
    return snap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.smoke and not args.submit:
        raise SystemExit("Choose --smoke or --submit.")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = freeze()
        check = check_donor()
        (OUT / "controlled_recipe.json").write_text(json.dumps({"study": "compact_lj_mp2d", "variant": "mp2_r16_d2", "seeds": list(SEEDS), "resources": {"queue": "mendels_q", "place": "free:shared", "ncpus": 8, "memory_gb": 16, "walltime": "24:00:00"}, "protocol": "same as compact_lj_reconstruction/mp2_r16_d4 except latent_dim=2 and output paths", "selection": "worst retained-source validation coordinate reconstruction only", "frames": 200, "evaluation_frames": [25, 50, 100, 150, 199], "final_test_used": False, "response_selection": False, **check}, indent=2, sort_keys=True) + "\n")
        ledger = OUT / "jobs.jsonl"
        previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        if args.smoke:
            if any(row.get("kind") == "smoke" and row.get("status") == "submitted" and row.get("code_snapshot") == str(frozen) for row in previous):
                print("smoke already submitted")
                return
            job = qsub(["-v", "LSS_SEED=123,LSS_SMOKE=1", str(frozen / "run_mp2d.pbs")])
            append(ledger, {"kind": "smoke", "status": "submitted", "job_id": job, "seed": 123, "code_snapshot": str(frozen), "submitted_at": now(), **check})
            print(job)
            return
        submitted = []
        for seed in SEEDS:
            if any(row.get("kind") == "train" and row.get("seed") == seed and row.get("status") == "submitted" for row in previous):
                continue
            run = OUT / f"mp2_r16_d2_s{seed}_{RUN_VERSION}"
            if run.exists():
                raise RuntimeError(f"Run directory already exists without a submitted ledger entry: {run}")
            job = qsub(["-v", f"LSS_SEED={seed},LSS_SMOKE=0", str(frozen / "run_mp2d.pbs")])
            row = {"kind": "train", "status": "submitted", "job_id": job, "seed": seed, "variant": "mp2_r16_d2", "run_dir": str(run), "code_snapshot": str(frozen), "submitted_at": now(), **check}
            append(ledger, row); previous.append(row); submitted.append((seed, job))
            print(job, seed, flush=True)
        evaluator = snapshot_evaluator(frozen)
        evaluation_jobs = []
        for seed, train_job in submitted:
            run = OUT / f"mp2_r16_d2_s{seed}_{RUN_VERSION}"
            out = OUT / "pratio" / f"mp2_r16_d2_s{seed}_{RUN_VERSION}"
            out.mkdir(parents=True, exist_ok=True)
            env = ",".join((f"LSS_PROJECT_ROOT={ROOT}", f"LSS_EVAL_CODE_ROOT={evaluator}", f"LSS_EVAL_RUNNER={evaluator / 'run_pratio_mp2d.py'}", f"CHECKPOINT={run / 'ae.pt'}", f"RECIPE={run / 'recipe.json'}", f"OUT={out}"))
            job = qsub(["-W", f"depend=afterok:{train_job}", "-v", env, str(evaluator / "run_pratio_mp2d.pbs")])
            append(ledger, {"kind": "pratio", "status": "submitted", "job_id": job, "depends_on": train_job, "seed": seed, "variant": "mp2_r16_d2", "out": str(out), "snapshot": str(evaluator), "submitted_at": now()})
            evaluation_jobs.append(job)
            print(job, f"pratio seed={seed} afterok={train_job}", flush=True)
        if evaluation_jobs:
            collector = snapshot_collector()
            job = qsub(["-W", "depend=afterany:" + ":".join(evaluation_jobs), "-v", f"LSS_COLLECTOR={collector / 'collect.py'}", str(frozen / "run_collect.pbs")])
            append(ledger, {"kind": "collector", "status": "submitted", "job_id": job, "depends_on": evaluation_jobs, "snapshot": str(collector), "collector_sha256": sha(collector / "collect.py"), "submitted_at": now()})
            print(job, "collector afterany=" + ":".join(evaluation_jobs), flush=True)


if __name__ == "__main__":
    main()
