#!/usr/bin/env python3
"""Freeze, assert, and submit the isolated joint-original uniform-frame pilot."""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "hpc/experiments/mini_shared_joint_original_uniform"
DATA_OUT = ROOT / "notebooks/results/mini_shared"
OUT = ROOT / "notebooks/results/mini_shared"
RESULT = OUT / "joint_original_uniform"
CODE = OUT / "code_uniform_v1"
ARM = "joint_original_uniform"

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def assertions() -> dict:
    protocol = json.loads((HERE / "protocol.json").read_text())
    manifest = json.loads((DATA_OUT / "data_manifest.json").read_text())
    assert protocol["arms"] == {ARM: ["stiffness_original", "node_original"]}
    assert protocol["ae"]["weight_normalization"] == "every source/frame weight is exactly 1"
    assert manifest["status"] == "complete"
    for name in protocol["arms"][ARM]:
        entry = manifest["datasets"][name]
        assert sha(Path(entry["trainval_path"])) == entry["trainval_sha256"]
    return {"protocol_sha256": sha(HERE / "protocol.json"), "data_manifest_sha256": sha(DATA_OUT / "data_manifest.json"),
            "sources": protocol["arms"][ARM], "mixed_excluded": True,
            "frame_weights": "all 200 post-warmup source/frame weights exactly 1"}

def qsub(script: Path, variables: dict[str, str], dependency: str | None, name: str) -> str:
    encoded = ",".join(f"{key}={value}" for key, value in variables.items())
    command = ["qsub", "-N", name]
    if dependency:
        command += ["-W", f"depend={dependency}"]
    command += ["-v", encoded, str(script)]
    job = subprocess.check_output(command, text=True).strip()
    if not re.fullmatch(r"\d+\.[\w.-]+", job): raise ValueError(f"Unexpected qsub output: {job!r}")
    return job

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true")
    args = parser.parse_args(); checked = assertions()
    if args.check:
        print(json.dumps({"status": "passed", **checked}, indent=2, sort_keys=True)); return
    ledger_path = OUT / "uniform_job_ledger.json"
    if CODE.exists() or ledger_path.exists():
        if not (CODE.exists() and ledger_path.exists() and not RESULT.exists()):
            raise FileExistsError("Uniform pilot snapshot or ledger already exists; preserving prior submission.")
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("jobs"):
            raise FileExistsError("Uniform pilot already has submitted jobs; preserving prior submission.")
        ledger.setdefault("submission_attempts", []).append({"status": "retrying_after_unaccepted_submission", "at": datetime.now(timezone.utc).isoformat()})
    else:
        CODE.mkdir(parents=True)
        shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("train.py", "collect.py", "evaluate.py", "pipeline.pbs", "collect.pbs", "protocol.json", "submit.py"):
            shutil.copyfile(HERE / name, CODE / name)
        shutil.copyfile(DATA_OUT / "data_manifest.json", CODE / "data_manifest.json")
        files = {str(path.relative_to(CODE)): sha(path) for path in sorted(CODE.rglob("*")) if path.is_file()}
        (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
        ledger = {"study": ARM, "seed": 123, "snapshot": str(CODE.relative_to(ROOT)),
            "code_manifest_sha256": sha(CODE / "manifest.json"), **checked,
            "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"},
            "postfit_primary": "physical endpoint fixed-frame0 sides q0.10 p-ratio R2 at frame199; AE reconstruction and autonomous rollout separately",
            "selection": "ordinary uniform validation coordinate MSE and latent-delta validation loss only; p-ratio never used",
            "smoke_gate": "same job: train smoke, endpoint diagnostic collection through frame199, then full fit only after both succeed",
            "jobs": {}, "status": "submitting", "submitted_at": datetime.now(timezone.utc).isoformat(), "submission_attempts": []}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    common = {"LSS_CODE_ROOT": str(CODE), "LSS_ARM": ARM}
    ledger["jobs"]["fit_with_smoke_gate"] = qsub(CODE / "pipeline.pbs", common, None, "mini_joint_uniform")
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ledger["jobs"]["postfit_collector"] = qsub(CODE / "collect.pbs", common, f"afterok:{ledger['jobs']['fit_with_smoke_gate']}", "mini_joint_uniform_eval")
    ledger["collector_dependency"] = f"afterok:{ledger['jobs']['fit_with_smoke_gate']}"; ledger["status"] = "submitted"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n"); print(json.dumps(ledger, indent=2, sort_keys=True))

if __name__ == "__main__": main()
