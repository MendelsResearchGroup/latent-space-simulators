#!/usr/bin/env python3
"""Freeze the source-identity-only retry after the code-v1 smoke receipt failure."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path("/rg/mendels_prj/alexander.z/latent-space-simulators")
HERE = ROOT / "hpc/experiments/all4_ae_low_prop_1000"
OUT = ROOT / "notebooks/results/all4_ae_low_prop_1000"
OLD = OUT / "code_v1"
CODE = OUT / "code_v2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if CODE.exists():
        raise FileExistsError(CODE)
    shutil.copytree(OLD, CODE)
    shutil.copy2(HERE / "train.py", CODE / "train.py")
    manifest = CODE / "manifest.json"
    manifest.unlink()
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    manifest.write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    ledger_path = OUT / "job_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["code_v1_failure"] = {
        "fit": "4687497.zeus-master",
        "collector": "4687510.zeus-master",
        "stage": "post-smoke exposure receipt",
        "cause": "direct graph lacks optional source_sim_id",
        "model_result": False,
    }
    ledger["retry_snapshot"] = str(CODE.relative_to(ROOT))
    ledger["retry_code_manifest_sha256"] = sha(manifest)
    ledger["status"] = "retry_prepared"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(sha(manifest))


if __name__ == "__main__":
    main()
