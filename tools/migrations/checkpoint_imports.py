"""One-time conversion of trusted PyTorch ZIP checkpoints to current imports.

Usage: python tools/migrations/checkpoint_imports.py SOURCE DESTINATION
This rewrites pickle GLOBAL references without unpickling objects or modifying
any tensor storage. Historical source files are never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickletools
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RETIRED = {"models", "config", "runner", "training", "hessian", "matched_gnn", "peptide", "peptide_tm", "inverse_design_barostat", "inverse_design_evaluation"}
MODEL_MODULES = {
    item.name: f"lss.dynamics.{module}"
    for module in ("autoencoder", "propagator")
    for item in ast.parse((ROOT / f"src/lss/dynamics/{module}.py").read_text()).body
    if isinstance(item, (ast.ClassDef, ast.FunctionDef))
}


def canonical_global(module: str, name: str) -> tuple[str, str]:
    if module in {"lss.latent.models", "lss.dynamics.models"}:
        return MODEL_MODULES[name], name
    if module == "lss.latent" or module.startswith("lss.latent."):
        return module.replace("lss.latent", "lss.dynamics", 1), name
    if module.startswith("lss.") and module.split(".")[1] in RETIRED:
        return module.replace("lss.", "lss.past_experiments.", 1), name
    if (module, name) in {("auxetic.network", "Box"), ("network", "Box")}:
        return "graph_utils.box", "Box"
    return module, name


def rewrite_pickle(data: bytes) -> tuple[bytes, list[dict]]:
    chunks, changes, previous = [], [], 0
    for opcode, argument, position in pickletools.genops(data):
        if opcode.name in {"STACK_GLOBAL", "FRAME"}:
            raise ValueError("Expected an unframed PyTorch protocol-2/3 pickle; refusing an unsupported encoding")
        if opcode.name == "GLOBAL":
            module, name = argument.split(" ")
            replacement_module, replacement_name = canonical_global(module, name)
            if (replacement_module, replacement_name) != (module, name):
                original = f"c{module}\n{name}\n".encode()
                assert data[position:position + len(original)] == original
                chunks.extend((data[previous:position], f"c{replacement_module}\n{replacement_name}\n".encode()))
                previous = position + len(original)
                changes.append({"from": f"{module}.{name}", "to": f"{replacement_module}.{replacement_name}"})
    chunks.append(data[previous:])
    return b"".join(chunks), changes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    records = []
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination, "x") as new:
        for entry in old.infolist():
            if entry.filename.endswith("/data.pkl"):
                payload, changes = rewrite_pickle(old.read(entry))
                new.writestr(entry, payload)
                records.extend(changes)
            else:
                with old.open(entry) as src, new.open(entry, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination) as new:
        assert old.namelist() == new.namelist()
        for entry in old.infolist():
            if not entry.filename.endswith("/data.pkl"):
                other = new.getinfo(entry.filename)
                assert (entry.CRC, entry.file_size) == (other.CRC, other.file_size)
    assert sha256(source) == source_hash
    record = {"source": str(source), "source_sha256": source_hash,
              "destination": str(destination), "destination_sha256": sha256(destination),
              "converter_sha256": sha256(Path(__file__)), "globals_changed": records,
              "all_other_zip_members_unchanged": True}
    destination.with_suffix(destination.suffix + ".migration.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.destination), indent=2))
