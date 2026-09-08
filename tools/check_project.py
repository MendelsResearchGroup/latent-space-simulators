"""Check local assets and notebook syntax without launching training or PBS jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-data', action='store_true', help='Also verify full dataset SHA-256 hashes.')
    args = parser.parse_args()
    errors = []
    sys.path.insert(0, str(ROOT / 'src'))
    import lss
    if not Path(lss.__file__).resolve().is_relative_to(ROOT):
        errors.append(f'lss imported outside this project: {lss.__file__}')
    from IPython.core.inputtransformer2 import TransformerManager
    transform = TransformerManager()
    notebooks = sorted((ROOT / 'notebooks').rglob('*.ipynb'))
    cells = 0
    for path in notebooks:
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook.get('cells', [])):
            if cell.get('cell_type') != 'code':
                continue
            try:
                compile(transform.transform_cell(''.join(cell.get('source', []))), f'{path.name}:cell{index}', 'exec')
                cells += 1
            except SyntaxError as exc:
                errors.append(str(exc))
    manifest = json.loads((ROOT / 'data' / 'manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / 'data' / item['name']
        if not path.is_file():
            errors.append(f'Missing data: {item["name"]}')
        elif path.stat().st_size != item['bytes']:
            errors.append(f'Data size mismatch: {item["name"]}')
        elif args.hash_data and sha256(path) != item['sha256']:
            errors.append(f'Data hash mismatch: {item["name"]}')
    print(f'Checked local import, {len(notebooks)} notebooks ({cells} code cells), and {len(manifest["files"])} data files.')
    for error in errors:
        print(error, file=sys.stderr)
    print('PASS' if not errors else f'FAIL: {len(errors)} issue(s)')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
