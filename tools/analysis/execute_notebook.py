#!/usr/bin/env python3
"""Execute one repository notebook in place with the active Python kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    path = args.notebook.resolve()
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()
    nbformat.write(notebook, path)
    print(f"executed {path} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
