#!/usr/bin/env python
"""Execute the tutorial notebooks on real data and save their outputs in place.

Read the Docs shows notebooks with the outputs stored in them, so run this after
changing a tutorial (a GPU is recommended), check the results, and commit.

    python scripts/execute_tutorials.py                 # all tutorials
    python scripts/execute_tutorials.py quickstart citeseq

Files the notebooks write (reference bundles) go to docs/tutorials/_run/, which
is git-ignored.
"""
import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs" / "tutorials"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="notebook names without .ipynb (default: all)")
    parser.add_argument("--timeout", type=int, default=6 * 3600, help="seconds per cell")
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()

    paths = [TUTORIALS / f"{n}.ipynb" for n in args.names] or sorted(TUTORIALS.glob("*.ipynb"))
    workdir = TUTORIALS / "_run"
    workdir.mkdir(exist_ok=True)
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        started = time.time()
        print(f"Executing {path.name} ...", flush=True)
        NotebookClient(nb, timeout=args.timeout, kernel_name=args.kernel,
                       resources={"metadata": {"path": str(workdir)}}).execute()
        nbformat.write(nb, path)
        print(f"  done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
