#!/usr/bin/env python
"""Package AnnData files for a Zenodo upload and register them in univi.datasets.

Step 1 - prepare the files (checks pairing and raw counts, optionally adds a
split column, writes compressed copies, and records checksums):

    python scripts/prepare_zenodo_release.py prepare hao_citeseq_pbmc \\
        --file rna=/data/Hao_RNA_data.h5ad --file adt=/data/Hao_ADT_data.h5ad \\
        --labels celltype.l1 celltype.l2 celltype.l3 \\
        [--splits splits.tsv] --outdir zenodo/hao_citeseq_pbmc

Step 2 - upload every .h5ad in the output directory to a new Zenodo record and
publish it (use the generated DESCRIPTION.md as a starting point).

Step 3 - register the published files (writes URLs and md5 checksums into
univi/datasets/registry.json):

    python scripts/prepare_zenodo_release.py register zenodo/hao_citeseq_pbmc --record-id 1234567

``--splits`` takes a two-column TSV (cell id, split) and stores it as
``obs["split"]`` in every file, so users can recover the exact train/val/test
assignment used in the paper.
"""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "univi" / "datasets" / "registry.json"


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_counts(x):
    v = x.data if sp.issparse(x) else np.asarray(x).ravel()
    v = v[:200_000]
    return v.size > 0 and bool(np.all(v >= 0) and np.all(np.mod(v, 1) == 0))


def prepare(args):
    files = dict(item.split("=", 1) for item in args.file)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    adatas = {k: ad.read_h5ad(v) for k, v in files.items()}
    paired = args.paired if args.paired is not None else list(adatas)

    for key, a in adatas.items():
        if not a.obs_names.is_unique:
            raise SystemExit(f"{key}: obs_names are not unique")
        has_counts = "counts" in a.layers and _is_counts(a.layers["counts"])
        if not has_counts and not _is_counts(a.X):
            print(f"warning: {key} has no raw integer counts in .X or layers['counts']")
        missing = [c for c in args.labels if c not in a.obs]
        if missing and key in paired:
            print(f"warning: {key} is missing label columns {missing}")
    if len(paired) > 1:
        first = adatas[paired[0]].obs_names
        for key in paired[1:]:
            if not adatas[key].obs_names.equals(first):
                raise SystemExit(f"{key} and {paired[0]} do not share identical, identically ordered obs_names; "
                                 "align them first (univi.data.align_paired_obs_names).")
    if args.splits:
        table = pd.read_csv(args.splits, sep="\t", index_col=0).iloc[:, 0].astype(str)
        for key, a in adatas.items():
            a.obs["split"] = pd.Categorical(table.reindex(a.obs_names).fillna("unassigned"))

    manifest = {"name": args.name, "paired": paired, "labels": args.labels, "files": {}}
    for key, a in adatas.items():
        filename = f"{args.name}_{key}.h5ad"
        path = out / filename
        a.write_h5ad(path, compression="gzip")
        manifest["files"][key] = {
            "filename": filename, "md5": _md5(path), "size_mb": round(path.stat().st_size / 1e6, 1),
            "n_obs": int(a.n_obs), "n_vars": int(a.n_vars), "obs_columns": list(map(str, a.obs.columns)),
        }
        print(f"wrote {path} ({manifest['files'][key]['size_mb']} MB)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = [f"# {args.name}", "", "Processed AnnData (.h5ad) files used with UniVI.", "", "| file | cells | features |",
             "| --- | --- | --- |"]
    lines += [f"| {f['filename']} | {f['n_obs']} | {f['n_vars']} |" for f in manifest["files"].values()]
    lines += ["", "Raw counts are in `.X` or `.layers['counts']`. Load with:", "",
              "```python", "import univi.datasets as uds", f"data = uds.load(\"{args.name}\")", "```", "",
              "Please cite the original data source and the UniVI article (doi:10.1101/gr.281431.125)."]
    (out / "DESCRIPTION.md").write_text("\n".join(lines) + "\n")
    print(f"next: upload the .h5ad files in {out} to Zenodo, publish, then run `register`")


def register(args):
    out = Path(args.outdir)
    manifest = json.loads((out / "manifest.json").read_text())
    payload = json.loads(REGISTRY.read_text())
    entry = payload["datasets"].setdefault(manifest["name"], {})
    entry["paired"] = manifest["paired"]
    if manifest["labels"]:
        entry["labels"] = manifest["labels"]
    entry["doi"] = args.doi or f"10.5281/zenodo.{args.record_id}"
    entry["record_url"] = f"https://zenodo.org/records/{args.record_id}"
    entry["files"] = {
        key: {"filename": f["filename"],
              "url": f"https://zenodo.org/records/{args.record_id}/files/{f['filename']}?download=1",
              "hash": f"md5:{f['md5']}", "size_mb": f["size_mb"]}
        for key, f in manifest["files"].items()
    }
    first = next(iter(manifest["files"].values()))
    entry.setdefault("n_cells", first["n_obs"])
    REGISTRY.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"updated {REGISTRY.relative_to(ROOT)} entry '{manifest['name']}'. "
          "Check that the md5 values match those shown on the Zenodo record, then commit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("name")
    p.add_argument("--file", action="append", required=True, metavar="KEY=PATH")
    p.add_argument("--paired", nargs="*", default=None, help="keys that share cells (default: all)")
    p.add_argument("--labels", nargs="*", default=[])
    p.add_argument("--splits", help="TSV: cell id <tab> split")
    p.add_argument("--outdir", required=True)
    p.set_defaults(func=prepare)
    r = sub.add_parser("register")
    r.add_argument("outdir")
    r.add_argument("--record-id", required=True)
    r.add_argument("--doi", help="defaults to 10.5281/zenodo.<record-id>")
    r.set_defaults(func=register)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
