#!/usr/bin/env python
"""Package AnnData files for a Zenodo upload and register them in univi.datasets.

Step 1 - prepare the files (checks pairing and raw counts, optionally adds a
split column, writes compressed copies, and records checksums):

    python scripts/prepare_zenodo_release.py prepare hao_citeseq_pbmc \\
        --file rna=/data/Hao_RNA_data.h5ad --file adt=/data/Hao_ADT_data.h5ad \\
        --labels celltype.l1 celltype.l2 celltype.l3 \\
        [--splits splits.tsv] --outdir zenodo/hao_citeseq_pbmc

Step 2 - upload the files to a Zenodo draft (scripts/zenodo_upload.py), review
the draft on zenodo.org, and publish it.

Step 3 - register the published files (writes URLs and md5 checksums into
univi/datasets/registry.json):

    python scripts/prepare_zenodo_release.py register zenodo/hao_citeseq_pbmc --record-id 1234567

``--splits`` takes a two-column TSV (cell id, split) and stores it as
``obs["split"]`` in every file, so users can recover the exact train/val/test
assignment used in the paper.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "univi" / "datasets" / "registry.json"


def prepare(args):
    import anndata as ad
    import pandas as pd
    from univi.datasets import export_dataset

    adatas = {k: ad.read_h5ad(v) for k, v in (item.split("=", 1) for item in args.file)}
    splits = None
    if args.splits:
        table = pd.read_csv(args.splits, sep="\t", index_col=0).iloc[:, 0].astype(str)
        splits = {name: table.index[table == name] for name in table.unique()}
    export_dataset(args.name, adatas, args.outdir, paired=args.paired, labels=args.labels,
                   splits=splits, title=args.title, source=args.source, overwrite=args.overwrite)
    print(f"next: upload the .h5ad files in {args.outdir} (scripts/zenodo_upload.py), then run `register`")


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
    p.add_argument("--title")
    p.add_argument("--source", help="original data source, e.g. 'GEO GSE164378'")
    p.add_argument("--overwrite", action="store_true")
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
