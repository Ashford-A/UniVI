# Datasets

`univi.datasets` downloads the datasets used in the tutorials and the paper, verifies each file's checksum, and caches it (in `~/.cache/univi`, or `UNIVI_DATA_DIR`).

```python
import univi.datasets as uds

uds.list_datasets()                      # name, contents, size, availability, figures
data = uds.pbmc_multiome_10k()           # {"rna": AnnData, "atac": AnnData}
data = uds.load("pbmc_multiome_10k", keys=["rna"])
uds.dataset_info("pbmc_multiome_10k")    # description, license, DOI, files
```

Paired modalities are returned with identical cell order. If `.X` holds integer counts and there is no `layers["counts"]`, the loader copies `.X` there so the preprocessors find raw counts.

## Catalog

| Name | Contents | Paper | Hosted | Original source |
| --- | --- | --- | --- | --- |
| `pbmc_multiome_10k` | 10x Multiome PBMCs, RNA + ATAC, 9,631 cells, `cell_type` | Figs. 4, 8–10; S2, S8–S11; Supp. Notebook S1 | [Zenodo 10.5281/zenodo.19581816](https://doi.org/10.5281/zenodo.19581816) | 10x Genomics |
| `hao_citeseq_pbmc` | CITE-seq PBMCs, RNA + 228 ADTs, `celltype.l1/l2/l3` | Figs. 2–3; S1 | [Zenodo 10.5281/zenodo.22883697](https://doi.org/10.5281/zenodo.22883697) | GEO [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) |
| `pbmc_multiome_bridge` | Multiome reference + Ding RNA + Satpathy ATAC | Fig. 5; S5 | [Zenodo 10.5281/zenodo.22883390](https://doi.org/10.5281/zenodo.22883390) | 10x Genomics; GEO [GSE132044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132044), [GSE129785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129785) |
| `teaseq_pbmc` | TEA-seq PBMCs, RNA + ADT + ATAC | Fig. 6; S6 | [Zenodo 10.5281/zenodo.22883520](https://doi.org/10.5281/zenodo.22883520) | GEO [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013) |
| `aml_mosaic` | AML CITE-seq bridge, van Galen scRNA-seq, DAb-seq | Fig. 7; S7 | [Zenodo 10.5281/zenodo.22883601](https://doi.org/10.5281/zenodo.22883601) | GEO [GSE220474](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220474), [GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256); BioProject [PRJNA602320](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA602320) |
| `shareseq_mouse_skin` | SHARE-seq mouse skin, RNA + ATAC | S3 | [Zenodo 10.5281/zenodo.22883542](https://doi.org/10.5281/zenodo.22883542) | GEO [GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203) |
| `scnmt_gastrulation` | scNMT-seq mouse gastrulation, RNA + CpG + GpC, 1,140 paired cells | S4 | [Zenodo 10.5281/zenodo.22885463](https://doi.org/10.5281/zenodo.22885463) | GEO [GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708) |

All seven datasets in this catalog have published download URLs in the bundled registry.

Please cite the original data generators as well as UniVI (see [Citation](../citation.md)).

## scNMT-seq mouse gastrulation (Supplemental Fig. S4)

The processed, paired dataset is published at [Zenodo 10.5281/zenodo.22885463](https://doi.org/10.5281/zenodo.22885463) ([record and supporting files](https://zenodo.org/records/22885463)). Load it through the built-in API:

```python
import univi.datasets as uds
from univi.datasets import build_univi_inputs_from_scnmt_triplet

data = uds.load("scnmt_gastrulation")  # or uds.scnmt_gastrulation()
rna, cpg, gpc = data["rna"], data["cpg"], data["gpc"]
built = build_univi_inputs_from_scnmt_triplet(rna, cpg, gpc)
adata_dict = built["adata_dict"]
recon_targets_spec = built["recon_targets_spec"]
```

All three files contain the same 1,140 cells in the same order. The published manifest and `release_audit.tsv` report:

| Modality | Download | Features | `.X` | Required layers |
| --- | --- | --- | --- | --- |
| RNA | [scnmt_gastrulation_rna.h5ad](https://zenodo.org/records/22885463/files/scnmt_gastrulation_rna.h5ad?download=1) | 22,084 | Counts | `counts` |
| CpG methylation | [scnmt_gastrulation_cpg.h5ad](https://zenodo.org/records/22885463/files/scnmt_gastrulation_cpg.h5ad?download=1) | 18,285 | Gene-body methylation fractions | `meth_successes`, `meth_total_count` |
| GpC accessibility | [scnmt_gastrulation_gpc.h5ad](https://zenodo.org/records/22885463/files/scnmt_gastrulation_gpc.h5ad?download=1) | 18,325 | Gene-body accessibility fractions | `acc_successes`, `acc_total_count` |

The three `.h5ad` files total about 224 MB; the 33 GB source archive is not needed for the hosted-data workflow. The loader verifies each file against the published MD5 checksum. `.obs["split"]` stores the seed-0 85/5/10 train/validation/test assignment (969/57/114 cells).

The source was parsed with `require_qc=False`, `filter_features=False`, `min_cov_cpg=1`, and `min_cov_gpc=1`. `build_univi_inputs_from_scnmt_triplet` log-normalizes RNA while retaining raw counts and leaves CpG/GpC fractions and success/coverage layers available for beta-binomial reconstruction. See the [S4 API notebook](api/figS4_scnmt.ipynb) for the complete workflow. Cite Argelaguet et al. (2019), [GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708), this dataset DOI, and the [UniVI article](https://doi.org/10.1101/gr.281431.125).

## Sharing your own datasets

`export_dataset` packages AnnData objects (raw counts, metadata, an optional split) with checksums, ready to upload; see [How the hosted datasets are built](building_datasets.md) for the full workflow, including the Zenodo upload script.

```python
uds.export_dataset("my_cite", {"rna": rna, "adt": adt}, "export/my_cite",
                   labels=["cell_type"], splits={"train": train_ids, "val": val_ids, "test": test_ids})
```

## Registering datasets

Register files for the current session, or point `UNIVI_DATASET_REGISTRY` at a JSON file with the same structure (useful for lab mirrors or offline clusters):

```python
uds.register_dataset("my_cite", {
    "paired": ["rna", "adt"],
    "files": {
        "rna": {"filename": "rna.h5ad", "url": "https://example.org/rna.h5ad", "hash": "sha256:..."},
        "adt": {"filename": "adt.h5ad", "url": "https://example.org/adt.h5ad", "hash": "sha256:..."},
    },
})
data = uds.load("my_cite")
```

`url` may be `https://`, `http://`, `ftp://` or `file://`; `urls` accepts a list of mirrors tried in order. Hashes are written `md5:<hex>` or `sha256:<hex>`.
