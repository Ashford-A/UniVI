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
| `hao_citeseq_pbmc` | CITE-seq PBMCs, RNA + 228 ADTs, `celltype.l1/l2/l3` | Figs. 2–3; S1 | not yet | GEO [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) |
| `pbmc_multiome_bridge` | Multiome reference + Ding RNA + Satpathy ATAC | Fig. 5; S5 | not yet | 10x Genomics; GEO [GSE132044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132044), [GSE129785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129785) |
| `teaseq_pbmc` | TEA-seq PBMCs, RNA + ADT + ATAC | Fig. 6; S6 | not yet | GEO [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013) |
| `aml_mosaic` | AML CITE-seq bridge, van Galen scRNA-seq, DAb-seq | Fig. 7; S7 | not yet | GEO [GSE220474](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220474), [GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256); BioProject [PRJNA602320](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA602320) |
| `shareseq_mouse_skin` | SHARE-seq mouse skin, RNA + ATAC | S3 | not yet | GEO [GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203) |
| `scnmt_gastrulation` | scNMT-seq mouse gastrulation, RNA + CpG + GpC | S4 | EBI (parsed bundle) | GEO [GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708) |

Datasets marked "not yet" are registered but raise `DatasetNotAvailableError` until processed copies are published; until then, obtain them from the original source. The scNMT-seq bundle is downloaded from the EBI and has no registered checksum.

Please cite the original data generators as well as UniVI (see [Citation](../citation.md)).

## Your own datasets

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
