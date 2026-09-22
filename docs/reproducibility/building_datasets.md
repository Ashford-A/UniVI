# How the hosted datasets are built

The datasets in `univi.datasets` are the objects the archived analysis notebooks worked with, taken at the point where quality control, cell filtering, and annotation are done but before any learned transform (feature selection, scaling, TF-IDF/LSI). Each file stores raw counts in `.X`, the cell metadata in `.obs` and, where the analysis defined one, the train/validation/test assignment in `.obs["split"]`. Anyone with the original data can rebuild them from the archived notebooks as described below.

`univi.datasets.export_dataset` writes the files, a `manifest.json` with their md5 checksums, and a `DESCRIPTION.md`. `scripts/zenodo_upload.py` uploads them to a Zenodo draft, and `scripts/prepare_zenodo_release.py register` adds the published record to the registry.

## Exporting from the archived notebooks

Run the named notebook up to the point given, then run the export code in a new cell (it does not need to be saved in the notebook). All snippets start with:

```python
from univi.datasets import export_dataset
```

### `hao_citeseq_pbmc` (Figs. 2–3, S1)

Notebook `UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb`, through the cell that defines `train_idx`, `val_idx`, `test_idx`:

```python
export_dataset(
    "hao_citeseq_pbmc", {"rna": rna, "adt": adt}, "zenodo/hao_citeseq_pbmc",
    labels=["celltype.l1", "celltype.l2", "celltype.l3"],
    splits={"train": train_idx, "val": val_idx, "test": test_idx},
    title="Hao et al. (2021) PBMC CITE-seq (RNA + ADT)", source="GEO GSE164378",
)
```

### `pbmc_multiome_bridge` (Figs. 5, S5)

Notebook `UniVI_manuscript_GR-Figure__5__Multiome_bridge_mapping_and_fine-tuning.ipynb`, through the cell that adds `celltype_harmonized_coarse`. The Multiome reference has no curated cell-type labels; the classifier head is trained on the harmonized labels of the Ding and Satpathy cohorts, so confirm those columns first:

```python
objs = {"rna": rna, "atac": atac, "ding_rna": uni_rna, "satpathy_atac": uni_atac}
for k, a in objs.items():
    print(k, a.shape, [c for c in a.obs.columns if "celltype" in c])

export_dataset(
    "pbmc_multiome_bridge", objs, "zenodo/pbmc_multiome_bridge", paired=["rna", "atac"],
    labels=["celltype_harmonized_coarse"], splits={"train": idx_train, "val": idx_val},
    title="10x Multiome PBMC reference with Ding et al. (2020) scRNA-seq and Satpathy et al. (2019) scATAC-seq",
    source="10x Genomics; GEO GSE132044; GEO GSE129785",
)
```

### `teaseq_pbmc` (Figs. 6, S6)

Notebook `UniVI_manuscript_GR-Figure__6__TEA-seq_tri-modal.ipynb`, through the ATAC cell that builds the tile matrices (`tile_mats`). The per-well raw objects are stacked, the protein panel is restricted to markers present in every well, and tiles are filtered as in the notebook (open in 0.5–80% of training-well cells), which keeps the file a manageable size:

```python
import anndata as ad
import numpy as np

wells = list(teaseq_prefixes)

def stack(parts):
    out = ad.concat(parts, label="_well", keys=wells, index_unique=None)
    out.obs["sample_id"] = out.obs.pop("_well").astype(str)
    return out

rna_parts = [rna_raw_dict[p].copy() for p in wells]
for a in rna_parts:
    a.var_names_make_unique()
panel = sorted(set.intersection(*[set(adt_raw_dict[p].var_names) for p in wells]))
rna_all = stack(rna_parts)
adt_all = stack([adt_raw_dict[p][:, panel].copy() for p in wells])
atac_all = stack([ad.AnnData(X=tile_mats[p][1], obs=tile_mats[p][0].obs.copy(), var=tile_mats[p][0].var.copy())
                  for p in wells])

in_train = (atac_all.obs["sample_id"] != HOLDOUT_PREFIX).to_numpy()
frac = np.asarray((atac_all.X[in_train] > 0).mean(axis=0)).ravel()
atac_all = atac_all[:, (frac >= MIN_TILE_FRAC) & (frac <= MAX_TILE_FRAC)]

common = rna_all.obs_names.intersection(adt_all.obs_names).intersection(atac_all.obs_names)
export_dataset(
    "teaseq_pbmc", {"rna": rna_all[common], "adt": adt_all[common], "atac": atac_all[common]}, "zenodo/teaseq_pbmc",
    title="TEA-seq PBMCs (RNA + ADT + ATAC tiles), four wells", source="GEO GSE158013",
)
```

### `aml_mosaic` (Figs. 7, S7)

Notebook `UniVI_manuscript_GR-Figure__7__AML_bridge_mapping_and_fine-tuning.ipynb`. Right after the cell that defines the grouped `splits`, keep a copy of them:

```python
aml_splits = {k: list(splits[k]) for k in ("train_obs", "val_obs", "test_obs")}
```

Then continue through the cells that define `find_mut_col_dab_obs` and `coerce_to_nullable_boolean`, and export. DAb-seq genotype calls are written as `mut_<GENE>` columns (1, 0, or missing); van Galen cells keep their `MutTranscripts` / `WtTranscripts` fields:

```python
import pandas as pd

dab = dab_adt_al.copy()
for gene in ["NPM1", "DNMT3A", "FLT3", "TP53", "TET2", "IDH2"]:
    col = find_mut_col_dab_obs(dab, gene)
    if col is not None:
        calls = coerce_to_nullable_boolean(dab.obs[col])
        dab.obs[f"mut_{gene}"] = pd.to_numeric(calls.map({True: 1.0, False: 0.0}), errors="coerce")

export_dataset(
    "aml_mosaic",
    {"cite_rna": cite_rna_al, "cite_adt": cite_adt_al, "vangalen_rna": vg_rna_al, "dabseq_adt": dab},
    "zenodo/aml_mosaic", paired=["cite_rna", "cite_adt"], labels=["sample_id"],
    splits={"train": aml_splits["train_obs"], "val": aml_splits["val_obs"], "test": aml_splits["test_obs"]},
    title="AML mosaic: CITE-seq bridge (Knorr et al. 2023), van Galen et al. (2019) scRNA-seq, DAb-seq (Demaree et al. 2021)",
    source="GEO GSE220474; GEO GSE116256; BioProject PRJNA602320",
)
```

### `shareseq_mouse_skin` (Supplemental Fig. S3)

Notebook `UniVI_manuscript_GR-Supple_____mouse_skin_SHARE-seq_integration.ipynb`, through the stratified split cell (`splits`):

```python
export_dataset(
    "shareseq_mouse_skin", {"rna": skin_rna, "atac": skin_atac}, "zenodo/shareseq_mouse_skin",
    labels=["cell_type"], splits={k: splits[k] for k in ("train", "val", "test")},
    title="SHARE-seq mouse skin, late anagen (RNA + ATAC)", source="GEO GSE140203 (GSM4156597, GSM4156608)",
)
```

### Already available

`pbmc_multiome_10k` is on [Zenodo](https://doi.org/10.5281/zenodo.19581816).

`scnmt_gastrulation` is available at [Zenodo 10.5281/zenodo.22885463](https://doi.org/10.5281/zenodo.22885463). The three `.h5ad` files contain 1,140 paired cells (22,084 RNA genes, 18,285 CpG features, and 18,325 GpC features), including the success/coverage layers and the seed-0 85/5/10 split. Load the published files with `uds.load("scnmt_gastrulation")`; see [file links and representation](datasets.md#scnmt-seq-mouse-gastrulation-supplemental-fig-s4).

To rebuild these files from the source archive: the dataset was parsed from the EBI's feature-level bundle (Argelaguet et al. 2019) with the settings of the archived Supplemental Fig. S4 notebook, and written directly, keeping the layers the beta-binomial likelihoods need (`export_dataset` is not used here because it keeps only `.X`):

```python
from univi.datasets import load_scnmt_gastrulation_genebody_triplet

triplet = load_scnmt_gastrulation_genebody_triplet("scnmt_gastrulation.tar.gz", require_qc=False,
                                                   filter_features=False, min_cov_cpg=1, min_cov_gpc=1)
# obs["split"]: seed-0 85/5/10 split; then write each object with .write_h5ad(..., compression="gzip")
```

## Uploading and registering

```bash
python scripts/zenodo_upload.py zenodo/hao_citeseq_pbmc --creator "Ashford, Andrew J." --orcid 0000-0002-1234-2118
# review the draft on zenodo.org, then publish it there
python scripts/prepare_zenodo_release.py register zenodo/hao_citeseq_pbmc --record-id <record id>
python -c "import univi.datasets as u; print(u.list_datasets())"
```

`zenodo_upload.py` needs a Zenodo token with the `deposit:write` scope in the `ZENODO_TOKEN` environment variable; `--sandbox` does a dry run on sandbox.zenodo.org. It verifies each uploaded file's checksum and links the record to the article's DOI. `register` writes the download URLs and md5 checksums into `univi/datasets/registry.json`, which ships with the next UniVI release. Until then, the updated registry file can be used directly with the `UNIVI_DATASET_REGISTRY` environment variable.

Before publishing, confirm that each original source permits redistribution of processed data, and keep the source citations in the record description.
