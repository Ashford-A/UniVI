# Datasets

`univi.datasets` downloads the datasets used in the tutorials and the Genome Research article, verifies each file against its registered MD5 checksum, and caches it in `~/.cache/univi/<name>/` (or under `UNIVI_DATA_DIR`). Every dataset is a Zenodo record of `.h5ad` files, one per modality or cohort.

```python
import univi.datasets as uds

uds.list_datasets()                      # one row per dataset: files, cells, size, figures
data = uds.pbmc_multiome_10k()           # {"rna": AnnData, "atac": AnnData}
data = uds.load("pbmc_multiome_10k", keys=["rna"])
uds.dataset_info("pbmc_multiome_10k")    # registry entry: description, DOI, files, checksums
```

Please cite the studies that generated the data as well as UniVI (full references on the [citation page](../citation.md)).

## Catalog

| Dataset | Assay | Keys | Cells | Download | Paper | Zenodo |
| --- | --- | --- | --- | --- | --- | --- |
| [`hao_citeseq_pbmc`](#hao_citeseq_pbmc) | CITE-seq, human PBMCs | `rna`, `adt` | 161,764 | 799 MB | Figs. 2–3; S1 | [22883697](https://doi.org/10.5281/zenodo.22883697) |
| [`pbmc_multiome_10k`](#pbmc_multiome_10k) | 10x Multiome, human PBMCs | `rna`, `atac` | 9,631 | 205 MB | Figs. 4, 8–10; S2, S8–S11; Supp. Notebook S1 | [19581816](https://doi.org/10.5281/zenodo.19581816) |
| [`pbmc_multiome_bridge`](#pbmc_multiome_bridge) | 10x Multiome reference + scRNA-seq + scATAC-seq, human PBMCs | `rna`, `atac`, `ding_rna`, `satpathy_atac` | 12,012 paired; 30,495 RNA; 47,148 ATAC | 699 MB | Fig. 5; S5 | [22883390](https://doi.org/10.5281/zenodo.22883390) |
| [`teaseq_pbmc`](#teaseq_pbmc) | TEA-seq, human PBMCs | `rna`, `adt`, `atac` | 29,894 | 420 MB | Fig. 6; S6 | [22883520](https://doi.org/10.5281/zenodo.22883520) |
| [`aml_mosaic`](#aml_mosaic) | CITE-seq + scRNA-seq + DAb-seq, human AML | `cite_rna`, `cite_adt`, `vangalen_rna`, `dabseq_adt` | 43,179 paired; 37,627 RNA; 40,239 protein | 300 MB | Fig. 7; S7 | [22883601](https://doi.org/10.5281/zenodo.22883601) |
| [`shareseq_mouse_skin`](#shareseq_mouse_skin) | SHARE-seq, mouse skin | `rna`, `atac` | 31,220 | 293 MB | S3 | [22883542](https://doi.org/10.5281/zenodo.22883542) |
| [`scnmt_gastrulation`](#scnmt_gastrulation) | scNMT-seq, mouse embryos | `rna`, `cpg`, `gpc` | 1,140 | 224 MB | S4 | [22885463](https://doi.org/10.5281/zenodo.22885463) |

Download sizes are the sum of the registered file sizes.

## What the files contain

The hosted files are the objects the archived analysis notebooks worked with, taken after quality control, cell filtering, and annotation, and before any learned transform (feature selection, normalization, scaling, TF-IDF/LSI). Fit those transforms on the training cells yourself; the [API notebooks](api/index.md) show how for each analysis.

- **`.X`** holds raw counts, with two exceptions: the scNMT-seq CpG and GpC files hold methylation and accessibility fractions (with the read counts in layers), and the DAb-seq file of `aml_mosaic` holds the processed protein values that came with the original DAb-seq objects.
- **Paired keys** contain the same cells, and `load` returns them in identical order. Keys that are independent cohorts (for example `ding_rna` in `pbmc_multiome_bridge`) are returned as stored.
- **`layers["counts"]`**: if `.X` holds integer counts and there is no `counts` layer, `load` copies `.X` there, which is where the UniVI preprocessors look. Pass `counts_layer=None` to leave the objects untouched.
- **`obs["split"]`** is present when the analysis used a fixed train/validation/test assignment. Otherwise the dataset's section says how the split is built.

Each section below lists the files, what `.X` and `.obs` hold, how the split is defined, and the original data. The collapsed **How these files were exported** block gives the export recipe: run the named archived notebook (in [`notebooks/GR_manuscript_reproducibility/`](https://github.com/Ashford-A/UniVI/tree/main/notebooks/GR_manuscript_reproducibility)) up to the cell given, then run the snippet in a new cell. Every snippet starts with:

```python
from univi.datasets import export_dataset
```

## `hao_citeseq_pbmc`

Paired RNA and surface-protein (CITE-seq) profiles of 161,764 human PBMCs from Hao et al. (2021), with the authors' three-level cell-type annotation. Used for Figs. 2–3 and Supplemental Fig. S1.

**Record:** [Zenodo 10.5281/zenodo.22883697](https://doi.org/10.5281/zenodo.22883697) · **Original data:** Hao et al. (2021), GEO [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) · **Notebook:** [Paired CITE-seq](api/fig2_3_citeseq.ipynb), [CITE-seq tutorial](../tutorials/citeseq.ipynb)

```python
data = uds.hao_citeseq_pbmc()            # same as uds.load("hao_citeseq_pbmc")
rna, adt = data["rna"], data["adt"]
```

| Key | File | Cells | Features | `.X` | `.obs` |
| --- | --- | --- | --- | --- | --- |
| `rna` | [hao_citeseq_pbmc_rna.h5ad](https://zenodo.org/records/22883697/files/hao_citeseq_pbmc_rna.h5ad?download=1) (748.7 MB) | 161,764 | 20,729 genes | Raw counts | `celltype.l1` (8 types), `celltype.l2` (31), `celltype.l3` (58), `split`; `donor`, `time`, `lane`, `orig.ident` (24 samples), `Phase`, and RNA/ADT/SCT QC columns |
| `adt` | [hao_citeseq_pbmc_adt.h5ad](https://zenodo.org/records/22883697/files/hao_citeseq_pbmc_adt.h5ad?download=1) (49.9 MB) | 161,764 | 228 antibody-derived tags | Raw ADT counts | Same columns as `rna` |

**Split.** `obs["split"]` holds the split used in the article: stratified by `celltype.l3`, at most 1,200 cells per type in the train/validation pool (80/10), and all remaining cells in test, which gives 37,862 / 4,722 / 119,180 cells.

:::{dropdown} How these files were exported
Notebook `UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb`, through the cell that defines `train_idx`, `val_idx`, `test_idx`:

```python
export_dataset(
    "hao_citeseq_pbmc", {"rna": rna, "adt": adt}, "zenodo/hao_citeseq_pbmc",
    labels=["celltype.l1", "celltype.l2", "celltype.l3"],
    splits={"train": train_idx, "val": val_idx, "test": test_idx},
    title="Hao et al. (2021) PBMC CITE-seq (RNA + ADT)", source="GEO GSE164378",
)
```
:::

## `pbmc_multiome_10k`

Paired RNA and chromatin accessibility (10x Multiome) for 9,631 human PBMCs after QC, annotated in `obs["cell_type"]` by Seurat label transfer against the Hao et al. (2021) PBMC reference. Used for Figs. 4 and 8–10, Supplemental Figs. S2 and S8–S11, and Supplemental Notebook S1, and by the [quickstart](../tutorials/quickstart.ipynb), [query mapping](../tutorials/query_mapping.ipynb), [supervised heads](../tutorials/supervised_heads.ipynb), and [generation](../tutorials/generation.ipynb) tutorials.

**Record:** [Zenodo 10.5281/zenodo.19581816](https://doi.org/10.5281/zenodo.19581816) (CC BY 4.0) · **Original data:** 10x Genomics, "PBMC from a Healthy Donor – Granulocytes Removed Through Cell Sorting (10k)", Cell Ranger ARC 1.0.0 · **Notebook:** [Paired 10x Multiome](api/fig4_multiome.ipynb)

```python
data = uds.pbmc_multiome_10k()           # same as uds.load("pbmc_multiome_10k")
rna, atac = data["rna"], data["atac"]
```

| Key | File | Cells | Features | `.X` | `.obs` |
| --- | --- | --- | --- | --- | --- |
| `rna` | [10x-Multiome-Pbmc10k-RNA.h5ad](https://zenodo.org/records/19581816/files/10x-Multiome-Pbmc10k-RNA.h5ad?download=1) (50.7 MB) | 9,631 | 29,095 genes | Raw counts | `cell_type` (19 types); Seurat QC and clustering columns (`nCount_RNA`, `nFeature_RNA`, `percent.mt`, `nCount_ATAC`, `nFeature_ATAC`, `seurat_clusters`, …) |
| `atac` | [10x-Multiome-Pbmc10k-ATAC.h5ad](https://zenodo.org/records/19581816/files/10x-Multiome-Pbmc10k-ATAC.h5ad?download=1) (153.9 MB) | 9,631 | 107,194 peaks (`.var` has `chrom`, `chromStart`, `chromEnd`) | Raw peak counts | Same columns as `rna` |

**Split.** None is stored. The Fig. 4 analysis stratifies by `cell_type`, capping each type at 800 training and 100 validation cells (80/10 otherwise) and sending the rest to test: `split_by_label(rna.obs["cell_type"], train_fraction=0.8, val_fraction=0.1, train_cap=800, val_cap=100, seed=0)` gives 5,777 / 717 / 3,137 cells. The quickstart uses an uncapped stratified 80/10/10 split.

These two files were published before `export_dataset` existed and are the same files the archived Fig. 4 notebook reads; the Zenodo record describes their provenance.

## `pbmc_multiome_bridge`

Three human PBMC cohorts for the bridge analysis: a paired 10x Multiome reference with no curated labels (12,012 cells, keys `rna` and `atac`), an independent scRNA-seq cohort from Ding et al. (2020) (30,495 cells), and an independent scATAC-seq cohort from Satpathy et al. (2019) (47,148 cells). The two cohorts carry harmonized cell-type labels, which the analysis uses for supervised refinement and then transfers back to the reference. Used for Fig. 5 and Supplemental Fig. S5.

**Record:** [Zenodo 10.5281/zenodo.22883390](https://doi.org/10.5281/zenodo.22883390) · **Original data:** 10x Genomics (Multiome reference); Ding et al. (2020), GEO [GSE132044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132044); Satpathy et al. (2019), GEO [GSE129785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129785) · **Notebook:** [Bridging unpaired RNA and ATAC cohorts](api/fig5_bridge.ipynb)

```python
data = uds.load("pbmc_multiome_bridge")
rna, atac = data["rna"], data["atac"]                    # paired reference
ding, satpathy = data["ding_rna"], data["satpathy_atac"] # unpaired query cohorts
```

| Key | File | Cells | Features | `.X` | `.obs` |
| --- | --- | --- | --- | --- | --- |
| `rna` | [pbmc_multiome_bridge_rna.h5ad](https://zenodo.org/records/22883390/files/pbmc_multiome_bridge_rna.h5ad?download=1) (46.6 MB) | 12,012 | 9,022 genes | Raw counts | `split`; `celltype_higher_res_pred` and the related `celltype_higher_res_*` columns (predictions archived from the manuscript workflow, not curated labels); columns from the source object (`barcode`, `source`, `rep`, `tech`, `celltype`, `_scvi_*`) |
| `atac` | [pbmc_multiome_bridge_atac.h5ad](https://zenodo.org/records/22883390/files/pbmc_multiome_bridge_atac.h5ad?download=1) (113.6 MB) | 12,012 | 85,485 peaks | Raw peak counts | Same columns as `rna` |
| `ding_rna` | [pbmc_multiome_bridge_ding_rna.h5ad](https://zenodo.org/records/22883390/files/pbmc_multiome_bridge_ding_rna.h5ad?download=1) (66.4 MB) | 30,495 | 94,507 (see below) | Raw counts | `celltype_harmonized_coarse` (8 types), `celltype_harmonized` (10), `celltype_raw` |
| `satpathy_atac` | [pbmc_multiome_bridge_satpathy_atac.h5ad](https://zenodo.org/records/22883390/files/pbmc_multiome_bridge_satpathy_atac.h5ad?download=1) (472.3 MB) | 47,148 | 94,507 (see below) | Raw counts | `celltype_harmonized_coarse` (7 types), `celltype_harmonized` (12), `celltype_raw` |

The two query files share one feature space of 94,507 features, the size of the reference's genes and peaks combined (9,022 + 85,485): `ding_rna` contains all 9,022 reference genes and `satpathy_atac` all 85,485 reference peaks. Select each cohort's own features before preprocessing, as the Fig. 5 notebook does:

```python
ding = ding[:, rna.var_names].copy()
satpathy = satpathy[:, atac.var_names].copy()
```

**Split.** `obs["split"]` on the reference holds the archived notebook's random 90/10 train/validation assignment (seed 42): 10,810 training and 1,201 validation cells, plus one leftover cell marked `unassigned`. The Ding and Satpathy cohorts are the query sets and have no split.

:::{dropdown} How these files were exported
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
:::

## `teaseq_pbmc`

Trimodal TEA-seq profiles of human PBMCs from Swanson et al. (2021): RNA, surface protein, and chromatin accessibility measured in the same 29,894 cells across four wells. Used for Fig. 6 and Supplemental Fig. S6.

**Record:** [Zenodo 10.5281/zenodo.22883520](https://doi.org/10.5281/zenodo.22883520) · **Original data:** Swanson et al. (2021), GEO [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013) · **Notebook:** [Trimodal TEA-seq, held-out well](api/fig6_teaseq.ipynb)

```python
data = uds.load("teaseq_pbmc")
rna, adt, atac = data["rna"], data["adt"], data["atac"]
```

| Key | File | Cells | Features | `.X` | `.obs` |
| --- | --- | --- | --- | --- | --- |
| `rna` | [teaseq_pbmc_rna.h5ad](https://zenodo.org/records/22883520/files/teaseq_pbmc_rna.h5ad?download=1) (76.5 MB) | 29,894 | 36,601 genes | Raw counts | `sample_id` (well), `barcode_stripped` |
| `adt` | [teaseq_pbmc_adt.h5ad](https://zenodo.org/records/22883520/files/teaseq_pbmc_adt.h5ad?download=1) (4.7 MB) | 29,894 | 47 antibody-derived tags (those measured in all four wells) | Raw ADT counts | `sample_id` |
| `atac` | [teaseq_pbmc_atac.h5ad](https://zenodo.org/records/22883520/files/teaseq_pbmc_atac.h5ad?download=1) (339.0 MB) | 29,894 | 155,930 genome-wide 500-bp tiles (open in 0.5–80% of training-well cells) | Raw tile counts | `sample_id`; per-cell ATAC QC and metadata from the original per-well tables (`n_fragments`, `frac_mito`, `TSSEnrichment`, `DoubletScore`, `well_id`, …) |

The three files store the same cells in the same order.

**Split.** None is stored; the split is by well. The article trains on wells 3, 4, and 6 and evaluates on held-out well 5 (`sample_id == "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea"`). The API notebook holds out a random 10% of training-well cells for early stopping, giving 20,226 training, 2,247 validation, and 7,421 held-out cells.

:::{dropdown} How these files were exported
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
:::

## `aml_mosaic`

A mosaic of three acute myeloid leukemia cohorts that were never measured together: paired CITE-seq from Knorr et al. (2023) (43,179 cells from 11 bone-marrow samples, 8 newly diagnosed AML and 3 age-matched normal donors; keys `cite_rna` and `cite_adt`), which trains the RNA–protein bridge; scRNA-seq from van Galen et al. (2019) (37,627 cells) with per-cell mutant and wild-type transcript calls; and DAb-seq from Demaree et al. (2021) (40,239 cells) with surface protein and single-cell genotype calls. Used for Fig. 7 and Supplemental Fig. S7.

**Record:** [Zenodo 10.5281/zenodo.22883601](https://doi.org/10.5281/zenodo.22883601) · **Original data:** Knorr et al. (2023), GEO [GSE220473](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220473) (the CITE-seq SubSeries of SuperSeries [GSE220474](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220474)); van Galen et al. (2019), GEO [GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256); Demaree et al. (2021), BioProject [PRJNA602320](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA602320) · **Notebook:** [AML mosaic integration and mutation heads](api/fig7_aml.ipynb)

```python
data = uds.load("aml_mosaic")
cite_rna, cite_adt = data["cite_rna"], data["cite_adt"]  # paired bridge
vg, dab = data["vangalen_rna"], data["dabseq_adt"]       # RNA-only and protein-only cohorts
```

| Key | File | Cells | Features | `.X` | `.obs` |
| --- | --- | --- | --- | --- | --- |
| `cite_rna` | [aml_mosaic_cite_rna.h5ad](https://zenodo.org/records/22883601/files/aml_mosaic_cite_rna.h5ad?download=1) (190.7 MB) | 43,179 | 21,782 genes (the same genes as `vangalen_rna`) | Raw counts | `sample_id` (11 samples), `library_id`, `split` |
| `cite_adt` | [aml_mosaic_cite_adt.h5ad](https://zenodo.org/records/22883601/files/aml_mosaic_cite_adt.h5ad?download=1) (3.8 MB) | 43,179 | 18 surface markers (the same panel as `dabseq_adt`) | Raw ADT counts | Same columns as `cite_rna` |
| `vangalen_rna` | [aml_mosaic_vangalen_rna.h5ad](https://zenodo.org/records/22883601/files/aml_mosaic_vangalen_rna.h5ad?download=1) (97.1 MB) | 37,627 | 21,782 genes | Raw counts | `MutTranscripts`, `WtTranscripts`, `CellType`, `PredictionRefined`, `orig.ident` (41 samples) |
| `dabseq_adt` | [aml_mosaic_dabseq_adt.h5ad](https://zenodo.org/records/22883601/files/aml_mosaic_dabseq_adt.h5ad?download=1) (8.4 MB) | 40,239 | 18 surface markers | Processed protein values from the original objects (not counts) | Genotype calls `NPM1 W288fs`, `DNMT3A R882H`, `FLT3-ITD`; `experiment` (8 experiments), `dab_source` (2), `leiden` |

Genotype truth differs by cohort. For van Galen cells it is derived per gene from the `MutTranscripts`/`WtTranscripts` strings; DAb-seq cells have direct calls for NPM1, DNMT3A, and FLT3 only, stored in the variant-level columns of the original objects (there are no gene-level `mut_<GENE>` columns), so other genes should be treated as unmeasured for them rather than wild type. Because `dabseq_adt` does not hold counts, `load` adds no `counts` layer to it; the Fig. 7 notebook standardizes those values within the cohort.

**Split.** `obs["split"]` in `cite_rna` and `cite_adt` assigns whole samples: 8 training, 1 validation, and 2 test samples (28,414 / 4,652 / 10,113 cells). `vangalen_rna` and `dabseq_adt` have no split; the API notebook's mutation heads hold out whole van Galen patients (`orig.ident`) and DAb-seq experiments (`experiment`).

:::{dropdown} How these files were exported
Notebook `UniVI_manuscript_GR-Figure__7__AML_bridge_mapping_and_fine-tuning.ipynb`. Right after the cell that defines the grouped `splits`, keep a copy of them:

```python
aml_splits = {k: list(splits[k]) for k in ("train_obs", "val_obs", "test_obs")}
```

Then export. `dab_adt_al` keeps the processed protein values and the variant-level genotype columns (`NPM1 W288fs`, `DNMT3A R882H`, `FLT3-ITD`) of the original DAb-seq objects, as in the hosted file; van Galen cells keep their `MutTranscripts` / `WtTranscripts` fields:

```python
export_dataset(
    "aml_mosaic",
    {"cite_rna": cite_rna_al, "cite_adt": cite_adt_al, "vangalen_rna": vg_rna_al, "dabseq_adt": dab_adt_al},
    "zenodo/aml_mosaic", paired=["cite_rna", "cite_adt"], labels=["sample_id"],
    splits={"train": aml_splits["train_obs"], "val": aml_splits["val_obs"], "test": aml_splits["test_obs"]},
    title="AML mosaic: CITE-seq bridge (Knorr et al. 2023), van Galen et al. (2019) scRNA-seq, DAb-seq (Demaree et al. 2021)",
    source="GEO GSE220473; GEO GSE116256; BioProject PRJNA602320",
)
```
:::

## `shareseq_mouse_skin`

Paired RNA and chromatin accessibility (SHARE-seq) from late-anagen mouse back skin (Ma et al. 2020): 31,220 cells in 22 annotated cell types, with sparser ATAC than 10x Multiome. Used for Supplemental Fig. S3.

**Record:** [Zenodo 10.5281/zenodo.22883542](https://doi.org/10.5281/zenodo.22883542) · **Original data:** Ma et al. (2020), GEO [GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203) (RNA: GSM4156608; ATAC: GSM4156597) · **Notebook:** [SHARE-seq mouse skin](api/figS3_shareseq.ipynb)

```python
data = uds.load("shareseq_mouse_skin")
rna, atac = data["rna"], data["atac"]
```

| Key | File | Cells | Features | `.X` | `.obs` |
| --- | --- | --- | --- | --- | --- |
| `rna` | [shareseq_mouse_skin_rna.h5ad](https://zenodo.org/records/22883542/files/shareseq_mouse_skin_rna.h5ad?download=1) (46.6 MB) | 31,220 | 23,296 genes | Raw counts | `cell_type` (22 types), `split`, `stage`; QC columns (`total_counts`, `n_genes_by_counts`, `pct_counts_mt`, …) |
| `atac` | [shareseq_mouse_skin_atac.h5ad](https://zenodo.org/records/22883542/files/shareseq_mouse_skin_atac.h5ad?download=1) (246.0 MB) | 31,220 | 340,341 peaks (after removing those overlapping the ENCODE mm10 blacklist) | Raw peak counts | `cell_type`, `split`, `stage`, `total_counts`, `n_genes_by_counts` |

The cells passed RNA QC (300–30,000 counts, at least 200 genes, at most 5% mitochondrial) and ATAC QC (250–12,500 peak counts), were measured in both modalities, and are not in the original authors' ambiguous "Mix" category. Peak-frequency filtering is left to the preprocessor; the S3 notebook keeps peaks open in 0.25–80% of training cells.

**Split.** `obs["split"]` holds 21,845 training, 3,112 validation, and 6,263 test cells. These sizes match a split stratified by `cell_type` at 70/10/20 and differ from the archived S3 notebook, which used a stratified 80/10/10 split (24,969 / 3,112 / 3,139 cells). The [S3 API notebook](api/figS3_shareseq.ipynb) uses the hosted split.

:::{dropdown} How these files were exported
Notebook `UniVI_manuscript_GR-Supple_____mouse_skin_SHARE-seq_integration.ipynb`, through the stratified split cell (`splits`):

```python
export_dataset(
    "shareseq_mouse_skin", {"rna": skin_rna, "atac": skin_atac}, "zenodo/shareseq_mouse_skin",
    labels=["cell_type"], splits={k: splits[k] for k in ("train", "val", "test")},
    title="SHARE-seq mouse skin, late anagen (RNA + ATAC)", source="GEO GSE140203 (GSM4156597, GSM4156608)",
)
```

Run as written, this exports the archived notebook's 80/10/10 split. The hosted files carry a split with 70/10/20 sizes instead; calling the notebook's `stratified_split_indices` with `train_frac=0.70, val_frac=0.10, test_frac=0.20` gives the same sizes.
:::

## `scnmt_gastrulation`

Paired RNA, CpG methylation, and GpC accessibility (scNMT-seq) for 1,140 cells from mouse gastrulation (Argelaguet et al. 2019), with CpG and GpC signal summarized over gene bodies. Used for Supplemental Fig. S4.

**Record:** [Zenodo 10.5281/zenodo.22885463](https://doi.org/10.5281/zenodo.22885463) ([files, manifest, and `release_audit.tsv`](https://zenodo.org/records/22885463)) · **Original data:** Argelaguet et al. (2019), GEO [GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708) · **Notebook:** [scNMT-seq mouse gastrulation](api/figS4_scnmt.ipynb)

```python
from univi.datasets import build_univi_inputs_from_scnmt_triplet

data = uds.scnmt_gastrulation()          # same as uds.load("scnmt_gastrulation")
rna, cpg, gpc = data["rna"], data["cpg"], data["gpc"]
built = build_univi_inputs_from_scnmt_triplet(rna, cpg, gpc)
adata_dict = built["adata_dict"]
recon_targets_spec = built["recon_targets_spec"]
```

| Key | File | Cells | Features | `.X` | Layers |
| --- | --- | --- | --- | --- | --- |
| `rna` | [scnmt_gastrulation_rna.h5ad](https://zenodo.org/records/22885463/files/scnmt_gastrulation_rna.h5ad?download=1) (46.8 MB) | 1,140 | 22,084 genes | Raw counts | `counts` |
| `cpg` | [scnmt_gastrulation_cpg.h5ad](https://zenodo.org/records/22885463/files/scnmt_gastrulation_cpg.h5ad?download=1) (74.4 MB) | 1,140 | 18,285 gene bodies | Methylation fractions | `meth_successes`, `meth_total_count` |
| `gpc` | [scnmt_gastrulation_gpc.h5ad](https://zenodo.org/records/22885463/files/scnmt_gastrulation_gpc.h5ad?download=1) (102.8 MB) | 1,140 | 18,325 gene bodies | Accessibility fractions | `acc_successes`, `acc_total_count` |

The three files contain the same cells in the same order, and each carries the EBI sample metadata (`stage` with 4 values, `lineage10x` with 23, `lineage10x_2`, `embryo`, `plate`, and the `pass_rnaQC` / `pass_metQC` / `pass_accQC` flags) plus `split`; the CpG and GpC files add per-cell `coverage_total` and `n_features_covered`. The success and coverage layers are what the beta-binomial likelihoods use: `build_univi_inputs_from_scnmt_triplet` log-normalizes RNA while keeping the raw counts, and returns the reconstruction targets that point the CpG and GpC decoders at those layers. The 33 GB source archive is not needed for this workflow.

**Split.** `obs["split"]` holds the article's seed-0 85/5/10 split (969 / 57 / 114 cells).

:::{dropdown} How these files were built
Unlike the other datasets, these files were not written with `export_dataset`, which keeps only `.X`. They were parsed from the EBI's feature-level bundle (Argelaguet et al. 2019) with the reader settings of the archived Supplemental Fig. S4 notebook (no QC or feature filtering, minimum coverage 1 for CpG and GpC) and written directly, keeping the layers the beta-binomial likelihoods need:

```python
from univi.datasets import load_scnmt_gastrulation_genebody_triplet

triplet = load_scnmt_gastrulation_genebody_triplet("scnmt_gastrulation.tar.gz", require_qc=False,
                                                   filter_features=False, min_cov_cpg=1, min_cov_gpc=1)
# obs["split"]: seed-0 85/5/10 split; then write each object with .write_h5ad(..., compression="gzip")
```
:::

## Loading options

`uds.load(name, keys=None, *, data_dir=None, counts_layer="counts", force=False, progress=True)` downloads what is missing, verifies it, and returns `{key: AnnData}`. `keys` loads a subset of files, `data_dir` overrides `UNIVI_DATA_DIR`, and `force=True` downloads again. A cached file is checked against its MD5 each time it is used and downloaded again if it does not match. `uds.fetch(name, keys)` does the download and verification only and returns `{key: Path}`, which is convenient for pointing other tools at the cached files.

## Using your own or mirrored data

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

## Packaging and publishing a dataset

`export_dataset` is how the hosted datasets above were written. It stores raw counts in `.X`, the selected `.obs` columns, and an optional split in `obs["split"]`, one `.h5ad` per key, together with a `manifest.json` (checksums, shapes, split counts) and a `DESCRIPTION.md` for the record:

```python
uds.export_dataset("my_cite", {"rna": rna, "adt": adt}, "export/my_cite",
                   labels=["cell_type"], splits={"train": train_ids, "val": val_ids, "test": test_ids})
```

`scripts/zenodo_upload.py` uploads the exported files to a Zenodo draft, and `scripts/prepare_zenodo_release.py register` adds the published record to the registry:

```bash
python scripts/zenodo_upload.py zenodo/hao_citeseq_pbmc --creator "Ashford, Andrew J." --orcid 0000-0002-1234-2118
# review the draft on zenodo.org, then publish it there
python scripts/prepare_zenodo_release.py register zenodo/hao_citeseq_pbmc --record-id <record id>
python -c "import univi.datasets as u; print(u.list_datasets())"
```

`zenodo_upload.py` needs a Zenodo token with the `deposit:write` scope in the `ZENODO_TOKEN` environment variable; `--sandbox` does a dry run on sandbox.zenodo.org. It verifies each uploaded file's checksum and links the record to the article's DOI. `register` writes the download URLs and MD5 checksums into `univi/datasets/registry.json`, which ships with the next UniVI release. Until then, the updated registry file can be used directly with the `UNIVI_DATASET_REGISTRY` environment variable.

Before publishing, confirm that each original source permits redistribution of processed data, and keep the source citations in the record description.
