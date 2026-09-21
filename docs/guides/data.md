# Data acquisition and preparation

The tutorials consume explicit AnnData files so that the statistical analysis is separated from dataset-specific downloads and file conversion. Real cohort data and trained publication checkpoints are not included in this documentation bundle. The attached notebooks contain the original acquisition/readers and remain the authoritative reference for source-specific parsing.

## Cohorts and accessions

| Tutorial | Dataset / source | Starting material and preparation |
| --- | --- | --- |
| CITE-seq | Hao et al., [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) | Paired RNA and antibody counts with hierarchical cell-type labels |
| Figure 4 / Figure 8 | [10x public datasets](https://www.10xgenomics.com/datasets), “10k Human PBMCs, Multiome v1.0, Chromium X” | RNA and peak matrices; the annotated version used in the notebooks supplies `cell_type` |
| Figure 5 reference | 10x, “PBMC from a Healthy Donor – No Cell Sorting (10k)” | A distinct paired reference; do not substitute Figure 4's dataset without recording the change |
| Figure 5 RNA query | Ding et al., [GSE132044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132044) | RNA counts, technology, harmonized coarse immune identities |
| Figure 5 ATAC query | Satpathy et al., [GSE129785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129785) | Accessibility quantified in the reference peak universe; harmonized labels |
| TEA-seq | Swanson et al., [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013) | RNA, ADT, and a common ATAC tile matrix across wells 3–6 |
| AML bridge | Knorr et al., [GSE220474](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220474) | Paired RNA/ADT and sample assignments |
| AML RNA | van Galen et al., [GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256) | RNA counts, cell states and sparse targeted transcript genotype calls |
| AML protein/genotype | Demaree et al., [PRJNA602320](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA602320) | Protein measurements and targeted genotypes; track raw versus processed values |
| SHARE-seq skin | Ma et al., [GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203), sample GSM4156597 | Paired RNA/ATAC, original cell-type annotation; mm10 blacklist filtering |
| scNMT-seq | Argelaguet et al., [GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708) | [Parsed EBI bundle](https://ftp.ebi.ac.uk/pub/databases/scnmt_gastrulation/scnmt_gastrulation.tar.gz), as cited in the supplement |

These accessions are transcribed from the published manuscript. Verify the chosen processed release and record its filenames/checksums; an accession may contain multiple samples, pipelines, or processed matrices. A raw-data accession is not automatically a direct download of the notebook's prepared `.h5ad` objects.

## Common AnnData contract

Each object must have cells in rows and features in columns. Store raw counts/abundance in `layers["counts"]` before any normalization. Use unique cell IDs and unique, version-consistent feature IDs. The tutorials set model `.X` through the fitted preprocessing helpers.

```python
import scanpy as sc
from univi.data import align_paired_obs_names

rna = sc.read_h5ad("prepared_rna.h5ad")
adt = sc.read_h5ad("prepared_adt.h5ad")
# Only do this if .X really contains the original raw counts.
rna.layers["counts"] = rna.X.copy()
adt.layers["counts"] = adt.X.copy()
paired = align_paired_obs_names({"rna": rna, "adt": adt})
paired["rna"].write_h5ad("data/tutorials/cite/rna.h5ad")
paired["adt"].write_h5ad("data/tutorials/cite/adt.h5ad")
```

Create the destination directory first. Never declare normalized, negative CLR, or scaled values to be raw counts merely to satisfy the filename contract.

For a 10x HDF5 feature-barcode matrix:

```python
import scanpy as sc
matrix = sc.read_10x_h5("filtered_feature_bc_matrix.h5", gex_only=False)
# Inspect matrix.var["feature_types"] before selecting; source labels vary.
rna = matrix[:, matrix.var["feature_types"] == "Gene Expression"].copy()
adt = matrix[:, matrix.var["feature_types"] == "Antibody Capture"].copy()
for obj in (rna, adt):
    obj.layers["counts"] = obj.X.copy()
```

For a peak matrix, preserve interval identifiers and genome build. Do not make repeated gene symbols unique by arbitrary suffixing and then expect them to match another cohort's unsuffixed symbols. Choose an explicit aggregation or stable-ID mapping and retain it.

## Required tutorial files and metadata

| Input directory | Files | Metadata |
| --- | --- | --- |
| `cite/` | `rna.h5ad`, `adt.h5ad` | `celltype.l1`, `.l2`, `.l3` copied consistently to paired objects |
| `multiome/` | `rna.h5ad`, `atac.h5ad` | `cell_type` |
| `bridge/` | `rna.h5ad`, `atac.h5ad`, `ding_rna.h5ad`, `satpathy_atac.h5ad` | Query `celltype_harmonized`, `technology`; reference labels optional |
| `tea/` | `rna.h5ad`, `adt.h5ad`, `atac.h5ad` | `well`; optional `n_fragments` on ATAC |
| `aml/` | `rna.h5ad`, `adt.h5ad`, `van_galen_rna.h5ad`, `dab_adt.h5ad`, `splits.tsv` | Bridge `sample_id`; query `mut_NPM1`, etc.; optional cell states/patient/disease fields |
| `benchmark/` | `rna.h5ad`, `atac.h5ad` | `cell_type` |
| `share/` | `rna.h5ad`, `atac.h5ad` | `cell_type`; ATAC `uns["blacklist_removed"]=True` after actual filtering |
| `scnmt/` | `scnmt_gastrulation.tar.gz` | Metadata read from the bundle |
| `raw/` | `rna.h5ad`, `atac.h5ad`, `peak_gene_links.tsv` | `cell_type`; ATAC interval columns |

`UNIVI_DATA` points to the parent of these directories. Prepared filenames are tutorial conventions, not claims about the names deposited by the original authors.

### Preserve barcode identity when joining annotations

```python
import pandas as pd
metadata = pd.read_csv("cell_metadata.tsv", sep="\t", index_col=0)
assert metadata.index.is_unique
missing = rna.obs_names.difference(metadata.index)
assert len(missing) == 0, f"Missing metadata for {len(missing)} cells"
rna.obs = rna.obs.join(metadata, how="left")
```

Check duplicate columns before joining. For paired modalities, propagate annotations by barcode, not by whichever row order a CSV happened to use. For multiple wells, prefix barcodes with the well/sample ID before concatenation, consistently across all paired assays.

### Barcode split maps

The CITE, Multiome, paired bridge, AML reference, SHARE-seq, scNMT, and raw-feature tutorials can read `splits.tsv`:

```text
cell_id	split
sampleA:AAAC...	train
sampleA:AAAG...	val
sampleB:TTGC...	test
```

The table must cover each retained paired barcode exactly once. Supply it after the intended QC/filtering. For AML, explicitly group the paired reference by sample/patient; the example rejects patient overlap across partitions. Make mutation-head splits separately and report their grouping unit. Their cell-level tutorial splits do not inherit the bridge's patient-level validity automatically.

TEA-seq constructs its split directly from the well field: 80% of reference-well cells train, 10% validate, the remaining 10% of those wells are unused, and well 5 tests. The Figure 8 illustration constructs per-seed splits in its loop. To reproduce archived barcodes in these two scripts, replace that explicit split-construction cell and retain the stated well/group restrictions.

## Dataset-specific preparation that should remain explicit

### Bridge ATAC features

Use the Figure 5 notebook's preprocessed/recounted input route, or recount query fragments in the exact reference genomic intervals before fitting/applying LSI. Equal dimensionality is insufficient. An ATAC matrix with a different feature order must be reordered by ID; genuinely missing reference peaks should trigger a data-preparation decision rather than silent zero filling.

### TEA-seq tiles

The source Figure 6 notebook reads wells separately, harmonizes barcodes per sample, and builds a global tile universe. Its holdout prefix is `X066-MP0C1W5`. Preserve that mapping when populating the simplified `well` field. Create a single tile vocabulary across all relevant assays, then learn frequencies, IDF and SVD from training cells only. Defining common coordinate bins is distinct from fitting their statistical transforms.

### AML proteins and genotypes

Use Figure 7's `canon_from_varname` and `align_cite_dab_adts` logic as the source reference for panel matching; inspect antibody clones/replicates instead of stripping every suffix indiscriminately. The paper uses 18 shared markers, but a different release can yield a different intersection and should be reported.

The portable tutorial requires nonnegative raw antibody abundances for both cohorts. If only already transformed DAb values are available, do not apply CLR again. Recreate the original notebook's explicitly processed-input branch and document that change, or obtain the raw matrix. That branch is a cohort-specific choice, not a safe automatic guess.

Populate mutation columns from the original per-cell evidence. For a binary table with explicit missing values:

```python
mutation = pd.read_csv("validated_mutation_calls.tsv", sep="\t", index_col=0)
assert mutation.index.is_unique
for gene in ["NPM1", "DNMT3A", "FLT3", "TP53", "NRAS", "TET2", "IDH2"]:
    if gene in mutation:
        calls = mutation[gene].reindex(van_galen.obs_names)
        assert calls.dropna().isin([0, 1]).all()
        van_galen.obs[f"mut_{gene}"] = calls
```

Do not use `.fillna(0)` on mutation calls. For transcript strings and assay-specific DAb annotations, use the source notebook's `build_YM_from_mut_wt_strings`, `build_YM_from_obs_cols`, and label-coverage reports as parsing references. Ambiguous or conflicting labels need an explicit resolution rule.

### SHARE-seq blacklist

Remove peaks overlapping the ENCODE mm10 blacklist using the same genome assembly as the peak matrix. The source notebook contains the interval filtering and RNA/ATAC QC sequence. Preserve the excluded peak list, counts before/after filtering, and the original `Mix` annotation; the tutorial then applies its stated cell filters.

### scNMT bundle

The promoted adapter reads the tar members directly, without extracting the archive. The RNA file is a wide gene-by-cell matrix; the genebody files are headerless long tables with cell ID, feature ID, feature class, successes, coverage, and percent. It maps RNA IDs through metadata before pairing modalities. Use successes and coverage, not a percent-only matrix, for the beta-binomial targets.

The adapter can load the source proof of concept without additional QC/filtering. If you enable further feature filtering before splitting, that selection uses all supplied cells; for a strictly inductive analysis, derive new filters from the training partition only.

## Data provenance to save

Keep accession, release, file checksums, genome annotation/build, feature mapping, antibody panel, QC rules, barcode namespace, split map and transform bundle together. The tutorials save model-side state, but they cannot infer upstream decisions from a matrix alone.
