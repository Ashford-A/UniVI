# Preparing data

## One AnnData per modality

UniVI takes a dictionary that maps a modality name to an AnnData object:

```python
adata_by_mod = {"rna": rna, "adt": adt}            # CITE-seq
adata_by_mod = {"rna": rna, "atac": atac}          # Multiome, SHARE-seq
adata_by_mod = {"rna": rna, "adt": adt, "atac": atac}  # TEA-seq, DOGMA-seq
```

The names are your choice; they must match the `name` of each `ModalityConfig` and are used everywhere else (`modality="rna"`, `src_mod="atac"`, ...).

**Paired modalities must describe the same cells in the same order.** `univi.datasets` returns aligned objects. For your own data, intersect and order the barcodes once:

```python
from univi.data import align_paired_obs_names
adata_by_mod = align_paired_obs_names({"rna": rna, "atac": atac})  # keeps shared cells, same order
```

`MultiModalDataset` (used by `make_loader`) raises an error if the `obs_names` of paired modalities differ.

## Where UniVI reads values

| What | Where | Why |
| --- | --- | --- |
| Raw counts | `.layers["counts"]` | read by the preprocessors; never overwritten |
| Model input | `.X`, or an `.obsm` key | what the encoder sees and the decoder reconstructs |
| Labels, batches, metadata | `.obs` | used for splits, evaluation, heads |

To feed an `.obsm` matrix instead of `.X`, pass its key per modality: `MultiModalDataset(adata_by_mod, X_key={"atac": "X_lsi"})`. The UniVI preprocessors put their output in `.X`, so with them this is not needed.

## Split first, then fit preprocessing

Anything learned from data (highly variable genes, scaling statistics, TF-IDF weights, the LSI basis) should be fit on training cells and then applied unchanged to validation, test, and query cells. The preprocessors in `univi.preprocessing` follow a `fit` / `transform` pattern for exactly this reason:

```python
from univi.preprocessing import RNAPreprocessor, ADTPreprocessor, ATACPreprocessor, split_by_label

split = split_by_label(rna.obs["cell_type"], train_fraction=0.8, val_fraction=0.1, seed=0)
rna_prep = RNAPreprocessor(n_hvg=2000, scale=True).fit(rna[split["train"]])
rna_train = rna_prep.transform(rna[split["train"]])
rna_test = rna_prep.transform(rna[split["test"]])
```

`split_by_label` stratifies by label so rare populations appear in every split. `max_per_label` caps how many cells of an abundant type enter the train/validation pool (the overflow goes to test), and `train_cap` / `val_cap` cap the training and validation sets directly.

### The preprocessors

| Class | Input | Output in `.X` | Key options |
| --- | --- | --- | --- |
| `RNAPreprocessor` | raw counts | log-normalized expression of selected genes, optionally z-scored | `n_hvg` (Seurat v3 HVGs on raw training counts; `None` keeps all genes), `scale`, `target_sum`, `exclude_genes`, `clip` |
| `ADTPreprocessor` | raw antibody counts | per-cell CLR, optionally z-scored per protein | `scale`, `clip` |
| `ATACPreprocessor` | raw peak or tile counts | TF-IDF then truncated SVD (LSI), optionally standardized | `n_components` (fit before dropping), `drop_first` (drop the depth-correlated first component), `method` (`"sklearn"`, `"signac"`, `"tea"`), `min_fraction` / `max_fraction` peak filters |

Transformed objects keep the raw counts in `layers["counts"]` (and log-normalized RNA in `layers["log1p"]`, CLR protein in `layers["clr"]`). Seurat-v3 HVG selection needs the `scikit-misc` package (`pip install "univi[tutorials]"`); the preprocessor raises an error rather than silently choosing another method.

You do not have to use these classes. Any preprocessing works as long as `.X` holds what you want the model to see and the likelihood matches it.

## Matching features for new data

`transform` refuses data that lacks features the preprocessor was fit on, rather than filling them silently.

- **RNA**: harmonize identifiers (symbols vs Ensembl IDs, annotation version). If some reference genes are genuinely absent from a query, refit the reference on the shared genes.
- **Protein**: harmonize antibody names across panels and restrict both datasets to shared markers before fitting.
- **ATAC**: quantify query accessibility on the reference peak set (for example with Signac `FeatureMatrix` or SnapATAC2). Independently called peaks are not comparable, even with the same count.

## Loaders

`make_loader` wraps a modality dictionary in a PyTorch `DataLoader` with the collate function UniVI's trainer expects:

```python
from univi.workflows import make_loader
train_loader = make_loader(train, batch_size=256, shuffle=True, drop_last=True)
val_loader = make_loader(val, batch_size=1024)
```

- `drop_last=True` avoids a final batch of one cell, which BatchNorm cannot train on.
- `labels={"head_name": codes}` adds integer labels for classification heads (`-1` = unknown).
- `recon_targets_spec` supplies extra reconstruction targets for binomial-type likelihoods (see [Advanced features](advanced.md#methylation-and-other-successestrials-data)).

## Supported assays at a glance

| Assay | Typical input to UniVI | Likelihood |
| --- | --- | --- |
| scRNA-seq | log-normalized HVGs, z-scored | `gaussian` |
| CITE-seq proteins | CLR, z-scored | `gaussian` |
| scATAC-seq | LSI (drop first component), standardized | `gaussian` |
| Raw counts of any kind | counts in `.X` | `nb`, `zinb`, `poisson` |
| Binarized peaks | 0/1 matrix | `bernoulli` |
| DNA methylation, allele counts | fraction in `.X`; successes and coverage in layers | `beta_binomial` (or `binomial`) |
| Proportions | values in (0, 1) | `beta` |
| Categorical annotation as a modality | one-hot matrix | `categorical` |
