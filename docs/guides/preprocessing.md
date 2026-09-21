# Preprocessing and feature compatibility

The model's decoder predicts the space used for its reconstruction target. Decide what biological output you need before choosing a likelihood and representation.

| Assay representation | Likelihood used in these tutorials | What a prediction means |
| --- | --- | --- |
| RNA log-normalized expression, optionally scaled | Gaussian | Log-expression or standardized log-expression |
| ADT CLR, optionally scaled | Gaussian | CLR abundance or standardized CLR abundance |
| ATAC TF-IDF/LSI | Gaussian | Accessibility coordinates |
| RNA raw counts | Negative binomial | Count-like decoder mean |
| ATAC raw peak counts | Poisson | Peak-count-like decoder mean |
| ATAC binary peaks, optional alternative | Bernoulli | Per-peak probability |
| CpG/GpC successes + coverage | Beta-binomial | Fraction/probability together with a coverage-aware training loss |

## Fit once, apply unchanged

```python
from univi.preprocessing import RNAPreprocessor, ADTPreprocessor, ATACPreprocessor

rna_transform = RNAPreprocessor(2000, scale=True).fit(rna_train)
rna_train_input = rna_transform.transform(rna_train)
rna_test_input = rna_transform.transform(rna_test)
rna_query_input = rna_transform.transform(rna_query)
```

The fitted RNA object retains selected features and, when requested, training normalization/scaling state. The ATAC object retains reference features, IDF, SVD, and scaling. The ADT object retains the fixed antibody panel and training scale. Reordering query features is supported; missing reference features raise an error.

Save these objects with the model using `save_reference`. A weight checkpoint without its feature list and transforms is insufficient for meaningful query inference.

## RNA normalization denominator

HVG selection and library normalization are separate choices. `normalize_on_selected=True` normalizes over the selected gene panel, following visible CITE/Multiome notebook code. `False` normalizes over the complete saved reference panel before selecting output features. The latter requires that normalization panel to be available in the query as well.

The new preprocessor selects Seurat v3 HVGs from raw training counts, never log-transformed counts. It preserves raw counts in `layers["counts"]` and log-normalized values in `layers["log1p"]`. If scaling is enabled, `.X` contains the scaled values. The tutorial feature table says which form is decoded.

Feature selection can exclude relevant markers. A missing gene should be reported as outside the decoder's output panel. Showing its observed expression from a larger raw object does not create a model prediction for it.

## ATAC variants are not interchangeable

`ATACPreprocessor` exposes three named conventions:

- `sklearn`: `TfidfTransformer` with smoothed IDF and L2 normalization, used by the paired Multiome notebook.
- `tea`: L1-normalized counts times `log1p(n/(1+df))`, then L2 normalization.
- `signac`: the SHARE-seq notebook's log-TF-IDF variant, applying `log1p(TF * IDF * 1e4)` with its recorded IDF convention, then L2 normalization.

The name `signac` identifies this notebook-derived variant, not a claim that every Signac version has the identical formula. `n_components` counts fitted components **before** optional removal of component 0. Figure 4/5 retain 100 fitted components; SHARE-seq fits 101 and drops the first. Keep this choice with the checkpoint.

An inverse linear projection from an LSI prediction is not a validated peak-level generative decoder. Train a feature-level model when peak interpretation is the goal.

## Biological count and missingness semantics

Per-cell CLR should be applied to an agreed antibody panel. Adding or dropping proteins changes its denominator. Already CLR-normalized input should not be transformed again.

For methylation/accessibility proportions, missing coverage differs from an observed zero. Keep successes and coverage in separate layers and validate `0 <= successes <= coverage`. Fraction inputs can encode missing entries as zero for the encoder while the reconstruction loss masks them using coverage; the mask must remain available.

For missing modalities, use only observed assay keys. Do not use an all-zero vector to mean “no assay” unless the model has an explicit mask that makes that interpretation valid. For independent query cohorts, use separate unimodal loaders.

## Memory

Feature-level count decoders can create large dense outputs. Batch the forward pass and write results by assay or cell block for large analyses. Several existing evaluation functions convert the source matrix to dense before inference, so their `batch_size` limits model-batch size rather than guaranteeing bounded host memory. The LSI tutorials avoid a dense output over all ATAC peaks; the raw-feature tutorial may require substantial RAM/VRAM on full datasets.
