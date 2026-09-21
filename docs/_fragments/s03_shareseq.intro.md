# S3: SHARE-seq mouse skin and differentiation structure

**Biological question:** does paired alignment extend beyond blood, across developmental lineages and a sparse accessibility assay?

This workflow adapts the paired RNA–ATAC analysis to late-anagen mouse skin: epidermal/hair-follicle states, mesenchymal, vascular, neural-crest-derived and immune populations. It uses the assay-specific QC, gene-family exclusions, log-TF-IDF and first-component removal recorded in the SHARE-seq notebook.

**Inputs:** `data/tutorials/share/rna.h5ad` and `atac.h5ad`, raw counts, and `cell_type`; ENCODE mm10 blacklist peaks must already be removed. Record that upstream operation in `atac.uns["blacklist_removed"]`. Do not merely set the flag to bypass filtering.

RNA cells pass total-count, detected-gene, and mitochondrial-fraction limits. ATAC cells pass count limits; paired barcodes are retained together. Author-labeled `Mix` cells are excluded for correspondence with this analysis, not because every transitional state is necessarily an artifact.

Fit 5,000 eligible RNA HVGs on training counts and exclude the notebook's mitochondrial/ribosomal/Rik/pseudogene families. Retain ATAC features accessible in 0.25–80% of training cells, fit 101 components, drop component 0, and reuse all transformations unchanged. This recipe uses the notebook's `log1p(n/(1+df))` IDF followed by `log1p(TF*IDF*1e4)` and L2 normalization.

S3A–B become the stacked modality/cell-type views; S3C–D become the per-feature RNA→LSI and ATAC→RNA correlation/MSE distributions.
**Notebook reference:** [UniVI_manuscript_GR-Supple_____mouse_skin_SHARE-seq_integration.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Supple_____mouse_skin_SHARE-seq_integration.ipynb), zero-based cells QC and blacklist cells, 32–49, 54–63. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-s3.png
:alt: Published UniVI Figure s3, supplied manuscript reference
:class: published-figure

Published Figure s3, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
