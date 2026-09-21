# Figure 6 and S6: trimodal TEA-seq with a held-out well

**Biological question:** do RNA, surface protein, and chromatin accessibility describe concordant immune compartments when all three are measured?

Train using wells 3, 4, and 6 and evaluate well 5. The three assays enter one model through separate encoders and likelihood-appropriate decoders. Evaluate all three paired relationships rather than selecting only the best pair.

| Panels | Tutorial analysis |
| --- | --- |
| 6A | RNA–ADT, RNA–ATAC, and ADT–ATAC FOSCTTM |
| 6B–C | Cross-modality neighbor fractions, entropy, and distance distributions |
| 6D–F | Stacked UMAP, Leiden communities, modality composition per community |
| 6G | Cytotoxic-lymphocyte RNA/protein markers and representative LSI coordinates |
| S6 | Native-space cluster labels and B-cell, T-cell, myeloid/DC markers |

**Inputs:** paired `rna.h5ad`, `adt.h5ad`, and `atac.h5ad` under `data/tutorials/tea/`, with `obs["well"]` normalized to strings `3`, `4`, `5`, `6`. Provide a common ATAC tile universe; the source notebook constructs this using SnapATAC2. When `n_fragments` is supplied, the tutorial applies the notebook's threshold of 1,500.

A well holdout tests a capture/library partition within a run. It does not establish robustness to a new person or laboratory. The tutorial fits HVGs, scales, TF-IDF and LSI using training cells alone, then applies them unchanged to validation and well 5. This is stricter than several visible original preprocessing cells, including per-well ADT standardization.

The recipe uses notebook `gamma=1.45` and RNA/ADT/ATAC reconstruction weights 1.0/2.0/1.45. The supplemental table lists a different gamma. All these settings remain visible in the shared example helper.
**Notebook reference:** [UniVI_manuscript_GR-Figure__6__TEA-seq_tri-modal.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__6__TEA-seq_tri-modal.ipynb), zero-based cells 10–20, 29–37, 53–80, 101–121, 126–133. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-6.png
:alt: Published UniVI Figure 6, supplied manuscript reference
:class: published-figure

Published Figure 6, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s6.png
:alt: Published UniVI Figure s6, supplied manuscript reference
:class: published-figure

Published Figure s6, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
