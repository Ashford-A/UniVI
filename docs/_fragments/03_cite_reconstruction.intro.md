# Figure 3 and S1: cross-modal reconstruction and marker biology

**Biological question:** can one assay recover the lineage-associated features measured by the other?

Use the paired CITE-seq model from [Figure 2](02_citeseq.md). RNA→ADT prediction uses RNA as the only encoder input; ADT→RNA prediction uses ADT alone. The observed target assay is reserved for comparison. Each observed/predicted pair uses the same target-cell UMAP coordinates and shared color limits.

| Panel family | Biological markers | Output |
| --- | --- | --- |
| 3A–B | Cell-type-aggregated RNA / ADT marker means | Paired group heatmaps and mean tables |
| 3C | Coarse immune identities | Shared stacked embedding from tutorial 2 |
| 3D–K | CD79A/CD19, LYZ/CD14, NKG7/CD56, TRAC/CD3 | Observed/predicted marker overlays |
| S1 | Expanded RNA and protein marker sets; `celltype.l2` | Additional overlays and subtype mean tables |

**Inputs:** the saved reference, held-out AnnData files, and stacked UMAP from tutorial 2. No second model fit is needed. Predictions are in the target decoder's training space: log-normalized RNA and CLR ADT in this notebook-derived recipe. They are not raw UMI counts.

CD56 and CD3 antibody features must retain their actual assay IDs. `NCAM1` and `CD3E` describe corresponding genes; they should not silently replace antibody feature names such as `CD56_1` and `CD3_1`. Missing requested markers are written to a report rather than replaced by a different feature.
**Notebook reference:** [UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb), zero-based cells 65–77, 80–101. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-3.png
:alt: Published UniVI Figure 3, published figure reference
:class: published-figure

Published Figure 3, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s1.png
:alt: Published UniVI Figure s1, published figure reference
:class: published-figure

Published Figure s1, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
