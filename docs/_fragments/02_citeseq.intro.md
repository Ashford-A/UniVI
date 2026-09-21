# Figure 2: paired RNA–protein integration

**Biological question:** do RNA and surface-protein measurements recover compatible immune identities for the same PBMCs?

Train on the Hao CITE-seq reference, encode held-out cells separately from RNA and ADT, and evaluate correspondence and cell-type transfer. Each cell contributes two points to the stacked embedding. Its fused representation, by contrast, would contribute one point.

| Published panels | Tutorial output | Meaning |
| --- | --- | --- |
| 2A–B | `test-stacked.png` and `.h5ad` | Modality and immune-lineage views of the same coordinates |
| 2C–D | Directional confusion matrices | Which immune identities transfer across assays |
| 2E–F | Per-class F1 plots and CSVs | Performance on rare populations as well as abundant ones |
| Paired correspondence | `rna-adt-metrics.json` | FOSCTTM and Recall@10 from true barcode pairs |

**Inputs:** `data/tutorials/cite/rna.h5ad` and `adt.h5ad`; raw counts in `layers["counts"]`; RNA annotations `celltype.l1`, `celltype.l2`, `celltype.l3`. Cells must genuinely be paired. Feature IDs may differ across modalities. [Data preparation](../guides/data.md).

The Figure 3 notebook supplies the shared CITE workflow here because it also contains the biological reconstructions and S1. It stratifies by `celltype.l3`, uses seed 42, and caps the per-label pool at 1,200 **before** allocating 80/10/10 fractions. This is different from capping the training partition at 1,200. If you have the original barcode split map, provide `splits.tsv`; that is the stronger reproduction record.

RNA uses training-selected HVGs, HVG-first library normalization, and log1p; ADT uses per-cell CLR. The visible preprocessing cells do not apply the Z scaling suggested by some prose descriptions. The tutorial preserves those code semantics and keeps raw counts intact.
**Notebook reference:** [UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb), zero-based cells 15–22, 28–38, 43–61. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-2.png
:alt: Published UniVI Figure 2, published figure reference
:class: published-figure

Published Figure 2, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
