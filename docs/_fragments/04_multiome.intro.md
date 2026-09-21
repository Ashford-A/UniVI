# Figure 4 and S2: paired RNA–ATAC integration

**Biological question:** does sparse chromatin accessibility align with expression at cell and lineage resolution?

Fit UniVI to paired 10x Multiome PBMCs. RNA enters as training-selected, log-normalized HVGs. ATAC enters as a training-fitted TF-IDF/LSI representation. Modality-specific posterior means support paired retrieval and cross-modal label transfer; their stacked embedding supports visual inspection.

| Panels | Analysis |
| --- | --- |
| 4A–B | Stacked latent geometry by immune identity and modality |
| 4C,E,G,I | ATAC→RNA reconstruction of CD79A, LYZ, NKG7, TRAC |
| 4D,F,H,J | RNA→ATAC reconstruction in LSI coordinates |
| 4K–L | Directional label-transfer confusion, `k=3` |
| S2 | An additional 3D view of the same high-dimensional latent data |

**Inputs:** `data/tutorials/multiome/rna.h5ad`, `atac.h5ad`, raw counts, and `obs["cell_type"]`. Distinguish this annotated dataset from the separate reference used in Figure 5.

The visible notebook call fits **100 LSI components and retains component 0**. The tutorial follows that call; it does not substitute the README's common 101-then-drop-first recipe. RNA is not Z scaled in the visible transformation function. See the [provenance audit](../reference/notebook-provenance.md) before interpreting differences from the supplement's general description.
**Notebook reference:** [UniVI_manuscript_GR-Figure__4__Multiome_paired.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__4__Multiome_paired.ipynb), zero-based cells 15–22, 27–38, 51–68, 3D visualization cells. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-4.png
:alt: Published UniVI Figure 4, supplied manuscript reference
:class: published-figure

Published Figure 4, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s2.png
:alt: Published UniVI Figure s2, supplied manuscript reference
:class: published-figure

Published Figure s2, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
