# Figure 7 and S7: AML mosaic integration and mutation-aware refinement

**Biological question:** can paired RNA–protein anchors connect independent AML cohorts and expose genotype-associated and differentiation-related structure?

Train a bridge on paired Knorr AML CITE-seq. Project van Galen RNA and DAb-seq proteins with their respective encoders. Genotypes are optional supervised targets and validation annotations; they are not falsely presented as a third measurement available in every cell.

| Panel family | Analysis |
| --- | --- |
| 7A,D | Cohort and van Galen cell-state structure before refinement |
| 7B–F | Observed NPM1 calls and directional neighborhood-probability transfer |
| 7G–K | Refined geometry, observed NPM1 status, and LSC17 expression signature |
| 7L | Held-out mutation AUC/AP, with per-gene label denominators |
| S7 | Patient/disease context and predicted driver landscapes over all cells |

**Inputs:** paired `rna.h5ad`/`adt.h5ad`, `van_galen_rna.h5ad`, `dab_adt.h5ad`, and a patient/sample-level bridge `splits.tsv` under `data/tutorials/aml/`. Reference `sample_id` must identify the actual grouping unit. Query `mut_GENE` columns contain `0`, `1`, or missing; cell-state and sample fields remain metadata. Canonicalize antibodies and document raw versus already processed DAb values before applying a reference transform.

The bridge uses all shared genes rather than silently replacing them with 2,000 HVGs. Per-gene binary heads use `[64,32]` hidden layers, LayerNorm, dropout 0.1, training-only class weights, 10 warmup epochs, head LR 1e-4, encoder LR 1e-5, and latent-preservation weight 5.0, following the visible refinement call. Missing mutation calls never enter a gene's supervised loss.
**Notebook reference:** [UniVI_manuscript_GR-Figure__7__AML_bridge_mapping_and_fine-tuning.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__7__AML_bridge_mapping_and_fine-tuning.ipynb), zero-based cells 17–39, 55–63, 75–88, 107–128. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-7.png
:alt: Published UniVI Figure 7, supplied manuscript reference
:class: published-figure

Published Figure 7, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s7.png
:alt: Published UniVI Figure s7, supplied manuscript reference
:class: published-figure

Published Figure s7, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
