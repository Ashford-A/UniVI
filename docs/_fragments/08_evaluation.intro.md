# Figure 8: evaluate integration and biological structure

**Question:** is the representation useful across complementary biological and cell-correspondence diagnostics?

Figure 8 compares integration methods rather than introducing a new biological dataset. This tutorial runs the UniVI evaluation arm and exports the metric families underlying all ten panels. The archived Python and R benchmark notebooks retain the full competitor-specific setup and cross-validation orchestration.

| Panels | Metric family | Desired direction |
| --- | --- | --- |
| A | Fused-space held-out label transfer | Higher |
| B–C | Fused-space k-means ARI / NMI | Higher |
| D | Ground-truth-label silhouette | Context dependent; higher separation |
| E | Model-fit wall time | Lower at comparable performance |
| F–G | Paired FOSCTTM / Recall@10 | Lower / higher |
| H–I | Cross-modality label transfer accuracy / macro-F1 | Higher |
| J | Stacked modality mixing | Compare with representation and composition |

**Inputs:** the annotated Multiome counts under `data/tutorials/benchmark/`, with `cell_type`. Refit all learned transforms per split. The three example seeds demonstrate evaluation mechanics; they do not constitute the published complete cross-validation sweep.

The source runner sets a 30-dimensional latent, `beta=1.35`, `gamma=3.75`, and 200 epochs. This tutorial keeps those code-derived defaults while exposing the preprocessing and evaluation boundaries. Only methods with comparable train/test scope, input measurements, and label access should appear on a common ranking.
**Notebook reference:** [UniVI_manuscript_GR-Figure__8__benchmarking_against_pytorch_tools.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__8__benchmarking_against_pytorch_tools.ipynb), zero-based cells 18–29, evaluation runner and cross-validation cells. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-8.png
:alt: Published UniVI Figure 8, supplied manuscript reference
:class: published-figure

Published Figure 8, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
