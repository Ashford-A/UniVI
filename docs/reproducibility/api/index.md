# Paper analyses with the public API

These notebooks run the analyses of the Genome Research article through UniVI's public API: `univi.datasets` for the data, the fitted preprocessors in `univi.preprocessing`, `UniVITrainer`, `UniVIRefiner`, and `univi.evaluation`. They are the place to start if you want to repeat, extend, or adapt an analysis from the paper without reading the archived notebooks' internal helper code.

Each notebook uses the model settings of the corresponding archived notebook (summarized in Supplemental Table S8) and the preprocessing described in Supplemental Table S7. Because preprocessing and refinement go through UniVI's reusable components rather than the archived notebooks' own code, results are close to, but not identical with, the published numbers. To regenerate the published numbers exactly, run the archived notebooks with UniVI 0.4.7 (see [Paper reproduction](../index.md)).

| Notebook | Figures | Data (`univi.datasets`) |
| --- | --- | --- |
| [Paired CITE-seq](fig2_3_citeseq.ipynb) | 2, 3, S1 | `hao_citeseq_pbmc` |
| [Paired 10x Multiome](fig4_multiome.ipynb) | 4, S2 | `pbmc_multiome_10k` |
| [Bridging unpaired RNA and ATAC cohorts](fig5_bridge.ipynb) | 5, S5 | `pbmc_multiome_bridge` |
| [Trimodal TEA-seq, held-out well](fig6_teaseq.ipynb) | 6, S6 | `teaseq_pbmc` |
| [AML mosaic integration and mutation heads](fig7_aml.ipynb) | 7, S7 | `aml_mosaic` |
| [SHARE-seq mouse skin](figS3_shareseq.ipynb) | S3 | `shareseq_mouse_skin` |
| [scNMT-seq mouse gastrulation](figS4_scnmt.ipynb) | S4 | `scnmt_gastrulation` — [Zenodo 10.5281/zenodo.22885463](https://doi.org/10.5281/zenodo.22885463) |

Training uses the archived notebooks' epoch limits with early stopping; a GPU is recommended. Every notebook has one cell of settings at the top (for example `N_EPOCHS`) that you can lower for a quick first run. The benchmarks and sweeps of Figs. 8–10 and Supplemental Figs. S8–S11 depend on other methods and large parameter grids and are only available as archived notebooks.

Check [Datasets](../datasets.md) (or `univi.datasets.list_datasets()`) for which datasets are downloadable.

```{toctree}
:hidden:

fig2_3_citeseq
fig4_multiome
fig5_bridge
fig6_teaseq
fig7_aml
figS3_shareseq
figS4_scnmt
```
