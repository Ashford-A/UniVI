# Extended and experimental tutorials

These notebooks go beyond the analyses in the Genome Research article. They use only the public UniVI API
and the datasets in `univi.datasets`, and they open in Colab like the core tutorials.

:::{admonition} What "experimental" means here
:class: warning

- The methods in these notebooks were **not** used for the published results, and their settings are
  reasonable starting values rather than tuned ones.
- Each notebook builds in the checks needed to interpret its output: held-out ground truth, empirical
  nulls, permutation tests, or a second model trained independently. Read those checks before the
  headline plots.
- In-silico perturbations describe what a trained model has learned to associate. They are hypotheses
  to test, not measured causal effects.
:::

| Tutorial | What it does | Data |
| --- | --- | --- |
| [In-silico chromatin perturbation of transcription regulators](tf_perturbation_multiome.ipynb) | trains a peak-level model, closes the peaks around a regulator's gene or carrying its JASPAR motif, and reads out predicted RNA per cell type against matched random peak sets; adds latent shifts, dose-response, a two-regulator interaction, and agreement with the paper's LSI model | 10x Multiome PBMC |
| [Transformer encoders and a fused multimodal transformer](transformer_encoders.ipynb) | compares MLP, per-modality transformer, and fused transformer encoders on the same inputs, then inspects the fused model's attention, including a permutation-calibrated test of whether genes attend to nearby peaks | 10x Multiome PBMC |
| [A unified RNA + protein + chromatin PBMC atlas](pbmc_mosaic_atlas.ipynb) | trains one model on CITE-seq, Multiome and TEA-seq together, tests on a held-out TEA-seq well whether a model that never saw protein and chromatin together can predict one from the other, transfers the Hao et al. annotation, and predicts surface protein for Multiome cells | Hao et al. CITE-seq, 10x Multiome, TEA-seq |
| [How much should you trust a cross-modal prediction?](prediction_uncertainty.ipynb) | samples the posterior to get prediction uncertainty, calibrates it into conformal prediction intervals with checked coverage, and tests whether uncertainty flags a cell type left out of training | 10x Multiome PBMC |

## Before you run them

- **Hardware.** A GPU is strongly recommended; in Colab choose *Runtime → Change runtime type → GPU*. Each
  notebook trains one to four models. The fused transformer's memory grows with the square of the number
  of tokens per cell (`N_TOKENS_RNA + N_TOKENS_ATAC`).
- **Downloads.** The Multiome data are about 205 MB. The atlas notebook also downloads the CITE-seq
  (about 800 MB) and TEA-seq (about 420 MB) data; loading the full CITE-seq RNA matrix needs several GB of
  RAM before it is subsampled. The optional motif scan in the perturbation notebook downloads the hg38
  genome from UCSC (about 940 MB) and needs `pyjaspar`; set `RUN_MOTIF_SCAN = False` to skip it.
- **Parameters.** All settings are in one cell near the top of each notebook, tagged `parameters`.

Start with the [quickstart](../quickstart.ipynb) if you have not trained a UniVI model before; these
notebooks assume its preprocessing and training steps.

```{toctree}
:hidden:
:maxdepth: 1

tf_perturbation_multiome
transformer_encoders
pbmc_mosaic_atlas
prediction_uncertainty
```
