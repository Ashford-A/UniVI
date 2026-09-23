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
| [In-silico chromatin perturbation of transcription regulators](tf_perturbation_multiome.ipynb) | trains a peak-level model, closes the peaks around a regulator's gene or carrying its JASPAR motif, and reads out predicted RNA per cell type against matched random peak sets; maps the effects along the genome, runs an in-silico single-peak tiling screen at each regulator's locus, separates a stress and activation module, and adds latent shifts, dose-response, a two-regulator interaction, and agreement with the paper's LSI model | 10x Multiome PBMC |
| [Transformer encoders and a fused multimodal transformer](transformer_encoders.ipynb) | compares MLP, per-modality transformer (with and without genomic position), and fused transformer encoders on the same inputs, then opens up the fused model: attention maps, entropy, attention rollout and whether it adds anything to token selection, cross-modal attention by cell type, sinks, pair-specific links, and a permutation-calibrated cis test with untrained-model controls | 10x Multiome PBMC |
| [A unified RNA + protein + chromatin PBMC atlas](pbmc_mosaic_atlas.ipynb) | trains one model on CITE-seq, Multiome and TEA-seq together and tunes an adversarial cohort head on validation cells; tests on a held-out TEA-seq well whether a model that never saw protein and chromatin together can predict one from the other; shows one cell through three assays, modality disagreement, which modality steers the fused embedding, and a latent walk between T-cell states; and predicts surface protein for Multiome cells | Hao et al. CITE-seq, 10x Multiome, TEA-seq |
| [How much should you trust a cross-modal prediction?](prediction_uncertainty.ipynb) | samples the posterior to get prediction uncertainty, calibrates it into conformal prediction intervals with checked coverage, and tests whether uncertainty flags a cell type left out of training, against its closest relatives and across several held-out types | 10x Multiome PBMC |
| [Quality control from cross-modal disagreement](cross_modal_qc.ipynb) | scores how much a barcode's RNA and ATAC disagree, benchmarks the scores on constructed mis-paired barcodes and doublets, and flags barcodes with conformal p-values at a controlled false discovery rate | 10x Multiome PBMC |

## Before you run them

- **Hardware.** A GPU is strongly recommended; in Colab choose *Runtime → Change runtime type → GPU*. Each
  notebook trains one to four models (the uncertainty notebook trains one more per extra held-out type). The fused transformer's memory grows with the square of the number
  of tokens per cell (`N_TOKENS_RNA + N_TOKENS_ATAC`).
- **Downloads.** The Multiome data are about 205 MB. The atlas notebook also downloads the CITE-seq
  (about 800 MB) and TEA-seq (about 420 MB) data; loading the full CITE-seq RNA matrix needs several GB of
  RAM before it is subsampled. The optional motif scan in the perturbation notebook downloads the hg38
  genome from UCSC (about 940 MB) and needs `pyjaspar`; set `RUN_MOTIF_SCAN = False` to skip it.
- **Parameters.** All settings are in one cell near the top of each notebook, tagged `parameters`.
- **Outputs.** Files the notebooks write (tables, models, reference bundles) go to `univi_outputs/` next to
  the notebook; that folder is ignored by git.

Start with the [quickstart](../quickstart.ipynb) if you have not trained a UniVI model before; these
notebooks assume its preprocessing and training steps.

```{toctree}
:hidden:
:maxdepth: 1

tf_perturbation_multiome
transformer_encoders
pbmc_mosaic_atlas
prediction_uncertainty
cross_modal_qc
```
