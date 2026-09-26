# Tutorials

Each tutorial is a Jupyter notebook that downloads its data, trains a model, and explains every step. Use the **Open in Colab** badge at the top of a notebook (or the rocket icon in the page header) to run it in the browser, or download it with the button in the header.

| Tutorial | You will learn to | Data |
| --- | --- | --- |
| [Quickstart](quickstart.ipynb) | load paired data, preprocess without leakage, train, embed, check alignment, predict RNA from ATAC, save a reference | 10x Multiome PBMC |
| [CITE-seq: RNA + protein](citeseq.ipynb) | build a fused embedding, predict protein from RNA and RNA from protein, denoise, read per-cell modality contributions | Hao et al. 2021 CITE-seq |
| [Map query data onto a reference](query_mapping.ipynb) | embed RNA-only and ATAC-only cohorts, transfer labels with confidence scores, spot unfamiliar cells, impute the missing modality | 10x Multiome PBMC |
| [Cell-type heads and refinement](supervised_heads.ipynb) | attach a classifier to a trained reference, refine with 20% of labels, predict from one modality | 10x Multiome PBMC |
| [Imputation, denoising, generation](generation.ipynb) | denoise, evaluate prediction per feature, generate cells from the prior or a cell type, run an in-silico perturbation | 10x Multiome PBMC |
| [Custom modalities and likelihoods](custom_modalities.ipynb) | combine three modalities, use count and beta-binomial likelihoods, learn per-cell modality gates, try a transformer encoder | synthetic |

The quickstart is the best starting point; the others can be read in any order.

## Extended and experimental

Notebooks that go beyond the paper, with the checks needed to interpret them built in. See
[Extended and experimental tutorials](experimental/index.md) for what "experimental" means and what each
notebook needs to run.

| Tutorial | You will learn to | Data |
| --- | --- | --- |
| [In-silico chromatin perturbation](experimental/tf_perturbation_multiome.ipynb) | close a regulator's cis or motif peaks and read out predicted RNA per cell type against matched random peak sets | 10x Multiome PBMC |
| [Transformer encoders](experimental/transformer_encoders.ipynb) | benchmark MLP, per-modality transformer, and fused transformer encoders, and test the fused model's attention | 10x Multiome PBMC |
| [Unified PBMC atlas from three assays](experimental/pbmc_mosaic_atlas.ipynb) | integrate CITE-seq, Multiome and TEA-seq in one model and validate on a held-out trimodal well | CITE-seq, Multiome, TEA-seq |
| [Prediction uncertainty](experimental/prediction_uncertainty.ipynb) | attach calibrated intervals to cross-modal predictions and flag cells unlike the reference | 10x Multiome PBMC |
| [Cross-modal quality control](experimental/cross_modal_qc.ipynb) | find mis-paired barcodes and doublets from cross-modal disagreement, with a controlled false discovery rate | 10x Multiome PBMC |

For the analyses in the Genome Research article, see [Paper reproduction](../reproducibility/index.md).

```{toctree}
:hidden:
:maxdepth: 1

quickstart
citeseq
query_mapping
supervised_heads
generation
custom_modalities
experimental/index
experimental/aml_genotype_latent
```
