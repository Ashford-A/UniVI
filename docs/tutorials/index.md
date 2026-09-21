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
```
