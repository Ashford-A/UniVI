# UniVI

<div class="univi-bands"><span></span><span></span><span></span></div>

<p class="univi-lede">
UniVI learns one latent space from single-cell modalities measured in the same cells (RNA, surface
protein, chromatin accessibility, DNA methylation, and others). Each modality has its own encoder and
decoder, so a trained model can align cells across assays, predict one modality from another, place new
single-modality datasets onto a multimodal reference, and generate cells.
</p>

```{image} _static/univi_architecture_light.png
:class: only-light univi-hero
:alt: UniVI architecture: one encoder and one decoder per modality, joined through a shared latent space
```
```{image} _static/univi_architecture_dark.png
:class: only-dark univi-hero
:alt: UniVI architecture: one encoder and one decoder per modality, joined through a shared latent space
```

## Install

```bash
pip install univi
```

UniVI needs PyTorch; if `import torch` fails, install the build for your platform from [pytorch.org](https://pytorch.org/get-started/locally/). See [Installation](installation.md) for conda, GPUs, and Colab.

## A first model

```python
import univi.datasets as uds
from univi import ModalityConfig, TrainingConfig, UniVIConfig, UniVIMultiModalVAE, UniVITrainer
from univi.evaluation import cross_modal_predict, encode_adata
from univi.preprocessing import ATACPreprocessor, RNAPreprocessor, split_by_label
from univi.workflows import make_loader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
data = uds.pbmc_multiome_10k()                       # paired RNA + ATAC, downloaded once
rna, atac = data["rna"], data["atac"]
split = split_by_label(rna.obs["cell_type"], seed=0)  # 80/10/10, stratified

rna_prep = RNAPreprocessor(n_hvg=2000, scale=True).fit(rna[split["train"]])
atac_prep = ATACPreprocessor(n_components=101, drop_first=True).fit(atac[split["train"]])
parts = {k: {"rna": rna_prep.transform(rna[i]), "atac": atac_prep.transform(atac[i])} for k, i in split.items()}

cfg = UniVIConfig(latent_dim=30, beta=1.25, gamma=4.35, modalities=[
    ModalityConfig("rna", parts["train"]["rna"].n_vars, [512, 256, 128], [128, 256, 512]),
    ModalityConfig("atac", parts["train"]["atac"].n_vars, [128, 64], [64, 128]),
])
model = UniVIMultiModalVAE(cfg)
UniVITrainer(model, make_loader(parts["train"], shuffle=True, drop_last=True), make_loader(parts["val"]),
             TrainingConfig(n_epochs=300, early_stopping=True, patience=50, device=device)).fit()

z_atac = encode_adata(model, parts["test"]["atac"], modality="atac", latent="modality_mean", device=device)
rna_from_atac = cross_modal_predict(model, parts["test"]["atac"], src_mod="atac", tgt_mod="rna", device=device)
```

The [quickstart](tutorials/quickstart.ipynb) walks through this with plots and metrics.

## Where to go next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
End-to-end notebooks on real data: RNA + ATAC, CITE-seq, query mapping, cell-type heads, generation, custom assays. Each opens in Colab.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc
How to prepare data, choose likelihoods and hyperparameters, read the latent space, evaluate an integration, and save a reference.
:::

:::{grid-item-card} Paper reproduction
:link: reproducibility/index
:link-type: doc
The Genome Research analyses figure by figure: notebooks, datasets, settings, and how to rerun them.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
Every public class and function, with signatures and docstrings.
:::

:::{grid-item-card} Extended and experimental
:link: tutorials/experimental/index
:link-type: doc
Beyond the paper: in-silico regulator perturbation, transformer encoders, a CITE-seq + Multiome + TEA-seq atlas, calibrated prediction uncertainty, and cross-modal quality control.
:::
::::

## What UniVI does

- **Joint embedding** of paired multimodal data (CITE-seq, 10x Multiome, SHARE-seq, TEA-seq, scNMT-seq), with any number of modalities.
- **Cross-modal prediction**: RNA from ATAC, protein from RNA, and so on, for every cell.
- **Reference mapping**: embed RNA-only, ATAC-only, or protein-only cohorts into a reference trained on paired data, then transfer labels or impute the missing modality.
- **Supervised refinement**: add cell-type, genotype, or other heads to a trained reference, with missing labels masked.
- **Likelihoods matched to the data**: Gaussian, negative binomial, zero-inflated NB, Poisson, Bernoulli, beta, binomial and beta-binomial (for methylation), categorical.
- **Evaluation**: FOSCTTM, Recall@k, modality mixing, label transfer, reconstruction metrics.

## Citation

If you use UniVI, please cite:

> Ashford AJ, Enright T, Somers J, Nikolova O, Demir E. Unifying multimodal single-cell data with a mixture-of-experts β-variational autoencoder framework. *Genome Research* (2026). [doi:10.1101/gr.281431.125](https://doi.org/10.1101/gr.281431.125)

BibTeX and dataset citations are on the [citation page](citation.md).

```{toctree}
:hidden:
:caption: Getting started

installation
tutorials/quickstart
```

```{toctree}
:hidden:
:caption: Learn

tutorials/index
user_guide/index
```

```{toctree}
:hidden:
:caption: Genome Research

reproducibility/index
reproducibility/datasets
reproducibility/api/index
```

```{toctree}
:hidden:
:caption: Reference

api
changelog
citation
contributing
```
