# UniVI

[![Paper](https://img.shields.io/badge/Genome%20Research-10.1101%2Fgr.281431.125-1f5f8b)](https://doi.org/10.1101/gr.281431.125)
[![Docs](https://readthedocs.org/projects/univi/badge/?version=latest)](https://univi.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/univi)](https://pypi.org/project/univi/)
[![PyPI downloads](https://img.shields.io/pepy/dt/univi?label=pypi%20downloads)](https://pepy.tech/project/univi)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/univi?cacheSeconds=3600)](https://anaconda.org/conda-forge/univi)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/univi?label=conda-forge%20downloads&cacheSeconds=300)](https://anaconda.org/conda-forge/univi)
[![Python](https://img.shields.io/pypi/pyversions/univi.svg)](https://pypi.org/project/univi/)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22886719.svg)](https://doi.org/10.5281/zenodo.22886719)
[![License: MIT](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)

**UniVI** integrates single-cell modalities measured in the same cells (RNA, surface protein, chromatin accessibility, DNA methylation, and others) into one latent space. Each modality has its own encoder and decoder, so a trained model can align cells across assays, predict one modality from another, map new single-modality datasets onto a multimodal reference, and generate cells.

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v1.1.0/assets/figures/univi_overview_dark.png">
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v1.1.0/assets/figures/univi_overview_light.png"
       alt="UniVI architecture, training objective, and evaluation" width="100%">
</picture>

**Documentation, tutorials, and API reference: [univi.readthedocs.io](https://univi.readthedocs.io/)**

## Installation

```bash
pip install univi                      # or: conda install -c conda-forge univi
pip install "univi[tutorials]"         # adds Jupyter and scikit-misc for the tutorials
```

UniVI requires PyTorch. If `import torch` fails, or you need a specific CUDA build, install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/).

## Quickstart

Train on paired RNA + ATAC from 10x Multiome PBMCs (downloaded and cached on first use), then embed held-out cells and predict their gene expression from chromatin accessibility:

```python
import torch
import univi.datasets as uds
from univi import ModalityConfig, TrainingConfig, UniVIConfig, UniVIMultiModalVAE, UniVITrainer
from univi.evaluation import cross_modal_predict, encode_adata, evaluate_alignment
from univi.preprocessing import ATACPreprocessor, RNAPreprocessor, split_by_label
from univi.workflows import make_loader, save_reference

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Paired data: one AnnData per modality, same cells in the same order
data = uds.pbmc_multiome_10k()
rna, atac = data["rna"], data["atac"]

# 2. Split, then fit preprocessing on training cells only
split = split_by_label(rna.obs["cell_type"], train_fraction=0.8, val_fraction=0.1, seed=0)
rna_prep = RNAPreprocessor(n_hvg=2000, scale=True).fit(rna[split["train"]])
atac_prep = ATACPreprocessor(n_components=101, drop_first=True).fit(atac[split["train"]])
parts = {k: {"rna": rna_prep.transform(rna[i]), "atac": atac_prep.transform(atac[i])}
         for k, i in split.items()}
train, val, test = parts["train"], parts["val"], parts["test"]

# 3. One encoder/decoder per modality, a shared 30-dimensional latent space
cfg = UniVIConfig(
    latent_dim=30, beta=1.25, gamma=4.35, encoder_dropout=0.1, decoder_dropout=0.05,
    kl_anneal_start=50, kl_anneal_end=85, align_anneal_start=75, align_anneal_end=110,
    modalities=[
        ModalityConfig("rna", train["rna"].n_vars, [512, 256, 128], [128, 256, 512], likelihood="gaussian"),
        ModalityConfig("atac", train["atac"].n_vars, [128, 64], [64, 128], likelihood="gaussian"),
    ],
)
model = UniVIMultiModalVAE(cfg)

# 4. Train with early stopping on the validation cells
UniVITrainer(
    model,
    train_loader=make_loader(train, batch_size=256, shuffle=True, drop_last=True),
    val_loader=make_loader(val, batch_size=1024),
    train_cfg=TrainingConfig(n_epochs=400, lr=1e-3, weight_decay=1e-4, device=device,
                             early_stopping=True, patience=50, best_epoch_warmup=110),
).fit()

# 5. Embed each modality, measure alignment, predict RNA from ATAC
z_rna = encode_adata(model, test["rna"], modality="rna", latent="modality_mean", device=device)
z_atac = encode_adata(model, test["atac"], modality="atac", latent="modality_mean", device=device)
labels = test["rna"].obs["cell_type"].astype(str).to_numpy()
metrics = evaluate_alignment(Z1=z_rna, Z2=z_atac, labels_source=labels, labels_target=labels)
print("FOSCTTM:", metrics["foscttm_mean"], "label transfer:", metrics["label_transfer_acc"])

rna_from_atac = cross_modal_predict(model, test["atac"], src_mod="atac", tgt_mod="rna", device=device)

# 6. Save weights + configuration + fitted preprocessing as a reusable reference
save_reference("multiome_reference", model, preprocessors={"rna": rna_prep, "atac": atac_prep})
```

The [quickstart notebook](https://univi.readthedocs.io/en/latest/tutorials/quickstart.html) runs the same steps with plots and explanations.

## What you can do with UniVI

| Task | Tutorial |
| --- | --- |
| Integrate paired RNA + ATAC; check alignment; predict one modality from another | [Quickstart](https://univi.readthedocs.io/en/latest/tutorials/quickstart.html) |
| CITE-seq: fused embeddings, protein prediction, denoising, per-cell modality weights | [CITE-seq](https://univi.readthedocs.io/en/latest/tutorials/citeseq.html) |
| Map RNA-only or ATAC-only data onto a reference; transfer labels; flag unfamiliar cells | [Query mapping](https://univi.readthedocs.io/en/latest/tutorials/query_mapping.html) |
| Add cell-type or genotype heads to a trained reference, with partial labels | [Heads and refinement](https://univi.readthedocs.io/en/latest/tutorials/supervised_heads.html) |
| Denoise, generate cells, run in-silico perturbations | [Generation](https://univi.readthedocs.io/en/latest/tutorials/generation.html) |
| Three or more modalities; count, methylation (beta-binomial) and other likelihoods | [Custom modalities](https://univi.readthedocs.io/en/latest/tutorials/custom_modalities.html) |

Every tutorial opens in Google Colab. The [user guide](https://univi.readthedocs.io/en/latest/user_guide/index.html) covers data preparation, likelihood and hyperparameter choices, evaluation metrics, and saving models.

## Reproducing the Genome Research analyses

The notebooks that produced every figure of the article are in [`notebooks/GR_manuscript_reproducibility/`](notebooks/GR_manuscript_reproducibility), with parameter files in `parameter_files/` and a driver script in `scripts/revision_reproduce_all.sh`. The article was prepared against release **v0.4.7** (`pip install univi==0.4.7`, or `envs/univi_v0.4.7_env.yml`). The [reproduction guide](https://univi.readthedocs.io/en/latest/reproducibility/index.html) maps each figure to its notebook and dataset.

Processed datasets download with `univi.datasets` (`uds.list_datasets()` shows what is available); see the [dataset catalog](https://univi.readthedocs.io/en/latest/reproducibility/datasets.html) for all seven records. The annotated Multiome PBMC data are on [Zenodo](https://doi.org/10.5281/zenodo.19581816), and the paired scNMT-seq RNA/CpG/GpC data for Supplemental Fig. S4 are on [Zenodo 10.5281/zenodo.22885463](https://doi.org/10.5281/zenodo.22885463). Load the latter with `uds.load("scnmt_gastrulation")` (1,140 paired cells; success/coverage layers and train/validation/test assignments included).

## Citation

> Ashford AJ, Enright T, Somers J, Nikolova O, Demir E. Unifying multimodal single-cell data with a mixture-of-experts β-variational autoencoder framework. *Genome Research* (2026). [doi:10.1101/gr.281431.125](https://doi.org/10.1101/gr.281431.125)

```bibtex
@article{Ashford2026UniVI,
  title   = {Unifying multimodal single-cell data with a mixture-of-experts
             {$\beta$}-variational autoencoder framework},
  author  = {Ashford, Andrew J. and Enright, Trevor and Somers, Julia
             and Nikolova, Olga and Demir, Emek},
  journal = {Genome Research},
  year    = {2026},
  doi     = {10.1101/gr.281431.125},
  url     = {https://doi.org/10.1101/gr.281431.125}
}
```

## Repository layout

```text
univi/        the Python package
docs/         documentation source; tutorials are the notebooks in docs/tutorials/
notebooks/    archived analysis notebooks for the Genome Research article
parameter_files/, scripts/   JSON configurations and command-line entry points
envs/         conda environment files (including the v0.4.7 environment used for the article)
tests/        unit tests and tutorial execution tests
```

## Contributing and support

Questions and bug reports are welcome as [GitHub issues](https://github.com/Ashford-A/UniVI/issues); please include the UniVI version (`python -c "import univi; print(univi.__version__)"`) and a minimal example. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup.

UniVI is released under the [MIT License](LICENSE).
