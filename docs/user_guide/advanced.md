# Advanced features

## Classification heads and refinement

Heads are small networks on the latent space that predict labels (cell type, genotype, disease status). Use them in two ways:

- **Refine a trained reference** (`model.add_classification_head(...)` then `UniVIRefiner`): the head is trained with encoders frozen, then selected encoders are fine-tuned at a low learning rate while decoders stay frozen. `RefinementConfig` controls the stages (`warmup_epochs`, `max_epochs`, `lr_head`, `lr_encoder`), keeps the latent space anchored (`latent_weight`), and can keep optimizing the original objective on paired data (`replay_weight` with a `replay_loader`). Unknown labels (`-1` or `nan`) are masked per head.
- **Train jointly** by listing heads in `UniVIConfig(class_heads=[...])` and passing labels to `make_loader(..., labels={...})`.

`ClassHeadConfig` options: `n_classes`, `head_type` (`"categorical"` or `"binary"`), `hidden_dims`, `dropout`, `batchnorm`, `layernorm`, `loss_weight`, `ignore_index`, `pos_weight` (binary), `adversarial` and `adv_lambda` (gradient reversal, to remove a nuisance variable during joint training). Predict with `predict_heads_adata(model, adata, modality)` or `model.predict_heads(x_dict)`.

The [heads tutorial](../tutorials/supervised_heads.ipynb) covers both.

## Methylation and other successes/trials data

For `binomial` and `beta_binomial` likelihoods, the decoder is scored against success and trial counts stored as layers, while the encoder reads `.X` (typically the observed fraction):

```python
recon_targets = {"cpg": {"successes_layer": "meth_successes", "total_count_layer": "meth_total_count"}}
loader = make_loader(adata_by_mod, batch_size=128, shuffle=True, recon_targets_spec=recon_targets)
```

The trainer passes these targets to the model automatically. For the scNMT-seq gastrulation data used in the paper, `univi.datasets.scnmt_gastrulation()` builds the three modalities with these layers, and `build_univi_inputs_from_scnmt_triplet` returns the matching `recon_targets_spec`.

## A categorical modality

A categorical annotation can be a full modality with its own encoder and decoder (rather than a head): give it a one-hot matrix and `likelihood="categorical"`. It then takes part in fusion and alignment like any other modality.

## Transformer encoders (experimental)

Any modality can use a transformer encoder over tokens built from its features:

```python
from univi.config import TokenizerConfig, TransformerConfig
ModalityConfig("rna", n_genes, [512, 256], [256, 512], encoder_type="transformer",
               tokenizer=TokenizerConfig(mode="topk_channels", n_tokens=256, channels=("value", "rank", "dropout")),
               transformer=TransformerConfig(d_model=128, num_heads=4, num_layers=2, dim_feedforward=256))
```

`UniVIConfig(fused_encoder_type="multimodal_transformer", fused_transformer=TransformerConfig(...))` goes further and encodes all modalities jointly with one transformer over their concatenated tokens (every fused modality needs a `tokenizer`); use it with `loss_mode="v2"`. `univi.interpretability` has helpers for attention maps and integrated-gradient feature importance. These components are experimental and were not used for the results in the paper.

## Label-informed fusion

`UniVIMultiModalVAE(cfg, use_label_encoder=True, n_label_classes=K, label_moe_weight=..., unlabeled_logvar=...)` adds a label encoder whose posterior joins the fusion as an extra expert for labeled cells. It is a semi-supervised option for joint training; `UniVIRefiner` does not support it.

## Matching unpaired cells

`univi.matching` provides helpers for pairing cells across unpaired datasets in latent space (bipartite matching, stratified matching within labels, mutual nearest neighbors, cluster-centroid matching, and Gromov-Wasserstein optimal transport).

## Scripts and command line

The repository's `scripts/` directory has entry points driven by JSON parameter files in `parameter_files/`:

```bash
python scripts/train_univi.py --config parameter_files/defaults_multiome_v1.json --outdir runs/multiome \
    --data-root /path/to/data [--device cuda] [--seed 0]
python scripts/evaluate_univi.py --help
```

The installed `univi` command offers `univi encode` (write latent embeddings to `.npz`) and `univi export-s1` (write an environment/hyperparameter/dataset summary table), both driven by the same parameter files. Run `univi <command> --help` for options.
