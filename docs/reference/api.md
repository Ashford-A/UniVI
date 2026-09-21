# API reference

Generated from the package source using Python AST, without importing the model during documentation builds. The new preprocessing, refinement, workflow, and perturbation modules and component-freezing methods require UniVI 0.5.0 or later. Public helpers are available from their modules and through lazy top-level imports.

## `univi.config`

### `ModalityConfig`

```python
name: str
input_dim: int
encoder_hidden: List[int]
decoder_hidden: List[int]
likelihood: str = 'gaussian'
recon_weight: float = 1.0
dispersion: str = 'gene'
init_log_theta: float = 0.0
use_library_size: bool = False
library_key: Optional[str] = None
library_log1p: bool = True
predict_library: bool = False
use_zero_inflation: Optional[bool] = None
hurdle: bool = False
clip_targets_min: Optional[float] = None
clip_targets_max: Optional[float] = None
concentration_init: float = 0.0
total_count_key: Optional[str] = None
successes_are_fraction: bool = False
positive_eps: float = 1e-08
ignore_index: int = -1
input_kind: Literal['matrix', 'obs'] = 'matrix'
obs_key: Optional[str] = None
mask_key: Optional[str] = None
sample_weight_key: Optional[str] = None
encoder_type: Literal['mlp', 'transformer'] = 'mlp'
transformer: Optional[TransformerConfig] = None
tokenizer: Optional[TokenizerConfig] = None
decoder_kwargs: Optional[Dict[str, Any]] = None
```

```text
Configuration for a single modality.

Backwards compatible with prior versions.

Added decoder/loss-related optional knobs so the model can support more
likelihoods without changing the config schema again.
```

### `ClassHeadConfig`

```python
name: str
n_classes: int
loss_weight: float = 1.0
ignore_index: int = -1
from_mu: bool = True
warmup: int = 0
adversarial: bool = False
adv_lambda: float = 1.0
head_type: Optional[Literal['categorical', 'binary']] = None
hidden_dims: Optional[List[int]] = None
dropout: Optional[float] = None
batchnorm: Optional[bool] = None
activation: Optional[Literal['relu', 'gelu', 'elu', 'leakyrelu', 'silu', 'tanh']] = None
pos_weight: float = 1.0
layernorm: bool = False
```

```text
Supervised / adversarial head configuration.

Backwards compatible defaults preserve the old behavior:
- head_type defaults to categorical (even if n_classes==2)
```

### `UniVIConfig`

```python
latent_dim: int
modalities: List[ModalityConfig]
beta: float = 1.0
gamma: float = 1.0
encoder_dropout: float = 0.0
decoder_dropout: float = 0.0
encoder_batchnorm: bool = True
decoder_batchnorm: bool = False
kl_anneal_start: int = 0
kl_anneal_end: int = 0
align_anneal_start: int = 0
align_anneal_end: int = 0
class_heads: Optional[List[ClassHeadConfig]] = None
label_head_name: str = 'label'
fused_encoder_type: Literal['moe', 'multimodal_transformer'] = 'moe'
fused_transformer: Optional[TransformerConfig] = None
fused_modalities: Optional[Sequence[str]] = None
fused_add_modality_embeddings: bool = True
fused_require_all_modalities: bool = True
use_moe_gating: bool = False
moe_gating_type: Literal['per_modality', 'shared'] = 'per_modality'
moe_gating_hidden: Optional[List[int]] = None
moe_gating_dropout: float = 0.0
moe_gating_batchnorm: bool = False
moe_gating_activation: Literal['relu', 'gelu', 'elu', 'leakyrelu', 'silu', 'tanh'] = 'relu'
moe_gate_eps: float = 1e-06
default_dispersion: Optional[str] = None
default_use_library_size: Optional[bool] = None
default_positive_eps: Optional[float] = None
model_kwargs: Optional[Dict[str, Any]] = None
```

### `TrainingConfig`

```python
n_epochs: int = 200
batch_size: int = 256
lr: float = 0.001
weight_decay: float = 0.0
device: str = 'cpu'
log_every: int = 10
grad_clip: Optional[float] = None
num_workers: int = 0
seed: int = 0
early_stopping: bool = False
patience: int = 20
min_delta: float = 0.0
best_epoch_warmup: int = 0
```

## `univi.models.univi`

### `UniVIMultiModalVAE`

```text
Multi-modal β-VAE with per-modality encoders and decoders.
```

#### `UniVIMultiModalVAE.__init__`

```python
__init__(self, cfg: UniVIConfig, *, loss_mode: str='v1', v1_recon: str='avg', v1_recon_mix: float=0.0, normalize_v1_terms: bool=True, recon_normalize_by_dim: Optional[bool]=None, recon_dim_power: Optional[float]=None, n_label_classes: int=0, label_loss_weight: float=1.0, use_label_encoder: bool=False, label_moe_weight: float=1.0, unlabeled_logvar: float=20.0, label_encoder_warmup: int=0, label_ignore_index: int=-1, classify_from_mu: bool=True, label_head_name: Optional[str]=None)
```

#### `UniVIMultiModalVAE.add_classification_head`

```python
add_classification_head(self, head: ClassHeadConfig, *, label_names=None)
```

```text
Attach a head without rebuilding the encoders or decoders.

Attach heads before constructing an optimizer. Existing names are rejected.
The updated configuration is sufficient to rebuild the head at load time.
```

#### `UniVIMultiModalVAE.freeze_decoders`

```python
freeze_decoders(self, modalities=None)
```

```text
Freeze decoder parameters, Dropout, and BatchNorm running statistics.
```

#### `UniVIMultiModalVAE.unfreeze_decoders`

```python
unfreeze_decoders(self, modalities=None)
```

```text
Enable decoder gradients; rebuild an optimizer if parameters were filtered.
```

#### `UniVIMultiModalVAE.freeze_encoders`

```python
freeze_encoders(self, modalities=None)
```

```text
Freeze selected modality encoders and their projection heads.
```

#### `UniVIMultiModalVAE.unfreeze_encoders`

```python
unfreeze_encoders(self, modalities=None)
```

```text
Enable selected modality encoders and their projection heads.
```

#### `UniVIMultiModalVAE.train`

```python
train(self, mode: bool=True)
```

```text
Respect explicit component freezes when a trainer calls model.train().
```

#### `UniVIMultiModalVAE.encode_fused`

```python
encode_fused(self, x_dict: Dict[str, torch.Tensor], *, epoch: int=0, y: Optional[YType]=None, use_mean: bool=True, inject_label_expert: bool=True, attn_bias_cfg: Optional[Mapping[str, Any]]=None, return_gates: bool=False, return_gate_logits: bool=False)
```

#### `UniVIMultiModalVAE.predict_heads`

```python
predict_heads(self, x_dict: Dict[str, torch.Tensor], *, epoch: int=0, y: Optional[YType]=None, use_mean: bool=True, inject_label_expert: bool=True, return_probs: bool=True, attn_bias_cfg: Optional[Mapping[str, Any]]=None)
```

## `univi.preprocessing`

### `RNAPreprocessor`

```text
RNA counts -> log1p normalized features -> optional train-fitted Z scores.

Seurat v3 HVGs are selected on raw training counts. ``normalize_on_selected``
reproduces the HVG-first normalization in the CITE/Multiome notebook code;
False normalizes over the complete reference feature panel before subsetting.
``n_hvg=None`` keeps all features. No fallback HVG method is chosen silently.
```

#### `RNAPreprocessor.__init__`

```python
__init__(self, n_hvg=2000, *, layer='counts', target_sum=10000.0, scale=False, normalize_on_selected=True, clip=None, exclude_genes=(), min_cells=1)
```

#### `RNAPreprocessor.fit`

```python
fit(self, adata)
```

#### `RNAPreprocessor.transform`

```python
transform(self, adata)
```

#### `RNAPreprocessor.fit_transform`

```python
fit_transform(self, adata)
```

### `ADTPreprocessor`

```text
Fixed-panel per-cell CLR followed by optional training-fitted scaling.

Input must be nonnegative raw abundance, not already CLR-transformed data.
Restrict/harmonize the antibody panel before fitting this transform.
```

#### `ADTPreprocessor.__init__`

```python
__init__(self, *, layer='counts', scale=False, clip=None)
```

#### `ADTPreprocessor.fit`

```python
fit(self, adata)
```

#### `ADTPreprocessor.transform`

```python
transform(self, adata)
```

#### `ADTPreprocessor.fit_transform`

```python
fit_transform(self, adata)
```

### `ATACPreprocessor`

```text
Reference peak counts -> fixed TF-IDF/LSI basis, optionally dropping LSI_0.

``method='sklearn'`` matches the Multiome notebook's TfidfTransformer.
``'signac'`` uses the SHARE-seq notebook's log1p(TF * IDF * 1e4),
with IDF=log1p(n/(1+df)), then L2 normalization.
``'tea'`` uses L1 TF * log1p(n/(1+df)), then L2 normalization.
``n_components`` is the number fit BEFORE dropping the first component.
```

#### `ATACPreprocessor.__init__`

```python
__init__(self, n_components=100, *, layer='counts', method='sklearn', drop_first=False, scale=True, min_fraction=0.0, max_fraction=1.0, random_state=42)
```

#### `ATACPreprocessor.fit`

```python
fit(self, adata)
```

#### `ATACPreprocessor.transform`

```python
transform(self, adata)
```

#### `ATACPreprocessor.fit_transform`

```python
fit_transform(self, adata)
```

### `split_by_label`

```python
split_by_label(labels, *, train_fraction=0.8, val_fraction=0.1, train_cap=None, val_cap=None, max_per_label=None, seed=0)
```

```text
Stratify cells; capped training/validation leftovers go to the test set.

``max_per_label`` caps the pool before allocating train/validation fractions,
and adds overflow to test, matching the CITE/Multiome notebook helper with
unused_to_test=True. It is mutually exclusive with
caps applied directly to training and validation counts.
This is a portable deterministic split, not a replacement for archived
manuscript barcode maps. Tiny classes may be absent from validation.
```

## `univi.refinement`

### `RefinementConfig`

```python
max_epochs: int = 1000
warmup_epochs: int = 10
lr_head: float = 0.0001
lr_encoder: float = 1e-05
weight_decay_head: float = 1e-06
weight_decay_encoder: float = 1e-06
latent_weight: float = 0.0
replay_weight: float = 0.0
replay_epoch: int = 10000
patience: int = 50
min_delta: float = 0.0
grad_clip: float = 5.0
log_every: int = 25
```

```text
Settings for head warmup followed by encoder/head fine-tuning.

``max_epochs`` includes warmup. ``latent_weight`` anchors posterior means
to an eval-mode copy of the original reference. ``replay_weight`` adds the
original generative objective on a separate paired training loader.
```

### `UniVIRefiner`

```text
Refine in place; retain a copy of the original model for comparisons.

Loaders yield ``(x_dict, targets_dict)`` from ``MultiModalDataset`` with
``collate_multimodal_xy_recon``. Use separate loaders for independent
RNA-only / ATAC-only / ADT-only cohorts. Shorter training loaders cycle so
each cohort contributes the same number of optimization steps per epoch.
Validation uses every labeled entry once. Labels must not be modalities.

The supported path is the analytic fusion model without a label encoder or
adversarial heads. Routers and all generative decoders stay frozen.
```

#### `UniVIRefiner.__init__`

```python
__init__(self, model, train_loaders, val_loaders, *, config=None, device='cpu', encoder_modalities=None, replay_loader=None)
```

#### `UniVIRefiner.evaluate`

```python
evaluate(self)
```

```text
Observed-label-weighted validation CE/BCE; fail on all-unlabeled data.
```

#### `UniVIRefiner.fit`

```python
fit(self)
```

```text
Train both stages and restore the lowest supervised validation loss.
```

### `predict_heads_adata`

```python
predict_heads_adata(model, adata, modality, *, device='cpu', batch_size=512)
```

```text
Return per-head probabilities in the input AnnData row order, from .X.
```

## `univi.workflows`

### `make_loader`

```python
make_loader(adata_by_mod, *, batch_size=256, shuffle=False, labels=None, recon_targets_spec=None, drop_last=False, seed=0)
```

```text
Build a CPU dataset with the required UniVI collator.

A mapping with multiple modalities must contain genuinely paired cells.
Use separate loaders for unrelated cohorts. BatchNorm training requires
at least two cells per batch; use drop_last=True for a singleton remainder.
```

### `stack_embeddings`

```python
stack_embeddings(model, datasets, *, device='cpu', batch_size=1024)
```

```text
Encode (cohort, modality, AnnData) entries and stack modality means.

Each input cell contributes one row per supplied modality. Features in the
result are latent dimensions, not a union of incompatible assay features.
``cell_id`` retains the input barcode and ``block`` the tuple's cohort label.
No UMAP is fit and no biological pairing is inferred by this function.
```

### `save_reference`

```python
save_reference(directory, model, *, preprocessors=None, metadata=None)
```

```text
Save model configuration, objective switches, labels, and fitted transforms.

The directory contains model.pt and optional preprocessing.joblib. Save
cell split maps and acquisition details in metadata. No optimizer state is
saved: this bundle is for inference/new training stages, not exact resume.
```

### `load_reference`

```python
load_reference(directory, *, device='cpu', load_preprocessors=True)
```

```text
Load a bundle created by save_reference; use only trusted local bundles.

Joblib preprocessing files use Python pickle. The returned tuple is
(model, preprocessors, metadata). Restore saved transforms; do not refit them.
```

## `univi.datasets`

### `load_scnmt_gastrulation_genebody_triplet`

```python
load_scnmt_gastrulation_genebody_triplet(tar_path: Path, rna_member: str='rna/counts.txt.gz', cpg_member: str='met/feature_level/genebody.tsv.gz', gpc_member: str='acc/feature_level/genebody.tsv.gz', metadata_member: str='sample_metadata.txt', require_qc: bool=True, min_cov_cpg: int=1, min_cov_gpc: int=1, filter_features: bool=True, rna_min_cells: int=10, rna_min_counts: int=20, meth_min_cells_covered: int=10, meth_min_total_coverage: float=50.0, verbose: bool=True)
```

### `build_univi_inputs_from_scnmt_triplet`

```python
build_univi_inputs_from_scnmt_triplet(rna: ad.AnnData, cpg: ad.AnnData, gpc: ad.AnnData, rna_target_sum: float=10000.0, log1p_rna: bool=True, copy: bool=True)
```

```text
Returns:
  - adata_dict (paired, same obs_names expected after alignment)
  - recon_targets_spec (for beta_binomial on cpg/gpc)
```

## `univi.perturbation`

### `predict_feature_perturbation`

```python
predict_feature_perturbation(model, adata, *, source_modality, target_modality, features, mode='off', value=0.0, device='cpu', batch_size=256)
```

```text
Compare baseline and edited-input decoder means without retraining.

``adata.X`` must already be in the model's input space. ``features`` are exact
source var_names, not latent coordinates or genes mapped heuristically to
peaks. Supported edits are off, set, add, scale. This reports model sensitivity,
not a causal perturbation effect. The input object is never modified.
```

## `univi.evaluation`

### `add_lsc17_scores`

```python
add_lsc17_scores(adata, layer: Optional[str]=None, gene_key: Optional[str]=None, prefix: str='LSC17')
```

### `compute_foscttm`

```python
compute_foscttm(Z1: np.ndarray, Z2: np.ndarray, metric: str='euclidean', block_size: int=512, return_sem: bool=False, return_per_cell: bool=False)
```

### `compute_match_recall_at_k`

```python
compute_match_recall_at_k(Z1: np.ndarray, Z2: np.ndarray, k: int=10, metric: str='euclidean', block_size: int=512, return_sem: bool=False, return_per_cell: bool=False)
```

### `label_transfer_knn`

```python
label_transfer_knn(Z_source: np.ndarray, labels_source: np.ndarray, Z_target: np.ndarray, labels_target: Optional[np.ndarray]=None, k: int=15, metric: str='euclidean', return_label_order: bool=False, return_f1: bool=False)
```

### `reconstruction_metrics`

```python
reconstruction_metrics(x_true: np.ndarray, x_pred: np.ndarray, *, kind: str='continuous')
```

### `encode_adata`

```python
encode_adata(model, adata, modality: str, device: str='cpu', layer: Optional[str]=None, X_key: str='X', batch_size: int=1024, latent: str='moe_mean', random_state: int=0)
```

```text
Encode a *single* modality into z.

latent:
  - "moe_mean" / "moe_sample": fused MoE/PoE posterior using only provided experts
  - "modality_mean" / "modality_sample": that modality posterior only
```

### `cross_modal_predict`

```python
cross_modal_predict(model, adata_src, src_mod: str, tgt_mod: str, device: str='cpu', layer: Optional[str]=None, X_key: str='X', batch_size: int=512, use_moe: bool=True, decoder_prefer: str='auto', decoder_link: Optional[str]=None)
```

```text
Encode src_mod then decode tgt_mod.

Robust to decoder outputs that are dicts (NB/ZINB/Poisson/Bernoulli/...).
Returns a mean-like matrix for evaluation/plotting.
```

### `encode_moe_gates_from_tensors`

```python
encode_moe_gates_from_tensors(model, x_dict: Mapping[str, Union[np.ndarray, 'sp.spmatrix', torch.Tensor]], *, device: str='cpu', batch_size: int=1024, modality_order: Optional[Sequence[str]]=None, kind: str='effective_precision', return_logits: bool=True, eps: float=1e-08)
```

```text
Compute per-cell modality *contribution weights* (aka "gates") for analytic fusion diagnostics.

This helper is designed for the **analytic fusion** path (MoE/PoE-style), where each modality
has a Gaussian posterior (mu_m, logvar_m) in latent space. We summarize how much each modality
contributes to the fused posterior on a per-cell basis.

Importantly, this function is intentionally **independent of cfg.use_moe_gating**:
  - If your model exposes router logits (best-effort via model.encode_fused(..., return_gate_logits=True)),
    we can optionally incorporate them.
  - If logits are unavailable, we still provide meaningful diagnostics using precision-only weights.

Parameters
----------
x_dict:
  dict {modality: X} with shape (n_cells, n_features) for each modality.
  Arrays may be numpy, scipy sparse, or torch tensors.
  All modalities must have the same n_cells.

modality_order:
  Order to report/stack modalities in outputs. If None, uses model cfg order if possible,
  otherwise uses sorted(x_dict.keys()).

kind:
  - "effective_precision":
      Precision-only weights derived from per-modality posterior uncertainty:
        s_m = sum_d exp(-logvar_m[d])
        w_m = s_m / sum_j s_j
      Always available (no router needed).
  - "router_x_precision":
      If router logits are available, combine router probabilities with effective precision:
        r = softmax(logits)
        w_m ∝ s_m * r_m
      If logits are unavailable, this silently falls back to "effective_precision".
  - "uniform":
      Equal weights across modalities (sanity check).

return_logits:
  If True, attempt to retrieve router logits. If not available, "logits" will be None.

Returns
-------
dict with:
  - weights: (n_cells, n_modalities) float32, rows sum to 1
  - logits: (n_cells, n_modalities) float32 or None
  - modality_order: list[str]
  - kind: str (the *effective* kind actually used; may downgrade router_x_precision -> effective_precision)
  - requested_kind: str
  - per_modality_mean: dict {modality: mean weight}
```

### `encode_fused_adata_pair`

```python
encode_fused_adata_pair(model, adata_by_mod: Mapping[str, Any], *, device: str='cpu', batch_size: int=1024, use_mean: bool=True, return_gates: bool=True, return_gate_logits: bool=True, write_to_adatas: bool=True, fused_obsm_key: str='X_univi_fused', gate_prefix: str='gate', layer_by_mod: Optional[Mapping[str, Optional[str]]]=None, X_key_by_mod: Optional[Mapping[str, str]]=None)
```

```text
Encode a fused posterior for paired/multi-observed cells.

Expected:
  - all AnnData share identical obs_names in the same order.

Returns dict with keys:
  - "Z_fused" (n_cells, latent_dim)
  - "mu", "logvar"
  - "gates" (or None)
  - "gate_logits" (or None)
  - "modality_order" (the order corresponding to columns of gates)
```

### `denoise_adata`

```python
denoise_adata(model, adata, *, modality: str, device: str='cpu', out_layer: str='denoised_fused', overwrite_X: bool=False, batch_size: int=512, adata_by_mod: Optional[Mapping[str, Any]]=None, layer_by_mod: Optional[Mapping[str, Optional[str]]]=None, X_key_by_mod: Optional[Mapping[str, str]]=None, use_mean: bool=True, decoder_prefer: str='auto', decoder_link: Optional[str]=None)
```

```text
Denoise (reconstruct) a modality.

- If adata_by_mod is None: encodes only this modality, decodes this modality (self denoise).
- If adata_by_mod is provided: encodes a fused posterior from adata_by_mod (paired/multi-observed),
  then decodes `modality` and writes into `adata.layers[out_layer]`.

Returns the modified `adata` (for chaining).
```

### `denoise_from_multimodal`

```python
denoise_from_multimodal(model, adata, *, modality: str, adata_by_mod: Mapping[str, Any], device: str='cpu', out_layer: str='denoised_fused', overwrite_X: bool=False, batch_size: int=512, layer_by_mod: Optional[Mapping[str, Optional[str]]]=None, X_key_by_mod: Optional[Mapping[str, str]]=None, use_mean: bool=True, decoder_prefer: str='auto', decoder_link: Optional[str]=None)
```

```text
README alias: true multimodal denoising via fused latent.
```

### `evaluate_alignment`

```python
evaluate_alignment(*, Z1: np.ndarray, Z2: np.ndarray, metric: str='euclidean', recall_ks: Sequence[int]=(1, 5, 10), k_mixing: int=20, k_entropy: int=30, labels_source: Optional[np.ndarray]=None, labels_target: Optional[np.ndarray]=None, compute_bidirectional_transfer: bool=True, k_transfer: int=15, json_safe: bool=True, block_size: int=512)
```

```text
README-facing alignment evaluation.

Computes:
  - FOSCTTM (mean + SEM)
  - Recall@k (mean + SEM) for each k
  - modality mixing + entropy in the *stacked* space
  - label transfer (source->target), and optionally bidirectional summary

Returns a flat dict with stable keys used by README examples.
```

### `generate_from_latent`

```python
generate_from_latent(model, n: Optional[int]=None, *, target_mod: Optional[str]=None, device: str='cpu', z_source: str='prior', z: Optional[np.ndarray]=None, batch_size: int=512, return_mean: bool=True, sample_likelihood: bool=False, decoder_prefer: str='auto', decoder_link: Optional[str]=None, random_state: int=0)
```

```text
README-compatible generator.

Notes
-----
- If z is None, samples z ~ N(0, I) when z_source == "prior".
- return_mean is kept for API parity (decoders return mean-like outputs).
- sample_likelihood is accepted but not implemented here.
```

### `fit_label_latent_gaussians`

```python
fit_label_latent_gaussians(Z: np.ndarray, labels: Sequence[Any], *, min_n: int=20, shrink: float=0.001)
```

### `sample_latent_by_label`

```python
sample_latent_by_label(gauss: Dict[str, Dict[str, Any]], label: str, n: int, *, random_state: int=0)
```
