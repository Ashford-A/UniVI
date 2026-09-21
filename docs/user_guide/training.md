# Training and tuning

## The training loop

```python
from univi import TrainingConfig, UniVITrainer
trainer = UniVITrainer(model, train_loader, val_loader, TrainingConfig(
    n_epochs=400, batch_size=256, lr=1e-3, weight_decay=1e-4, device="cuda",
    early_stopping=True, patience=50, best_epoch_warmup=110, grad_clip=None, log_every=25))
history = trainer.fit()     # {"train_loss", "val_loss", "beta", "gamma"} per epoch
```

- With a validation loader, the trainer tracks the best validation loss and **restores those weights** at the end of `fit()`. `trainer.best_epoch` records which epoch that was.
- `best_epoch_warmup` delays best-epoch tracking (and patience counting). Set it to the end of your annealing windows: validation loss uses the full `beta` and `gamma`, so it is high and not comparable while they are still ramping up.
- `early_stopping=True` stops after `patience` epochs without improvement (by more than `min_delta`). `n_epochs` is then an upper bound.
- `grad_clip` clips the gradient norm; useful if the loss turns `nan`.
- `UniVITrainer(..., use_amp=True, amp_dtype="bf16")` enables mixed precision on CUDA.
- The trainer uses Adam with the configured learning rate and weight decay.

## Starting hyperparameters

The paper's settings are a good starting point (Supplemental Table S8 lists them for every analysis):

| Data | `latent_dim` | `beta` | `gamma` | Dropout (enc/dec) | Annealing (KL; alignment) |
| --- | --- | --- | --- | --- | --- |
| CITE-seq (RNA + ADT) | 30 | 1.5 | 5.0 | 0.10 / 0.00 | none |
| 10x Multiome (RNA + ATAC) | 30 | 1.25 | 4.35 | 0.10 / 0.05 | epochs 50–85; 75–110 |

Encoders of `[512, 256, 128]` for about 2,000 genes and `[128, 64]` for protein or 100-dimensional LSI inputs work well; decoders mirror them.

In the paper's sensitivity analyses on Multiome PBMCs (Supplemental Figs. S8–S10):

- performance was stable over a broad region of `beta` and `gamma`, and degraded when either was set to 0;
- cross-modal correspondence and label transfer were similar for latent dimensions from about 10 to 50, while clustering agreement (NMI) declined as the dimension grew;
- dropout between 0 and about 0.15 had small effects.

## Reading the training curve

- **Validation loss far above training loss in the first epochs**: expected while `beta`/`gamma` anneal from 0.
- **Loss becomes `nan`**: lower the learning rate, set `grad_clip=5.0`, and check the inputs for `inf`/`nan` values or a likelihood that does not match the data (for example `nb` on z-scored values).
- **Validation loss rises steadily after an early minimum**: overfitting; early stopping handles it, and more dropout or weight decay can help.

## Common problems

| Symptom | Things to try |
| --- | --- |
| Modalities form separate clouds in a joint UMAP | increase `gamma`; check that the paired `obs_names` really match; train longer after the alignment ramp |
| Cell types blur together | lower `gamma` or `beta`; increase `latent_dim`; check preprocessing (HVGs, scaling) |
| One modality dominates | raise the other modality's `recon_weight`, or use `recon_normalize_by_dim=True` |
| Poor cross-modal prediction of some features | expected for features the source modality does not inform; judge with per-feature correlations and cell-type means |
| `ValueError` about missing features at `transform` | the query lacks reference features; see [matching features](data.md#matching-features-for-new-data) |
| BatchNorm error with a batch of size 1 | `make_loader(..., drop_last=True)` for training |
| Out of memory on the GPU | smaller `batch_size`; fewer input features (HVGs, LSI instead of raw peaks) |

## Hyperparameter search

`univi.hyperparam_optimization` contains random-search helpers for common data types, called from Python with training and validation AnnData objects:

```python
from univi.hyperparam_optimization import run_multiome_hparam_search
best_config, results = run_multiome_hparam_search(rna_train, atac_train, rna_val, atac_val,
                                                  celltype_key="cell_type", device="cuda", max_configs=50)
```

Similar functions exist for CITE-seq (`run_citeseq_hparam_search`), TEA-seq, and single modalities (RNA, ADT, ATAC). They expect raw counts in `layers["counts"]`.

## Reproducibility

`univi.utils.seed.set_seed(0)` seeds Python, NumPy, and PyTorch; `set_seed(0, deterministic=True)` also requests deterministic cuDNN kernels. Results can still differ slightly across hardware and PyTorch versions. Save the split (`split_by_label` output or `obs_names`), the fitted preprocessors, and the model together, which `save_reference` does.
