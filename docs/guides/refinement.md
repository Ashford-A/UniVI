# Decoder freezing and staged supervised refinement

Use supervised refinement when you have a clear target—harmonized cell identity or observed mutation status—and can reserve labels for evaluation. Save the unrefined reference first.

## Freeze decoder parameters and behavior

```python
model.freeze_decoders()
model.train()
# Decoders remain in evaluation mode; BatchNorm statistics and Dropout stay fixed.

model.freeze_encoders(["rna"])
model.unfreeze_encoders(["rna"])
```

Setting only `requires_grad=False` is insufficient to preserve a module with BatchNorm: its running statistics can still update in training mode. The added helpers clear stale gradients and maintain frozen modules in eval mode even when a trainer calls `model.train()`. They can select specific modality lists.

Freezing does not block gradients from passing **through** a decoder to an unfrozen encoder. This permits reference reconstruction replay while leaving decoder weights unchanged. It also does not guarantee unchanged decoded profiles for cells whose encoders move.

If you manually unfreeze parameters after constructing an optimizer that filtered them out, rebuild the optimizer or add the parameter group. `UniVIRefiner` rebuilds its AdamW optimizer at the stage boundary.

## Attach a public head to an existing reference

```python
from univi import ClassHeadConfig

model.add_classification_head(
    ClassHeadConfig(
        name="celltype", n_classes=len(classes),
        hidden_dims=[64, 64, 32], dropout=0.1,
        batchnorm=False, layernorm=True,
    ),
    label_names=classes,
)
```

This preserves existing generative weights and updates the serializable model configuration. Duplicate head names are rejected. Define the category order from training labels and save it; independent `.astype("category").cat.codes` calls on different cohorts can assign incompatible integers to the same biological labels.

For a binary mutation target, use `head_type="binary"`, `n_classes=2`, and one output logit. `label_names=["WT", "MUT"]` records the biological mapping, while the probability output has shape `(n_cells, 1)`.

## Supervised loaders

```python
from univi.workflows import make_loader

rna_loader = make_loader(
    {"rna": rna_train},
    labels={"NPM1": npm1_train_codes},  # 0, 1, or -1 for unknown
    batch_size=128, shuffle=True,
)
```

Use one loader per independent cohort. All assays in a multi-key dictionary must be genuinely paired. Missing labels are masked per head. Do not pass labels as an input modality for these refinement tutorials; the target would then participate in representation construction.

## Two-stage optimization

```python
from univi.refinement import UniVIRefiner, RefinementConfig

refiner = UniVIRefiner(
    model, train_loaders=[rna_loader, adt_loader],
    val_loaders=[rna_val_loader, adt_val_loader],
    encoder_modalities=["rna", "adt"], device="cuda",
    config=RefinementConfig(
        max_epochs=1000, warmup_epochs=10,
        lr_head=1e-4, lr_encoder=1e-5,
        latent_weight=5.0, patience=50,
    ),
)
report = refiner.fit()
```

Warmup trains only heads. Fine-tuning trains selected encoders and heads; generative decoders and routers remain frozen. The latent-preservation term compares the student's posterior mean to a frozen eval-mode copy of the reference. It regularizes movement without claiming exact preservation.

For Figure 5-style paired replay, supply a separate paired **training** loader and `replay_weight=2.0`. The replay term evaluates the original generative objective at the specified annealing epoch. It uses no query test labels.

The refiner alternates cohorts and cycles shorter training loaders, balancing the number of updates per cohort. Within a batch it divides supervised loss by the actual number of observed, active head targets. Validation includes each observed target once and selects the checkpoint with the lowest weighted supervised loss. A head with no labels contributes no false WT examples; all-unlabeled validation raises an error. The lowest-loss checkpoint can legitimately be a warmup checkpoint.

These semantics are explicit: they are a reusable implementation inspired by the notebook loops, not a promise of bit-identical optimizer trajectories. The original general-purpose `UniVITrainer` and its objective are preserved. `UniVIRefiner` is limited to analytic fusion without a label encoder or adversarial heads.

## Predict and save

```python
from univi.refinement import predict_heads_adata
from univi.workflows import save_reference

p = predict_heads_adata(model, query_rna, "rna", device="cuda")
query_rna.obs["p_NPM1"] = p["NPM1"].ravel()
save_reference("results/refined-reference", model,
               preprocessors=transforms, metadata=run_metadata)
```

Probabilities are model estimates. Softmax confidence and sigmoid probabilities are not automatically calibrated, and supervised separation is not independent validation of the target. Use held-out labels, cohort/patient stratification, and the original biological marker structure together.
