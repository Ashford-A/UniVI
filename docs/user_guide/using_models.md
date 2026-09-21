# Using a trained model

All functions below take the model, AnnData objects already transformed by the fitted preprocessors, and a `device`.

## Embed

```python
from univi.evaluation import encode_adata, encode_fused_adata_pair
from univi.workflows import stack_embeddings

z_rna = encode_adata(model, rna, modality="rna", latent="modality_mean")        # (cells, latent_dim)
fused = encode_fused_adata_pair(model, {"rna": rna, "atac": atac})              # writes .obsm["X_univi_fused"]
joint = stack_embeddings(model, [("ref", "rna", rna), ("ref", "atac", atac)])   # AnnData, .obsm["X_univi"]
```

See [which latent representation to use](model.md#which-latent-representation-to-use).

## Predict one modality from another

```python
from univi.evaluation import cross_modal_predict
adt_from_rna = cross_modal_predict(model, rna, src_mod="rna", tgt_mod="adt")    # (cells, n_proteins)
```

Predictions are in the target modality's **model input space** (for example z-scored CLR protein, or LSI components for ATAC). For count, Bernoulli and beta-binomial decoders the function returns the decoder mean (expected counts, probabilities, or fractions).

## Denoise

```python
from univi.evaluation import denoise_adata
denoise_adata(model, rna, modality="rna", out_layer="denoised")                              # from RNA alone
denoise_adata(model, rna, modality="rna", out_layer="denoised", adata_by_mod={"rna": rna, "atac": atac})  # both
```

The result is written to `rna.layers[out_layer]`.

## Generate

```python
from univi.evaluation import generate_from_latent, fit_label_latent_gaussians, sample_latent_by_label

cells = generate_from_latent(model, n=1000, z_source="prior")          # {"rna": ..., "atac": ...}
gauss = fit_label_latent_gaussians(z_rna, rna.obs["cell_type"].astype(str).to_numpy())
z_b = sample_latent_by_label(gauss, label="Naive B", n=500)
b_rna = generate_from_latent(model, z=z_b, target_mod="rna")
```

## Perturb an input feature

```python
from univi.perturbation import predict_feature_perturbation
res = predict_feature_perturbation(model, rna, source_modality="rna", target_modality="rna",
                                   features=["MS4A1"], mode="off")    # also "set", "add", "scale" with value=
res["delta"]   # perturbed minus baseline decoder mean, per cell and output feature
```

This measures the model's learned associations, not causal effects.

## Modality contributions

```python
from univi.evaluation import encode_moe_gates_from_tensors
w = encode_moe_gates_from_tensors(model, {"rna": rna.X, "atac": atac.X}, kind="effective_precision")
w["weights"], w["per_modality_mean"]
```

See [modality weights and gating](model.md#modality-weights-and-gating).

## Save and reload

A **reference bundle** keeps everything needed to use the model on new data:

```python
from univi.workflows import save_reference, load_reference

save_reference("my_reference", model, preprocessors={"rna": rna_prep, "atac": atac_prep},
               metadata={"dataset": "...", "split": {"train": list(train_cells)}})
model, preprocessors, metadata = load_reference("my_reference", device="cuda")
query = preprocessors["rna"].transform(new_rna)
```

The bundle holds the weights, the full configuration, classification heads and their label names, the fitted preprocessors (`preprocessing.joblib`, which uses pickle, so only load bundles you trust), and your metadata. It does not store optimizer state; it is for inference and further training stages, not for resuming an interrupted run. To resume training, use `trainer.save(path)` / `trainer.load(path)`, which include the optimizer.
