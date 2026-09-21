# Reconstruction, imputation, and generation

These operations all use a decoder, but their biological meaning depends on how the latent vector was obtained.

| Operation | Information supplied | Meaning of output |
| --- | --- | --- |
| Self reconstruction | One measured assay | Same-assay model reconstruction |
| Fused reconstruction | Multiple paired assays | Reconstruction informed by all supplied assays |
| Cross-modal prediction | Source assay only | Model estimate of the target assay |
| Latent generation | Prior or explicitly supplied latent vectors | Synthetic decoder mean profiles |

## Preserve measured data

```python
from univi.evaluation import denoise_adata, denoise_from_multimodal, cross_modal_predict

denoise_adata(model, rna_test, modality="rna", device="cuda",
             out_layer="self_reconstruction", overwrite_X=False)
denoise_from_multimodal(
    model, rna_test, modality="rna",
    adata_by_mod={"rna": rna_test, "adt": adt_test}, device="cuda",
    out_layer="paired_reconstruction", overwrite_X=False,
)
adt_test.layers["predicted_from_rna"] = cross_modal_predict(
    model, rna_test, "rna", "adt", device="cuda",
)
```

The last assignment requires paired and identically ordered test rows. For an RNA-only query, construct a new target-feature AnnData with the query's observation index and the saved target feature order. Never attach predictions to unrelated measured cells simply because the shapes match.

Retain `layers['counts']` and the original measured assay. The [CITE marker tutorial](../tutorials/03_cite_reconstruction.md) uses the same spatial coordinates and comparable color scales for measured and predicted values. Its feature tables expose both favorable and unfavorable reconstructions.

## Interpret the target representation

Gaussian decoders return values in the fitted continuous input space. A Z-scored RNA target requires the saved RNA scaler's inverse transform to recover log-normalized units. Those values are still not raw UMI counts. A CLR protein target is not an antibody count, and a decoded LSI component is not a peak probability or gene-activity score.

The raw-feature Supplemental Notebook S1 workflow instead uses NB RNA and Poisson ATAC likelihoods. Its perturbations act on the actual modeled peak columns. Use the [preprocessing guide](preprocessing.md) to identify the representation before interpreting a decoder output.

For scNMT, distinguish success counts, coverage, and predicted fractions. Evaluate fractions only where target coverage is positive. Zero coverage denotes no observation, not zero methylation or accessibility.

## Generate profiles from latent vectors

```python
from univi.evaluation import (
    generate_from_latent, fit_label_latent_gaussians, sample_latent_by_label,
)

# Model-prior profiles; outputs are decoder means.
prior_profiles = generate_from_latent(model, n=100, device="cuda", random_state=0)

# An empirical, label-conditioned latent distribution fit on training cells.
distributions = fit_label_latent_gaussians(z_train, train_labels, min_n=20)
z_sample = sample_latent_by_label(distributions, label="CD4 T", n=100, random_state=0)
rna_profiles = generate_from_latent(model, z=z_sample, target_mod="rna", device="cuda")
```

The second example is conditional on an empirical Gaussian fit to a labeled training population; it does not add a conditional generative architecture to UniVI. Verify that the chosen label exists and meets the minimum sample requirement.

The current `generate_from_latent` returns mean-like decoder outputs. `sample_likelihood=True` is explicitly unimplemented. Generated profiles are useful for model inspection, but are not independent biological replicates and should not inflate differential-expression sample sizes.
