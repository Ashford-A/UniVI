# Representations, fusion, and modality weights

Choose the representation that answers the biological question. A plotting function cannot make an inappropriate representation into a valid evaluation.

| Representation | Rows | Public entry point | Appropriate use |
| --- | --- | --- | --- |
| Modality-specific mean | One per cell in one assay | `encode_adata(..., latent="modality_mean")` | Paired retrieval, cross-modality transfer, unimodal projection |
| Fused mean | One per genuinely paired cell | `encode_fused_adata_pair(...)` | Joint neighborhood analyses, one representation per cell |
| Stacked means | One per cell per assay | `stack_embeddings(...)` | Modality interleaving, assay-specific offsets, paired links |
| Decoder mean | One per input cell, columns in target training space | `cross_modal_predict(...)` | Cross-assay prediction and validation |

## A single-modality “MoE mean” is still single-modality input

`encode_adata` supplies only the selected assay. With one expert, the analytic fused mean equals that expert's mean; the function does not acquire information from a missing second assay. This makes it suitable for projecting a new unimodal cohort into a pretrained reference.

`encode_fused_adata_pair` accepts a dictionary with two **or more** modalities despite its name. All objects must share the same ordered cell identifiers. Do not pass independent cohorts to obtain a supposed fused cell.

## Why duplicated fused embeddings are misleading

If the same fused matrix is assigned to RNA and ADT and then stacked, their paired distances become zero by construction. A modality-colored plot will look perfectly mixed even if the two original encoders disagree. Use the separate encoder means to assess correspondence; use one fused point per cell to describe joint biology.

## Precision fusion versus learned gates

For an encoder posterior with log variance $\ell_{m,d}$, precision is $\tau_{m,d}=\exp(-\ell_{m,d})$. With learned routing disabled, the fused mean in dimension $d$ is:

$$\mu_{f,d}=\frac{\sum_m\tau_{m,d}\mu_{m,d}}{\sum_m\tau_{m,d}}.$$

A learned router, when enabled, multiplies each modality's precision by a per-cell softmax weight before fusion. These are different quantities:

| Quantity | Interpretation | Caveat |
| --- | --- | --- |
| Router probability | A network output over available modalities | Enabling a router does not prove it received a meaningful training gradient |
| Per-dimension normalized precision | Each expert's relative precision in each latent dimension | Depends on learned variance and calibration |
| Effective-precision summary | Sum precision over dimensions, normalize over modalities | A compact diagnostic, not the complete per-dimension fusion weights |
| Uniform fallback gate | Equal weights returned when no learned router exists | Does not mean the fused representation ignored posterior precision |

For the scNMT tutorial, use:

```python
weights = encode_moe_gates_from_tensors(
    model, {mod: a.X for mod, a in paired.items()},
    modality_order=["rna", "cpg", "gpc"],
    kind="effective_precision", return_logits=False,
)
```

The explicit `kind` names what the plotted heatmap measures. In the current implementation, plain v1 self/cross reconstruction and posterior alignment do not train a router through fused reconstruction. The visible scNMT notebook leaves learned gating off. Its precision summaries can still be nonuniform because encoder posterior variances differ.

## Interpreting UMAP

Compute metrics in latent space. Use UMAP for exploration and display, and retain its seed and input representation. Do not independently fit UMAPs for observed versus predicted marker maps when the goal is to compare values at the same cells.

Before/after refinement UMAPs have their own coordinate fits; axis rotations and apparent cluster distances are not directly comparable. Compare quantitative neighborhood/label metrics and inspect multiple seeds or a fixed reference projection if you need to isolate geometric movement.
