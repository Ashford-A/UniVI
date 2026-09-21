# Figure 1: what UniVI learns

UniVI uses modality-specific encoders and decoders connected through a shared latent coordinate system. A paired cell can produce an RNA posterior, a protein posterior, an accessibility posterior, or another supported combination. Those posteriors are distributions, with both means and variances.

```{figure} ../_static/figures/figure-1.png
:alt: Published Figure 1, UniVI model and evaluation roadmap
:class: published-figure

Figure 1 from the supplied manuscript. This is the framework overview, not a biological training tutorial.
```

## The manuscript objective

The biological tutorials explicitly construct the model with:

```python
model = UniVIMultiModalVAE(
    config,
    loss_mode="v1",
    v1_recon="avg",
    normalize_v1_terms=True,
)
```

In this mode, training combines reconstruction, KL regularization toward a shared prior, and paired posterior alignment. For multiple observed modalities, the `avg` reconstruction setting balances self-reconstruction with directed cross-reconstruction. The fused representation is constructed separately; it does not replace those per-modality reconstruction paths.

| Task | Input to the model | Output and interpretation |
| --- | --- | --- |
| Encode RNA | RNA only | RNA-specific latent posterior mean |
| Predict protein from RNA | RNA only, protein decoder | Model-predicted protein features |
| Fuse a paired cell | All observed, genuinely paired modalities | One precision-weighted representation per cell |
| Stack modalities | Separate encoder outputs | One point per cell per modality |
| Bridge a new cohort | Query's measured modality | Projection through a pretrained encoder |
| Refine with labels | Measured input plus separate masked targets | A supervised change to encoder geometry and head parameters |

## Pairing is a biological statement

Equal row counts do not establish pairing. In paired training, row `i` must represent the same cell in every supplied modality. `align_paired_obs_names` returns a new aligned dictionary and should be assigned. It can intersect barcodes; it cannot prove that those barcodes identify the same biological cells across unrelated experiments.

Independent cohorts belong in separate loaders and separate encoder calls. A shared feature panel allows a compatible input representation; it does not manufacture cell pairs.

## Optional supervision

Cell-type and mutation heads can make a latent representation more useful for a chosen target. That changes the evidential meaning of target separation: it becomes partly optimized for the supplied labels. Preserve the unsupervised reference, hold out labels, and report which stage generated each panel.

The [representation guide](../guides/representations.md) explains fused versus stacked embeddings, posterior precision, and router weights. The [refinement guide](../guides/refinement.md) explains the new public functions used to replace notebook-level model edits.
