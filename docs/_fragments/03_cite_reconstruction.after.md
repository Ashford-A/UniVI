## Interpret smooth reconstructions carefully

Successful cross-prediction should preserve lineage contrasts and plausible within-lineage structure. Smoothing alone is not evidence of improvement: a decoder can create clean-looking marker maps while missing cell-level variation. Review the per-feature Pearson correlations and MSE tables, including weakly predicted features.

The group heatmaps use the observed group means to set one standardization for both observed and predicted values. The original notebook also explores separately standardized views; these can emphasize pattern concordance but hide amplitude differences. The tutorial's choice is explicit and reusable.

These RNA-derived protein predictions are model outputs. Do not treat them as additional independent protein measurements in hypothesis tests. For denoising and model-based synthetic sampling, see [additional analyses](../guides/denoising-generation.md).
