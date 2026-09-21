# Quickstart: train, encode, and predict

Run a small paired RNA–protein analysis on a CPU. This example creates synthetic counts, holds out cells, fits normalization on the training partition, trains UniVI, and predicts protein from RNA alone. It is a complete installation check without a download.

**What you learn:** the public configuration classes, the required collator through `make_loader`, modality-specific encoding, and cross-modal prediction. The shapes at the end should be `(10, 4)` for RNA latent coordinates and `(10, 6)` for predicted proteins. Random synthetic counts have no intended biological signal.

Install the [tutorial source version](../installation.md) before running this example. Subsequent tutorials expect real, prepared data with the [documented input contracts](../guides/data.md).
