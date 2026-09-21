# FAQ

**Does UniVI need paired data?**
Training needs cells with at least two modalities measured together; that is what teaches the model how modalities correspond. Once trained, single-modality data can be embedded with the matching encoder, so unimodal cohorts can be mapped onto a paired reference ([tutorial](../tutorials/query_mapping.ipynb)).

**How many paired cells do I need?**
In the paper's overlap sweep on Multiome PBMCs (Fig. 9), correspondence collapsed when only a few percent of cells were paired and stabilized with modest overlap: with 10% of cells paired, bidirectional label-transfer macro-F1 was about 0.70 and clustering NMI about 0.72. The paired cells should cover the cell types you care about.

**Raw counts or normalized data?**
Keep raw counts in `layers["counts"]` and give the model normalized inputs with a Gaussian likelihood for integration work (the paper's default). Use count likelihoods (`nb`, `poisson`) on raw counts when you need decoders that output counts.

**Can I use more than two modalities?**
Yes, any number: add one `ModalityConfig` per modality ([tutorial](../tutorials/custom_modalities.ipynb)).

**How do I handle batches or donors?**
Train on data that spans them; the shared latent space is often robust to moderate batch effects. For stronger effects, preprocess with a batch-aware method, or add an adversarial head (`ClassHeadConfig(..., adversarial=True)`) for the batch variable.

**Are predicted values real measurements?**
No. Cross-modal predictions and denoised values are model estimates. They are good for visualization, annotation, and hypothesis generation; statistical tests should use measured data.

**Why does my joint UMAP look mixed but cell types are blurred?**
Mixing alone is easy to achieve by collapsing structure. Check label transfer and per-cell-type structure too; lower `gamma` if the embedding over-aligns.

**Where are downloaded datasets stored?**
In `~/.cache/univi` unless `UNIVI_DATA_DIR` is set. Files are verified by checksum and reused.

**How do I cite UniVI?**
See [Citation](../citation.md).
