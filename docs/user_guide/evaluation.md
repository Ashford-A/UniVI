# Evaluating an integration

Paired data let you measure alignment directly, because each cell's true partner in the other modality is known. Compute metrics on held-out test cells.

## One call

```python
from univi.evaluation import evaluate_alignment
labels = test_rna.obs["cell_type"].astype(str).to_numpy()
m = evaluate_alignment(Z1=z_rna, Z2=z_atac, labels_source=labels, labels_target=labels, recall_ks=(1, 10, 50))
m["foscttm_mean"], m["recall_at_k"]["10"]["mean"], m["label_transfer_acc"], m["worst_direction_macro_f1"]
```

`Z1` and `Z2` are per-modality embeddings (`latent="modality_mean"`) of the same cells in the same order.

## What the metrics mean

| Metric | Function | Meaning | Better |
| --- | --- | --- | --- |
| FOSCTTM | `compute_foscttm` | for each cell, the fraction of cells closer to it than its true partner in the other modality; 0 is perfect, about 0.5 is random | lower |
| Recall@k | `compute_match_recall_at_k` | fraction of cells whose true partner is among their k nearest cross-modal neighbors | higher |
| Modality mixing | `compute_modality_mixing` | how often a cell's neighbors in a stacked embedding come from the other modality | higher, up to the balanced value |
| Modality entropy | `compute_modality_entropy` | entropy of modality labels among neighbors | higher |
| Label transfer | `label_transfer_knn` | k-NN classification of one modality's cells from the other's labels (accuracy, macro-F1) | higher |
| Reconstruction | `reconstruction_metrics`, `evaluate_cross_reconstruction` | per-feature MSE and Pearson correlation between observed and predicted values | lower MSE, higher r |

`evaluate_alignment` reports label transfer in both directions and summarizes the worse one (`worst_direction_macro_f1`), which is a stricter test than either direction alone.

## Practical notes

- FOSCTTM and Recall@k compare every cell with every other cell. They run in blocks, but for very large test sets a random subsample of 10,000–20,000 paired cells gives stable estimates.
- Mixing and entropy reward interleaving, but perfect mixing can also come from an embedding that has collapsed biological structure. Always read them together with label transfer or clustering agreement.
- Use per-modality embeddings for alignment metrics. The fused embedding is the same point for both modalities.
- For cross-modal prediction, per-feature correlation is more informative than a single average, and cell-type means (as in the [CITE-seq tutorial](../tutorials/citeseq.ipynb)) show whether biological patterns are recovered.
