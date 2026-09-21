# Evaluate biology, correspondence, and generalization

An integration can mix modalities while losing biological structure, or preserve broad cell types while failing to match individual cells. Evaluate both properties in the original latent space. Use UMAP to inspect the result, not as the input to the reported latent-space metrics.

## Choose a metric for the question

| Question | Quantity | Interpretation and limitation |
| --- | --- | --- |
| Does each cell retrieve its measured counterpart? | Directional FOSCTTM, Recall@k | Requires paired cells in identical row order. Lower FOSCTTM and higher recall are better. |
| Are annotated populations transferable? | Cross-modality kNN accuracy, macro-F1, confusion matrix | Broad shared labels can transfer even when exact-cell correspondence is poor. Macro-F1 exposes rare-class failures. |
| Is biological structure retained? | Fused train-to-test label transfer, ARI, NMI, silhouette | Record label granularity and clustering procedure. These are different estimands, not interchangeable scores. |
| Do modalities occupy shared neighborhoods? | Stacked-space mixing and modality entropy | Report cohort balance, neighborhood size, and cell-type stratification. Mixing alone does not establish a useful integration. |
| Can a missing assay be predicted? | Per-feature correlation, MSE, marker maps | Evaluate measured held-out targets in their training representation. A smooth decoder output can look convincing despite poor prediction. |
| Do supervised mutation heads generalize? | Held-out ROC-AUC and average precision | Report positives, negatives, prevalence, and split unit. One-class test targets have undefined AUC. |

For directional FOSCTTM, each source cell's true target distance is compared with all other candidate targets. UniVI's implementation counts **strictly closer** targets and divides by `n - 1`. Ties do not count as closer. Compute the reverse direction separately. Candidate-pool size changes Recall@k and can change FOSCTTM; record any subsampling and its seed.

```python
from univi.evaluation import encode_adata, compute_foscttm, compute_match_recall_at_k

assert rna_test.obs_names.equals(atac_test.obs_names)
zr = encode_adata(model, rna_test, modality="rna", device="cuda")
za = encode_adata(model, atac_test, modality="atac", device="cuda")
scores = {
    "rna_to_atac_foscttm": compute_foscttm(zr, za),
    "atac_to_rna_foscttm": compute_foscttm(za, zr),
    "rna_to_atac_recall10": compute_match_recall_at_k(zr, za, k=10),
}
```

The tutorial helper caps expensive pair-ranking evaluations at 20,000 paired cells using a fixed subsample. It retains the full held-out cohorts for label transfer. At this size, pair ranking remains quadratic in time; blocked computation limits temporary distance-matrix memory rather than eliminating that cost.

## Paired, unpaired, and supervised comparisons

In Figure 5, Ding RNA and Satpathy ATAC are independent cells. There is no true cross-cohort pair index, so FOSCTTM between those cohorts is undefined. Evaluate harmonized-label transfer and cell-type-conditioned mixing instead. Do not align row numbers and call them pairs.

A fused embedding uses all supplied assays for each cell. It is appropriate for joint clustering, but a fused vector containing the target assay is not an independent cross-modality prediction. Use the source-only encoder when assessing a missing target assay.

Reserve test labels before head training. When comparing pre- and post-refinement results, retain the same query split, label vocabulary, feature transforms, metric definitions, and evaluation rows. Compare supervised results against the unrefined projection and state which labels trained the model.

Cell-level AML splits assess within-cohort behavior. Patient-grouped splits answer a different question. An all-cell scNMT view is transductive because it includes training cells; keep it separate from the held-out score.

## Figure 8 scope

The [Figure 8 tutorial](../tutorials/08_evaluation.md) exports the ten panel metric families for UniVI. It uses three seeds to demonstrate the workflow, rather than recreating the publication's full cross-validation experiment. The original Python and R benchmark notebooks remain the references for competitor installation, method-specific training, folds, and result aggregation.

Combine methods only after matching splits, preprocessing access, labels used during fitting, representation type, direction, candidate pool, hardware, and timing boundaries. Report unavailable modality-specific metrics as missing. A tool exposing only a fused embedding cannot supply a meaningful separate-modality pair-retrieval score.
