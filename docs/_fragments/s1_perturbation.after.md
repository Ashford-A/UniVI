## Treat the result as model sensitivity

A decoder response is a learned association under an altered input, not experimental evidence that a peak regulates a gene. Large count additions can move cells far outside the training distribution. The source notebook includes exploratory large-addition examples; this tutorial starts with an explicit ablation and matched nulls.

Inspect the unperturbed cross-prediction first. A poorly calibrated model is a weak foundation for perturbation interpretation. Report affected peaks, missing target genes, direction/magnitude, and null matching. The empirical tail fraction measures departure from these model-based null edits; it is not a clinical or causal significance claim.

For Bernoulli models, inputs must remain binary; for Gaussian LSI models, changing a component is not a peak edit. The public perturbation helper validates the requested feature IDs, preserves the input AnnData, and leaves model parameters unchanged. Experimental perturbation data, donor-held-out models, and out-of-distribution checks would strengthen this exploratory analysis.
