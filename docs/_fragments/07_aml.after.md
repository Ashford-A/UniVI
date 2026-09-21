## Separate measurement, prediction, and supervision

The saved table reports the number of labeled test cells, the positive count, prevalence, AUC, and average precision for each cohort/gene. AUC is undefined for a one-class test set; sparse genes are explicitly marked unevaluable. Average precision should be read relative to prevalence. The published dense DAb targets perform better than several sparse van Galen targets; annotation quality and label availability matter.

A missing call is not wild type. In targeted transcript-derived genotyping, even an apparent negative can mean failure to detect a mutant transcript. Patient-level labels attached to all cells also have a different meaning from measured cell-level genotype. Record which type of evidence each column represents.

The example's mutation-head split is at cell level, matching the scope discussed in the supplement. Its performance is not a new-patient estimate. A stronger future study would group mutation-head splits by patient and evaluate unseen patients with adequate positives/negatives; do not silently replace a failed patient split with a cell split.

The LSC17 analysis here is the arithmetic mean of available panel genes in log-normalized expression, with a within-cohort Z score. It is not the clinically weighted LSC17 prognostic assay. ADT-only cells receive an explicitly labeled **imputed-RNA signature**. Their signature and mutation-head probabilities are model outputs, not independent molecular measurements.

After mutation supervision, better genotype separation is expected from the objective. The most informative checks are held-out labels, preserved unsupervised biology, and comparison with the frozen-projection baseline. [Refinement guide](../guides/refinement.md).
