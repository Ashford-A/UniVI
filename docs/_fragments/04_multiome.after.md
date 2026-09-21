## What the ATAC decoder predicts

This model's ATAC output has one column per LSI component. A change in `LSI_3` is a change in a learned accessibility coordinate; it is not accessibility at a specific peak or a promoter. The decoder has no peak-level output in this recipe.

The paper's reference test set contains 3,137 paired cells and reports FOSCTTM 0.0479, with label-transfer accuracies near 0.96 in both directions. The tutorial checks the same biological relationships, but altered preprocessing or splits need not yield those values.

The 3D tutorial panel is explicitly a PCA projection of latent coordinates with two camera views. It offers another diagnostic view; paired-retrieval metrics in the full latent space provide the quantitative evidence. For peak-level biological sensitivity analyses, use the [raw-feature S1 notebook tutorial](s1_perturbation.md).
