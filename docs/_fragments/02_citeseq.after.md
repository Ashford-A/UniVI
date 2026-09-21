## Read the results

The manuscript reports a held-out set of 112,359 pairs, FOSCTTM 0.0209 on a 20,000-pair subsample, and RNA→ADT / ADT→RNA accuracies of 0.958 / 0.962. These are published reference values, not acceptance thresholds for a newly trained model. Feature panels, archived splits, checkpoint selection, and software versions affect the result.

Read confusion matrices alongside cell counts and per-class F1. A strong overall accuracy can coexist with weaker minority-state performance. Neighbor transfer within paired test data is an alignment diagnostic; it does not by itself measure transfer to a different donor or platform.

The script saves its model, objective switches, transforms, feature order, and split barcodes. Continue directly to [Figure 3 and S1](03_cite_reconstruction.md) using those files.
