# Troubleshooting biological workflows

| Symptom | Likely cause and action |
| --- | --- |
| `freeze_decoders` or `univi.refinement` is missing | Install UniVI 0.5.0 or the updated source checkout, then restart the notebook kernel. See [Installation](../installation.md). |
| `No module named _common` | Download the complete tutorial bundle and run notebooks from its extracted directory. Keep `_common.py` beside them. |
| Cell order or pairing error | Intersect and reorder by actual barcode identity. A multi-key loader requires the same cells in the same order. Independent cohorts need separate loaders. |
| Missing fitted genes, proteins, or peaks | Apply the saved reference feature universe. Resolve identifier/build differences upstream; do not silently replace unmeasured features with zeros. |
| Missing `counts` layer | Recover raw nonnegative counts before preprocessing. Do not rename normalized values to counts or apply CLR twice. |
| HVG LOESS error | Check raw counts, enough cells/variable genes, and installed `scikit-misc`. The workflow does not silently substitute a different HVG algorithm. |
| Too few cells or features for LSI | Reduce the requested SVD dimension deliberately for a small pilot. Published settings assume the full dataset. |
| BatchNorm fails on a singleton | Choose a batch size without a singleton remainder or use `drop_last=True` for the training loader. Keep complete validation/test data. |
| CPU/GPU out of memory | Reduce batch size and pair-ranking sample size. Some existing AnnData evaluation helpers densify the input matrix; batching alone may not solve host-memory pressure. |
| AML AUC is missing | Inspect per-gene test label counts and classes. Missing or one-class outcomes cannot support ROC-AUC. |
| All-unlabeled refinement error | Supply observed training and validation labels for active heads. Unknown mutation calls must remain unknown. |
| scNMT success/coverage error | Align cell IDs and gene-body features, confirm integer counts and `0 ≤ successes ≤ coverage`, and retain zero coverage as unobserved. |
| Refined model looks unchanged | The best validation checkpoint may come from head-only warmup. Inspect `best_epoch` and history before extending training. |
| Frozen decoder outputs change | Frozen decoder weights can receive different latent inputs after encoder refinement. Compare parameter/buffer equality separately from output equality. |
| Predicted LSI values look negative | LSI is a signed continuous representation. These values are not raw accessibility counts. |

## Minimum useful diagnostic record

Report the package commit, full exception, input shapes, feature names/types, split sizes, likelihoods, and the smallest example that reproduces the problem. Include whether `.X` is raw counts, log-normalized, CLR, Z-scored, LSI, or a coverage-aware fraction. Avoid sharing identifiable sample information in a public issue.

For an initial environment check, run the [synthetic quickstart](../tutorials/00_quickstart.md). For a documentation build error, use `python -m sphinx -n -W --keep-going -b html docs docs/_build/html` from the repository root; warnings are intentionally treated as failures.
