# New public APIs in 0.5.0

UniVI 0.5.0 packages the biological tutorial APIs on top of the audited 0.4.7 repository commit `8353ec8d422841e756b3abe4e9a5c4286c0c81dd`. The release bundle includes a wheel and source distribution containing every new module. All helpers below are also available through lazy top-level imports such as `from univi import UniVIRefiner, save_reference`. The version number in a local build does not imply that it has already been uploaded to PyPI or conda-forge.

| Addition | Biological workflow it makes reusable |
| --- | --- |
| `model.freeze_decoders()` / `unfreeze_decoders()` | Keep decoder weights, BatchNorm statistics, and Dropout behavior fixed while adapting encoders |
| `model.freeze_encoders()` / `unfreeze_encoders()` | Head-only warmup and modality-specific adaptation |
| `model.add_classification_head()` | Add categorical or binary supervision without rebuilding generative weights |
| `ClassHeadConfig.layernorm` | Express the notebook-style LayerNorm heads through public configuration |
| `UniVIRefiner` / `RefinementConfig` | Mask unknown labels, warm up heads, adapt selected encoders, preserve reference latents, and optionally replay paired training data |
| Fitted RNA, ADT, and ATAC preprocessors | Apply one saved training transformation to independent cohorts |
| `make_loader()` and `stack_embeddings()` | Use the required collator and combine modality embeddings without fabricating pairing |
| `save_reference()` / `load_reference()` | Carry model settings, transforms, feature order, head vocabulary, and run metadata together |
| scNMT archive adapters | Build RNA, success-count, and coverage matrices from the source notebook's input layout |
| `predict_feature_perturbation()` | Compare a baseline and explicit feature edit without mutating input data or reaching into model internals |

The patch also fixes recursive lazy submodule imports and handles `None` head dropout/BatchNorm defaults by inheriting the model setting. The existing generative trainer and manuscript notebooks are retained.

## What to prioritize next

The most valuable next addition is a small, redistributable **real-data fixture with a pinned reference bundle**. It would let CI verify actual feature mappings and inference outputs beyond the current synthetic tests. Archived barcode split maps and antibody harmonization tables would substantially strengthen numerical reproducibility as well.

For future analyses, patient-grouped mutation evaluation, probability calibration, uncertainty summaries for cross-modal prediction, and explicit out-of-reference query diagnostics would add useful evidence. Each requires its own validated design; the new refinement helper does not claim to solve them automatically.

A future resumed-training API could save optimizer/scheduler/RNG state. The reference bundle intentionally documents that it supports inference and new stages, rather than exact resume.
