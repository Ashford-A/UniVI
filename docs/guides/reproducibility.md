# Reuse a workflow and reproduce a result

The tutorials make the manuscript's biological tasks accessible through public functions. Their source notebooks establish what the original analyses did. Exact numerical replication additionally requires the original processed feature matrices, barcode splits, annotations, environment, model state or training run, and evaluation sampling choices.

## Three distinct outputs

| Output | What it establishes |
| --- | --- |
| Published reference figure | What appeared in the supplied manuscript or supplement; embedded here with attribution |
| Executable tutorial | How to carry out the corresponding biological task with the public UniVI APIs |
| Synthetic verification | That the code paths execute and produce expected classes of output on small constructed inputs |

Synthetic verification does not establish the published biological findings or numerical performance. These tutorials ship without fabricated execution outputs. The provided executed Supplemental Notebook S1 is an additional record of the original exploratory analysis.

## Preserve the analysis identity

Use the pinned notebook links and [provenance audit](../reference/notebook-provenance.md). The audit records source-code discrepancies instead of resolving them by reading parameter JSONs. The portable workflows deliberately use training-only learned transforms; this differs from some original notebook preprocessing.

Each training tutorial saves a reference directory with model configuration, objective switches, weights, head-label mappings, fitted transforms, and metadata. Split barcodes and model feature lists are included in tutorial metadata. The supplied split-map interface allows reuse of original barcodes when available.

Keep these additional records with a real run:

1. Raw accessions, download versions, genome build, QC thresholds, cell/feature identifiers, and any antibody-name mapping.
2. The original and post-QC cell counts, excluded barcodes, grouping variables, and precise train/validation/test maps.
3. The software commit, Python/package environment, device, training seed, optimizer settings, and stopping epoch.
4. Metric direction, candidate-pool size, label definitions, neighborhood sizes, and any analysis subsamples.

```bash
git rev-parse HEAD > results/source-commit.txt
python -m pip freeze > results/environment.txt
```

Seed controls make a run easier to reproduce, but do not promise bit-identical GPU kernels or UMAP coordinates across software and hardware. Do not diagnose a different UMAP orientation as an integration failure; compare the original latent representations and biological metrics.

## Reload a reference for a new cohort

```python
from univi.workflows import load_reference
from univi.evaluation import encode_adata

model, transforms, metadata = load_reference("results/reference", device="cuda")
query = transforms["rna"].transform(query_raw)
query.obsm["X_univi"] = encode_adata(model, query, modality="rna", device="cuda")
```

Use a trusted bundle: preprocessing objects are serialized with joblib. Loading saved transforms is essential; refitting HVGs, IDF, SVD, CLR panel selection, or scaling on the query changes the reference input space. A new feature universe generally requires a new reference model.

The bundle does not save optimizer state and is not an exact interrupted-training checkpoint. It supports inference and a new refinement stage. Retain original trainer checkpoints if exact training continuation is needed.
