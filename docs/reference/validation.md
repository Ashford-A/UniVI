# Validation and current limits

The original tutorial source was checked locally on 20 September 2026; release packaging and public imports were checked again for 0.5.0 on 21 September 2026. The checks below validate software behavior and documentation integrity; the manuscript's large biological experiments were not rerun.

## Automated helper checks

The targeted test suite contains 17 cases, including parameterized cases. It covers frozen decoder parameter/buffer preservation across training calls, gradients reaching unfrozen encoders, public head attachment without reinitializing the reference, unknown-label masking, staged refinement with replay, save/load parity, training-fitted preprocessing and feature-order handling, actual pairing checks, capped split semantics, fractional perturbations on integer-typed inputs, safe rejection of a stale reference overwrite, and lazy top-level public API imports.

```bash
python -m pytest -q tests/test_biological_workflows.py
```

## Executable tutorial checks

All 11 tutorial scripts were executed on small synthetic inputs using CPU and shortened training schedules. The fixtures cover paired RNA/ADT, RNA/ATAC, independent bridge queries, TEA-seq well metadata, missing AML mutation labels, scNMT tar members with success/coverage matrices, and peak-linked perturbations. The marker reconstruction script reuses the CITE reference produced by the preceding script.

```bash
python tests/smoke_examples.py
```

This exercises input preparation, model construction, training/refinement, checkpoint round trips, analysis outputs, and plotting. The synthetic runs use smaller architectures, fewer components, and fewer epochs; they do not establish real-data memory requirements, convergence, or biological accuracy. Generated tutorial notebooks are intentionally unexecuted and have no synthetic plots passed off as biological results.

## Documentation checks

```bash
python scripts/build_tutorial_docs.py --check
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

The site uses committed generated pages/notebooks and an AST-derived API reference so documentation builds do not need Torch or biological data. The prepared GitHub workflow runs these checks and the helper tests. Local validation does not imply that a remote GitHub Actions or Read the Docs build has run.

The strict HTML build passed with no warnings. All 11 notebook files passed format and code-syntax validation. A static scan checked 6,177 local links and asset references across the 31 HTML files and found no missing targets. Those files comprise 29 content pages plus search and the generated index. Check the rendered layout, navigation, and downloads after deployment.

## Release artifact checks

For release 0.5.0, both the wheel and source distribution passed strict Twine metadata checks and a comparison of all 39 package Python files against the checkout. The installed wheel was imported outside the source tree and passed all 17 helper tests. The conda-forge recipe's version, runtime dependencies, and source SHA-256 were checked, and its public API/reference-bundle smoke script passed against that installed wheel. A full conda build and remote feedstock CI were not run locally.

## Biological verification still needed

Real-data replication requires downloading and preparing the datasets in the [data guide](../guides/data.md), recovering original split maps where exact correspondence matters, reviewing the [documented notebook differences](notebook-provenance.md), and running the full training schedules. Original model checkpoints and complete processed input matrices are not included in the tutorial distribution.

The included published figure images are source references. No newly generated biological score, UMAP, mutation probability, or perturbation result is represented as a verified replication of the paper. The [reproducibility guide](../guides/reproducibility.md) explains what to preserve for a real validation run.
