# Releasing UniVI

Tags use the form `vX.Y.Z` (for example `v1.1.0`). The version must match in `pyproject.toml`, `univi/__init__.py`, `CITATION.cff`, `conda.recipe/meta.yaml`, and the top entry of `CHANGELOG.md`.

## One-time setup

- **PyPI**: an API token from <https://pypi.org/manage/account/token/>, saved in `~/.pypirc`:

  ```ini
  [pypi]
  username = __token__
  password = pypi-...
  ```

- **GitHub CLI** (optional, for creating releases from the terminal): `gh auth login`.
- **Zenodo software DOIs** (optional): sign in at <https://zenodo.org> with GitHub, open *GitHub* in your account settings, and switch the `Ashford-A/UniVI` repository on. Every GitHub release published afterwards is archived with its own DOI. This must be enabled *before* the release is created.
- **Read the Docs**: in the project's admin settings, set the default version to `stable`, which follows the newest release tag.

## 1. Optional: publish more datasets

The export code for each paper dataset is in `docs/reproducibility/building_datasets.md`. For files you already have on disk:

```bash
python scripts/prepare_zenodo_release.py prepare hao_citeseq_pbmc \
    --file rna=/path/Hao_RNA_data.h5ad --file adt=/path/Hao_ADT_data.h5ad \
    --labels celltype.l1 celltype.l2 celltype.l3 --outdir zenodo/hao_citeseq_pbmc
python scripts/zenodo_upload.py zenodo/hao_citeseq_pbmc --creator "Ashford, Andrew J."   # then publish the draft on zenodo.org
python scripts/prepare_zenodo_release.py register zenodo/hao_citeseq_pbmc --record-id <RECORD_ID>
python -c "import univi.datasets as u; print(u.list_datasets())"
```

Registry changes ship with the package, so do this before building the release.

## 2. Optional: refresh tutorial outputs

Read the Docs shows the outputs stored in the notebooks. After changing a tutorial, execute it on real data (GPU recommended) and review the results:

```bash
python scripts/execute_tutorials.py            # or: python scripts/execute_tutorials.py quickstart
git diff --stat docs/tutorials
```

## 3. Check everything locally

```bash
python -m pip install -e ".[tutorials,test,docs]" build twine
python -m pytest -q                                           # unit tests + tutorial notebooks
python -m sphinx -W --keep-going -b html docs docs/_build/html
rm -rf dist && python -m build --outdir dist
python -m twine check --strict dist/*
python scripts/verify_release.py dist
```

## 4. Commit and push to `main`

```bash
git checkout main
git pull
git add -A
git status                      # review the file list
git commit -m "Release v1.1.0"
git push origin main
```

Wait for the GitHub Actions checks to pass.

## 5. Tag

```bash
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0
```

## 6. Upload to PyPI

Upload the files built and checked in step 3 (rebuild them if anything changed since):

```bash
python -m twine upload dist/univi-1.1.0.tar.gz dist/univi-1.1.0-py3-none-any.whl
```

Published files cannot be replaced; a mistake needs a new version number.

## 7. Create the GitHub release

```bash
awk '/^## 1.1.0/{f=1; next} /^## /{f=0} f' CHANGELOG.md > /tmp/univi-notes.md
gh release create v1.1.0 dist/univi-1.1.0.tar.gz dist/univi-1.1.0-py3-none-any.whl \
    --title "v1.1.0" --notes-file /tmp/univi-notes.md
```

or use *Releases → Draft a new release* on GitHub with the same tag, title, notes, and files. With the Zenodo integration on, the release is archived and receives a DOI within a few minutes.

## 8. Verify the PyPI package

```bash
python -m venv /tmp/univi-check && source /tmp/univi-check/bin/activate
python -m pip install "univi==1.1.0"
python -c "import univi, univi.datasets as u; print(univi.__version__); print(u.list_datasets())"
deactivate
```

## 9. Update conda-forge

The feedstock is <https://github.com/conda-forge/univi-feedstock>. After the PyPI release, the conda-forge bot (`regro-cf-autotick-bot`) normally opens a pull request titled `univi v1.1.0`, usually within hours.

1. Open the pull request and check the diff: the version, the `sha256` of the PyPI sdist, and the `run` requirements (compare with `dependencies` in `pyproject.toml`; unchanged since 0.5.0). Make sure the `host` requirement reads `setuptools >=77` (needed since 1.0.0, which uses an SPDX license expression); if it is still plain `setuptools`, push that change to the bot's branch before merging.
2. Wait for the feedstock CI to pass, then merge.
3. The package appears on the conda-forge channel after the upload finishes (often within an hour of merging).

If no bot pull request appears, update the recipe yourself:

```bash
git clone https://github.com/<your-username>/univi-feedstock.git     # your fork of the feedstock
cd univi-feedstock
git checkout -b v1.1.0
python /path/to/UniVI/scripts/update_conda_recipe.py --feedstock . --version 1.1.0
# make sure the host requirement in recipe/meta.yaml reads `setuptools >=77`
git diff                                                               # version, sha256, setuptools
git commit -am "univi v1.1.0"
git push origin v1.1.0
```

Open a pull request against `conda-forge/univi-feedstock`, comment `@conda-forge-admin, please rerender`, and merge after CI passes. Reset `build: number:` to 0 for a new version. The script reads the published sdist from PyPI, so run it after step 6.

Verify:

```bash
conda create -n univi-check -c conda-forge "univi=1.1.0" && conda activate univi-check
python -c "import univi; print(univi.__version__)"
```

## 10. Documentation

Pushing to `main` rebuilds the `latest` docs; pushing the tag updates `stable`. On Read the Docs, open *Versions*, confirm both built, and activate `v1.1.0` if you want a permanent page for that release. Check the tutorials, the API reference, and the Colab buttons.

## 11. Afterwards

- If Zenodo minted a software DOI, add its badge to `README.md` and the DOI to `CITATION.cff` (as `doi:`).
- On the GitHub repository page, set *About → Website* to `https://univi.readthedocs.io` and add topics (for example `single-cell`, `multi-omics`, `variational-autoencoder`, `pytorch`, `anndata`).
- For the next development cycle, add an `## Unreleased` section at the top of `CHANGELOG.md`.
