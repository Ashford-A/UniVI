# Publish UniVI 0.5.0

This guide is for **Windows Command Prompt**, using Python 3.12. The new release
includes the complete documentation update and all new public APIs. The earlier
`UniVI_ReadTheDocs.zip` is not a PyPI package. The versioned release bundle
supersedes it for installation and publishing.

No external push or upload is performed by the supplied helper scripts.

## Version choice

0.5.0 is a feature release beyond 0.4.7. Publication is a reasonable occasion to
consider 1.0.0, but [Semantic Versioning](https://semver.org/) ties 1.0 to a defined,
stable public API. Use 1.0 when you intend to maintain compatibility throughout
1.x, with incompatible changes requiring a later major release. These new
workflow APIs are suitable for a 0.5.0 feature release while their real-data use
is validated. A patch version such as 0.4.8 understates the new functionality.

Package version: `0.5.0`. Git tag: `v0.5.0`. Keep the existing 0.4.7 release and
tag intact as the manuscript-era software record.

## 1. Extract and apply the release bundle

Extract `UniVI_v0.5.0_release.zip` so its `START_HERE.html` and `apply_release.py`
are directly in the folder assigned below. Adjust the path if necessary. A fresh
clone avoids mixing the release with unrelated work in your existing checkout.

```bat
set "UNIVI_RELEASE=%USERPROFILE%\Downloads\UniVI_v0.5.0_release"
cd /d "%USERPROFILE%"
git clone https://github.com/Ashford-A/UniVI.git UniVI-release-0.5.0
cd UniVI-release-0.5.0
git switch main
git pull --ff-only origin main
python "%UNIVI_RELEASE%\apply_release.py"
set "UNIVI_SOURCE=%CD%"
```

The helper checks for a clean repository and automatically chooses the full
patch or the incremental patch for the earlier documentation update. It refuses
an incompatible checkout and does not overwrite unrelated modifications.
If you already committed the earlier update in your own checkout, you can use
that clean checkout instead. Do not apply both patches manually.

If `python` is unavailable, open your usual Python/conda terminal first. One way
to obtain a release environment with conda is:

```bat
conda create -n univi-release-050 -c conda-forge python=3.12 pip -y
conda activate univi-release-050
```

## 2. Install, verify, and build

From the updated UniVI checkout, using a Python 3.12 environment:

```bat
python -m pip install --upgrade pip build twine
python -m pip install -e ".[tutorials,test]"
python -m pip install -r docs\requirements.txt
python -m pytest -q tests
python scripts\build_tutorial_docs.py --check
python -m sphinx -n -W --keep-going -b html docs docs\_build\html
python -m build --outdir dist\0.5.0
python -m twine check --strict dist\0.5.0\*
python scripts\verify_release.py dist\0.5.0
```

Stop if a check fails. The verifier checks version consistency and confirms that
every `univi/*.py` source file is present and current in both distributions.
The package contains the new API modules; docs and downloadable notebooks live
in the GitHub/Read the Docs sources. Existing CUDA users can install their desired
PyTorch build using the [official selector](https://pytorch.org/get-started/locally/).

The release archive also contains prebuilt, checked files in `dist/`. Building
from your final reviewed checkout is preferred if you changed anything. Use only
the two artifacts from the same build; do not mix old and rebuilt files.

## 3. Push the source to GitHub main

```bat
git diff --check
git status --short
git add -A
git commit -m "Release UniVI 0.5.0: biological tutorials and public refinement APIs"
git push origin main
```

The fresh checkout and ignored build/output directories keep this commit scoped
to the release. If you use an existing checkout, inspect the file list before
`git add -A`. Wait for the GitHub documentation, helper-test, and packaging checks
to pass before publishing the package. The test workflow covers Python 3.10 and
3.12 on Linux; a configured CI matrix is not a claim that it has already run.

If branch protection requires a pull request, use a release branch and merge it
through GitHub instead of forcing a push. Keep the release tag on the commit
actually merged into main.

```bat
git tag -a v0.5.0 -m "Release v0.5.0"
git push origin v0.5.0
```

## 4. Upload the wheel and source distribution to PyPI

```bat
python -m twine upload --repository-url https://upload.pypi.org/legacy/ --username __token__ dist\0.5.0\univi-0.5.0-py3-none-any.whl dist\0.5.0\univi-0.5.0.tar.gz
```

When prompted for the password, enter a PyPI API token scoped to `univi`, including
its `pypi-` prefix. Enter it at the local prompt; it does not belong in source code
or the command itself. Existing configured credentials may suppress the prompt.
See [PyPI token instructions](https://pypi.org/help/#apitoken) and the
[official packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

You cannot replace an uploaded file by changing it while retaining the same
version/filename. If a published release needs a fix, issue a new version. If only
one upload succeeded because the connection failed, upload the missing artifact
from the same build rather than rebuilding the already published version.

Verify from **outside the repository**, so its source directory cannot shadow
the installed wheel:

```bat
cd /d "%TEMP%"
python -m venv univi-verify-050
univi-verify-050\Scripts\python -m pip install "univi==0.5.0"
univi-verify-050\Scripts\python -c "import univi; from univi import UniVIRefiner, RefinementConfig, RNAPreprocessor, save_reference; print(univi.__version__, univi.__file__); assert hasattr(univi.UniVIMultiModalVAE, 'freeze_decoders')"
cd /d "%UNIVI_SOURCE%"
```

This clean install downloads the runtime dependencies. The same installed-wheel
import pattern was checked locally when preparing the bundle.

## 5. Create the GitHub Release

Use the [GitHub CLI](https://cli.github.com/) from the UniVI checkout after the
tag has been pushed. This matches the repository's
[v0.4.7 release](https://github.com/Ashford-A/UniVI/releases/tag/v0.4.7): the title
is the tag name, the notes give a short description followed by a
**Full Changelog** comparison link, and the wheel and source archive are attached.
The annotated tag message also follows the existing `Release v0.4.7` pattern.
Use the exact same artifacts that you uploaded to PyPI.

```bat
gh auth login
gh release create v0.5.0 --repo Ashford-A/UniVI --verify-tag --title "v0.5.0" --notes-file RELEASE_NOTES_v0.5.0.md dist\0.5.0\univi-0.5.0-py3-none-any.whl dist\0.5.0\univi-0.5.0.tar.gz
gh release view v0.5.0 --repo Ashford-A/UniVI --web
```

`--verify-tag` stops if the remote tag does not exist. The separate
`RELEASE_NOTES_v0.5.0.md` keeps the GitHub release in your existing concise format;
`CHANGELOG.md` retains the more detailed API and compatibility notes.

## 6. Update conda-forge

The official channel is built from the separate
[`conda-forge/univi-feedstock`](https://github.com/conda-forge/univi-feedstock)
repository. Updating UniVI's own `conda.recipe/` does not publish to conda-forge.
That in-repo recipe is now explicitly for local checkout builds.

After PyPI is live, the conda-forge bot may open a version PR automatically.
Inspect open PRs first. If a 0.5.0 PR already exists, update that PR instead of
creating a duplicate. It must include `joblib>=1.3` in runtime requirements and
the new API tests, not just the new version/hash.

```bat
gh pr list --repo conda-forge/univi-feedstock --state open
```

For a manual PR, create a branch in your personal fork. These commands follow
the [conda-forge maintenance workflow](https://conda-forge.org/docs/maintainer/updating_pkgs/):

```bat
cd /d "%USERPROFILE%"
gh repo fork conda-forge/univi-feedstock --clone=false
git clone https://github.com/Ashford-A/univi-feedstock.git univi-feedstock-release-0.5.0
cd univi-feedstock-release-0.5.0
git remote add upstream https://github.com/conda-forge/univi-feedstock.git
git fetch upstream
git switch -c update-0.5.0 upstream/main
python "%UNIVI_SOURCE%\scripts\update_conda_recipe.py" --feedstock . --version 0.5.0
git diff --check
git diff -- recipe
git add recipe
git commit -m "Update univi to 0.5.0 with public workflow API checks"
git push -u origin update-0.5.0
gh pr create --repo conda-forge/univi-feedstock --base main --head Ashford-A:update-0.5.0 --title "univi 0.5.0" --body-file "%UNIVI_RELEASE%\conda-forge\PR_BODY.md"
```

`update_conda_recipe.py` downloads the **published** source archive, verifies its
PyPI SHA-256 and package version, updates the recipe hash/version, resets build
number to zero, adds `joblib`, and adds import/reference-roundtrip tests. This
avoids a hash mismatch if your locally rebuilt tarball differs from the supplied
one. The bundle includes a concrete recipe prepared against its own tarball as a
review reference; prefer refreshing from PyPI immediately before submitting.

Use the PR number returned by GitHub in place of `PR_NUMBER`:

```bat
gh pr comment PR_NUMBER --repo conda-forge/univi-feedstock --body "@conda-forge-admin, please rerender"
gh pr checks PR_NUMBER --repo conda-forge/univi-feedstock --watch
```

Review the rerendered changes and merge after the required checks pass. The
feedstock then builds and publishes the package to conda-forge. The supplied
recipe was parsed and its Python smoke check run locally; a full conda build and
feedstock CI still need to run. Once the new build is available:

```bat
conda search -c conda-forge "univi=0.5.0"
conda create -n univi-050 -c conda-forge python=3.12 univi=0.5.0 scikit-misc jupyterlab
conda activate univi-050
python -c "import univi; from univi import UniVIRefiner; print(univi.__version__, univi.__file__)"
```

## 7. Activate the documentation

Import `Ashford-A/UniVI` into Read the Docs, or trigger a build of the existing
project. Activate main and the `v0.5.0` version as appropriate. The checked-in
`.readthedocs.yaml` supplies the build configuration. After the live URL works,
replace the documentation-source URL in README/project metadata with that actual
URL. No project slug or live documentation URL is assumed in this bundle.

The guide at `docs/guides/publishing.md` covers local preview and maintenance.
