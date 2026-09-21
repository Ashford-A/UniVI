# Release maintenance

This guide describes the maintainer workflow for publishing UniVI releases.

## Prepare a release

1. Update the version in `pyproject.toml` and `univi/__init__.py`.
2. Update `CHANGELOG.md`, installation examples, and release notes.
3. Regenerate the tutorial pages and API reference, then run the checks below.
4. Review and merge the release changes. Require the GitHub Actions checks to pass.

```bash
python -m pip install --upgrade build twine
python -m pip install -e ".[tutorials,test]"
python -m pip install -r docs/requirements.txt
python -m pytest -q tests
python scripts/build_tutorial_docs.py --check
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

## Build and publish

Build the wheel and source distribution from the final release commit into a
version-specific output directory. Run `python -m twine check --strict` on both
files and `python scripts/verify_release.py` on that directory.

Create an annotated `vX.Y.Z` tag with the message `Release vX.Y.Z`, then push the
tag. Upload the checked distribution files to PyPI using Twine. Create a GitHub
Release titled `vX.Y.Z`, attach the same files, and include a summary of changes
and a comparison link to the previous release.

Published distribution files and release tags should remain immutable. Verify
the PyPI installation in a fresh environment outside the source checkout.

## Update conda-forge

Conda-forge releases are maintained in
[univi-feedstock](https://github.com/conda-forge/univi-feedstock).
Check for an existing version-update pull request before opening a new one.
From a feedstock checkout, run UniVI's `scripts/update_conda_recipe.py` with
`--feedstock . --version X.Y.Z` to retrieve and verify the published PyPI source
archive and update the recipe and API checks.

Review dependency changes, request a rerender, and merge after the required
feedstock checks pass. Verify the published package in a fresh conda environment.
The `conda.recipe/` directory in this repository is for local checkout builds.

## Publish documentation

Activate the release tag on Read the Docs and verify its build. Use `stable`
for released documentation and `latest` for the default development branch.
Check navigation, figures, and notebook downloads after deployment.
See [documentation maintenance](docs/guides/publishing.md) for build and editing
instructions.
