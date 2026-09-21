# Build and publish the documentation

The documentation is a Sphinx project using MyST Markdown and the Read the Docs theme. Its pages, figures, downloadable notebooks, API reference, and navigation are committed source assets. A documentation build does not download biological datasets, import Torch, or train a model.

## Build locally

```bash
python -m pip install -r docs/requirements.txt nbformat
python scripts/build_tutorial_docs.py
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000` in a browser to preview the documentation.

## Connect Read the Docs

After the reviewed source change is on GitHub, import `Ashford-A/UniVI` into your Read the Docs account and select the documentation branch or merged default branch. The project slug determines the public URL; no slug or live URL is assumed here.

The repository-root `.readthedocs.yaml` selects Ubuntu 24.04, Python 3.12, `docs/conf.py`, and `docs/requirements.txt`. It treats warnings as build failures and enables downloadable HTML archives. The configuration follows the [official configuration v2 reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html); account import, webhook, and version activation are described in the [official Read the Docs tutorial](https://docs.readthedocs.com/platform/stable/tutorial/index.html).

Activate the intended branch/version, trigger a build, and verify navigation, search, image loading, and notebook downloads. Add the resulting public URL to the README only after that build succeeds. The prepared GitHub workflow separately checks generated-source consistency, strict HTML compilation, and helper tests on pushes and pull requests.

## Edit the maintainable source

| Change | Edit |
| --- | --- |
| Tutorial analysis code and notebook cells | `docs/examples/*.py` using `# %%` cell markers |
| Tutorial introduction, panel map, interpretation, provenance | Matching `docs/_fragments/*.intro.md` and `*.after.md` |
| Shared tutorial plotting/training recipe | `docs/examples/_common.py` |
| General guide | Its Markdown file under `docs/guides/` |
| Public helper API | Source/docstrings under `univi/`; regenerate the API page |
| Theme or navigation | `docs/conf.py`, `docs/_static/custom.css`, `docs/index.md` |

Run `python scripts/build_tutorial_docs.py` after source edits. It regenerates the tutorial pages, downloadable notebooks, source-derived API page, and tutorial ZIP. Commit generated assets so Read the Docs can build without notebook execution. `python scripts/build_tutorial_docs.py --check` verifies text/notebook freshness without writing files.

The scripts and notebooks share one source to avoid code drifting between a web example and its download. Manuscript notebook snapshots remain untouched as historical references. New biological outputs belong in a separately identified run, with their environment and source commit recorded.
