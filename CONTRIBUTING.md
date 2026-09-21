# Contributing

Thanks for helping improve UniVI. Bug reports, questions, documentation fixes, and pull requests are all welcome.

## Reporting a problem

Open an [issue](https://github.com/Ashford-A/UniVI/issues) with:

- the UniVI version (`python -c "import univi; print(univi.__version__)"`), and your PyTorch, Python, and OS versions
- a minimal code example, and the full error message
- the shapes and a `print()` of the AnnData objects involved, if data are part of the problem

## Development setup

```bash
git clone https://github.com/Ashford-A/UniVI.git
cd UniVI
python -m pip install -e ".[tutorials,test,docs]"
python -m pytest -q                      # unit tests and tutorial notebooks (synthetic data)
```

`tests/test_tutorials.py` executes every notebook in `docs/tutorials/` against small synthetic stand-ins for the real datasets, with shortened training. It checks that the tutorials run with the current API, not their biological results.

## Documentation

The site is built with Sphinx, `sphinx-book-theme`, and MyST-NB:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
python -m http.server --directory docs/_build/html 8000
```

- Pages are Markdown files under `docs/`; tutorials are the notebooks in `docs/tutorials/`.
- Read the Docs renders notebooks with the outputs saved in them and never executes them. To refresh the outputs after changing a tutorial, run it on real data (a GPU helps) with `python scripts/execute_tutorials.py`, check the results, and commit the notebook.
- Every tutorial has one cell tagged `parameters` holding its training settings; the tests override these to run quickly.

## Pull requests

- Keep changes focused, and add or update tests for new behavior.
- Run `python -m pytest -q` before pushing.
- Do not modify the archived notebooks in `notebooks/GR_manuscript_reproducibility/`; they are the record of the published analyses.

Maintainer release steps are in [RELEASE.md](https://github.com/Ashford-A/UniVI/blob/main/RELEASE.md).
