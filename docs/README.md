# UniVI documentation maintenance

The source of each executable tutorial is `examples/<name>.py`. Edit its narrative
in `_fragments/<name>.intro.md` and `_fragments/<name>.after.md`, then regenerate:

```bash
python -m pip install -r docs/requirements.txt nbformat
python scripts/build_tutorial_docs.py
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

The generated `.ipynb`, tutorial Markdown, API reference, and downloadable tutorial
ZIP are committed. Read the Docs builds do not execute notebooks or import UniVI.
Preserve historical notebooks under `notebooks/GR_manuscript_reproducibility/`.

Code changes require `python -m pytest -q tests`. For a manual integration check,
`python tests/smoke_examples.py` executes all 11 scripts with synthetic inputs.
This is not verification of the publication's biological results.

See `guides/publishing.md` for Read the Docs import and version activation.
