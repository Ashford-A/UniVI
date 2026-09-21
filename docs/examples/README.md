# Downloadable UniVI tutorials

Keep all files together: notebooks import `_common.py` from this directory.
Install UniVI 0.5.0 (or this release's source) with its `tutorials` extra before running.

The notebooks are generated from matching percent-cell Python scripts. They are
unexecuted portable tutorials; synthetic script verification is reported in the
documentation. Biological datasets are not bundled or downloaded automatically.

Set `UNIVI_DATA` to the directory containing the task subfolders described in the
data guide and `UNIVI_OUTPUT` to an output directory. These can be absolute paths.
The small `00_quickstart.ipynb` needs no external data. Run `02_citeseq` before
`03_cite_reconstruction`; the latter loads the former's fitted reference.

From the repository root, regenerate these notebooks and web pages with:
`python scripts/build_tutorial_docs.py`.
