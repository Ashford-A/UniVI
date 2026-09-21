# Installation and versions

Use Python 3.10 or newer for UniVI. The documentation build is pinned to Python 3.12. Keep the analysis environment separate from the documentation environment: building the website does not need a GPU, datasets, or PyTorch.

## Install UniVI 0.5.0

Create a fresh environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows Command Prompt
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install PyTorch for your platform using its [official installation selector](https://pytorch.org/get-started/locally/), then install UniVI:

```bash
python -m pip install --upgrade pip
python -m pip install "univi[tutorials]==0.5.0"
```

To develop UniVI from source:

```bash
git clone https://github.com/Ashford-A/UniVI.git
cd UniVI
python -m pip install -e ".[tutorials]"
```

The Python package contains the APIs. Download the tutorial notebooks and shared helpers from the tutorial pages, or use the repository files under `docs/examples/`.

After the conda-forge feedstock update is merged and built:

```bash
conda create -n univi-tutorials -c conda-forge python=3.12 univi=0.5.0 scikit-misc jupyterlab
conda activate univi-tutorials
```

`scikit-misc` is included because Seurat v3 HVG selection needs it. The new preprocessor fails explicitly if that calculation fails; it does not silently replace Seurat v3 with a different method. `joblib` saves the learned preprocessing transforms. JupyterLab and notebook-format support make the downloads convenient to run locally.

The historical package remains available with `pip install univi==0.4.7`; it does not install these new APIs. Verify the installed release:

```python
import univi
from univi import UniVIMultiModalVAE, UniVIRefiner, RefinementConfig
print(univi.__version__, univi.__file__)
assert hasattr(UniVIMultiModalVAE, "freeze_decoders")
assert hasattr(UniVIMultiModalVAE, "add_classification_head")
```

## Run a biological tutorial

From the repository root:

```bash
python docs/examples/02_citeseq.py
python docs/examples/03_cite_reconstruction.py
```

Defaults are `data/tutorials/` for inputs and `results/tutorials/` for outputs. In Windows Command Prompt:

```bash
set UNIVI_DATA=D:\UniVI_data\tutorials
set UNIVI_OUTPUT=D:\UniVI_results\tutorials
python docs\examples\05_bridge.py
```

On macOS/Linux:

```bash
export UNIVI_DATA=/path/to/tutorial-data
export UNIVI_OUTPUT=/path/to/tutorial-results
python docs/examples/05_bridge.py
```

The common helper chooses CUDA, then Apple MPS, then CPU. Some likelihood operations can have backend-specific limitations; use CPU or supported CUDA if MPS fails on an operation. GPU memory requirements depend strongly on feature dimensions: raw peak decoding can be much more expensive than 100-dimensional LSI decoding. Start with the quickstart, then an intentionally small development subset, before committing to a full dataset.

A short development run can set `UNIVI_EPOCHS=10`. This changes the analysis and cannot reproduce a 3,000–5,000-epoch-cap manuscript recipe. `UNIVI_SMOKE=1` is reserved for the included software tests: it changes network dimensions and must never be used to report biological performance.

## Open the notebooks

```bash
jupyter lab docs/examples
```

The notebooks and `_common.py` should remain together. Launching inside `docs/examples/` changes relative default paths; set `UNIVI_DATA` and `UNIVI_OUTPUT` to absolute paths before launching Jupyter, or edit them at the top of `_common.py`. Each notebook is generated from its readable Python script to keep the two versions synchronized.

## Build only the website

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` locally. Read the Docs uses the checked-in `.readthedocs.yaml`; see [publishing](guides/publishing.md).
