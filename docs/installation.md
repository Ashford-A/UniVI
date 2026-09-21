# Installation

UniVI supports Python 3.10 and newer on Linux, macOS, and Windows.

## With pip

```bash
python -m pip install univi
```

To run the tutorial notebooks locally, include the `tutorials` extra (Jupyter and `scikit-misc`, which Scanpy's Seurat-v3 highly-variable-gene selection needs):

```bash
python -m pip install "univi[tutorials]"
```

### PyTorch and GPUs

UniVI depends on PyTorch. `pip install univi` pulls the default PyTorch wheel for your platform, which on Linux includes CUDA support. For a specific CUDA version, or a CPU-only build, install PyTorch first with the selector at [pytorch.org](https://pytorch.org/get-started/locally/), then install UniVI.

Pass `device="cuda"`, `"mps"` (Apple silicon), or `"cpu"` to `TrainingConfig` and the encoding functions. The tutorials pick automatically:

```python
import torch
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

Training runs on a CPU, just more slowly. If an operation is not supported on MPS, fall back to CPU.

## With conda

```bash
conda install -c conda-forge univi
```

conda-forge releases follow PyPI releases, usually within a few days.

## Google Colab

Every tutorial has an **Open in Colab** badge and a first cell that installs UniVI when it detects Colab. Choose *Runtime → Change runtime type → GPU* for faster training.

## From source

```bash
git clone https://github.com/Ashford-A/UniVI.git
cd UniVI
python -m pip install -e ".[tutorials,test]"
python -m pytest -q
```

## Check the installation

```python
import univi
print(univi.__version__)

import univi.datasets as uds
print(uds.list_datasets())
```

Datasets are cached in `~/.cache/univi` by default; set the `UNIVI_DATA_DIR` environment variable to use another location (for example a scratch disk on a cluster).

## Which version to use

| Goal | Install |
| --- | --- |
| New analyses, tutorials | latest release: `pip install univi` |
| Rerun the archived Genome Research notebooks exactly | `pip install univi==0.4.7` (the release the article was prepared against) |

The tutorials on this site use APIs added after 0.4.7 (fitted preprocessors, reference bundles, refinement, dataset downloads). The [reproduction section](reproducibility/index.md) explains how the archived notebooks relate to them.
