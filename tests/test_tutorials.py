"""Execute the tutorial and paper-reproduction notebooks end to end on small synthetic stand-in data.

This checks that every tutorial runs against the current API; it does not
check biological results. Datasets are served from local synthetic files via
``UNIVI_DATASET_REGISTRY`` and training is shortened by overriding the cell
tagged ``parameters`` in each notebook.

Run with:  pytest tests/test_tutorials.py     (skipped unless nbclient is installed)
"""
import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (sorted((ROOT / "docs" / "tutorials").glob("*.ipynb"))
             + sorted((ROOT / "docs" / "tutorials" / "experimental").glob("*.ipynb"))
             + sorted((ROOT / "docs" / "reproducibility" / "api").glob("*.ipynb")))

# Small values injected after each notebook's "parameters" cell.
OVERRIDES = """
N_EPOCHS = 3
REFINE_EPOCHS = 3
HEAD_WARMUP_EPOCHS = 3
ENCODER_PATIENCE = 3
BRIDGE_EPOCHS = 3
N_HVG = 300
N_LSI = 21
MAX_CELLS = 1500
N_GENERATED = 200
MAX_TEST_CELLS = 400
MIN_PER_CLASS = 5
HIGHRES_HEAD_EPOCHS = 3
N_PEAKS = 800
N_NULL = 5
RUN_MOTIF_SCAN = False
N_TOKENS_RNA = 32
N_TOKENS_ATAC = 32
MAX_ATTENTION_CELLS = 300
N_PERMUTATIONS = 20
N_CITE_CELLS = 900
N_CITE_TEST = 300
N_SAMPLES = 5
N_NULL_LSI = 3
EXTRA_HOLDOUTS = 1
N_UNTRAINED_CONTROLS = 1
ADV_LAMBDAS = [1.0]
N_TILING_NULL = 15
"""


@pytest.fixture(scope="module")
def synthetic_registry(tmp_path_factory):
    from synthetic_data import write_registry

    root = tmp_path_factory.mktemp("univi-synthetic")
    return write_registry(root), root


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_tutorial_executes(path, synthetic_registry, tmp_path, monkeypatch):
    registry, root = synthetic_registry
    monkeypatch.setenv("UNIVI_DATASET_REGISTRY", str(registry))
    monkeypatch.setenv("UNIVI_DATA_DIR", str(root / "cache"))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    nb = nbformat.read(path, as_version=4)
    idx = next((i for i, c in enumerate(nb.cells) if "parameters" in c.metadata.get("tags", [])), None)
    assert idx is not None, f"{path.name} needs a cell tagged 'parameters'"
    nb.cells.insert(idx + 1, nbformat.v4.new_code_cell(OVERRIDES))

    client = nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                     resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
