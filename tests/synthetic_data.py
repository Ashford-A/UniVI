"""Small synthetic stand-ins for the tutorial datasets.

They reproduce the *schema* of the real files (modality keys, obs label
columns, raw integer counts in ``.X``, marker gene/protein names) so the
tutorial notebooks can be executed end to end in CI without network access.
They carry cell-type structure so models have something to learn, but they are
not biologically meaningful and must never be used to report results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CELL_TYPES = ["CD14 Mono", "CD16 Mono", "CD4 Naive", "CD4 TCM", "CD8 Naive",
              "CD8 TEM_1", "NK", "Naive B", "Memory B", "cDC", "pDC", "Treg"]
L1 = {"CD14 Mono": "Mono", "CD16 Mono": "Mono", "CD4 Naive": "CD4 T", "CD4 TCM": "CD4 T",
      "CD8 Naive": "CD8 T", "CD8 TEM_1": "CD8 T", "NK": "NK", "Naive B": "B",
      "Memory B": "B", "cDC": "DC", "pDC": "DC", "Treg": "CD4 T"}
MARKER_GENES = ["MS4A1", "CD79A", "CD3D", "CD3E", "TRAC", "IL7R", "CD8A", "NKG7", "GNLY",
                "LYZ", "CD14", "FCGR3A", "LST1", "S100A8", "CLEC10A", "FCER1A", "LILRA4",
                "FOXP3", "CCR7", "GZMB", "PRF1", "BANK1", "PAX5", "CD74", "HLA-DRA"]
MARKER_ADTS = ["CD3-1", "CD4", "CD8a", "CD19", "CD20", "CD14", "CD16", "CD56-1",
               "CD45RA", "CD45RO", "CD11c", "HLA-DR", "CD127", "CD25"]


def _cells(n, rng, types=CELL_TYPES):
    probs = rng.dirichlet(np.ones(len(types)) * 2)
    return rng.choice(types, size=n, p=probs)


def _counts(labels, n_features, rng, *, rate=1.0, sparsity=0.0, types=CELL_TYPES):
    k = len(types)
    programs = rng.normal(0, 1.2, size=(k, n_features))
    base = rng.normal(-1.0, 1.0, size=n_features)
    idx = np.array([types.index(t) for t in labels])
    logmu = base + programs[idx] + rng.normal(0, 0.3, size=(len(labels), 1))
    x = rng.poisson(rate * np.exp(logmu)).astype(np.float32)
    if sparsity:
        x *= rng.random(x.shape) > sparsity
    return sp.csr_matrix(x)


def make_multiome(n_cells=1200, n_genes=600, n_peaks=1500, seed=0):
    rng = np.random.default_rng(seed)
    labels = _cells(n_cells, rng)
    obs = pd.DataFrame({"cell_type": pd.Categorical(labels)},
                       index=[f"AAAC{i:06d}-1" for i in range(n_cells)])
    genes = MARKER_GENES + [f"GENE{i}" for i in range(n_genes - len(MARKER_GENES))]
    peaks = [f"chr{1 + i % 22}-{10_000 + 700 * i}-{10_500 + 700 * i}" for i in range(n_peaks)]
    rna = ad.AnnData(_counts(labels, n_genes, rng, rate=2.0), obs=obs.copy(),
                     var=pd.DataFrame(index=genes))
    atac = ad.AnnData(_counts(labels, n_peaks, rng, rate=0.6, sparsity=0.5), obs=obs.copy(),
                      var=pd.DataFrame(index=peaks))
    return {"rna": rna, "atac": atac}


def make_citeseq(n_cells=1500, n_genes=600, seed=1):
    rng = np.random.default_rng(seed)
    labels = _cells(n_cells, rng)
    obs = pd.DataFrame({
        "celltype.l1": pd.Categorical([L1[t] for t in labels]),
        "celltype.l2": pd.Categorical(labels),
        "celltype.l3": pd.Categorical(labels),
        "donor": pd.Categorical([f"P{1 + i % 8}" for i in range(n_cells)]),
    }, index=[f"L1_AAAC{i:06d}" for i in range(n_cells)])
    genes = MARKER_GENES + [f"GENE{i}" for i in range(n_genes - len(MARKER_GENES))]
    adts = MARKER_ADTS + [f"ADT{i}" for i in range(40 - len(MARKER_ADTS))]
    rna = ad.AnnData(_counts(labels, n_genes, rng, rate=2.0), obs=obs.copy(),
                     var=pd.DataFrame(index=genes))
    adt = ad.AnnData(_counts(labels, len(adts), rng, rate=20.0), obs=obs.copy(),
                     var=pd.DataFrame(index=adts))
    return {"rna": rna, "adt": adt}


def write_registry(root: Path) -> Path:
    """Write synthetic files plus a registry JSON; return the registry path.

    Point ``UNIVI_DATASET_REGISTRY`` at the returned file so that
    ``univi.datasets.load(...)`` serves these files instead of downloading.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name, (builder, paired) in {
        "pbmc_multiome_10k": (make_multiome, ["rna", "atac"]),
        "hao_citeseq_pbmc": (make_citeseq, ["rna", "adt"]),
    }.items():
        files = {}
        for key, adata in builder().items():
            path = root / f"{name}_{key}.h5ad"
            adata.write_h5ad(path)
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            files[key] = {"filename": path.name, "url": path.as_uri(), "hash": f"md5:{digest}"}
        datasets[name] = {"title": f"synthetic {name}", "paired": paired, "files": files}
    reg = root / "registry.json"
    reg.write_text(json.dumps({"datasets": datasets}, indent=2))
    return reg


if __name__ == "__main__":  # pragma: no cover
    import sys
    print(write_registry(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_univi_data")))
