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


TEASEQ_WELLS = ["GSM5123951_X066-MP0C1W3_leukopak_perm-cells_tea", "GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea",
                "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea", "GSM5123954_X066-MP0C1W6_leukopak_perm-cells_tea"]
AML_GENES = ["NPM1", "DNMT3A", "FLT3", "TP53", "TET2", "IDH2"]
SKIN_TYPES = ["Basal", "Spinous", "Granular", "TAC-1", "TAC-2", "IRS", "Medulla", "Dermal Fibroblast"]
LINEAGES = ["Epiblast", "Primitive_Streak", "Nascent_mesoderm", "Mature_mesoderm", "Visceral_endoderm"]


def _obs(labels, prefix, **cols):
    df = pd.DataFrame({"cell_type": pd.Categorical(labels), **cols},
                      index=[f"{prefix}{i:06d}" for i in range(len(labels))])
    return df


def make_bridge(n_ref=900, n_ding=500, n_sat=500, n_genes=500, n_peaks=1200, seed=2):
    rng = np.random.default_rng(seed)
    genes = MARKER_GENES + [f"GENE{i}" for i in range(n_genes - len(MARKER_GENES))]
    peaks = [f"chr{1 + i % 22}-{10_000 + 700 * i}-{10_500 + 700 * i}" for i in range(n_peaks)]
    out = {}
    lab = _cells(n_ref, rng)
    obs = _obs(lab, "MULTI_", celltype_harmonized_coarse=pd.Categorical([L1[t] for t in lab]))
    out["rna"] = ad.AnnData(_counts(lab, n_genes, rng, rate=2.0), obs=obs.copy(), var=pd.DataFrame(index=genes))
    out["atac"] = ad.AnnData(_counts(lab, n_peaks, rng, rate=0.6, sparsity=0.5), obs=obs.copy(),
                             var=pd.DataFrame(index=peaks))
    lab = _cells(n_ding, rng)
    keep_g = genes[: n_genes - 20]                              # query lacks a few reference genes
    out["ding_rna"] = ad.AnnData(_counts(lab, len(keep_g), rng, rate=2.0),
                                 obs=_obs(lab, "DING_", celltype_harmonized_coarse=pd.Categorical([L1[t] for t in lab])),
                                 var=pd.DataFrame(index=keep_g))
    lab = _cells(n_sat, rng)
    keep_p = peaks[30:] + [f"chrX-{i}-{i + 500}" for i in range(40)]  # partial peak overlap
    out["satpathy_atac"] = ad.AnnData(_counts(lab, len(keep_p), rng, rate=0.6, sparsity=0.5),
                                      obs=_obs(lab, "SAT_", celltype_harmonized_coarse=pd.Categorical([L1[t] for t in lab])),
                                      var=pd.DataFrame(index=keep_p))
    return out


def make_teaseq(n_cells=1400, seed=3):
    rng = np.random.default_rng(seed)
    lab = _cells(n_cells, rng)
    wells = rng.choice(TEASEQ_WELLS, size=n_cells)
    obs = pd.DataFrame({"sample_id": pd.Categorical(wells)},
                       index=[f"AAAC{i:06d}__{w}" for i, w in enumerate(wells)])
    genes = MARKER_GENES + [f"GENE{i}" for i in range(500 - len(MARKER_GENES))]
    adts = ["CD3", "CD4", "CD8a", "CD19", "CD14", "CD16", "CD45RA", "CD45RO", "CD27", "CD127", "IgD", "IgM",
            "CD38", "CD11c", "HLA-DR"] + [f"ADT{i}" for i in range(15)]
    tiles = [f"chr{1 + i % 22}:{500 * i}-{500 * (i + 1)}" for i in range(1500)]
    return {"rna": ad.AnnData(_counts(lab, len(genes), rng, rate=2.0), obs=obs.copy(), var=pd.DataFrame(index=genes)),
            "adt": ad.AnnData(_counts(lab, len(adts), rng, rate=20.0), obs=obs.copy(), var=pd.DataFrame(index=adts)),
            "atac": ad.AnnData(_counts(lab, len(tiles), rng, rate=0.5, sparsity=0.6), obs=obs.copy(),
                               var=pd.DataFrame(index=tiles))}


def make_aml(n_cite=900, n_vg=600, n_dab=600, seed=4):
    rng = np.random.default_rng(seed)
    genes = MARKER_GENES + ["CD34", "MPO", "KIT"] + [f"GENE{i}" for i in range(400)]
    adts = ["CD3", "CD4", "CD8", "CD19", "CD14", "CD16", "CD33", "CD34", "CD38", "CD45RA", "CD56", "CD11b",
            "CD117", "HLA-DR", "CD13", "CD64", "CD71", "CD123"]
    out = {}
    lab = _cells(n_cite, rng)
    samples = rng.choice([f"aml{i}" for i in range(1, 9)] + ["control_1", "control_2", "aml9"], size=n_cite)
    obs = pd.DataFrame({"sample_id": pd.Categorical(samples)}, index=[f"CITE_{i:06d}" for i in range(n_cite)])
    out["cite_rna"] = ad.AnnData(_counts(lab, len(genes), rng, rate=2.0), obs=obs.copy(), var=pd.DataFrame(index=genes))
    out["cite_adt"] = ad.AnnData(_counts(lab, len(adts), rng, rate=20.0), obs=obs.copy(), var=pd.DataFrame(index=adts))
    lab = _cells(n_vg, rng)
    mut = rng.choice(["", "NPM1 W288fs", "DNMT3A R882H", "NPM1 W288fs,FLT3-ITD", "TP53 R273L"], size=n_vg)
    wt = rng.choice(["", "NPM1", "DNMT3A", "FLT3", "TET2"], size=n_vg)
    out["vangalen_rna"] = ad.AnnData(_counts(lab, len(genes), rng, rate=2.0),
                                     obs=pd.DataFrame({"patient": rng.choice(["AML419A", "AML556", "BM1"], size=n_vg),
                                                       "CellType": lab, "MutTranscripts": mut, "WtTranscripts": wt},
                                                      index=[f"VG_{i:06d}" for i in range(n_vg)]),
                                     var=pd.DataFrame(index=genes))
    lab = _cells(n_dab, rng)
    dab_obs = pd.DataFrame({"experiment": rng.choice(["fig3_ab_geno", "fig4_ab_geno"], size=n_dab)},
                           index=[f"DAB_{i:06d}" for i in range(n_dab)])
    for g in ["NPM1", "DNMT3A", "FLT3"]:
        dab_obs[f"mut_{g}"] = rng.choice([0.0, 1.0, np.nan], size=n_dab)
    out["dabseq_adt"] = ad.AnnData(_counts(lab, len(adts), rng, rate=20.0), obs=dab_obs, var=pd.DataFrame(index=adts))
    return out


def make_shareseq(n_cells=1200, seed=5):
    rng = np.random.default_rng(seed)
    lab = _cells(n_cells, rng, types=SKIN_TYPES)
    split = rng.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1], size=n_cells)
    obs = pd.DataFrame({"cell_type": pd.Categorical(lab), "split": pd.Categorical(split)},
                       index=[f"R1.01.R2.{i:05d}" for i in range(n_cells)])
    genes = ["Krt14", "Krt10", "Lor", "Lef1", "Msx2", "mt-Co1", "Rps3", "Rpl7", "Gm1234Rik", "Tmsb4x-ps1"] + \
            [f"Gene{i}" for i in range(490)]
    peaks = [f"chr{1 + i % 19}:{10_000 + 700 * i}-{10_500 + 700 * i}" for i in range(1500)]
    return {"rna": ad.AnnData(_counts(lab, len(genes), rng, rate=2.0, types=SKIN_TYPES), obs=obs.copy(),
                              var=pd.DataFrame(index=genes)),
            "atac": ad.AnnData(_counts(lab, len(peaks), rng, rate=0.4, sparsity=0.6, types=SKIN_TYPES),
                               obs=obs.copy(), var=pd.DataFrame(index=peaks))}


def make_scnmt(n_cells=300, seed=6):
    rng = np.random.default_rng(seed)
    lab = _cells(n_cells, rng, types=LINEAGES)
    obs = pd.DataFrame({"lineage10x": pd.Categorical(lab), "stage": rng.choice(["E4.5", "E5.5", "E6.5", "E7.5"], size=n_cells),
                        "embryo": rng.choice(["E6.5_embryo1", "E7.5_embryo2"], size=n_cells)},
                       index=[f"E{i:04d}" for i in range(n_cells)])
    out = {"rna": ad.AnnData(_counts(lab, 400, rng, rate=2.0, types=LINEAGES).toarray(), obs=obs.copy(),
                             var=pd.DataFrame(index=[f"Gene{i}" for i in range(400)]))}
    out["rna"].layers["counts"] = out["rna"].X.copy()
    for mod, prefix in [("cpg", "meth"), ("gpc", "acc")]:
        cov = rng.poisson(6, (n_cells, 200)).astype(np.float32)
        p = 1 / (1 + np.exp(-rng.normal(0, 1.5, (len(LINEAGES), 200))[[LINEAGES.index(t) for t in lab]]))
        suc = rng.binomial(cov.astype(int), p).astype(np.float32)
        a = ad.AnnData(np.divide(suc, cov, out=np.full_like(suc, 0.5), where=cov > 0), obs=obs.copy(),
                       var=pd.DataFrame(index=[f"{prefix}_Gene{i}" for i in range(200)]))
        a.layers[f"{prefix}_successes"], a.layers[f"{prefix}_total_count"] = suc, cov
        out[mod] = a
    return out


PAPER_BUILDERS = {
    "pbmc_multiome_bridge": (make_bridge, ["rna", "atac"]),
    "teaseq_pbmc": (make_teaseq, ["rna", "adt", "atac"]),
    "aml_mosaic": (make_aml, ["cite_rna", "cite_adt"]),
    "shareseq_mouse_skin": (make_shareseq, ["rna", "atac"]),
    "scnmt_gastrulation": (make_scnmt, ["rna", "cpg", "gpc"]),
}


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
        **PAPER_BUILDERS,
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
