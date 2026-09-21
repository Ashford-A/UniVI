"""Readers for the parsed scNMT-seq mouse gastrulation bundle (Argelaguet et al. 2019).

The bundle (``scnmt_gastrulation.tar.gz``) is read in place, without extraction,
and converted into paired RNA, CpG-methylation and GpC-accessibility AnnData
objects. Methylation and accessibility keep per-feature success and coverage
counts in layers so they can be modelled with a beta-binomial likelihood.

The Genome Research analysis (Supplemental Fig. S4) used
``require_qc=False`` and ``filter_features=False``. The optional feature filters
are computed over all supplied cells; apply them after splitting if you need
training-only feature selection.
"""
from __future__ import annotations
import gzip
import tarfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

def _open_tar_member_stream(tar: tarfile.TarFile, member_name: str):
    """
    Return a readable file-like stream for a tar member.
    Handles .gz members transparently.
    """
    fobj = tar.extractfile(member_name)
    if fobj is None:
        raise FileNotFoundError(f"Member not found in tar: {member_name}")
    if member_name.endswith(".gz"):
        return gzip.GzipFile(fileobj=fobj)
    return fobj

def list_tar_members(tar_path: Path, n_preview: int = 80) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]

    names = [m.name for m in members]
    manifest = pd.DataFrame({
        "member_name": names,
        "size_bytes": [m.size for m in members],
    })

    print(f"Manifest files: {len(manifest)}")
    print("Preview:")
    for n in manifest["member_name"].head(n_preview):
        print(" ", n)

    return manifest

def read_rna_counts_from_tar(
    tar_path: Path,
    member_name: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Reads rna/counts.txt.gz from the scnmt_gastrulation parsed tar.
    Returns DataFrame with shape (cells x genes), index=id_rna strings, columns=ens_id.
    """
    if verbose:
        print(f"\nReading RNA matrix: {member_name}")

    with tarfile.open(tar_path, "r:*") as tar:
        with _open_tar_member_stream(tar, member_name) as fh:
            df = pd.read_csv(fh, sep="\t")

    if verbose:
        print("Raw shape:", df.shape)
        print(df.head())

    # First column is gene ID
    gene_col = df.columns[0]
    if str(gene_col).lower() not in {"ens_id", "gene", "gene_id", "feature"}:
        warnings.warn(f"Unexpected first RNA column: {gene_col}")

    df = df.set_index(gene_col)

    # rows=genes, cols=cells -> transpose to cells x genes
    df = df.T
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if verbose:
        print(f"Assumed rows=features, cols=cells -> transposed to cells x features: {df.shape}")

    return df

def read_feature_level_long_from_tar(
    tar_path: Path,
    member_name: str,
    modality: str,                     # "cpg" or "gpc"
    feature_subset: Optional[List[str]] = None,
    cell_subset: Optional[List[str]] = None,
    min_coverage: int = 1,
    chunksize: int = 500_000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Read long-format feature-level table from scnmt_gastrulation parsed tar.

    Expected columns (headerless):
      0: cell_id
      1: feature_id
      2: feature_class (e.g. genebody)
      3: successes
      4: coverage
      5: percent (0-100)

    Returns dict with:
      - frac_df: cells x features (fractions)
      - successes_df: cells x features
      - coverage_df: cells x features
      - meta: summary
    """
    if verbose:
        print(f"\nReading feature-level long table: {member_name} ({modality})")

    feature_subset_set = set(map(str, feature_subset)) if feature_subset is not None else None
    cell_subset_set = set(map(str, cell_subset)) if cell_subset is not None else None

    chunk_results = []
    total_rows = 0
    kept_rows = 0

    with tarfile.open(tar_path, "r:*") as tar:
        with _open_tar_member_stream(tar, member_name) as fh:
            reader = pd.read_csv(
                fh,
                sep="\t",
                header=None,
                names=["cell_id", "feature_id", "feature_class", "successes", "coverage", "percent"],
                chunksize=chunksize,
                dtype={
                    "cell_id": "string",
                    "feature_id": "string",
                    "feature_class": "string",
                    "successes": "float32",
                    "coverage": "float32",
                    "percent": "float32",
                },
            )

            for i, chunk in enumerate(reader, start=1):
                total_rows += len(chunk)

                # basic cleaning
                chunk = chunk.dropna(subset=["cell_id", "feature_id", "successes", "coverage"])
                chunk["cell_id"] = chunk["cell_id"].astype(str)
                chunk["feature_id"] = chunk["feature_id"].astype(str)

                # filter by coverage
                chunk = chunk[chunk["coverage"] >= min_coverage]

                # optional subsets
                if cell_subset_set is not None:
                    chunk = chunk[chunk["cell_id"].isin(cell_subset_set)]
                if feature_subset_set is not None:
                    chunk = chunk[chunk["feature_id"].isin(feature_subset_set)]

                if len(chunk) == 0:
                    continue

                # fraction = successes / coverage
                cov = chunk["coverage"].to_numpy(dtype=np.float32)
                suc = chunk["successes"].to_numpy(dtype=np.float32)
                frac = np.divide(
                    suc,
                    cov,
                    out=np.zeros(len(chunk), dtype=np.float32),
                    where=(cov > 0),
                )

                chunk = chunk.copy()
                chunk["fraction"] = frac

                chunk_results.append(chunk[["cell_id", "feature_id", "successes", "coverage", "fraction"]])
                kept_rows += len(chunk)

                if verbose and (i <= 3 or i % 25 == 0):
                    print(f"  chunk {i}: input_rows={len(chunk):,}, retained_total={kept_rows:,}")

    if len(chunk_results) == 0:
        raise ValueError(f"No rows retained from {member_name}. Check subsets/min_coverage.")

    long_df = pd.concat(chunk_results, axis=0, ignore_index=True)

    if verbose:
        print(f"Total rows read: {total_rows:,}")
        print(f"Rows retained:   {kept_rows:,}")
        print("Retained long_df shape:", long_df.shape)
        print(long_df.head())

    # Aggregate duplicates if present
    dup_mask = long_df.duplicated(subset=["cell_id", "feature_id"], keep=False)
    if dup_mask.any():
        if verbose:
            print(f"Found duplicated (cell_id, feature_id) rows -> aggregating")
        long_df = (
            long_df.groupby(["cell_id", "feature_id"], as_index=False)[["successes", "coverage"]]
            .sum()
        )
        cov = long_df["coverage"].to_numpy(dtype=np.float32)
        suc = long_df["successes"].to_numpy(dtype=np.float32)
        long_df["fraction"] = np.divide(suc, cov, out=np.zeros(len(long_df), dtype=np.float32), where=(cov > 0))

    # Pivot wide: cells x features
    successes_df = long_df.pivot(index="cell_id", columns="feature_id", values="successes")
    coverage_df  = long_df.pivot(index="cell_id", columns="feature_id", values="coverage")
    frac_df      = long_df.pivot(index="cell_id", columns="feature_id", values="fraction")

    # Align and fill
    cells = successes_df.index.astype(str)
    feats = successes_df.columns.astype(str)

    successes_df = successes_df.reindex(index=cells, columns=feats).fillna(0).astype(np.float32)
    coverage_df  = coverage_df.reindex(index=cells, columns=feats).fillna(0).astype(np.float32)
    frac_df      = frac_df.reindex(index=cells, columns=feats).fillna(0).astype(np.float32)

    if verbose:
        print(f"Wide matrices built (cells x features): {frac_df.shape}")
        nnz_cov = int((coverage_df.to_numpy() > 0).sum())
        print(f"Coverage nonzero entries: {nnz_cov:,}")

    return {
        "frac_df": frac_df,
        "successes_df": successes_df,
        "coverage_df": coverage_df,
        "meta": {
            "member_name": member_name,
            "modality": modality,
            "n_cells": int(frac_df.shape[0]),
            "n_features": int(frac_df.shape[1]),
            "total_rows_read": int(total_rows),
            "rows_retained": int(kept_rows),
            "min_coverage": int(min_coverage),
        },
    }

def read_metadata_from_tar(
    tar_path: Path,
    member_name: str = "sample_metadata.txt",
    verbose: bool = True,
) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:*") as tar:
        with _open_tar_member_stream(tar, member_name) as fh:
            meta = pd.read_csv(fh, sep="\t")

    # normalize common string columns
    for c in ["sample", "id_rna", "id_met", "id_acc", "embryo", "plate", "stage", "lineage10x", "lineage10x_2"]:
        if c in meta.columns:
            meta[c] = meta[c].astype("string")

    if verbose:
        print("Metadata shape:", meta.shape)
        print("Metadata columns:", list(meta.columns))
        print(meta.head())

    return meta

def _df_to_anndata_dense_or_sparse(
    df: pd.DataFrame,
    modality: str,
    feature_type: str,
    copy_to_counts: bool = False,
) -> ad.AnnData:
    x = df.copy()
    x.index = x.index.astype(str)
    x.columns = x.columns.astype(str)
    x = x.fillna(0.0)

    arr = x.to_numpy(dtype=np.float32)
    zero_frac = float((arr == 0).mean()) if arr.size else 0.0
    X = sp.csr_matrix(arr) if zero_frac > 0.5 else arr

    a = ad.AnnData(X=X)
    a.obs_names = pd.Index(x.index.astype(str), name="cell_id")
    a.var_names = pd.Index(x.columns.astype(str), name="feature")

    a.obs["modality"] = modality
    a.var["feature_type"] = feature_type

    if copy_to_counts:
        a.layers["counts"] = a.X.copy()

    return a

def build_methyl_like_anndata_from_long_feature(
    parsed: Dict[str, Any],
    modality: str,               # "cpg" or "gpc"
    successes_layer: str,
    coverage_layer: str,
    feature_type: str = "genebody",
) -> ad.AnnData:
    frac_df = parsed["frac_df"]
    suc_df = parsed["successes_df"]
    cov_df = parsed["coverage_df"]

    # ensure same order
    cells = frac_df.index.astype(str)
    feats = frac_df.columns.astype(str)
    suc_df = suc_df.reindex(index=cells, columns=feats).fillna(0).astype(np.float32)
    cov_df = cov_df.reindex(index=cells, columns=feats).fillna(0).astype(np.float32)

    a = _df_to_anndata_dense_or_sparse(
        frac_df,
        modality=modality,
        feature_type=feature_type,
        copy_to_counts=False,
    )

    suc_arr = suc_df.to_numpy(dtype=np.float32)
    cov_arr = cov_df.to_numpy(dtype=np.float32)

    a.layers[successes_layer] = sp.csr_matrix(suc_arr) if (suc_arr == 0).mean() > 0.5 else suc_arr
    a.layers[coverage_layer]  = sp.csr_matrix(cov_arr) if (cov_arr == 0).mean() > 0.5 else cov_arr

    # QC summaries from coverage
    cov_mat = a.layers[coverage_layer]
    if sp.issparse(cov_mat):
        cov_sum_cell = np.asarray(cov_mat.sum(axis=1)).ravel()
        cov_nnz_cell = np.asarray((cov_mat > 0).sum(axis=1)).ravel()
        cov_sum_feat = np.asarray(cov_mat.sum(axis=0)).ravel()
    else:
        cov_sum_cell = cov_mat.sum(axis=1)
        cov_nnz_cell = (cov_mat > 0).sum(axis=1)
        cov_sum_feat = cov_mat.sum(axis=0)

    a.obs["coverage_total"] = cov_sum_cell.astype(np.float32)
    a.obs["n_features_covered"] = cov_nnz_cell.astype(np.int32)
    a.var["coverage_total"] = cov_sum_feat.astype(np.float32)

    return a

def attach_metadata_and_align_modalities(
    rna: Optional[ad.AnnData],
    cpg: Optional[ad.AnnData],
    gpc: Optional[ad.AnnData],
    meta: pd.DataFrame,
    require_qc: bool = True,
    verbose: bool = True,
) -> Dict[str, ad.AnnData]:
    """
    - RNA obs_names are id_rna strings -> map to metadata['sample'] via metadata['id_rna']
    - cpg/gpc obs_names are sample IDs already (for parsed genebody files)
    - attach metadata to each modality
    - intersect common cells across available modalities
    """
    meta = meta.copy()

    if "sample" not in meta.columns:
        raise ValueError("metadata missing 'sample' column")

    # joint QC filter if requested
    if require_qc:
        mask = pd.Series(True, index=meta.index)
        for qc_col in ["pass_rnaQC", "pass_metQC", "pass_accQC"]:
            if qc_col in meta.columns:
                mask &= meta[qc_col].fillna(False).astype(bool)
        meta_qc = meta.loc[mask].copy()
        if verbose:
            print(f"QC filter retained {meta_qc.shape[0]} / {meta.shape[0]} metadata rows")
    else:
        meta_qc = meta

    meta_by_sample = meta_qc.drop_duplicates(subset=["sample"]).set_index("sample")
    meta_by_sample.index = meta_by_sample.index.astype(str)

    # RNA: map id_rna -> sample
    if rna is not None:
        if "id_rna" not in meta_qc.columns:
            raise ValueError("metadata missing 'id_rna' for RNA mapping")

        rna_map = (
            meta_qc[["sample", "id_rna"]]
            .dropna(subset=["sample", "id_rna"])
            .drop_duplicates(subset=["id_rna"])
            .set_index("id_rna")["sample"]
            .astype(str)
        )
        rna_map.index = rna_map.index.astype(str)

        keep_rna = [ix for ix in rna.obs_names.astype(str) if ix in rna_map.index]
        rna = rna[keep_rna].copy()

        rna.obs["sample"] = rna_map.loc[rna.obs_names.astype(str)].to_numpy().astype(str)
        rna.obs_names = pd.Index(rna.obs["sample"].astype(str), name="cell_id")

        common = rna.obs_names.intersection(meta_by_sample.index)
        rna = rna[common].copy()
        rna.obs = rna.obs.join(meta_by_sample, how="left")

    def _attach_sample_metadata(a: Optional[ad.AnnData]) -> Optional[ad.AnnData]:
        if a is None:
            return None
        a = a.copy()
        a.obs_names = pd.Index(a.obs_names.astype(str), name="cell_id")
        a.obs["sample"] = a.obs_names.astype(str)
        common = a.obs_names.intersection(meta_by_sample.index)
        a = a[common].copy()
        a.obs = a.obs.join(meta_by_sample, how="left")
        return a

    cpg = _attach_sample_metadata(cpg)
    gpc = _attach_sample_metadata(gpc)

    available = {k: v for k, v in {"rna": rna, "cpg": cpg, "gpc": gpc}.items() if v is not None}
    if len(available) == 0:
        raise ValueError("No modalities available after parsing and metadata attachment.")

    common_cells = None
    for k, a in available.items():
        idx = pd.Index(a.obs_names.astype(str))
        common_cells = idx if common_cells is None else common_cells.intersection(idx)

    common_cells = pd.Index(common_cells.astype(str))

    if verbose:
        print(f"Common paired cells across {list(available.keys())}: {len(common_cells)}")

    out = {}
    for k, a in available.items():
        out[k] = a[common_cells].copy()
        if verbose:
            print(k, out[k])

    return out

def filter_rna_genes_basic(
    rna: ad.AnnData,
    min_cells: int = 10,
    min_counts: int = 20,
    verbose: bool = True,
) -> ad.AnnData:
    r = rna.copy()

    X = r.layers["counts"] if "counts" in r.layers else r.X
    if sp.issparse(X):
        n_cells_by_gene = np.asarray((X > 0).sum(axis=0)).ravel()
        total_counts_by_gene = np.asarray(X.sum(axis=0)).ravel()
    else:
        n_cells_by_gene = (X > 0).sum(axis=0)
        total_counts_by_gene = X.sum(axis=0)

    keep = (n_cells_by_gene >= min_cells) & (total_counts_by_gene >= min_counts)
    if verbose:
        print(f"RNA features before: {r.n_vars}")
        print(f"RNA features kept:   {int(keep.sum())}")

    r = r[:, keep].copy()
    return r

def filter_methyl_features_by_coverage(
    a: ad.AnnData,
    coverage_layer: str,
    min_cells_covered: int = 10,
    min_total_coverage: float = 50.0,
    verbose: bool = True,
) -> ad.AnnData:
    b = a.copy()
    cov = b.layers[coverage_layer]

    if sp.issparse(cov):
        cells_covered = np.asarray((cov > 0).sum(axis=0)).ravel()
        total_cov = np.asarray(cov.sum(axis=0)).ravel()
    else:
        cells_covered = (cov > 0).sum(axis=0)
        total_cov = cov.sum(axis=0)

    keep = (cells_covered >= min_cells_covered) & (total_cov >= min_total_coverage)
    if verbose:
        print(f"{b.obs['modality'].iloc[0]} features before: {b.n_vars}")
        print(f"{b.obs['modality'].iloc[0]} features kept:   {int(keep.sum())}")

    b = b[:, keep].copy()
    return b

def load_scnmt_gastrulation_genebody_triplet(
    tar_path: Path,
    rna_member: str = "rna/counts.txt.gz",
    cpg_member: str = "met/feature_level/genebody.tsv.gz",
    gpc_member: str = "acc/feature_level/genebody.tsv.gz",
    metadata_member: str = "sample_metadata.txt",
    require_qc: bool = True,
    min_cov_cpg: int = 1,
    min_cov_gpc: int = 1,
    filter_features: bool = True,
    rna_min_cells: int = 10,
    rna_min_counts: int = 20,
    meth_min_cells_covered: int = 10,
    meth_min_total_coverage: float = 50.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    # metadata first (lets us subset long tables to relevant cells)
    meta = read_metadata_from_tar(tar_path, metadata_member, verbose=verbose)

    meta_work = meta.copy()
    if require_qc:
        mask = pd.Series(True, index=meta_work.index)
        for qc_col in ["pass_rnaQC", "pass_metQC", "pass_accQC"]:
            if qc_col in meta_work.columns:
                mask &= meta_work[qc_col].fillna(False).astype(bool)
        meta_work = meta_work.loc[mask].copy()
        if verbose:
            print(f"Metadata rows after joint QC filter: {meta_work.shape[0]}")

    rna_keep_ids = meta_work["id_rna"].dropna().astype(str).unique().tolist() if "id_rna" in meta_work.columns else None
    sample_keep_ids = meta_work["sample"].dropna().astype(str).unique().tolist()

    # RNA
    rna_df = read_rna_counts_from_tar(tar_path, rna_member, verbose=verbose)
    if rna_keep_ids is not None:
        rna_df = rna_df.loc[rna_df.index.intersection(pd.Index(rna_keep_ids))].copy()
        if verbose:
            print("RNA after metadata/QC subset:", rna_df.shape)

    rna = _df_to_anndata_dense_or_sparse(
        rna_df,
        modality="rna",
        feature_type="gene",
        copy_to_counts=True,
    )

    # CpG (long -> wide)
    cpg_parsed = read_feature_level_long_from_tar(
        tar_path=tar_path,
        member_name=cpg_member,
        modality="cpg",
        cell_subset=sample_keep_ids,
        min_coverage=min_cov_cpg,
        chunksize=500_000,
        verbose=verbose,
    )
    cpg = build_methyl_like_anndata_from_long_feature(
        cpg_parsed,
        modality="cpg",
        successes_layer="meth_successes",
        coverage_layer="meth_total_count",
        feature_type="genebody",
    )

    # GpC (long -> wide)
    gpc_parsed = read_feature_level_long_from_tar(
        tar_path=tar_path,
        member_name=gpc_member,
        modality="gpc",
        cell_subset=sample_keep_ids,
        min_coverage=min_cov_gpc,
        chunksize=500_000,
        verbose=verbose,
    )
    gpc = build_methyl_like_anndata_from_long_feature(
        gpc_parsed,
        modality="gpc",
        successes_layer="acc_successes",
        coverage_layer="acc_total_count",
        feature_type="genebody",
    )

    # Attach metadata + intersect common paired cells
    aligned = attach_metadata_and_align_modalities(
        rna=rna,
        cpg=cpg,
        gpc=gpc,
        meta=meta,
        require_qc=require_qc,
        verbose=verbose,
    )
    rna = aligned["rna"]
    cpg = aligned["cpg"]
    gpc = aligned["gpc"]

    # Optional feature filtering (highly recommended for speed)
    if filter_features:
        if verbose:
            print("\nApplying feature filtering...")

        rna = filter_rna_genes_basic(
            rna, min_cells=rna_min_cells, min_counts=rna_min_counts, verbose=verbose
        )
        cpg = filter_methyl_features_by_coverage(
            cpg,
            coverage_layer="meth_total_count",
            min_cells_covered=meth_min_cells_covered,
            min_total_coverage=meth_min_total_coverage,
            verbose=verbose,
        )
        gpc = filter_methyl_features_by_coverage(
            gpc,
            coverage_layer="acc_total_count",
            min_cells_covered=meth_min_cells_covered,
            min_total_coverage=meth_min_total_coverage,
            verbose=verbose,
        )

        # Re-intersect cells just in case (vars changed only, but safe)
        common_cells = rna.obs_names.intersection(cpg.obs_names).intersection(gpc.obs_names)
        rna = rna[common_cells].copy()
        cpg = cpg[common_cells].copy()
        gpc = gpc[common_cells].copy()

    return {
        "rna": rna,
        "cpg": cpg,
        "gpc": gpc,
        "meta": meta,
        "cpg_parse_meta": cpg_parsed["meta"],
        "gpc_parse_meta": gpc_parsed["meta"],
    }

def build_univi_inputs_from_scnmt_triplet(
    rna: ad.AnnData,
    cpg: ad.AnnData,
    gpc: ad.AnnData,
    rna_target_sum: float = 1e4,
    log1p_rna: bool = True,
    copy: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      - adata_dict (paired, same obs_names expected after alignment)
      - recon_targets_spec (for beta_binomial on cpg/gpc)
    """
    r = rna.copy() if copy else rna
    m = cpg.copy() if copy else cpg
    a = gpc.copy() if copy else gpc

    # RNA preprocessing for model input in .X
    # Keep raw counts in .layers["counts"]
    sc.pp.normalize_total(r, target_sum=rna_target_sum)
    if log1p_rna:
        sc.pp.log1p(r)

    # cpg/gpc already have fractions in .X and successes/coverage in layers
    adata_dict = {"rna": r, "cpg": m, "gpc": a}

    recon_targets_spec = {
        "cpg": {
            "successes_layer": "meth_successes",
            "total_count_layer": "meth_total_count",
        },
        "gpc": {
            "successes_layer": "acc_successes",
            "total_count_layer": "acc_total_count",
        },
    }

    return {
        "adata_dict": adata_dict,
        "recon_targets_spec": recon_targets_spec,
    }
