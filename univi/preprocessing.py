"""Reusable fit-on-training/apply-to-query transforms for biological tutorials.

All transforms preserve feature order and reject missing reference features.
They deliberately keep raw counts separate from Gaussian model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize


def _counts(adata, layer):
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Use unique feature and cell identifiers before preprocessing.")
    x = adata.X if layer is None else adata.layers[layer]
    values = x.data if sp.issparse(x) else np.asarray(x)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative raw counts.")
    return sp.csr_matrix(x, dtype=np.float32)


def _indices(adata, names):
    if not adata.var_names.is_unique:
        raise ValueError("Feature identifiers must be unique.")
    indices = adata.var_names.get_indexer(names)
    if (indices < 0).any():
        missing = np.asarray(names)[indices < 0]
        raise ValueError(f"Query is missing {len(missing)} reference features; examples: {missing[:5].tolist()}")
    return indices


class RNAPreprocessor:
    """RNA counts -> log1p normalized features -> optional train-fitted Z scores.

    Seurat v3 HVGs are selected on raw training counts. ``normalize_on_selected``
    reproduces the HVG-first normalization in the CITE/Multiome notebook code;
    False normalizes over the complete reference feature panel before subsetting.
    ``n_hvg=None`` keeps all features. No fallback HVG method is chosen silently.
    """
    def __init__(self, n_hvg=2000, *, layer="counts", target_sum=1e4,
                 scale=False, normalize_on_selected=True, clip=None,
                 exclude_genes=(), min_cells=1):
        self.n_hvg, self.layer, self.target_sum = n_hvg, layer, target_sum
        self.scale, self.normalize_on_selected, self.clip = scale, normalize_on_selected, clip
        self.exclude_genes, self.min_cells = tuple(exclude_genes), min_cells

    def fit(self, adata):
        import scanpy as sc
        x = _counts(adata, self.layer)
        self.normalization_features_ = adata.var_names.astype(str).tolist()
        if self.n_hvg is None:
            self.features_ = self.normalization_features_.copy()
        else:
            eligible = np.asarray((x > 0).sum(axis=0)).ravel() >= self.min_cells
            eligible &= ~adata.var_names.isin(self.exclude_genes)
            candidates = AnnData(x[:, eligible], var=adata.var.iloc[np.flatnonzero(eligible)].copy())
            if candidates.n_vars == 0:
                raise ValueError("No eligible genes remain for HVG selection.")
            if candidates.n_vars <= self.n_hvg:
                self.features_ = candidates.var_names.astype(str).tolist()
            else:
                sc.pp.highly_variable_genes(candidates, flavor="seurat_v3", n_top_genes=int(self.n_hvg))
                self.features_ = candidates.var_names[candidates.var.highly_variable].astype(str).tolist()
        if self.scale:
            self.scaler_ = StandardScaler().fit(self._log(adata).toarray())
        return self

    def _log(self, adata):
        x = _counts(adata, self.layer)
        selected = x[:, _indices(adata, self.features_)]
        panel = selected if self.normalize_on_selected else x[:, _indices(adata, self.normalization_features_)]
        totals = np.asarray(panel.sum(axis=1)).ravel()
        normalized = selected.multiply((self.target_sum / np.maximum(totals, 1e-12))[:, None]).tocsr()
        normalized.data = np.log1p(normalized.data)
        return normalized

    def transform(self, adata):
        if not hasattr(self, "features_"):
            raise RuntimeError("Call fit on training cells first.")
        out = adata[:, self.features_].copy()
        out.layers["counts"] = _counts(adata, self.layer)[:, _indices(adata, self.features_)]
        out.layers["log1p"] = self._log(adata)
        out.X = (self.scaler_.transform(out.layers["log1p"].toarray()).astype(np.float32)
                 if self.scale else out.layers["log1p"].copy())
        if self.clip is not None:
            if sp.issparse(out.X):
                out.X.data = np.clip(out.X.data, -self.clip, self.clip)
            else:
                out.X = np.clip(out.X, -self.clip, self.clip)
        return out

    def fit_transform(self, adata):
        return self.fit(adata).transform(adata)


class ADTPreprocessor:
    """Fixed-panel per-cell CLR followed by optional training-fitted scaling.

    Input must be nonnegative raw abundance, not already CLR-transformed data.
    Restrict/harmonize the antibody panel before fitting this transform.
    """
    def __init__(self, *, layer="counts", scale=False, clip=None):
        self.layer, self.scale, self.clip = layer, scale, clip

    def _clr(self, adata):
        x = _counts(adata, self.layer)[:, _indices(adata, self.features_)].toarray()
        values = np.log1p(x)
        return values - values.mean(axis=1, keepdims=True)

    def fit(self, adata):
        self.features_ = adata.var_names.astype(str).tolist()
        if self.scale:
            self.scaler_ = StandardScaler().fit(self._clr(adata))
        else:
            self._clr(adata)
        return self

    def transform(self, adata):
        if not hasattr(self, "features_"):
            raise RuntimeError("Call fit on training cells first.")
        out = adata[:, self.features_].copy()
        out.layers["counts"] = _counts(adata, self.layer)[:, _indices(adata, self.features_)]
        out.layers["clr"] = self._clr(adata).astype(np.float32)
        out.X = (self.scaler_.transform(out.layers["clr"]).astype(np.float32)
                 if self.scale else out.layers["clr"].copy())
        if self.clip is not None:
            out.X = np.clip(out.X, -self.clip, self.clip)
        return out

    def fit_transform(self, adata):
        return self.fit(adata).transform(adata)


class ATACPreprocessor:
    """Reference peak counts -> fixed TF-IDF/LSI basis, optionally dropping LSI_0.

    ``method='sklearn'`` matches the Multiome notebook's TfidfTransformer.
    ``'signac'`` uses the SHARE-seq notebook's log1p(TF * IDF * 1e4),
    with IDF=log1p(n/(1+df)), then L2 normalization.
    ``'tea'`` uses L1 TF * log1p(n/(1+df)), then L2 normalization.
    ``n_components`` is the number fit BEFORE dropping the first component.
    """
    def __init__(self, n_components=100, *, layer="counts", method="sklearn",
                 drop_first=False, scale=True, min_fraction=0.0, max_fraction=1.0,
                 random_state=42):
        if method not in {"sklearn", "signac", "tea"}:
            raise ValueError("Unknown TF-IDF method.")
        if not 0 <= min_fraction < max_fraction <= 1:
            raise ValueError("Require 0 <= min_fraction < max_fraction <= 1.")
        self.n_components, self.layer, self.method = n_components, layer, method
        self.drop_first, self.scale = drop_first, scale
        self.min_fraction, self.max_fraction = min_fraction, max_fraction
        self.random_state = random_state

    def _tfidf(self, x):
        if self.method == "sklearn":
            return self.tfidf_.transform(x)
        tf = normalize(x, norm="l1", axis=1).multiply(self.idf_).tocsr()
        if self.method == "signac":
            tf.data = np.log1p(tf.data * 1e4)
        return normalize(tf, norm="l2", axis=1)

    def fit(self, adata):
        x = _counts(adata, self.layer)
        df = np.asarray((x > 0).sum(axis=0)).ravel()
        frequency = df / x.shape[0]
        keep = (frequency >= self.min_fraction) & (frequency <= self.max_fraction) & (df > 0)
        self.features_ = adata.var_names[keep].astype(str).tolist()
        x, df = x[:, keep], df[keep]
        if not 1 <= self.n_components < min(x.shape):
            raise ValueError("n_components must be smaller than training cell and retained feature counts.")
        if self.method == "sklearn":
            self.tfidf_ = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit(x)
        else:
            self.idf_ = np.log1p(x.shape[0] / (1 + df))
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        z = self.svd_.fit_transform(self._tfidf(x))
        if self.drop_first:
            z = z[:, 1:]
        if z.shape[1] == 0:
            raise ValueError("Dropping the first component leaves no model inputs.")
        if self.scale:
            self.scaler_ = StandardScaler().fit(z)
        return self

    def transform(self, adata):
        if not hasattr(self, "svd_"):
            raise RuntimeError("Call fit on training cells first.")
        x = _counts(adata, self.layer)[:, _indices(adata, self.features_)]
        z = self.svd_.transform(self._tfidf(x))
        first = int(self.drop_first)
        z = z[:, first:]
        if self.scale:
            z = self.scaler_.transform(z)
        return AnnData(X=z.astype(np.float32), obs=adata.obs.copy(),
                       var=pd.DataFrame(index=[f"LSI_{i}" for i in range(first, self.n_components)]))

    def fit_transform(self, adata):
        return self.fit(adata).transform(adata)


def split_by_label(labels, *, train_fraction=0.8, val_fraction=0.1,
                   train_cap=None, val_cap=None, max_per_label=None, seed=0):
    """Stratify cells; capped training/validation leftovers go to the test set.

    ``max_per_label`` caps the pool before allocating train/validation fractions,
    and adds overflow to test, matching the CITE/Multiome notebook helper with
    unused_to_test=True. It is mutually exclusive with
    caps applied directly to training and validation counts.
    This is a portable deterministic split, not a replacement for archived
    manuscript barcode maps. Tiny classes may be absent from validation.
    """
    labels = np.asarray(labels)
    if max_per_label is not None and (train_cap is not None or val_cap is not None):
        raise ValueError("Choose a pool cap or separate train/validation caps.")
    if pd.isna(labels).any():
        raise ValueError("Handle missing stratification labels explicitly.")
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Require positive training and test fractions.")
    rng = np.random.default_rng(seed)
    result = {key: [] for key in ("train", "val", "test")}
    for label in pd.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        used = len(idx) if max_per_label is None else min(len(idx), max_per_label)
        ntr = int(used * train_fraction)
        nva = int(used * val_fraction)
        if train_cap is not None:
            ntr = min(ntr, int(train_cap))
        if val_cap is not None:
            nva = min(nva, int(val_cap))
        result["train"].extend(idx[:ntr])
        result["val"].extend(idx[ntr:ntr+nva])
        result["test"].extend(idx[ntr+nva:])
    return {key: np.asarray(values, dtype=int) for key, values in result.items()}
