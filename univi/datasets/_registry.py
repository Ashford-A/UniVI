"""Download, verify, cache and load the datasets used in the UniVI tutorials.

Datasets are described in ``registry.json`` (shipped with the package). Each
entry lists one or more files with a download URL and a checksum. Files are
downloaded once into a cache directory and verified before use.

Configuration
-------------
``UNIVI_DATA_DIR``
    Cache directory. Defaults to ``$XDG_CACHE_HOME/univi`` or ``~/.cache/univi``.
``UNIVI_DATASET_REGISTRY``
    Optional path to a JSON file with extra or overriding dataset entries
    (same schema as the built-in registry). Useful for mirrors, offline
    clusters, or your own lab datasets.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import warnings
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

__all__ = [
    "DatasetNotAvailableError",
    "get_data_dir",
    "list_datasets",
    "dataset_info",
    "register_dataset",
    "fetch",
    "load",
    "pbmc_multiome_10k",
    "hao_citeseq_pbmc",
    "scnmt_gastrulation",
]

_USER_AGENT = "univi-datasets (+https://github.com/Ashford-A/UniVI)"
_RUNTIME_ENTRIES: Dict[str, Dict[str, Any]] = {}


class DatasetNotAvailableError(RuntimeError):
    """Raised when a registered dataset has no download location yet."""


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
def _builtin_registry() -> Dict[str, Dict[str, Any]]:
    text = resources.files("univi.datasets").joinpath("registry.json").read_text(encoding="utf-8")
    return json.loads(text)["datasets"]


def _registry() -> Dict[str, Dict[str, Any]]:
    reg = _builtin_registry()
    extra = os.environ.get("UNIVI_DATASET_REGISTRY")
    if extra:
        payload = json.loads(Path(extra).expanduser().read_text(encoding="utf-8"))
        reg.update(payload.get("datasets", payload))
    reg.update(copy.deepcopy(_RUNTIME_ENTRIES))
    return reg


def _entry(name: str) -> Dict[str, Any]:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(reg)}")
    return copy.deepcopy(reg[name])


def _is_available(entry: Mapping[str, Any]) -> bool:
    files = entry.get("files") or {}
    return bool(files) and all(_urls(f) for f in files.values())


def _urls(file_entry: Mapping[str, Any]) -> List[str]:
    urls = file_entry.get("urls")
    if urls is None:
        urls = [file_entry["url"]] if file_entry.get("url") else []
    return [u for u in urls if u]


def register_dataset(name: str, entry: Mapping[str, Any]) -> None:
    """Register (or override) a dataset for the current Python session.

    ``entry`` uses the same schema as the built-in registry, for example::

        register_dataset("my_citeseq", {
            "title": "My CITE-seq run",
            "paired": ["rna", "adt"],
            "files": {
                "rna": {"filename": "rna.h5ad", "url": "https://.../rna.h5ad",
                        "hash": "sha256:..."},
                "adt": {"filename": "adt.h5ad", "url": "https://.../adt.h5ad",
                        "hash": "sha256:..."},
            },
        })
    """
    if not isinstance(entry, Mapping) or "files" not in entry:
        raise ValueError("A dataset entry must be a mapping with a 'files' key.")
    _RUNTIME_ENTRIES[str(name)] = copy.deepcopy(dict(entry))


def list_datasets():
    """Return a table of registered datasets and whether they can be downloaded."""
    import pandas as pd

    rows = []
    for name, e in _registry().items():
        rows.append({
            "name": name,
            "title": e.get("title", ""),
            "modalities": ", ".join(e.get("files", {}).keys()),
            "n_cells": e.get("n_cells"),
            "size_mb": sum(float(f.get("size_mb") or 0) for f in e.get("files", {}).values()) or None,
            "available": _is_available(e),
            "used_in": ", ".join(e.get("used_in", [])),
        })
    return pd.DataFrame(rows).set_index("name")


def dataset_info(name: str) -> Dict[str, Any]:
    """Return the full registry entry (description, source, license, files)."""
    return _entry(name)


# -----------------------------------------------------------------------------
# Download + verification
# -----------------------------------------------------------------------------
def get_data_dir(data_dir: Optional[os.PathLike] = None) -> Path:
    """Resolve the dataset cache directory (created if needed)."""
    if data_dir is None:
        data_dir = os.environ.get("UNIVI_DATA_DIR")
    if data_dir is None:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
        data_dir = os.path.join(base, "univi")
    path = Path(data_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_hash(spec: Optional[str]):
    if not spec:
        return None, None
    algo, _, digest = str(spec).partition(":")
    if not digest:
        raise ValueError(f"Hash must look like 'md5:<hex>' or 'sha256:<hex>', got {spec!r}")
    algo = algo.lower()
    if algo not in ("md5", "sha256"):
        raise ValueError(f"Unsupported hash algorithm {algo!r}; use md5 or sha256.")
    return algo, digest.lower()


def _file_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, algo: Optional[str], progress: bool) -> Optional[str]:
    """Stream ``url`` to ``dest`` and return the digest (if ``algo``)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    hasher = hashlib.new(algo) if algo else None
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, prefix=dest.name, suffix=".part")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, tmp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            bar = None
            if progress:
                try:
                    from tqdm.auto import tqdm
                    bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
                except Exception:  # pragma: no cover - tqdm is a core dependency
                    bar = None
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                tmp.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
                if bar is not None:
                    bar.update(len(chunk))
            if bar is not None:
                bar.close()
        shutil.move(tmp.name, dest)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return hasher.hexdigest() if hasher is not None else None


def fetch(
    name: str,
    keys: Optional[Iterable[str]] = None,
    *,
    data_dir: Optional[os.PathLike] = None,
    force: bool = False,
    progress: bool = True,
) -> Dict[str, Path]:
    """Download (if needed) and verify dataset files; return local paths by key.

    Cached files are re-used when their checksum matches. A file whose checksum
    does not match is re-downloaded; a mismatch after download raises an error.
    """
    entry = _entry(name)
    files = entry.get("files") or {}
    if not _is_available(entry):
        record = entry.get("record_url") or "the dataset's Zenodo record"
        raise DatasetNotAvailableError(
            f"Dataset {name!r} is registered but has no download location yet. "
            f"See {record} or the UniVI documentation (Datasets page) for how to obtain it. "
            "You can also point to local/mirrored copies via UNIVI_DATASET_REGISTRY "
            "or univi.datasets.register_dataset()."
        )
    wanted = list(files) if keys is None else list(keys)
    unknown = sorted(set(wanted) - set(files))
    if unknown:
        raise KeyError(f"Dataset {name!r} has no files {unknown}; available: {list(files)}")

    root = get_data_dir(data_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for key in wanted:
        spec = files[key]
        dest = root / spec["filename"]
        algo, expected = _parse_hash(spec.get("hash"))
        if dest.exists() and not force:
            if algo is None or _file_digest(dest, algo) == expected:
                out[key] = dest
                continue
            warnings.warn(f"{dest.name}: cached file failed checksum; downloading again.")
        if algo is None:
            warnings.warn(f"{name}/{key}: no checksum registered; the download cannot be verified.")
        errors = []
        for url in _urls(spec):
            try:
                digest = _download(url, dest, algo, progress)
            except Exception as exc:  # try the next mirror
                errors.append(f"{url}: {exc}")
                continue
            if algo is not None and digest != expected:
                dest.unlink(missing_ok=True)
                raise IOError(
                    f"Checksum mismatch for {spec['filename']} from {url} "
                    f"(expected {algo}:{expected}, got {algo}:{digest})."
                )
            out[key] = dest
            break
        else:
            raise IOError(f"Could not download {name}/{key}:\n  " + "\n  ".join(errors))
    return out


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def _looks_like_counts(X, max_values: int = 200_000) -> bool:
    import numpy as np
    import scipy.sparse as sp

    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if values.size == 0:
        return False
    values = values[:max_values]
    return bool(np.all(values >= 0) and np.all(np.mod(values, 1) == 0))


def _align_paired(adatas: Dict[str, Any], keys: List[str]) -> None:
    first = adatas[keys[0]]
    for k in keys:
        if not adatas[k].obs_names.is_unique:
            raise ValueError(f"{k}: cell identifiers (obs_names) must be unique for pairing.")
    common = first.obs_names
    for k in keys[1:]:
        common = common[common.isin(adatas[k].obs_names)]
    if len(common) == 0:
        raise ValueError(f"No shared cell identifiers across {keys}.")
    dropped = {k: adatas[k].n_obs - len(common) for k in keys if adatas[k].n_obs != len(common)}
    if dropped:
        warnings.warn(f"Kept {len(common)} cells measured in all of {keys}; dropped {dropped}.")
    for k in keys:
        if not adatas[k].obs_names.equals(common):
            adatas[k] = adatas[k][common].copy()


def load(
    name: str,
    keys: Optional[Iterable[str]] = None,
    *,
    data_dir: Optional[os.PathLike] = None,
    counts_layer: Optional[str] = "counts",
    force: bool = False,
    progress: bool = True,
    **reader_kwargs: Any,
) -> Dict[str, Any]:
    """Download (if needed) and load a registered dataset as AnnData objects.

    Parameters
    ----------
    name
        Registry name, e.g. ``"pbmc_multiome_10k"``. See :func:`list_datasets`.
    keys
        Subset of files to load (e.g. ``["rna"]``). Defaults to all.
    data_dir
        Cache directory (overrides ``UNIVI_DATA_DIR``).
    counts_layer
        If this layer is missing and ``.X`` holds non-negative integers, copy
        ``.X`` into ``.layers[counts_layer]`` so the UniVI preprocessors can
        find raw counts. Set to ``None`` to leave objects untouched.
    force
        Re-download even if a verified cached copy exists.
    reader_kwargs
        Passed to the dataset-specific reader for archive datasets
        (e.g. scNMT-seq).

    Returns
    -------
    dict
        ``{key: AnnData}``. Keys listed under the entry's ``paired`` field share
        identical, identically ordered ``obs_names``.
    """
    import anndata as ad

    entry = _entry(name)
    paths = fetch(name, keys, data_dir=data_dir, force=force, progress=progress)

    reader = entry.get("reader")
    if reader == "scnmt_gastrulation":
        from .scnmt import load_scnmt_gastrulation_genebody_triplet

        kwargs = {"require_qc": False, "filter_features": False, "verbose": False}
        kwargs.update(reader_kwargs)
        triplet = load_scnmt_gastrulation_genebody_triplet(paths["bundle"], **kwargs)
        return {k: triplet[k] for k in ("rna", "cpg", "gpc")}
    if reader_kwargs:
        raise TypeError(f"Unexpected keyword arguments for {name!r}: {sorted(reader_kwargs)}")

    adatas: Dict[str, Any] = {}
    for key, path in paths.items():
        a = ad.read_h5ad(path)
        if counts_layer and counts_layer not in a.layers and _looks_like_counts(a.X):
            a.layers[counts_layer] = a.X.copy()
        adatas[key] = a

    paired = [k for k in entry.get("paired", []) if k in adatas]
    if len(paired) >= 2:
        _align_paired(adatas, paired)
    return adatas


# -----------------------------------------------------------------------------
# Named shortcuts (tab-completion friendly)
# -----------------------------------------------------------------------------
def pbmc_multiome_10k(**kwargs: Any) -> Dict[str, Any]:
    """Paired RNA + ATAC 10x Multiome PBMCs (9,631 cells, ``obs['cell_type']``).

    Used for Figs. 4 and 8–10 and Supplemental Figs. S2 and S8–S11 of the
    Genome Research article. Returns ``{"rna": AnnData, "atac": AnnData}`` with
    raw counts in ``.X`` and ``.layers["counts"]``.
    """
    return load("pbmc_multiome_10k", **kwargs)


def hao_citeseq_pbmc(**kwargs: Any) -> Dict[str, Any]:
    """Paired RNA + ADT CITE-seq PBMCs from Hao et al. (2021) with
    ``celltype.l1``/``l2``/``l3`` labels (Figs. 2–3, Supplemental Fig. S1).

    Returns ``{"rna": AnnData, "adt": AnnData}``.
    """
    return load("hao_citeseq_pbmc", **kwargs)


def scnmt_gastrulation(**kwargs: Any) -> Dict[str, Any]:
    """scNMT-seq mouse gastrulation (Argelaguet et al. 2019): RNA, CpG
    methylation and GpC accessibility (Supplemental Fig. S4).

    Returns ``{"rna", "cpg", "gpc"}`` AnnData objects with RNA counts, CpG/GpC
    fractions in ``.X``, and success/coverage layers for the beta-binomial
    likelihoods, as used in the article. Keyword arguments are passed to
    :func:`load`. To parse the original EBI bundle yourself, use
    :func:`load_scnmt_gastrulation_genebody_triplet`.
    """
    return load("scnmt_gastrulation", **kwargs)
