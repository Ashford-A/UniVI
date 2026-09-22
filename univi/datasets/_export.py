"""Package AnnData objects as a shareable, checksummed dataset for univi.datasets.

``export_dataset`` writes one ``.h5ad`` per modality with raw counts in ``.X``
(``univi.datasets.load`` restores ``layers["counts"]`` from it), selected ``.obs`` metadata, an optional split column,
plus ``manifest.json`` (checksums, shapes) and ``DESCRIPTION.md`` for the
hosting record. ``scripts/prepare_zenodo_release.py register`` turns the
manifest into a registry entry once the files are published.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

__all__ = ["export_dataset"]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _counts_matrix(adata, counts_layer: Optional[str]):
    from ._registry import _looks_like_counts

    if counts_layer and counts_layer in adata.layers:
        X = adata.layers[counts_layer]
    else:
        X = adata.X
    if not _looks_like_counts(X):
        warnings.warn("A matrix being exported does not look like raw integer counts; "
                      "it is exported as-is. Pass counts_layer=... if raw counts are in a layer.")
    return X


def _clean_obs(obs, columns: Optional[Sequence[str]]):
    import pandas as pd

    obs = obs.copy() if columns is None else obs.loc[:, [c for c in columns if c in obs.columns]].copy()
    for col in obs.columns:
        s = obs[col]
        if s.dtype == object:
            # h5ad needs homogeneous columns; keep missing values as missing
            obs[col] = s.map(lambda v: v if v is None or isinstance(v, str) or pd.isna(v) else str(v))
    return obs


def _split_labels(splits: Mapping[str, Iterable], obs_names, n_obs: int):
    import numpy as np
    import pandas as pd

    labels = pd.Series("unassigned", index=obs_names, dtype=object)
    for name, members in splits.items():
        members = np.asarray(list(members))
        if members.dtype.kind in "iu":
            if members.size and (members.min() < 0 or members.max() >= n_obs):
                raise IndexError(f"split {name!r}: positional indices out of range")
            labels.iloc[members] = name
        else:
            idx = pd.Index(members.astype(str))
            unknown = idx.difference(obs_names)
            if len(unknown):
                raise KeyError(f"split {name!r}: {len(unknown)} cell ids not found (e.g. {list(unknown[:3])})")
            labels.loc[idx] = name
    return pd.Categorical(labels.to_numpy())


def export_dataset(
    name: str,
    adatas: Mapping[str, Any],
    outdir,
    *,
    paired: Optional[Sequence[str]] = None,
    labels: Sequence[str] = (),
    obs_columns: Optional[Sequence[str]] = None,
    splits: Optional[Mapping[str, Iterable]] = None,
    split_key: str = "split",
    counts_layer: Optional[str] = "counts",
    title: Optional[str] = None,
    description: Optional[str] = None,
    source: Optional[str] = None,
    compression: Optional[str] = "gzip",
    overwrite: bool = False,
) -> Path:
    """Write AnnData objects as a hostable dataset; return the manifest path.

    Parameters
    ----------
    name
        Dataset name as it will appear in ``univi.datasets`` (e.g. ``"hao_citeseq_pbmc"``).
    adatas
        ``{key: AnnData}``, e.g. ``{"rna": rna, "adt": adt}``. Objects are not modified.
    outdir
        Output directory (created).
    paired
        Keys that describe the same cells. They must contain the same cell ids;
        they are written in the order of the first key. Default: all keys when
        they share identical cell ids, otherwise none.
    labels
        Label columns expected in ``.obs`` (checked and recorded in the manifest).
    obs_columns
        ``.obs`` columns to keep (default: all). ``split_key`` is always added
        when ``splits`` is given.
    splits
        ``{"train": ids, "val": ids, "test": ids}`` with cell ids or positional
        indices into the first paired object. Stored as ``obs[split_key]``
        (``"unassigned"`` elsewhere) in every paired object.
    counts_layer
        Layer holding raw counts; ``.X`` is used when the layer is missing.
    title, description, source
        Text for the manifest and ``DESCRIPTION.md``.
    """
    import anndata as ad
    import pandas as pd

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(adatas)
    if not keys:
        raise ValueError("adatas is empty")

    if paired is None:
        first = adatas[keys[0]].obs_names
        paired = keys if all(adatas[k].obs_names.equals(first) for k in keys) else []
    paired = list(paired)
    order = None
    if paired:
        order = adatas[paired[0]].obs_names
        if not order.is_unique:
            raise ValueError(f"{paired[0]}: obs_names are not unique")
        for k in paired[1:]:
            other = adatas[k].obs_names
            if len(other) != len(order) or not other.isin(order).all():
                raise ValueError(f"{k!r} and {paired[0]!r} do not contain the same cells; align them first "
                                 "(univi.data.align_paired_obs_names).")

    split_col = None
    if splits is not None:
        if not paired:
            raise ValueError("splits require at least one paired key")
        split_col = pd.Series(_split_labels(splits, order, len(order)), index=order)

    manifest: Dict[str, Any] = {"name": name, "title": title or name, "paired": paired,
                                "labels": list(labels), "source": source, "files": {}}
    for key in keys:
        a = adatas[key]
        if key in paired and not a.obs_names.equals(order):
            a = a[order]
        missing = [c for c in labels if c not in a.obs.columns]
        if missing and key in paired:
            warnings.warn(f"{key}: label columns {missing} not found in .obs")
        X = _counts_matrix(a, counts_layer)
        obs = _clean_obs(a.obs, obs_columns)
        if split_col is not None and key in paired:
            obs[split_key] = pd.Categorical(split_col.loc[a.obs_names].to_numpy())
        b = ad.AnnData(X=X, obs=obs, var=a.var.copy())
        filename = f"{name}_{key}.h5ad"
        path = out / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        b.write_h5ad(path, compression=compression)
        info = {"filename": filename, "md5": _md5(path), "size_mb": round(path.stat().st_size / 1e6, 1),
                "n_obs": int(b.n_obs), "n_vars": int(b.n_vars), "obs_columns": list(map(str, b.obs.columns))}
        if split_col is not None and key in paired:
            info["split_counts"] = {str(k): int(v) for k, v in b.obs[split_key].value_counts().items()}
        manifest["files"][key] = info
        print(f"wrote {path} ({info['n_obs']} x {info['n_vars']}, {info['size_mb']} MB)")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = [f"# {manifest['title']}", ""]
    if description:
        lines += [description.strip(), ""]
    if source:
        lines += [f"Original data: {source}", ""]
    lines += ["| file | cells | features |", "| --- | --- | --- |"]
    lines += [f"| {f['filename']} | {f['n_obs']} | {f['n_vars']} |" for f in manifest["files"].values()]
    lines += ["", "Raw counts are in `.X`."]
    if split_col is not None:
        lines += [f"`.obs['{split_key}']` holds the train/validation/test assignment used in the analysis."]
    lines += ["", "Load with:", "", "```python", "import univi.datasets as uds",
              f'data = uds.load("{name}")', "```", "",
              "If you use these data, please cite the original data source and the UniVI article: "
              "Ashford et al., Genome Research (2026), doi:10.1101/gr.281431.125."]
    (out / "DESCRIPTION.md").write_text("\n".join(lines) + "\n")
    return out / "manifest.json"
