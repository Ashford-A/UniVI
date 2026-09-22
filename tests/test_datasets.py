"""Tests for univi.datasets: registry, download/verification, and loading."""
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import univi.datasets as uds


def _write(tmp_path, name, obs_names, n_vars=5, counts=True):
    x = np.random.default_rng(0).poisson(2, (len(obs_names), n_vars)).astype(np.float32)
    if not counts:
        x = x + 0.5
    a = ad.AnnData(sp.csr_matrix(x), obs=pd.DataFrame({"cell_type": "T"}, index=obs_names))
    path = tmp_path / name
    a.write_h5ad(path)
    return path


def _md5(path):
    return "md5:" + hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    src = tmp_path / "mirror"
    src.mkdir()
    monkeypatch.setenv("UNIVI_DATA_DIR", str(tmp_path / "cache"))
    rna = _write(src, "rna.h5ad", ["c1", "c2", "c3", "c4"])
    adt = _write(src, "adt.h5ad", ["c4", "c2", "c1", "x9"])  # different order + extra cell
    uds.register_dataset("toy", {
        "paired": ["rna", "adt"],
        "files": {
            "rna": {"filename": "rna.h5ad", "url": rna.as_uri(), "hash": _md5(rna)},
            "adt": {"filename": "adt.h5ad", "url": adt.as_uri(), "hash": _md5(adt)},
        },
    })
    yield src


def test_builtin_registry_lists_hosted_multiome():
    table = uds.list_datasets()
    assert "pbmc_multiome_10k" in table.index
    assert bool(table.loc["pbmc_multiome_10k", "available"])
    info = uds.dataset_info("pbmc_multiome_10k")
    assert info["doi"] == "10.5281/zenodo.19581816"
    assert set(info["files"]) == {"rna", "atac"}


def test_unhosted_dataset_raises_helpful_error():
    uds.register_dataset("not_hosted_yet", {"files": {"rna": {"filename": "rna.h5ad", "url": None, "hash": None}}})
    with pytest.raises(uds.DatasetNotAvailableError, match="no download location"):
        uds.fetch("not_hosted_yet")


def test_load_aligns_pairs_and_adds_counts_layer(mirror):
    with pytest.warns(UserWarning, match="Kept 3 cells"):
        data = uds.load("toy", progress=False)
    assert list(data["rna"].obs_names) == ["c1", "c2", "c4"]
    assert data["rna"].obs_names.equals(data["adt"].obs_names)
    assert "counts" in data["rna"].layers


def test_cached_file_is_reused(mirror):
    first = uds.fetch("toy", ["rna"], progress=False)["rna"]
    (mirror / "rna.h5ad").unlink()  # mirror disappears; cache must still serve the file
    again = uds.fetch("toy", ["rna"], progress=False)["rna"]
    assert first == again and again.exists()


def test_checksum_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIVI_DATA_DIR", str(tmp_path / "cache"))
    path = _write(tmp_path, "rna.h5ad", ["a", "b"])
    uds.register_dataset("bad", {"files": {"rna": {
        "filename": "rna.h5ad", "url": path.as_uri(), "hash": "md5:" + "0" * 32}}})
    with pytest.raises(IOError, match="Checksum mismatch"):
        uds.fetch("bad", progress=False)
    assert not (tmp_path / "cache" / "bad" / "rna.h5ad").exists()


def test_falls_back_to_second_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIVI_DATA_DIR", str(tmp_path / "cache"))
    path = _write(tmp_path, "rna.h5ad", ["a", "b"])
    uds.register_dataset("mirrors", {"files": {"rna": {
        "filename": "rna.h5ad",
        "urls": [(tmp_path / "missing.h5ad").as_uri(), path.as_uri()],
        "hash": _md5(path)}}})
    assert uds.fetch("mirrors", progress=False)["rna"].exists()


def test_non_count_matrix_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIVI_DATA_DIR", str(tmp_path / "cache"))
    path = _write(tmp_path, "rna.h5ad", ["a", "b"], counts=False)
    uds.register_dataset("normalized", {"files": {"rna": {
        "filename": "rna.h5ad", "url": path.as_uri(), "hash": _md5(path)}}})
    assert "counts" not in uds.load("normalized", progress=False)["rna"].layers


def test_env_registry_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIVI_DATA_DIR", str(tmp_path / "cache"))
    path = _write(tmp_path, "rna.h5ad", ["a", "b"])
    reg = tmp_path / "extra.json"
    reg.write_text(json.dumps({"datasets": {"hao_citeseq_pbmc": {"files": {
        "rna": {"filename": "rna.h5ad", "url": path.as_uri(), "hash": _md5(path)}}}}}))
    monkeypatch.setenv("UNIVI_DATASET_REGISTRY", str(reg))
    data = uds.load("hao_citeseq_pbmc", progress=False)
    assert data["rna"].n_obs == 2


def test_scnmt_readers_still_importable():
    from univi import load_scnmt_gastrulation_genebody_triplet
    from univi.datasets import build_univi_inputs_from_scnmt_triplet  # noqa: F401
    assert callable(load_scnmt_gastrulation_genebody_triplet)
