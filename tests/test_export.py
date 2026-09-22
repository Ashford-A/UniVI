"""Tests for dataset export, the Zenodo uploader (against a local mock), and registration."""
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

import univi.datasets as uds
from synthetic_data import make_citeseq

ROOT = Path(__file__).resolve().parents[1]


def test_export_roundtrip(tmp_path, monkeypatch):
    d = make_citeseq(n_cells=120)
    d["adt"] = d["adt"][::-1].copy()                      # same cells, different order
    names = d["rna"].obs_names
    manifest = uds.export_dataset("toy_cite", d, tmp_path / "out", paired=["rna", "adt"],
                                  labels=["celltype.l2"], splits={"train": names[:80], "test": np.arange(100, 120)})
    m = json.loads(manifest.read_text())
    assert m["files"]["rna"]["split_counts"] == {"train": 80, "test": 20, "unassigned": 20}
    for key, f in m["files"].items():
        assert hashlib.md5((tmp_path / "out" / f["filename"]).read_bytes()).hexdigest() == f["md5"]
    files = {k: {"filename": f["filename"], "url": (tmp_path / "out" / f["filename"]).as_uri(),
                 "hash": "md5:" + f["md5"]} for k, f in m["files"].items()}
    monkeypatch.setenv("UNIVI_DATA_DIR", str(tmp_path / "cache"))
    uds.register_dataset("toy_cite", {"paired": ["rna", "adt"], "files": files})
    back = uds.load("toy_cite", progress=False)
    assert back["adt"].obs_names.equals(back["rna"].obs_names)
    assert list(back["rna"].obs_names[:3]) == list(names[:3])
    assert "counts" in back["rna"].layers and set(back["rna"].obs["split"].cat.categories) == {"train", "test", "unassigned"}


def test_export_rejects_unpaired(tmp_path):
    d = make_citeseq(n_cells=50)
    d["adt"] = d["adt"][:40].copy()
    with pytest.raises(ValueError, match="same cells"):
        uds.export_dataset("bad", d, tmp_path, paired=["rna", "adt"])


class _MockZenodo(BaseHTTPRequestHandler):
    store = {}

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dep(self):
        base = f"http://127.0.0.1:{self.server.server_port}"
        return {"id": 42, "record_id": 42, "links": {"bucket": f"{base}/files/b1", "html": f"{base}/deposit/42"}}

    def do_POST(self):
        assert self.headers["Authorization"] == "Bearer tok"
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        if self.path.endswith("/actions/publish"):
            self._send(202, {**self._dep(), "doi": "10.5072/zenodo.42"})
        else:
            self._send(201, self._dep())

    def do_PUT(self):
        n = int(self.headers["Content-Length"])
        data = self.rfile.read(n)
        if self.path.startswith("/files/"):
            _MockZenodo.store[self.path.rsplit("/", 1)[-1]] = data
            self._send(201, {"checksum": "md5:" + hashlib.md5(data).hexdigest()})
        else:
            _MockZenodo.store["metadata"] = json.loads(data)
            self._send(200, self._dep())


def test_zenodo_upload_and_register(tmp_path):
    d = make_citeseq(n_cells=60)
    out = tmp_path / "hao_citeseq_pbmc"
    uds.export_dataset("hao_citeseq_pbmc", d, out, labels=["celltype.l1"])
    server = HTTPServer(("127.0.0.1", 0), _MockZenodo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "zenodo_upload.py"), str(out),
                            "--creator", "Ashford, Andrew J.", "--base-url", f"http://127.0.0.1:{server.server_port}",
                            "--publish"], env={**os.environ, "ZENODO_TOKEN": "tok"}, capture_output=True, text=True)
    finally:
        server.shutdown()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "hao_citeseq_pbmc_rna.h5ad" in _MockZenodo.store
    meta = _MockZenodo.store["metadata"]["metadata"]
    assert meta["upload_type"] == "dataset" and meta["creators"][0]["name"] == "Ashford, Andrew J."
    assert json.loads((out / "zenodo.json").read_text())["deposition_id"] == 42
