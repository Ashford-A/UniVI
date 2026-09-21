#!/usr/bin/env python
"""Upload a dataset produced by univi.datasets.export_dataset to a Zenodo draft.

    set ZENODO_TOKEN=...            (Windows)   /   export ZENODO_TOKEN=...   (Linux/macOS)
    python scripts/zenodo_upload.py zenodo/hao_citeseq_pbmc --creator "Ashford, Andrew J." \
        --orcid 0000-0002-1234-2118 [--sandbox] [--deposition-id 1234567] [--publish]

Create a token at https://zenodo.org/account/settings/applications/tokens/new/
with the ``deposit:write`` (and, to publish from here, ``deposit:actions``) scopes.
Use --sandbox (https://sandbox.zenodo.org, separate account and token) for a dry run.

Without --publish the record stays a draft: review it on zenodo.org and press
Publish there. Publishing cannot be undone (files of a published record can only
change through a new version). After publishing, register the files with:

    python scripts/prepare_zenodo_release.py register zenodo/hao_citeseq_pbmc --record-id <id>

The deposition id is also written to zenodo.json in the dataset directory.
"""
import argparse
import html
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PAPER_DOI = "10.1101/gr.281431.125"


def _request(method, url, token, *, data=None, body_path=None, content_type="application/json"):
    headers = {"Authorization": f"Bearer {token}"}
    if body_path is not None:
        size = os.path.getsize(body_path)
        headers.update({"Content-Type": "application/octet-stream", "Content-Length": str(size)})
        fh = open(body_path, "rb")
        req = urllib.request.Request(url, data=fh, method=method, headers=headers)
    else:
        fh = None
        payload = None if data is None else json.dumps(data).encode()
        if payload is not None:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=payload, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            text = resp.read().decode() or "{}"
            return json.loads(text)
    except urllib.error.HTTPError as err:
        raise SystemExit(f"{method} {url} failed: HTTP {err.code}\n{err.read().decode(errors='replace')[:2000]}")
    finally:
        if fh is not None:
            fh.close()


def _description_html(markdown_text):
    """Minimal Markdown to HTML: paragraphs, and code blocks kept verbatim."""
    parts, in_code, buf = [], False, []
    for line in markdown_text.splitlines():
        if line.startswith("```"):
            if in_code:
                parts.append("<pre>" + html.escape("\n".join(buf)) + "</pre>")
                buf = []
            in_code = not in_code
            continue
        if in_code:
            buf.append(line)
        elif line.startswith("#"):
            parts.append(f"<p><strong>{html.escape(line.lstrip('# '))}</strong></p>")
        elif line.strip():
            parts.append(f"<p>{html.escape(line)}</p>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir", type=Path, help="directory written by export_dataset (has manifest.json)")
    parser.add_argument("--creator", action="append", required=True, help='"Family, Given" (repeatable)')
    parser.add_argument("--orcid", action="append", default=[], help="ORCID per --creator, in order")
    parser.add_argument("--affiliation", action="append", default=[])
    parser.add_argument("--title", help="record title (default: manifest title)")
    parser.add_argument("--license", default="cc-by-4.0")
    parser.add_argument("--deposition-id", help="upload into an existing draft instead of creating one")
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
    parser.add_argument("--token-env", default="ZENODO_TOKEN")
    parser.add_argument("--publish", action="store_true", help="publish immediately (irreversible)")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Set the {args.token_env} environment variable to a Zenodo access token.")
    base = args.base_url or ("https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api")
    manifest = json.loads((args.outdir / "manifest.json").read_text())
    description = (args.outdir / "DESCRIPTION.md").read_text() if (args.outdir / "DESCRIPTION.md").exists() else ""

    if args.deposition_id:
        dep = _request("GET", f"{base}/deposit/depositions/{args.deposition_id}", token)
    else:
        dep = _request("POST", f"{base}/deposit/depositions", token, data={})
        print(f"created draft deposition {dep['id']}")
    bucket = dep["links"]["bucket"]

    for key, f in manifest["files"].items():
        path = args.outdir / f["filename"]
        print(f"uploading {path.name} ({f['size_mb']} MB) ...", flush=True)
        res = _request("PUT", f"{bucket}/{path.name}", token, body_path=path)
        remote = str(res.get("checksum", "")).replace("md5:", "")
        if remote and remote != f["md5"]:
            raise SystemExit(f"checksum mismatch for {path.name}: local {f['md5']}, Zenodo {remote}")
        print(f"  ok (md5 {f['md5']})")

    creators = []
    for i, name in enumerate(args.creator):
        c = {"name": name}
        if i < len(args.orcid) and args.orcid[i]:
            c["orcid"] = args.orcid[i]
        if i < len(args.affiliation) and args.affiliation[i]:
            c["affiliation"] = args.affiliation[i]
        creators.append(c)
    metadata = {
        "upload_type": "dataset",
        "title": args.title or f"UniVI dataset: {manifest.get('title') or manifest['name']}",
        "creators": creators,
        "description": _description_html(description) or manifest["name"],
        "license": args.license,
        "access_right": "open",
        "keywords": ["single-cell", "multi-omics", "UniVI", "AnnData"],
        "related_identifiers": [{"identifier": PAPER_DOI, "relation": "isSupplementTo", "scheme": "doi"}],
    }
    dep = _request("PUT", f"{base}/deposit/depositions/{dep['id']}", token, data={"metadata": metadata})
    (args.outdir / "zenodo.json").write_text(json.dumps({"deposition_id": dep["id"], "base": base}, indent=2))
    print(f"metadata set; draft: {dep['links'].get('html', '')}")

    if args.publish:
        dep = _request("POST", f"{base}/deposit/depositions/{dep['id']}/actions/publish", token)
        print(f"published: DOI {dep.get('doi')}  record {dep.get('record_id', dep['id'])}")
    else:
        print("Review the draft on Zenodo and publish it there (or rerun with --publish).")
    print(f"Then: python scripts/prepare_zenodo_release.py register {args.outdir} "
          f"--record-id {dep.get('record_id', dep['id'])}")


if __name__ == "__main__":
    sys.exit(main())
