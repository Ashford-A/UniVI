"""Verify release metadata and that wheel/sdist include every package source file.

Run from the checkout: python scripts/verify_release.py dist/0.5.0
Uses only the standard library; does not import UniVI or require Torch.
"""
from pathlib import Path
from email.parser import BytesParser
import argparse
import ast
import re
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = re.search(r'^version\s*=\s*"([^"]+)"',
                        (root / 'pyproject.toml').read_text(encoding="utf-8"), re.MULTILINE).group(1)
    tree = ast.parse((root / 'univi/__init__.py').read_text(encoding="utf-8"))
    runtime_version = next(ast.literal_eval(n.value) for n in tree.body
                           if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    if runtime_version != version:
        raise SystemExit(f'Version mismatch: pyproject={version}, __init__={runtime_version}')
    wheel = args.directory / f'univi-{version}-py3-none-any.whl'
    sdist = args.directory / f'univi-{version}.tar.gz'
    actual = set(args.directory.glob('*.whl')) | set(args.directory.glob('*.tar.gz'))
    if actual != {wheel, sdist}:
        raise SystemExit('Expected exactly the wheel and sdist for this version in the release directory.')
    sources = {str(p.relative_to(root)).replace('\\', '/'): p.read_bytes()
               for p in (root / 'univi').rglob('*.py')}
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(archive.read(f'univi-{version}.dist-info/METADATA'))
        if metadata['Name'] != 'univi' or metadata['Version'] != version:
            raise SystemExit('Wheel metadata name/version mismatch.')
        if not any(re.match(r'joblib\s*[>=]', r) for r in metadata.get_all('Requires-Dist', [])):
            raise SystemExit('The wheel is missing its joblib runtime dependency.')
        for name, content in sources.items():
            if archive.read(name) != content:
                raise SystemExit(f'Wheel contains missing/stale source: {name}')
    with tarfile.open(sdist, 'r:gz') as archive:
        prefix = f'univi-{version}/'
        metadata = BytesParser().parsebytes(archive.extractfile(prefix + 'PKG-INFO').read())
        if metadata['Version'] != version:
            raise SystemExit('Source distribution version mismatch.')
        for name, content in sources.items():
            if archive.extractfile(prefix + name).read() != content:
                raise SystemExit(f'Source distribution contains stale source: {name}')
    print(f'UniVI {version}: wheel/sdist metadata and all {len(sources)} Python source files match.')


if __name__ == '__main__':
    main()
