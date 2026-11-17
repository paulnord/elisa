"""Download the TESS products recorded in products.json into a local directory."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def verify(path, product):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != product["bytes"] or digest.hexdigest() != product["sha256"]:
        raise ValueError(f"{path} differs from the pinned MAST product {product['uri']}")


def fetch(product, directory):
    path = directory / product["filename"]
    if path.exists():
        verify(path, product)
        return path
    url = "https://mast.stsci.edu/api/v0.1/Download/file?" + urlencode({"uri": product["uri"]})
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        partial = Path(temporary) / path.name
        with urlopen(url, timeout=90) as source, partial.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        verify(partial, product)
        partial.replace(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="local directory for the full FITS files")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = Path(__file__).with_name("products.json")
    for item in json.loads(manifest.read_text())["products"]:
        print(fetch(item, args.directory), flush=True)
