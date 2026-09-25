"""Download and safely extract the official BEIR SciFact archive."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARCHIVE = DATA_DIR / "scifact.zip"
TARGET = DATA_DIR / "scifact"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
EXPECTED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    actual = md5(ARCHIVE)
    if actual != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: expected {EXPECTED_MD5}, got {actual}")
    if not TARGET.exists():
        with zipfile.ZipFile(ARCHIVE) as handle:
            safe_extract(handle, DATA_DIR)
    print(f"SciFact ready at {TARGET} (MD5 {actual})")


if __name__ == "__main__":
    main()
