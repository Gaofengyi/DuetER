"""Download, verify, and safely extract selected official BEIR archives."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATASETS = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "fiqa": "17918ed23cd04fb15047f73e6c3bd9d9",
    "trec-covid": "ce62140cb23feb9becf6270d0d1fe6d1",
    "msmarco": "444067daf65d982533ea17ebd59501e4",
    "nq": "d4d3d2e48787a744b6f6e691ff534307",
    "hotpotqa": "f412724f78b0d91183a0e86805e16114",
}


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


def parallel_download(url: str, archive: Path, workers: int) -> None:
    """Resume an archive with independent HTTP byte ranges and atomic assembly."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        total = int(response.headers["Content-Length"])
        accepts_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    prefix = archive.stat().st_size if archive.exists() else 0
    if prefix == total:
        return
    if prefix > total:
        raise RuntimeError(f"local archive is larger than the remote object: {archive}")
    if not accepts_ranges:
        if prefix:
            raise RuntimeError(f"server cannot resume the partial archive: {archive}")
        urllib.request.urlretrieve(url, archive)
        return

    remaining = total - prefix
    part_count = min(max(1, workers), remaining)
    chunk = (remaining + part_count - 1) // part_count
    ranges: list[tuple[int, int, Path]] = []
    for number in range(part_count):
        start = prefix + number * chunk
        end = min(total - 1, start + chunk - 1)
        if start > end:
            break
        ranges.append((start, end, archive.with_name(f"{archive.name}.part{number:02d}")))

    def fetch(item: tuple[int, int, Path]) -> Path:
        start, end, part = item
        expected = end - start + 1
        for attempt in range(6):
            existing = part.stat().st_size if part.exists() else 0
            if existing > expected:
                raise RuntimeError(f"range part is too large: {part}")
            if existing == expected:
                return part
            resumed_start = start + existing
            ranged = urllib.request.Request(url, headers={"Range": f"bytes={resumed_start}-{end}"})
            try:
                with urllib.request.urlopen(ranged, timeout=60) as response, part.open("ab") as output:
                    if response.status != 206:
                        raise RuntimeError(f"server ignored byte range {resumed_start}-{end}")
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            except (OSError, urllib.error.URLError):
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))
        if part.stat().st_size != expected:
            raise RuntimeError(f"incomplete range {start}-{end}: {part.stat().st_size}/{expected}")
        print(f"Downloaded bytes {start}-{end}", flush=True)
        return part

    with concurrent.futures.ThreadPoolExecutor(max_workers=part_count) as executor:
        parts = list(executor.map(fetch, ranges))

    assembled = archive.with_name(f"{archive.name}.assembling")
    with assembled.open("wb") as output:
        if archive.exists():
            with archive.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if assembled.stat().st_size != total:
        raise RuntimeError(f"assembled size mismatch: {assembled.stat().st_size}/{total}")
    assembled.replace(archive)
    for part in parts:
        part.unlink()


def download(dataset: str, workers: int) -> None:
    expected = DATASETS[dataset]
    archive = DATA_DIR / f"{dataset}.zip"
    target = DATA_DIR / dataset
    url = f"{BASE_URL}/{dataset}.zip"
    print(f"Downloading/resuming {url} with {workers} range workers", flush=True)
    parallel_download(url, archive, workers)
    actual = md5(archive)
    if actual != expected:
        raise RuntimeError(
            f"{dataset}: MD5 mismatch: expected {expected}, got {actual}"
        )
    if not target.exists():
        with zipfile.ZipFile(archive) as handle:
            safe_extract(handle, DATA_DIR)
    required = [target / "corpus.jsonl", target / "queries.jsonl", target / "qrels" / "test.tsv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"{dataset}: incomplete extraction: {missing}")
    print(f"{dataset} ready at {target} (MD5 {actual})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(DATASETS),
        default=["nfcorpus", "fiqa", "trec-covid"],
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        download(dataset, args.workers)


if __name__ == "__main__":
    main()
