#!/usr/bin/env python3
"""Download the published reference network with strict HTTP range checks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import Request, urlopen

from cerebellar_reproduction.reference import PUBLISHED_MD5, PUBLISHED_SIZE, md5


URL = "https://zenodo.org/api/records/7230288/files/balanced.hdf5/content"
EXPECTED_SIZE = PUBLISHED_SIZE
EXPECTED_MD5 = PUBLISHED_MD5
CHUNK_SIZE = 8 * 1024 * 1024


def _fetch_range(url: str, start: int, end: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=120) as response:
        expected_range = f"bytes {start}-{end}/{EXPECTED_SIZE}"
        actual_range = response.headers.get("Content-Range")
        if response.status != 206 or actual_range != expected_range:
            raise RuntimeError(
                f"Unexpected range response: status={response.status}, "
                f"Content-Range={actual_range!r}; expected {expected_range!r}."
            )
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError(f"Range {start}-{end} returned {len(data)} bytes.")
    return data


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    existing = temporary.stat().st_size if temporary.exists() else 0
    if not 0 <= existing <= EXPECTED_SIZE:
        raise RuntimeError(f"Refusing to resume invalid partial file of {existing} bytes: {temporary}")
    mode = "ab" if existing else "wb"
    with temporary.open(mode) as writer:
        for start in range(existing, EXPECTED_SIZE, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, EXPECTED_SIZE) - 1
            writer.write(_fetch_range(URL, start, end))
            writer.flush()
            print(f"downloaded {end + 1:,}/{EXPECTED_SIZE:,} bytes")
    digest = md5(temporary)
    if digest != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch for {temporary}: {digest}; expected {EXPECTED_MD5}.")
    os.replace(temporary, destination)
    print(f"verified {destination} ({EXPECTED_MD5})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    download(parser.parse_args().destination)


if __name__ == "__main__":
    main()
