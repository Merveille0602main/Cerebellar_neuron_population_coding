"""Identity checks for the published De Schepper et al. network file."""

from __future__ import annotations

import hashlib
from pathlib import Path


PUBLISHED_SIZE = 129_879_036
PUBLISHED_MD5 = "f0e17f89718c05756eb07a5da77b5c4c"


def md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as reader:
        while chunk := reader.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_published_reference(path: str | Path) -> None:
    path = Path(path)
    size = path.stat().st_size
    if size != PUBLISHED_SIZE:
        raise ValueError(f"Reference size mismatch: {size}; expected {PUBLISHED_SIZE} bytes.")
    actual_md5 = md5(path)
    if actual_md5 != PUBLISHED_MD5:
        raise ValueError(f"Reference MD5 mismatch: {actual_md5}; expected {PUBLISHED_MD5}.")
