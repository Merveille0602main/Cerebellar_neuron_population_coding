#!/usr/bin/env python3
"""Fetch the exact paper-era DBBS ``GABA.granule`` MOD source."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen


COMMIT = "8d21c4a4f888f19a6531224f1eb08d6c833c44e5"
SOURCE_NAME = "glia__dbbs_mod_collection__GABA__granule.mod"
URL = (
    "https://raw.githubusercontent.com/dbbs-lab/dbbs-mod-collection/"
    f"{COMMIT}/dbbs_mod_collection/mod/{SOURCE_NAME}"
)
EXPECTED_SHA256 = "f99a72c520860e01d77ca2d34b4f7bade4796a65b6abcb5c6f88f5d79695bfae"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=root / "data/reference/dbbs_mod_collection" / SOURCE_NAME,
    )
    args = parser.parse_args()

    with urlopen(URL, timeout=120) as response:
        data = response.read()
    actual = sha256(data)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch: {actual}; expected {EXPECTED_SHA256}."
        )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(args.destination.suffix + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, args.destination)
    print(f"verified {args.destination} ({actual}, commit {COMMIT})")


if __name__ == "__main__":
    main()
