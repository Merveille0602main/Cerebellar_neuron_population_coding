#!/usr/bin/env python3
"""Compile exact GoC/GrC cell and paper DBBS synapse MOD sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--braincell-repo", type=Path, default=ROOT / "assets/braincell")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/reference/neuron_goc_mechanisms",
    )
    parser.add_argument(
        "--gaba-mod",
        type=Path,
        default=(
            ROOT
            / "data/reference/dbbs_mod_collection"
            / "glia__dbbs_mod_collection__GABA__granule.mod"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild an existing output directory.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    library = output / "x86_64/libnrnmech.so"
    if output.exists():
        if not args.force:
            if library.exists():
                print(json.dumps({"library": str(library), "reused": True}, indent=2))
                return
            raise SystemExit(f"Output exists without a compiled library: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    model_root = args.braincell_repo / "examples/neuron_compare/Cerebellum_mod"
    sources = sorted((model_root / "GoC/channel").glob("*.mod"))
    sources.extend(
        (
            model_root / "GoC/ion/CdpStC_MA20_GoC.mod",
            model_root / "GrC/synapse/GRANULE_Ampa_det_vi.mod",
            model_root / "GrC/synapse/GRANULE_Nmda_det_vi.mod",
            args.gaba_mod,
        )
    )
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise SystemExit(f"Missing MOD sources: {missing}")

    nrnivmodl = Path(sys.executable).with_name("nrnivmodl")
    if not nrnivmodl.exists():
        located = shutil.which("nrnivmodl")
        if located is None:
            raise SystemExit("nrnivmodl is unavailable; install the NEURON validation extra.")
        nrnivmodl = Path(located)
    subprocess.run(
        [str(nrnivmodl), *(str(path.resolve()) for path in sources)],
        cwd=output,
        check=True,
    )
    if not library.exists():
        raise SystemExit(f"nrnivmodl completed without creating {library}")

    import neuron

    manifest = {
        "neuron_version": neuron.__version__,
        "library": str(library),
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in sources
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
