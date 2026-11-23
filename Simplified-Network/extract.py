#!/usr/bin/env python3
"""Extract the original seed-318 benchmark subnetwork from balanced.hdf5."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np

from cerebellar_reproduction.bsb_legacy import read_published_network
from cerebellar_reproduction.export import write_spec
from cerebellar_reproduction.granular_layer import GOC, GRC, POPULATIONS, ROUTES
from cerebellar_reproduction.model import NetworkSpec
from cerebellar_reproduction.reference import PUBLISHED_MD5, verify_published_reference
from cerebellar_reproduction.validation import validate_export


ROOT = Path(__file__).resolve().parent
SEED = 318
SAMPLED_COUNTS = {GRC: 256, GOC: 12}
SOURCE = "https://doi.org/10.5281/zenodo.7230288"


def extract(source: Path, destination: Path) -> dict:
    verify_published_reference(source)
    original = read_published_network(source)
    rng = np.random.default_rng(SEED)
    populations = {name: original.populations[name] for name in POPULATIONS}
    for name, count in SAMPLED_COUNTS.items():
        population = populations[name]
        selected = np.sort(rng.choice(np.sort(population.source_ids), count, replace=False))
        indices = population.local_indices(selected)
        populations[name] = replace(
            population,
            source_ids=selected,
            positions=population.positions[indices],
        )

    connectivity = {}
    for route in ROUTES:
        connection = original.connectivity[route.name]
        pairs = connection.cell_pairs
        keep = np.isin(pairs[:, 0], populations[route.pre].source_ids) & np.isin(
            pairs[:, 1], populations[route.post].source_ids
        )
        connectivity[route.name] = replace(
            connection,
            cell_pairs=pairs[keep],
            compartment_pairs=None if connection.compartment_pairs is None else connection.compartment_pairs[keep],
            morphology_pairs=None if connection.morphology_pairs is None else connection.morphology_pairs[keep],
        )

    spec = NetworkSpec(populations=populations, connectivity=connectivity, source_path=SOURCE)
    spec_path = destination / "spec"
    manifest = write_spec(spec, spec_path)
    manifest["source_md5"] = PUBLISHED_MD5
    manifest["selection"] = {
        "algorithm": "numpy.default_rng.choice_without_replacement_on_sorted_source_ids",
        "seed": SEED,
        "sample_order": [GRC, GOC],
        "sampled_population_sizes": SAMPLED_COUNTS,
    }
    (spec_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_export(spec_path)

    with h5py.File(source, "r") as original_file, h5py.File(destination / "morphologies.hdf5", "w") as output:
        morphologies = output.create_group("morphologies")
        for name in ("GranuleCell", "GolgiCell"):
            original_file.copy(f"morphologies/{name}", morphologies)
        output.attrs["source"] = SOURCE
        output.attrs["source_md5"] = PUBLISHED_MD5
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="published balanced.hdf5")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    extract(args.source, args.output)
    print(f"Extracted and validated {args.output / 'spec'}")


if __name__ == "__main__":
    main()
