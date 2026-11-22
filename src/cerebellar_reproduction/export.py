"""Portable on-disk export of a :class:`~.model.NetworkSpec`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import NetworkSpec
from .reference import md5


FORMAT_VERSION = 1


def write_spec(spec: NetworkSpec, destination: str | Path) -> dict:
    """Write manifest plus lossless NumPy tables, returning the manifest."""
    spec.validate()
    destination = Path(destination)
    populations_dir = destination / "populations"
    connectivity_dir = destination / "connectivity"
    populations_dir.mkdir(parents=True, exist_ok=True)
    connectivity_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "braincell-cerebellar-network-spec",
        "format_version": FORMAT_VERSION,
        "source_path": spec.source_path,
        "populations": {},
        "connectivity": {},
    }
    if spec.source_path is not None and Path(spec.source_path).is_file():
        source = Path(spec.source_path)
        manifest["source_size"] = source.stat().st_size
        manifest["source_md5"] = md5(source)
    for name, population in sorted(spec.populations.items()):
        relative = Path("populations") / f"{name}.npz"
        arrays = {"source_ids": population.source_ids}
        if population.positions is not None:
            arrays["positions_um"] = population.positions
        np.savez_compressed(destination / relative, **arrays)
        manifest["populations"][name] = {"file": str(relative), "n_cell": population.size}
    for name, connection in sorted(spec.connectivity.items()):
        relative = Path("connectivity") / f"{name}.npz"
        local_pairs = np.column_stack((
            spec.populations[connection.pre_population].local_indices(connection.cell_pairs[:, 0]),
            spec.populations[connection.post_population].local_indices(connection.cell_pairs[:, 1]),
        ))
        edge_pairs, contact_to_edge = np.unique(local_pairs, axis=0, return_inverse=True)
        arrays = {
            "cell_pairs": connection.cell_pairs,
            "local_pairs": local_pairs,
            "edge_pairs": edge_pairs.astype(np.int32, copy=False),
            "contact_to_edge": contact_to_edge.astype(np.int32, copy=False),
        }
        if connection.compartment_pairs is not None:
            arrays["compartment_pairs"] = connection.compartment_pairs
        if connection.morphology_pairs is not None:
            arrays["morphology_pairs"] = connection.morphology_pairs
        np.savez_compressed(destination / relative, **arrays)
        manifest["connectivity"][name] = {
            "file": str(relative),
            "pre_population": connection.pre_population,
            "post_population": connection.post_population,
            "n_edge": int(len(edge_pairs)),
            "n_contact": connection.size,
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
