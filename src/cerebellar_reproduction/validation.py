"""Consistency checks for an exported portable network specification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validate_export(path: str | Path) -> dict:
    """Validate all manifest counts, index bounds, and edge/contact mappings."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    populations = manifest["populations"]
    for name, info in populations.items():
        with np.load(path / info["file"]) as data:
            source_ids = data["source_ids"]
            if source_ids.shape != (info["n_cell"],):
                raise ValueError(f"Population {name!r} count mismatch.")
            if len(np.unique(source_ids)) != len(source_ids):
                raise ValueError(f"Population {name!r} has duplicate source IDs.")
            if "positions_um" in data and data["positions_um"].shape != (info["n_cell"], 3):
                raise ValueError(f"Population {name!r} position shape mismatch.")
    for name, info in manifest["connectivity"].items():
        with np.load(path / info["file"]) as data:
            local_pairs = data["local_pairs"]
            edge_pairs = data["edge_pairs"]
            contact_to_edge = data["contact_to_edge"]
            if local_pairs.shape != (info["n_contact"], 2):
                raise ValueError(f"Connectivity {name!r} contact count mismatch.")
            if edge_pairs.shape != (info["n_edge"], 2):
                raise ValueError(f"Connectivity {name!r} edge count mismatch.")
            if contact_to_edge.shape != (info["n_contact"],):
                raise ValueError(f"Connectivity {name!r} contact mapping shape mismatch.")
            if len(edge_pairs) and not np.array_equal(edge_pairs[contact_to_edge], local_pairs):
                raise ValueError(f"Connectivity {name!r} edge/contact mapping is not lossless.")
            pre_size = populations[info["pre_population"]]["n_cell"]
            post_size = populations[info["post_population"]]["n_cell"]
            if len(edge_pairs) and (
                edge_pairs[:, 0].min() < 0
                or edge_pairs[:, 0].max() >= pre_size
                or edge_pairs[:, 1].min() < 0
                or edge_pairs[:, 1].max() >= post_size
            ):
                raise ValueError(f"Connectivity {name!r} has out-of-bounds local indices.")
    return manifest
