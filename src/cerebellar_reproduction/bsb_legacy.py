"""Reader for the BSB 3.x HDF5 layout used by the published network.

The 2022 reference file stores its data below ``/cells`` rather than in the
newer BSB storage layout.  This module intentionally depends only on h5py so
that extraction is reproducible without installing the historical BSB stack.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from .model import Connectivity, NetworkSpec, Population


# The source HDF5 exposes only the connection tag and global cell IDs.  These
# definitions retain the paper's semantic source/target types, including the
# entity-based mossy-fiber input population.
PAPER_CONNECTION_POPULATIONS = {
    "mossy_to_glomerulus": ("mossy_fibers", "glomerulus"),
    "glomerulus_to_granule": ("glomerulus", "granule_cell"),
    "golgi_to_glomerulus": ("golgi_cell", "glomerulus"),
    "glomerulus_to_golgi": ("glomerulus", "golgi_cell"),
    "golgi_to_granule": ("golgi_cell", "granule_cell"),
    "golgi_to_golgi": ("golgi_cell", "golgi_cell"),
    "gap_goc": ("golgi_cell", "golgi_cell"),
    "ascending_axon_to_golgi": ("granule_cell", "golgi_cell"),
    "parallel_fiber_to_golgi": ("granule_cell", "golgi_cell"),
    "ascending_axon_to_purkinje": ("granule_cell", "purkinje_cell"),
    "parallel_fiber_to_purkinje": ("granule_cell", "purkinje_cell"),
    "parallel_fiber_to_basket": ("granule_cell", "basket_cell"),
    "parallel_fiber_to_stellate": ("granule_cell", "stellate_cell"),
    "stellate_to_purkinje": ("stellate_cell", "purkinje_cell"),
    "basket_to_purkinje": ("basket_cell", "purkinje_cell"),
    "stellate_to_stellate": ("stellate_cell", "stellate_cell"),
    "basket_to_basket": ("basket_cell", "basket_cell"),
}


class BSBLayoutError(ValueError):
    """The input does not look like a readable published BSB reconstruction."""


def read_published_network(path: str | Path) -> NetworkSpec:
    """Read placements and all readable contact tables from a BSB 3.x file."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        _require_legacy_layout(handle)
        populations = _read_populations(handle)
        connectivity = _read_connectivity(handle, populations)
    spec = NetworkSpec(populations=populations, connectivity=connectivity, source_path=str(path))
    spec.validate()
    return spec


def _require_legacy_layout(handle: h5py.File) -> None:
    required = ("cells/placement", "cells/connections")
    missing = [name for name in required if name not in handle]
    if missing:
        raise BSBLayoutError("Not a supported BSB 3.x network; missing " + ", ".join(missing))


def _read_populations(handle: h5py.File) -> dict[str, Population]:
    """Read canonical simulation IDs from each placement group's identifier span."""
    if "cells/placement" in handle:
        populations: dict[str, Population] = {}
        for name, placement in handle["cells/placement"].items():
            identifiers = _as_integer_array(placement["identifiers"], f"placement {name} identifiers")
            if identifiers.shape != (2,):
                raise BSBLayoutError(f"placement {name} identifiers must contain [start, count].")
            start, count = map(int, identifiers)
            source_ids = np.arange(start, start + count, dtype=np.int64)
            positions = None
            if "positions" in placement:
                positions = np.asarray(placement["positions"], dtype=float)
                if positions.shape != (count, 3):
                    raise BSBLayoutError(
                        f"placement {name} positions has shape {positions.shape}; expected {(count, 3)}."
                    )
            populations[name] = Population(name=name, source_ids=source_ids, positions=positions)
        return populations

    # Fallback for early BSB 3.x files without per-type placement groups.
    positions = np.asarray(handle["cells/positions"])
    if positions.ndim != 2 or positions.shape[1] < 5:
        raise BSBLayoutError("/cells/positions must contain global ID, type, and xyz columns.")
    by_id = {int(row[0]): np.asarray(row[-3:], dtype=float) for row in positions}
    populations: dict[str, Population] = {}
    for dataset_name, dataset in handle["cells/type_maps"].items():
        if not dataset_name.endswith("_map"):
            continue
        name = dataset_name.removesuffix("_map")
        source_ids = _as_integer_array(dataset, f"type map {dataset_name}")
        cell_positions = [by_id.get(int(cell_id)) for cell_id in source_ids]
        known = all(position is not None for position in cell_positions)
        populations[name] = Population(
            name=name,
            source_ids=source_ids,
            positions=np.asarray(cell_positions, dtype=float) if known and len(source_ids) else None,
        )
    return populations


def _read_connectivity(
    handle: h5py.File, populations: dict[str, Population]
) -> dict[str, Connectivity]:
    result: dict[str, Connectivity] = {}
    pairs_group = handle["cells/connections"]
    compartments_group = handle.get("cells/connection_compartments")
    morphology_group = handle.get("cells/connection_morphologies")
    for name, dataset in pairs_group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        try:
            cell_pairs = _as_integer_pairs(dataset, f"connection {name}")
        except OSError as exc:
            raise BSBLayoutError(f"Cannot read connection table {name!r}; input may be truncated.") from exc
        pre_name, post_name = _connection_populations(name, cell_pairs, populations)
        _ensure_entity_population(populations, pre_name, cell_pairs[:, 0])
        _ensure_entity_population(populations, post_name, cell_pairs[:, 1])
        result[name] = Connectivity(
            name=name,
            pre_population=pre_name,
            post_population=post_name,
            cell_pairs=cell_pairs,
            compartment_pairs=_optional_pairs(compartments_group, name),
            morphology_pairs=_optional_pairs(morphology_group, name),
        )
    return result


def _connection_populations(
    name: str, pairs: np.ndarray, populations: dict[str, Population]
) -> tuple[str, str]:
    if name in PAPER_CONNECTION_POPULATIONS:
        return PAPER_CONNECTION_POPULATIONS[name]
    known = {
        cell_id: pop_name
        for pop_name, population in populations.items()
        for cell_id in population.source_ids
    }
    pre_names = Counter(known.get(int(cell_id)) for cell_id in pairs[:, 0])
    post_names = Counter(known.get(int(cell_id)) for cell_id in pairs[:, 1])
    if len(pre_names) == len(post_names) == 1 and None not in pre_names and None not in post_names:
        return next(iter(pre_names)), next(iter(post_names))
    raise BSBLayoutError(f"Cannot infer populations for unrecognised connection set {name!r}.")


def _ensure_entity_population(
    populations: dict[str, Population], name: str, ids: np.ndarray
) -> None:
    if name in populations and populations[name].size:
        return
    populations[name] = Population(name=name, source_ids=np.unique(ids), positions=None)


def _optional_pairs(group: h5py.Group | None, name: str) -> np.ndarray | None:
    if group is None or name not in group:
        return None
    return _as_integer_pairs(group[name], f"{group.name}/{name}")


def _as_integer_array(dataset: h5py.Dataset, context: str) -> np.ndarray:
    value = np.asarray(dataset)
    if value.ndim != 1:
        raise BSBLayoutError(f"{context} must be one-dimensional.")
    return _coerce_integers(value, context)


def _as_integer_pairs(dataset: h5py.Dataset, context: str) -> np.ndarray:
    value = np.asarray(dataset)
    if value.ndim != 2 or value.shape[1] != 2:
        raise BSBLayoutError(f"{context} must have shape (n, 2).")
    return _coerce_integers(value, context)


def _coerce_integers(value: np.ndarray, context: str) -> np.ndarray:
    if not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
        raise BSBLayoutError(f"{context} contains non-integer identifiers.")
    return value.astype(np.int64, copy=False)
