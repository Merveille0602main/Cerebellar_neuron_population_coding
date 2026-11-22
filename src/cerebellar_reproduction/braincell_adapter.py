"""Lower the portable topology into BrainCell declarations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .validation import validate_export
from .morphology_mapping import MorphologyMapping


@dataclass(frozen=True)
class ProjectionBinding:
    """BrainCell synapse binding for one exported anatomical connection set.

    ``synapse_index`` is deliberately explicit: BSB compartment identifiers
    are not automatically equivalent to indices in a BrainCell placed-synapse
    pool. The caller must supply an array or a callable that performs the
    morphology-specific mapping.
    """

    synapse: str
    synapse_index: np.ndarray | Callable[[Mapping[str, np.ndarray]], np.ndarray]
    weight: object | None = None
    delay: object | None = None


@dataclass(frozen=True)
class PlacedSynapsePool:
    """A placed BrainCell synapse pool indexed by BSB target compartment."""

    synapse: str
    compartment_to_synapse_index: Mapping[int, int]
    n_active: int

    def contact_indices(self, target_compartments: np.ndarray) -> np.ndarray:
        values = np.asarray(target_compartments, dtype=np.int64)
        try:
            return np.fromiter(
                (self.compartment_to_synapse_index[int(value)] for value in values),
                dtype=np.int32,
                count=len(values),
            )
        except KeyError as exc:
            raise KeyError(
                f"Target compartment {exc.args[0]!r} is absent from synapse pool {self.synapse!r}."
            ) from exc


def place_mapped_synapse_pool(
    cell,
    mapping: MorphologyMapping,
    target_compartments: np.ndarray,
    synapse,
) -> PlacedSynapsePool:
    """Place one named synapse layout at all mapped target control volumes.

    Multiple BSB compartments can collapse onto one BrainCell control volume.
    The returned indices follow BrainCell's actual sparse runtime ordering, not
    the raw BSB compartment identifiers or the input-location order.
    """

    values = np.asarray(target_compartments, dtype=np.int64)
    unique_compartments = np.unique(values)
    locations = [mapping.location(int(value)) for value in unique_compartments]
    locset = _explicit_locset(tuple(location.point for location in locations))
    cell.place(locset, synapse)

    from braincell._discretization.geometry import locate_cv_on_branch

    discretization = cell._discretization
    point_id_by_compartment = {}
    for compartment, location in zip(unique_compartments, locations):
        branch_cv_ids = discretization.cv_tree.branch_to_cv_ids[location.branch_id]
        cv_id = locate_cv_on_branch(
            branch_cv_ids,
            discretization.cvs,
            x=location.x,
        )
        point_id = int(discretization.node_tree.cv_to_mid_node_id[cv_id])
        point_id_by_compartment[int(compartment)] = point_id

    active_point_ids = sorted(set(point_id_by_compartment.values()))
    synapse_index_by_point = {
        point_id: index for index, point_id in enumerate(active_point_ids)
    }
    compartment_to_synapse_index = {
        compartment: synapse_index_by_point[point_id]
        for compartment, point_id in point_id_by_compartment.items()
    }
    return PlacedSynapsePool(
        synapse=synapse.instance_name,
        compartment_to_synapse_index=compartment_to_synapse_index,
        n_active=len(active_point_ids),
    )


def _explicit_locset(points: tuple[tuple[int, float], ...]):
    from dataclasses import dataclass

    from braincell.filter import LocsetExpr, LocsetMask
    from braincell.filter.helper import normalize_locset_points

    @dataclass(frozen=True)
    class ExplicitLocations(LocsetExpr):
        points: tuple[tuple[int, float], ...]

        def evaluate(self, morpho, cache=None):
            _ = cache
            normalized = normalize_locset_points(self.points)
            names = tuple(
                f"{morpho.branch(index=branch_id).name}({x:g})"
                for branch_id, x in normalized
            )
            return LocsetMask(points=normalized, display_names=names)

    return ExplicitLocations(points=points)


def add_populations(network, spec_path: str | Path, cells: Mapping[str, object]) -> dict:
    """Add prebuilt homogeneous BrainCell cells after checking population sizes."""
    spec_path = Path(spec_path)
    manifest = validate_export(spec_path)
    added = {}
    for name, info in sorted(manifest["populations"].items()):
        if name not in cells:
            continue
        cell = cells[name]
        pop_size = tuple(getattr(cell, "pop_size", ()))
        expected = (info["n_cell"],)
        if pop_size != expected:
            raise ValueError(f"Cell {name!r} has pop_size={pop_size}; expected {expected}.")
        added[name] = network.add_population(name, cell)
    return added


def add_edge_sets(network, spec_path: str | Path, names: set[str] | None = None) -> dict:
    """Add exact unique cell-level edge tables for populations already present."""
    import braincell.network as network_api

    spec_path = Path(spec_path)
    manifest = validate_export(spec_path)
    available = set(network.populations)
    added = {}
    for name, info in sorted(manifest["connectivity"].items()):
        if names is not None and name not in names:
            continue
        if info["pre_population"] not in available or info["post_population"] not in available:
            continue
        with np.load(spec_path / info["file"]) as data:
            edge_pairs = data["edge_pairs"]
        added[name] = network.add_edges(
            name=name,
            pre=info["pre_population"],
            post=info["post_population"],
            method=network_api.pairs(edge_pairs),
        )
    return added


def add_projections(
    network,
    spec_path: str | Path,
    bindings: Mapping[str, ProjectionBinding],
) -> dict:
    """Expand exported anatomical contacts onto BrainCell synapse pools."""
    import braincell.network as network_api

    spec_path = Path(spec_path)
    manifest = validate_export(spec_path)
    added = {}
    for name, binding in bindings.items():
        info = manifest["connectivity"][name]
        with np.load(spec_path / info["file"]) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        synapse_index = (
            binding.synapse_index(data)
            if callable(binding.synapse_index)
            else np.asarray(binding.synapse_index)
        )
        if synapse_index.shape != (info["n_contact"],):
            raise ValueError(
                f"Projection {name!r} synapse_index has shape {synapse_index.shape}; "
                f"expected {(info['n_contact'],)}."
            )
        kwargs = {}
        if binding.delay is not None:
            kwargs["delay"] = binding.delay
        added[name] = network.add_projection(
            name=name,
            edges=name,
            synapse=binding.synapse,
            method=network_api.explicit_contacts(
                source_edge=data["contact_to_edge"],
                synapse_index=synapse_index,
            ),
            weight=binding.weight,
            **kwargs,
        )
    return added
