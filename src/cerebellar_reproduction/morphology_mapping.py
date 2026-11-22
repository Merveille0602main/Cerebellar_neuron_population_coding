"""Map legacy BSB morphology compartments to BrainCell locations.

Legacy BSB stores a compartment as a flat segment index over morphology
branches.  BrainCell addresses a point by ``(branch_id, normalized_x)`` and
then lowers that point to the cell model's control-volume discretization.
This module bridges those two coordinate systems without treating a BSB
compartment identifier as a BrainCell synapse-pool index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class BSBBranch:
    """One branch loaded from the legacy BSB HDF5 morphology repository."""

    branch_id: int
    labels: tuple[str, ...]
    points_um: np.ndarray
    compartment_start: int

    @property
    def n_compartment(self) -> int:
        return max(0, int(len(self.points_um)) - 1)


@dataclass(frozen=True)
class MappedLocation:
    """A BSB segment midpoint expressed as a BrainCell branch location."""

    compartment_id: int
    bsb_branch_id: int
    bsb_segment_index: int
    labels: tuple[str, ...]
    branch_id: int
    branch_name: str
    x: float
    mapping_mode: str

    @property
    def point(self) -> tuple[int, float]:
        return (self.branch_id, self.x)


@dataclass(frozen=True)
class MorphologyMapping:
    """Mapping result for one named morphology."""

    morphology_name: str
    transform_name: str
    branch_matches: dict[int, int]
    locations: dict[int, MappedLocation]
    unmapped_compartments: tuple[int, ...]

    def location(self, compartment_id: int) -> MappedLocation:
        try:
            return self.locations[int(compartment_id)]
        except KeyError as exc:
            raise KeyError(
                f"BSB compartment {compartment_id!r} is not mapped for "
                f"{self.morphology_name!r}."
            ) from exc

    def map_compartments(self, compartment_ids: Iterable[int]) -> tuple[MappedLocation, ...]:
        return tuple(self.location(int(value)) for value in compartment_ids)


_TRANSFORMS = {
    "identity": np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    ),
    "rotate_z_minus_90": np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    ),
    "rotate_z_plus_90": np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    ),
    "rotate_z_180": np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    ),
}


def build_morphology_mapping(
    hdf5_path: str | Path,
    morphology_name: str,
    braincell_morphology,
    *,
    tolerance_um: float = 1e-3,
) -> MorphologyMapping:
    """Map all possible BSB compartments for one BrainCell morphology.

    Rigid z-axis rotations are tested because the published BSB repository
    stores some ASC morphologies in their placement orientation.  Translation
    is removed by centering the BrainCell morphology at the first soma point.
    GranuleCell is handled separately: the published scaffold used
    ``NoGeometry`` with four synthetic dendrites and a synthetic aa/pf, while
    the reduced template contains only soma and dendrites. Use anatomical
    coordinates for the full template and dendrite ordinals for the reduced one.
    """

    branches = _load_bsb_branches(hdf5_path, morphology_name)
    if morphology_name == "GranuleCell" and not any(
        branch.name == "ascending_axon" for branch in braincell_morphology.branches
    ):
        return _build_granule_mapping(branches, braincell_morphology)

    bc_branches = tuple(braincell_morphology.branches)
    origin = _points_um(bc_branches[0])[0]
    centered = {
        int(branch.index): _points_um(branch) - origin
        for branch in bc_branches
    }

    best = None
    for transform_name, matrix in _TRANSFORMS.items():
        transformed = {
            branch_id: points @ matrix
            for branch_id, points in centered.items()
        }
        matches = _match_branches(
            branches,
            bc_branches,
            transformed,
            tolerance_um=tolerance_um,
        )
        score = sum(branch.n_compartment for branch in branches if branch.branch_id in matches)
        candidate = (score, len(matches), transform_name, matches)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    assert best is not None
    _, _, transform_name, matches = best
    locations: dict[int, MappedLocation] = {}
    unmapped: list[int] = []
    bc_by_id = {int(branch.index): branch for branch in bc_branches}
    for branch in branches:
        match = matches.get(branch.branch_id)
        if match is None:
            unmapped.extend(_branch_compartment_ids(branch))
            continue
        branch_id, reversed_orientation = match
        target = bc_by_id[branch_id]
        for segment_index, compartment_id, x in _segment_midpoints(branch):
            if reversed_orientation:
                x = 1.0 - x
            locations[compartment_id] = MappedLocation(
                compartment_id=compartment_id,
                bsb_branch_id=branch.branch_id,
                bsb_segment_index=segment_index,
                labels=branch.labels,
                branch_id=branch_id,
                branch_name=str(target.name),
                x=float(np.clip(x, 0.0, 1.0)),
                mapping_mode="exact_geometry",
            )

    return MorphologyMapping(
        morphology_name=morphology_name,
        transform_name=transform_name,
        branch_matches={source: target[0] for source, target in matches.items()},
        locations=locations,
        unmapped_compartments=tuple(unmapped),
    )


def target_mapping_summary(
    mapping: MorphologyMapping,
    target_compartments: np.ndarray,
) -> dict[str, object]:
    """Return contact-level and unique-compartment mapping coverage."""

    values = np.asarray(target_compartments, dtype=np.int64)
    unique = np.unique(values)
    available = set(mapping.locations)
    mapped_mask = np.fromiter((int(value) in available for value in values), dtype=bool, count=len(values))
    unique_mapped = [int(value) for value in unique if int(value) in available]
    modes = sorted({mapping.locations[value].mapping_mode for value in unique_mapped})
    labels = sorted(
        {
            label
            for value in unique_mapped
            for label in mapping.locations[value].labels
        }
    )
    return {
        "n_contact": int(len(values)),
        "n_contact_mapped": int(mapped_mask.sum()),
        "contact_coverage": float(mapped_mask.mean()) if len(values) else 1.0,
        "n_unique_compartment": int(len(unique)),
        "n_unique_compartment_mapped": int(len(unique_mapped)),
        "unique_compartment_coverage": float(len(unique_mapped) / len(unique)) if len(unique) else 1.0,
        "mapping_modes": modes,
        "target_labels": labels,
        "unmapped_target_compartments": [int(value) for value in unique if int(value) not in available],
    }


def _load_bsb_branches(hdf5_path: str | Path, morphology_name: str) -> tuple[BSBBranch, ...]:
    with h5py.File(Path(hdf5_path), "r") as handle:
        group = handle[f"morphologies/{morphology_name}/branches"]
        output = []
        compartment_start = 0
        for branch_id in sorted(map(int, group.keys())):
            branch = group[str(branch_id)]
            points = np.column_stack((branch["x"][:], branch["y"][:], branch["z"][:])).astype(float)
            labels = tuple(_as_text(value) for value in branch.attrs.get("branch_labels", ()))
            item = BSBBranch(
                branch_id=branch_id,
                labels=labels,
                points_um=points,
                compartment_start=compartment_start,
            )
            output.append(item)
            compartment_start += item.n_compartment
    return tuple(output)


def _match_branches(
    source_branches: tuple[BSBBranch, ...],
    target_branches: tuple,
    transformed_points: dict[int, np.ndarray],
    *,
    tolerance_um: float,
) -> dict[int, tuple[int, bool]]:
    forward: dict[tuple, list[int]] = {}
    reverse: dict[tuple, list[int]] = {}
    for branch in target_branches:
        branch_id = int(branch.index)
        points = transformed_points[branch_id]
        forward.setdefault(_point_key(points, tolerance_um), []).append(branch_id)
        reverse.setdefault(_point_key(points[::-1], tolerance_um), []).append(branch_id)

    matches: dict[int, tuple[int, bool]] = {}
    used: set[int] = set()
    for source in source_branches:
        if len(source.points_um) == 0:
            continue
        key = _point_key(source.points_um, tolerance_um)
        candidates = [value for value in forward.get(key, ()) if value not in used]
        reversed_orientation = False
        if not candidates:
            candidates = [value for value in reverse.get(key, ()) if value not in used]
            reversed_orientation = True
        if len(candidates) != 1:
            continue
        target_id = candidates[0]
        used.add(target_id)
        matches[source.branch_id] = (target_id, reversed_orientation)
    return matches


def _build_granule_mapping(
    branches: tuple[BSBBranch, ...],
    braincell_morphology,
) -> MorphologyMapping:
    target_branches = tuple(braincell_morphology.branches)
    soma = next((branch for branch in target_branches if branch.type == "soma"), None)
    dendrites = sorted(
        (branch for branch in target_branches if branch.type == "dendrite"),
        key=lambda branch: int(branch.index),
    )
    if soma is None or len(dendrites) != 4:
        raise ValueError(
            "GranuleCell semantic mapping requires one soma and four BrainCell dendrites."
        )

    locations: dict[int, MappedLocation] = {}
    matches: dict[int, int] = {}
    unmapped: list[int] = []
    bsb_dendrites = [branch for branch in branches if "dendrites" in branch.labels]
    if len(bsb_dendrites) != 4:
        raise ValueError("Published GranuleCell morphology must contain four synthetic dendrites.")

    for branch in branches:
        if "soma" in branch.labels:
            target = soma
            mode = "semantic_soma"
        elif "dendrites" in branch.labels:
            target = dendrites[bsb_dendrites.index(branch)]
            mode = "semantic_dendrite_ordinal"
        else:
            unmapped.extend(_branch_compartment_ids(branch))
            continue
        matches[branch.branch_id] = int(target.index)
        for segment_index, compartment_id, x in _segment_midpoints(branch):
            locations[compartment_id] = MappedLocation(
                compartment_id=compartment_id,
                bsb_branch_id=branch.branch_id,
                bsb_segment_index=segment_index,
                labels=branch.labels,
                branch_id=int(target.index),
                branch_name=str(target.name),
                x=float(np.clip(x, 0.0, 1.0)),
                mapping_mode=mode,
            )

    return MorphologyMapping(
        morphology_name="GranuleCell",
        transform_name="semantic_no_geometry",
        branch_matches=matches,
        locations=locations,
        unmapped_compartments=tuple(unmapped),
    )


def _segment_midpoints(branch: BSBBranch):
    if branch.n_compartment == 0:
        return
    lengths = np.linalg.norm(np.diff(branch.points_um, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 0.0:
        xs = (np.arange(branch.n_compartment, dtype=float) + 0.5) / branch.n_compartment
    else:
        starts = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
        xs = (starts + 0.5 * lengths) / total
    for segment_index, x in enumerate(xs):
        yield (
            segment_index,
            branch.compartment_start + segment_index,
            float(x),
        )


def _branch_compartment_ids(branch: BSBBranch) -> range:
    return range(branch.compartment_start, branch.compartment_start + branch.n_compartment)


def _points_um(branch) -> np.ndarray:
    points = branch.points
    if hasattr(points, "to_decimal"):
        import brainunit as u

        points = points.to_decimal(u.um)
    return np.asarray(points, dtype=float)


def _point_key(points: np.ndarray, tolerance_um: float) -> tuple:
    decimals = max(0, int(round(-np.log10(tolerance_um))))
    rounded = np.round(np.asarray(points, dtype=float), decimals=decimals)
    rounded[np.abs(rounded) < tolerance_um / 2.0] = 0.0
    return (rounded.shape, tuple(rounded.ravel()))


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf8")
    return str(value)
