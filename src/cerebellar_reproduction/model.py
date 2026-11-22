"""Simulator-neutral in-memory representation of a reconstructed network."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Population:
    """One placed cell population with stable source IDs and coordinates."""

    name: str
    source_ids: np.ndarray
    positions: np.ndarray | None = None
    rotations: np.ndarray | None = None
    morphology: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.source_ids.ndim != 1:
            raise ValueError("Population source_ids must be one-dimensional.")
        if self.positions is not None and self.positions.shape != (len(self.source_ids), 3):
            raise ValueError("Population positions must have shape (n_cell, 3).")
        if len(np.unique(self.source_ids)) != len(self.source_ids):
            raise ValueError("Population source_ids must be unique.")

    @property
    def size(self) -> int:
        return int(len(self.source_ids))

    def local_indices(self, source_ids: np.ndarray) -> np.ndarray:
        """Convert this population's stable BSB IDs to BrainCell indices."""
        values = np.asarray(source_ids, dtype=np.int64)
        order = np.argsort(self.source_ids)
        sorted_ids = self.source_ids[order]
        found = np.searchsorted(sorted_ids, values)
        valid = found < len(sorted_ids)
        valid[valid] &= sorted_ids[found[valid]] == values[valid]
        if not valid.all():
            unknown = np.unique(values[~valid])[:5]
            raise ValueError(f"Unknown source IDs in population {self.name!r}: {unknown.tolist()}")
        return order[found].astype(np.int32, copy=False)


@dataclass(frozen=True)
class Connectivity:
    """A BSB connection set preserved as one row per anatomical contact."""

    name: str
    pre_population: str
    post_population: str
    cell_pairs: np.ndarray
    compartment_pairs: np.ndarray | None = None
    morphology_pairs: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.cell_pairs.ndim != 2 or self.cell_pairs.shape[1] != 2:
            raise ValueError("cell_pairs must have shape (n_contact, 2).")
        if self.compartment_pairs is not None and self.compartment_pairs.shape != self.cell_pairs.shape:
            raise ValueError("compartment_pairs must have shape (n_contact, 2).")
        if self.morphology_pairs is not None and self.morphology_pairs.shape != self.cell_pairs.shape:
            raise ValueError("morphology_pairs must have shape (n_contact, 2).")

    @property
    def size(self) -> int:
        return int(len(self.cell_pairs))


@dataclass(frozen=True)
class NetworkSpec:
    """Placed populations and exact anatomical-contact tables."""

    populations: dict[str, Population] = field(default_factory=dict)
    connectivity: dict[str, Connectivity] = field(default_factory=dict)
    source_path: str | None = None

    def validate(self) -> None:
        for name, connection in self.connectivity.items():
            if name != connection.name:
                raise ValueError(f"Connectivity map key {name!r} differs from its name.")
            try:
                pre = self.populations[connection.pre_population]
                post = self.populations[connection.post_population]
            except KeyError as exc:
                raise ValueError(f"{name} references an unknown population.") from exc
            for role, ids, population in (
                ("pre", connection.cell_pairs[:, 0], pre),
                ("post", connection.cell_pairs[:, 1], post),
            ):
                if not np.isin(ids, population.source_ids).all():
                    raise ValueError(f"{name} has {role} cell IDs outside its population.")
