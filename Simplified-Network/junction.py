"""Map the extracted Golgi contacts into the native BrainCell Junction API."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import brainunit as u
import numpy as np


# Default of the DBBS parallel gap-junction mechanism (in nS).
CONDUCTANCE_NS = 1.0
CV_VOLTAGE_RECORDING = "golgi_junction_cv_v"


def reciprocal_contacts(cells: np.ndarray, compartments: np.ndarray) -> np.ndarray:
    """Convert mirrored rows into physical contacts, retaining multiplicities."""
    rows = Counter(
        ((int(a), int(x)), (int(b), int(y)))
        for (a, b), (x, y) in zip(cells, compartments, strict=True)
    )
    pairs = []
    for (a, b), count in sorted(rows.items()):
        if a == b or count != rows[b, a]:
            raise ValueError(f"Expected distinct, reciprocally mirrored junction endpoints: {a}, {b}.")
        if a < b:
            pairs.extend([(a, b)] * count)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2, 2)


def mapped_contacts(morphology, spec: Path, morphologies: Path):
    """Read contact endpoints, member indices, branches and branch coordinates."""
    from cerebellar_reproduction.morphology_mapping import build_morphology_mapping

    with np.load(spec / "connectivity/gap_goc.npz", allow_pickle=False) as data:
        endpoints = reciprocal_contacts(data["cell_pairs"], data["compartment_pairs"])
    with np.load(spec / "populations/golgi_cell.npz", allow_pickle=False) as data:
        source_ids = np.sort(data["source_ids"])
    if not len(endpoints):
        raise ValueError("No mirrored Golgi junction contacts were loaded.")
    if not np.isin(endpoints[..., 0], source_ids).all():
        raise ValueError("Junction contacts contain unknown Golgi cells.")
    source_indices = np.searchsorted(source_ids, endpoints[..., 0])
    mapping = build_morphology_mapping(morphologies, "GolgiCell", morphology)
    locations = mapping.map_compartments(endpoints[..., 1].ravel())
    branches = np.asarray([location.branch_id for location in locations]).reshape(source_indices.shape)
    x = np.asarray([location.x for location in locations]).reshape(source_indices.shape)
    return endpoints, source_indices, branches, x


@dataclass(frozen=True)
class NativeJunction:
    """Native contacts and the solver CV indices used for post-step diagnostics."""

    endpoints: np.ndarray
    population_indices: np.ndarray
    cv_ids: np.ndarray
    voltage_shape: tuple[int, int]

    @classmethod
    def install(cls, network, cell, spec: Path, morphologies: Path) -> "NativeJunction":
        from braincell import mech, observe
        from braincell.event import _resolve_cell_location_cvs
        from braincell.filter import LocsetMask

        endpoints, members, branches, x = mapped_contacts(cell.morpho, spec, morphologies)
        ends = []
        cv_ids = []
        for side in range(2):
            locations = LocsetMask.from_columns(branches[:, side], x[:, side])
            ends.append(mech.JunctionEndpoint(cell, locations, members[:, side]))
            # Use the same ordered resolver as the native junction graph.
            cv_ids.append(_resolve_cell_location_cvs(cell, locations))
        network.connect_junction(
            "gap_goc", endpoint=ends[0],
            junction=mech.Junction(params={"g": CONDUCTANCE_NS * u.nS}, partner=ends[1]),
        )
        cell.record(CV_VOLTAGE_RECORDING, observe.state("v"))
        return cls(endpoints, members, np.stack(cv_ids, axis=-1), (cell.pop_size[0], cell.n_cv))

    def endpoint_voltage(self, sample, steps: int) -> np.ndarray:
        voltage = np.asarray(sample.values.to_decimal(u.mV)).reshape(steps, *self.voltage_shape)
        return voltage[:, self.population_indices, self.cv_ids]

    def metadata(self) -> dict:
        within_cell = self.endpoints[:, 0, 0] == self.endpoints[:, 1, 0]
        same_cv = within_cell & (self.cv_ids[:, 0] == self.cv_ids[:, 1])
        return {
            "runtime_electrical_coupling": bool(np.any(~same_cv)),
            "physical_contacts": len(self.endpoints),
            "intercellular_contacts": int((~within_cell).sum()),
            "intracellular_contacts": int(within_cell.sum()),
            "same_electrical_point_contacts": int(same_cv.sum()),
            "conductance_nS_per_contact": CONDUCTANCE_NS,
            "integration": "native_coupled_implicit_cable_voltage",
            "current_measurement": "reconstructed_from_post_step_cv_voltages",
        }
