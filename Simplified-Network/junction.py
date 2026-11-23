"""Two-ended Golgi junctions for BrainCell 0.2.0 point-current integration."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import brainunit as u
import numpy as np
from braincell._discretization.mechanism import merge_place_rules, normalize_place_rule
from braincell.filter import at
from braincell.mech import FunctionClamp

from cerebellar_reproduction.morphology_mapping import build_morphology_mapping


# Default of the DBBS parallel gap-junction mechanism (in nS).
CONDUCTANCE_NS = 1.0
RECORDING = "golgi_junction_current"


def reciprocal_contacts(cells: np.ndarray, compartments: np.ndarray) -> np.ndarray:
    """Convert mirrored directed rows into physical contacts with two endpoints."""
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


@dataclass(frozen=True)
class Junction:
    """A set of symmetric ohmic contacts bound to the solver's electrical points."""

    endpoints: np.ndarray  # Contact, endpoint, (source ID, BSB compartment).
    population_indices: np.ndarray
    point_ids: np.ndarray

    def _clamp(self, cell, contact: int, direction: int) -> FunctionClamp:
        # Capture the contact and sign in closure cells so BrainCell's layout
        # fingerprint distinguishes the two callbacks. Both read the same
        # pre-step voltages before any cell's membrane state advances.
        def current(_t, junction=self, owner=cell):
            # Version-specific bridge: the point voltage includes electrical
            # nodes that do not coincide with a control-volume midpoint.
            voltage = owner._point_voltage_for_mechanisms()[
                junction.population_indices[contact], junction.point_ids[contact]
            ]
            return (direction * CONDUCTANCE_NS * u.nS * (voltage[1] - voltage[0])).in_unit(u.nA)

        return FunctionClamp(current)

    def _place(self, cell, locations) -> None:
        # The 0.2.0 public per-member place() only accepts synapses. Lower
        # these clamps as member-specific point rules in one grid rebuild.
        rules = []
        declarations = []
        for row, pair in enumerate(locations):
            for side, location in enumerate(pair):
                clamp = self._clamp(cell, row, 1 if side == 0 else -1)
                member = int(self.population_indices[row, side])
                rules.append(normalize_place_rule(location, (clamp,), population_indices=(member,)))
                declarations.append(clamp)
        previous = cell.place_rules
        cell._place_rules = merge_place_rules(previous, tuple(rules))
        try:
            cell.discretize()
        except Exception:
            cell._place_rules = previous
            raise

        clamps = cell.clamps
        by_declaration = {record.declaration: record for record in clamps.instances}
        recording_ids = []
        for offset, declaration in enumerate(declarations):
            record = by_declaration[declaration]
            self.point_ids.flat[offset] = record.point_id
            recording_ids.append(record.id)
        self.point_ids.flags.writeable = False
        clamps.by_id(recording_ids).record(RECORDING)

    @classmethod
    def install(cls, cell, spec: Path, morphologies: Path) -> "Junction":
        with np.load(spec / "connectivity/gap_goc.npz", allow_pickle=False) as data:
            endpoints = reciprocal_contacts(data["cell_pairs"], data["compartment_pairs"])
        with np.load(spec / "populations/golgi_cell.npz", allow_pickle=False) as data:
            source_ids = np.sort(data["source_ids"])
        source_indices = np.searchsorted(source_ids, endpoints[..., 0])
        if not np.isin(endpoints[..., 0], source_ids).all():
            raise ValueError("Junction contacts contain unknown Golgi cells.")
        if not len(endpoints):
            raise ValueError("No mirrored Golgi junction contacts were loaded.")

        point_ids = np.full(source_indices.shape, -1, dtype=np.int64)
        junction = cls(endpoints, source_indices, point_ids)
        mapping = build_morphology_mapping(morphologies, "GolgiCell", cell.morphology)
        locations = [
            [at(location.branch_id, location.x) for location in mapping.map_compartments(pair)]
            for pair in endpoints[..., 1]
        ]
        junction._place(cell, locations)
        return junction

    def metadata(self) -> dict:
        within_cell = self.endpoints[:, 0, 0] == self.endpoints[:, 1, 0]
        same_point = within_cell & (self.point_ids[:, 0] == self.point_ids[:, 1])
        return {
            "runtime_electrical_coupling": True,
            "physical_contacts": len(self.endpoints),
            "intercellular_contacts": int((~within_cell).sum()),
            "intracellular_contacts": int(within_cell.sum()),
            "same_electrical_point_contacts": int(same_point.sum()),
            "conductance_nS_per_contact": CONDUCTANCE_NS,
            "integration": "explicit_start_of_step_point_voltage",
        }
