"""Published granular-layer circuit on the BrainCell 0.2 direct network API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

import braincell as bc
from braincell import mech
from braincell.filter import LocsetMask, at, branch_in
import brainunit as u
import numpy as np

from .morphology_mapping import build_morphology_mapping
from . import paper_synapses as receptors  # Register the DBBS point mechanisms.


@dataclass(frozen=True)
class Route:
    name: str
    pre: str
    post: str
    mechanisms: tuple[type[bc.Synapse], ...] = ()


MF = "mossy_fibers"
GLOM = "glomerulus"
GRC = "granule_cell"
GOC = "golgi_cell"
POPULATIONS = (MF, GLOM, GRC, GOC)
GAP_ROUTE = Route("gap_goc", GOC, GOC)
ROUTES = (
    Route("mossy_to_glomerulus", MF, GLOM),
    Route("glomerulus_to_granule", GLOM, GRC, (receptors.GrCAMPA2020, receptors.GrCNMDA2020)),
    Route("glomerulus_to_golgi", GLOM, GOC, (receptors.GoCAMPA_MF2020, receptors.GoCNMDA2020)),
    Route("ascending_axon_to_golgi", GRC, GOC, (receptors.GoCAMPA_AA2020, receptors.GoCNMDA2020)),
    Route("parallel_fiber_to_golgi", GRC, GOC, (receptors.GoCAMPAPF2020,)),
    Route("golgi_to_granule", GOC, GRC, (receptors.GrCGABA2020,)),
    Route("golgi_to_golgi", GOC, GOC, (receptors.GoCGABAAlpha1,)),
    GAP_ROUTE,
)
REFERENCE_THRESHOLD = -20.0 * u.mV
# See GRANULAR_LAYER_0_2.md for the two NEURON connection mechanisms.
RELAY_DELAY = 0.0 * u.ms
CHEMICAL_DELAY = 1.0 * u.ms


@dataclass(frozen=True)
class Contacts:
    """One row per published anatomical contact."""

    cells: np.ndarray
    compartments: np.ndarray | None


def _load_contacts(spec: Path) -> dict[str, Contacts]:
    result = {}
    for route in ROUTES:
        with np.load(spec / "connectivity" / f"{route.name}.npz", allow_pickle=False) as archive:
            result[route.name] = Contacts(
                archive["cell_pairs"],
                archive["compartment_pairs"] if route.mechanisms or route is GAP_ROUTE else None,
            )
    return result


def _pilot_populations(contacts: Mapping[str, Contacts]) -> dict[str, np.ndarray]:
    # Select a real closed motif with both GrC axon outputs to the same GoC,
    # feedback to that GrC, and at least one shared glomerulus.
    aa = contacts["ascending_axon_to_golgi"].cells
    pf = contacts["parallel_fiber_to_golgi"].cells
    go_gr = contacts["golgi_to_granule"].cells
    gl_gr = contacts["glomerulus_to_granule"].cells
    gl_go = contacts["glomerulus_to_golgi"].cells
    mf_gl = contacts["mossy_to_glomerulus"].cells
    go_go = contacts["golgi_to_golgi"].cells
    partners = {}
    for source, target in np.unique(go_go, axis=0):
        if source != target:
            partners.setdefault(source, []).append(target)
    motifs = set(map(tuple, aa)) & set(map(tuple, pf)) & set(map(tuple, go_gr[:, ::-1]))
    for granule, golgi in sorted(motifs):
        glomeruli = np.unique(gl_gr[gl_gr[:, 1] == granule, 0])
        if golgi in partners and np.intersect1d(glomeruli, gl_go[gl_go[:, 1] == golgi, 0]).size:
            mossy = np.unique(mf_gl[np.isin(mf_gl[:, 1], glomeruli), 0])
            return {
                MF: mossy, GLOM: glomeruli, GRC: np.array([granule]),
                GOC: np.sort([golgi, partners[golgi][0]]),
            }
    raise ValueError("The supplied network has no complete granular-layer pilot motif.")


def _all_populations(spec: Path) -> dict[str, np.ndarray]:
    ids = {}
    for name in POPULATIONS:
        with np.load(spec / "populations" / f"{name}.npz", allow_pickle=False) as archive:
            ids[name] = np.sort(archive["source_ids"])
    return ids


def _subset(
    contacts: Mapping[str, Contacts],
    populations: Mapping[str, np.ndarray],
) -> dict[str, Contacts]:
    result = {}
    for route in ROUTES:
        entry = contacts[route.name]
        keep = np.isin(entry.cells[:, 0], populations[route.pre]) & np.isin(
            entry.cells[:, 1], populations[route.post]
        )
        result[route.name] = Contacts(
            entry.cells[keep],
            None if entry.compartments is None else entry.compartments[keep],
        )
    return result


def _locations(mapping, compartment_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids, inverse = np.unique(compartment_ids, return_inverse=True)
    locations = mapping.map_compartments(ids)
    branch = np.array([location.branch_id for location in locations], dtype=np.int64)
    pos = np.array([location.x for location in locations], dtype=float)
    return branch[inverse], pos[inverse]


def _default_schedule(ids: np.ndarray) -> dict[int, np.ndarray]:
    return {int(source): np.arange(1.0, 11.0) for source in ids}


def _source_events(network, populations, contacts, schedules):
    mossy = populations[MF]
    unknown = set(schedules).difference(mossy)
    if unknown:
        raise ValueError(f"Pulse schedule contains mossy IDs outside the selection: {sorted(unknown)}.")
    times = {}
    for source_id in mossy:
        row = np.asarray(schedules.get(source_id, ()), dtype=float)
        if row.ndim != 1 or not np.isfinite(row).all() or np.any(row < 0):
            raise ValueError(f"Mossy fiber {source_id} requires finite non-negative event times.")
        times[source_id] = np.unique(row)
    relay = contacts["mossy_to_glomerulus"].cells
    target_ids, count = np.unique(relay[:, 1], return_counts=True)
    if not np.array_equal(target_ids, populations[GLOM]) or not np.all(count == 1):
        raise ValueError("Expected exactly one published mossy parent for each retained glomerulus.")
    parent = dict(zip(relay[:, 1], relay[:, 0], strict=True))
    network.add_population(
        MF, bc.EventSequence.from_times([times[ident] * u.ms for ident in mossy])
    )
    return network.add_population(
        GLOM,
        bc.EventSequence.from_times(
            [times[parent[ident]] * u.ms for ident in populations[GLOM]]
        ),
    )


def _make_cells(braincell_assets, populations, temperature_celsius, v_init_mv):
    # The 0.2.0 wheel does not bundle model examples and morphology fixtures.
    # Let the installed package resolve first, and use the checkout for assets.
    assets = str(Path(braincell_assets).resolve())
    if assets not in sys.path:
        sys.path.append(assets)
    from examples.neuron_compare.cell.goc_ma2020.goc_braincell import GoC
    from examples.neuron_compare.cell.goc_ma2020.parameters import (
        axon_region_name, dend_region_name, load_goc20_params,
    )
    from .paper_granule_cell import PaperGranuleCell

    class PaperGolgiCell(GoC):
        def _define_regions(self):
            names = {"soma": ["soma"]}
            for kind, expected, classify in (
                ("dendrite", 151, dend_region_name),
                ("axon", 75, axon_region_name),
            ):
                branches = [branch for branch in self.morph.branches if branch.type == kind]
                if len(branches) != expected:
                    raise ValueError(f"Expected {expected} Golgi {kind} sections, got {len(branches)}.")
                for ordinal, branch in enumerate(branches):
                    names.setdefault(classify(ordinal), []).append(branch.name)
            self.regions = {key: branch_in("name", tuple(items)) for key, items in names.items()}

    common = {"temperature_celsius": temperature_celsius, "v_init_mV": v_init_mv}
    grc = PaperGranuleCell(pop_size=(len(populations[GRC]),), name=GRC, **common).build()
    goc = PaperGolgiCell(
        params=load_goc20_params(), pop_size=(len(populations[GOC]),), name=GOC, **common
    ).build()
    return {GRC: grc.cell, GOC: goc.cell}, {GRC: grc.morpho, GOC: goc.morph}


@dataclass(frozen=True)
class PlacedRoute:
    target: bc.Cell
    source: bc.EventSource
    source_index: np.ndarray
    receiver_index: np.ndarray
    target_indices: np.ndarray
    mechanisms: dict[str, str]


def _place_route(
    relay, cells, mapping, populations, route, path, *, temperature_celsius, receiver_mode,
) -> PlacedRoute:
    if not path.cells.size:
        raise ValueError(f"The selection contains no published contacts for {route.name}.")
    compartments = path.compartments
    assert compartments is not None
    pre_local = np.searchsorted(populations[route.pre], path.cells[:, 0])
    post_local = np.searchsorted(populations[route.post], path.cells[:, 1])
    branch, x = _locations(mapping[route.post], compartments[:, 1])
    target = cells[route.post]

    if route.pre == GLOM:
        pre_events = relay.model
    else:
        pre_branch, _ = _locations(mapping[route.pre], compartments[:, 0])
        source_branches, pre_offset = np.unique(pre_branch, return_inverse=True)
        pre_events = bc.VoltageCrossingSource(
            cells[route.pre],
            location=LocsetMask.from_columns(source_branches, np.full(len(source_branches), 0.5)),
            threshold=REFERENCE_THRESHOLD,
            name=f"{route.name}_axon",
        )
        pre_local = pre_local * len(source_branches) + pre_offset

    # Contact mode retains an independent release history and receptor at each
    # published contact. Section mode aggregates both states within this route.
    section_receivers = receiver_mode == "section" and route.pre != GLOM
    if section_receivers:
        place, point_index = np.unique(
            np.column_stack((post_local, branch)), axis=0, return_inverse=True
        )
        routing = np.unique(np.column_stack((pre_local, point_index)), axis=0)
        source_index, receiver_index = routing.T
        place_cells, place_branches = place.T
        place_x = np.full(len(place), 0.5)
    else:
        order = np.argsort(post_local, kind="stable")
        place_cells, place_branches, place_x = post_local[order], branch[order], x[order]
        source_index = pre_local[order]
        receiver_index = np.arange(len(order))
    post_ids, start = np.unique(place_cells, return_index=True)
    end = np.r_[start[1:], len(place_cells)]
    locations = [
        LocsetMask.from_columns(place_branches[a:b], place_x[a:b])
        for a, b in zip(start, end, strict=True)
    ]
    mechanisms = {}
    specs = []
    synapse_spec = getattr(mech, "SynapseSpec", mech.Synapse)
    for mechanism in route.mechanisms:
        suffix = "nmda" if "NMDA" in mechanism.__name__ else (
            "gaba" if "GABA" in mechanism.__name__ else "ampa"
        )
        name = f"{route.name}_{suffix}"
        specs.append(synapse_spec(mechanism.__name__, name=name, temperature_celsius=temperature_celsius))
        mechanisms[suffix] = name
    target[post_ids].place(locations, *specs)
    return PlacedRoute(target, pre_events, source_index, receiver_index, place_cells, mechanisms)


def _connect_route(network, route, placed, *, record_synapses):
    connections = {}
    sources = placed.source[placed.source_index]
    for suffix, name in placed.mechanisms.items():
        endpoints = placed.target.synapses[name]
        if not np.array_equal(endpoints.population_index, placed.target_indices):
            raise RuntimeError(f"{name}: native receptor count or ordering differs from the placement plan.")
        connection = network.connect(
            name,
            source=sources,
            synapse=endpoints[placed.receiver_index],
            delay=RELAY_DELAY if route.pre == GLOM else CHEMICAL_DELAY,
        )
        if not np.array_equal(connection.source_index, sources.source_id):
            raise RuntimeError(f"{name}: event source order differs from the published wiring.")
        if not np.array_equal(connection.synapse_id, endpoints.id[placed.receiver_index]):
            raise RuntimeError(f"{name}: native connection order differs from the planned receptor order.")
        if record_synapses:
            placed.target.record(name, bc.observe.synapse(name=name).state("g"))
        connections[suffix] = {"receivers": len(endpoints), "event_rows": len(connection)}
    return connections


def build_granular_layer(
    spec_path: str | Path,
    source_hdf5: str | Path,
    braincell_assets: str | Path,
    *,
    full: bool = False,
    plan_only: bool = False,
    mossy_spike_times_ms: Mapping[int, Sequence[float]] | None = None,
    temperature_celsius: float = 32.0,
    v_init_mv: float = -65.0,
    receiver_mode: str = "contact",
    exercise_circuit: bool = False,
):
    """Build the published granular-layer motif or all selected population rows.

    Return the native BrainCell network and its provenance summary. A plan
    applies the same selection and maps every selected presynaptic and
    postsynaptic compartment without materializing the runtime connection table.
    """
    if receiver_mode not in {"section", "contact"}:
        raise ValueError("receiver_mode must be 'section' or 'contact'.")
    if full and exercise_circuit:
        raise ValueError("The diagnostic current clamps are available only for the pilot.")
    spec_path = Path(spec_path)
    contacts = _load_contacts(spec_path)
    populations = _all_populations(spec_path) if full else _pilot_populations(contacts)
    contacts = _subset(contacts, populations)
    cells, morphologies = _make_cells(braincell_assets, populations, temperature_celsius, v_init_mv)
    mapping = {
        name: build_morphology_mapping(source_hdf5, morphology, morphologies[name])
        for name, morphology in [(GRC, "GranuleCell"), (GOC, "GolgiCell")]
    }
    counts = {}
    for route in ROUTES:
        path = contacts[route.name]
        if path.compartments is not None:
            _locations(mapping[route.post], path.compartments[:, 1])
            if route.pre in mapping:
                _locations(mapping[route.pre], path.compartments[:, 0])
        counts[route.name] = {"anatomical_contacts": len(path.cells)}
    counts[GAP_ROUTE.name]["runtime_electrical_coupling"] = False
    info = {
        "braincell_version": bc.__version__,
        "braincell_module": bc.__file__,
        "selection": "full" if full else "published_motif",
        "population_counts": {name: len(ids) for name, ids in populations.items()},
        "selected_global_ids": None if full else {name: ids.tolist() for name, ids in populations.items()},
        "control_volumes_per_cell": {name: cell.n_cv for name, cell in cells.items()},
        "morphology_branches_per_cell": {name: len(morph.branches) for name, morph in morphologies.items()},
        "receiver_mode": receiver_mode,
        "routes": counts,
        "mossy_to_glomerulus": "published device relay expanded to a fixed event table",
        "ascending_axon_ampa": "Masoli et al. (2020) ModelDB 266806 syn_AA2GoC_det substitute",
        "golgi_to_golgi_gaba": "DBBS alpha1 kinetics with 3200 pS nominal gmax from De Schepper et al. (2022)",
        "diagnostic_current_clamps": exercise_circuit,
        "excluded": {
            "golgi_to_glomerulus": "structural scaffold relation; feedback is on golgi_to_granule",
            "gap_goc": "contact endpoints are retained; two-terminal electrical current is not simulated",
        },
    }
    if plan_only:
        return None, info

    network = bc.Network("published_granular_layer")
    if mossy_spike_times_ms is None:
        mossy_spike_times_ms = _default_schedule(populations[MF])
    relay = _source_events(network, populations, contacts, mossy_spike_times_ms)
    if not full:
        for cell in cells.values():
            cell.soma.record("soma_v", bc.observe.state("v"))
    if exercise_circuit:
        # A deterministic activity check of the live axonal and feedback
        # pathways, separate from the original mossy-fiber stimulus protocol.
        for kind, amplitude in [(GRC, 0.4), (GOC, 0.7)]:
            cells[kind].place(
                at("soma", 0.5),
                mech.CurrentClamp(delay=2.0 * u.ms, durations=2.0 * u.ms, amplitudes=amplitude * u.nA),
            )
    for name, cell in cells.items():
        network.add_population(name, cell)
    chemical_routes = [route for route in ROUTES if route.mechanisms]
    placements = {
        route.name: _place_route(
            relay, cells, mapping, populations, route, contacts[route.name],
            temperature_celsius=temperature_celsius, receiver_mode=receiver_mode,
        )
        for route in chemical_routes
    }
    for route in chemical_routes:
        counts[route.name]["mechanisms"] = _connect_route(
            network, route, placements[route.name], record_synapses=not full,
        )
    return network, info
