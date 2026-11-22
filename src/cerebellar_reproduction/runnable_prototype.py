"""Small runnable BrainCell network backed by one published BSB contact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import MethodType
from typing import Mapping, Sequence

import numpy as np

from .braincell_adapter import place_mapped_synapse_pool
from .morphology_mapping import build_morphology_mapping


@dataclass(frozen=True)
class PrototypeMetadata:
    connection: str
    source_contact_index: int
    source_pre_id: int
    source_post_id: int
    target_compartment: int
    target_branch_id: int
    target_branch_name: str
    target_x: float
    synapse_index: int
    note: str


@dataclass(frozen=True)
class SubnetworkMetadata:
    """Published identifiers and sizes retained by the runnable pilot."""

    connection: str
    source_pre_ids: tuple[int, ...]
    source_post_ids: tuple[int, ...]
    n_edge: int
    n_contact: int
    target_compartments: tuple[int, ...]
    synapse_mechanisms: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class InputDrivenSubnetworkMetadata:
    """Source IDs and relay topology retained by the input-driven pilot."""

    source_mossy_ids: tuple[int, ...]
    source_glomerulus_ids: tuple[int, ...]
    source_granule_ids: tuple[int, ...]
    mossy_to_glomerulus_edges: tuple[tuple[int, int], ...]
    n_glomerulus_to_granule_edge: int
    n_glomerulus_to_granule_contact: int
    mossy_spike_times_ms: tuple[tuple[float, ...], ...]
    glomerulus_parent_mossy_ids: tuple[int, ...]
    relay_mode: str
    synapse_mechanisms: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class GolgiSubnetworkMetadata:
    """Published identifiers retained by a glomerulus→GoC pilot."""

    source_glomerulus_ids: tuple[int, ...]
    source_golgi_ids: tuple[int, ...]
    n_edge: int
    n_contact: int
    target_compartments: tuple[int, ...]
    target_region_counts: tuple[tuple[str, int], ...]
    glomerulus_spike_times_ms: tuple[tuple[float, ...], ...]
    synapse_mechanisms: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class SharedInputSubnetworkMetadata:
    """Published IDs retained when one MF schedule drives GrC and GoC."""

    granule_pathway: InputDrivenSubnetworkMetadata
    source_golgi_ids: tuple[int, ...]
    n_glomerulus_to_golgi_edge: int
    n_glomerulus_to_golgi_contact: int
    golgi_target_compartments: tuple[int, ...]
    golgi_target_region_counts: tuple[tuple[str, int], ...]
    golgi_synapse_mechanisms: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class ChemicalLoopSubnetworkMetadata:
    """Published IDs retained by the no-gap granular-layer chemical loop."""

    shared_input: SharedInputSubnetworkMetadata
    n_golgi_to_granule_edge: int
    n_golgi_to_granule_contact: int
    gaba_target_compartments: tuple[int, ...]
    gaba_synapse_mechanism: str
    spike_output_mode: str
    note: str


def build_glomerulus_granule_prototype(
    spec_path: str | Path,
    source_hdf5: str | Path,
    braincell_repo: str | Path,
):
    """Return a runnable one-contact glomerulus→Granule BrainCell network.

    This is an integration prototype, not a calibrated paper experiment.  It
    uses an explicit one-event schedule at t=0 and BrainCell's generic AMPA
    model to prove that a published contact can drive a real multi-compartmental
    GrC.
    """

    import braincell
    import brainunit as u
    from braincell import mech
    from braincell.filter import at

    braincell_repo = Path(braincell_repo)
    if str(braincell_repo) not in sys.path:
        sys.path.insert(0, str(braincell_repo))
    from examples.neuron_compare.cell.grc_ma2020.grc_braincell import GrC
    from examples.neuron_compare.cell.grc_ma2020.parameters import load_grc20_params

    spec_path = Path(spec_path)
    with np.load(spec_path / "connectivity/glomerulus_to_granule.npz") as data:
        source_pair = data["cell_pairs"][0]
        target_compartment = int(data["compartment_pairs"][0, 1])

    relay = _build_scheduled_spike_source(((0.0,),), name="prototype_glomerulus")

    granule = GrC(
        params=load_grc20_params(),
        temperature_celsius=32.0,
        pop_size=(1,),
        name="prototype_granule",
    ).build()
    _restrict_spikes_to_soma(granule.cell)
    mapping = build_morphology_mapping(
        source_hdf5,
        "GranuleCell",
        granule.morpho,
    )
    location = mapping.location(target_compartment)
    synapse = mech.Synapse(
        "AMPA",
        name="glomerulus_ampa",
        g_max=0.001 * (u.mS / u.cm**2),
    )
    pool = place_mapped_synapse_pool(
        granule.cell,
        mapping,
        np.asarray([target_compartment]),
        synapse,
    )
    synapse_index = int(pool.contact_indices(np.asarray([target_compartment]))[0])
    target = at(location.branch_id, location.x)
    granule.cell.place(
        target,
        mech.MechanismProbe(
            name="ampa_g",
            mechanism="glomerulus_ampa",
            field="g",
        ),
    )
    granule.cell.place(
        at("soma", 0.5),
        mech.StateProbe(name="v_soma", field="v"),
    )

    network = braincell.Network(name="published-contact-prototype")
    network.add_population("glomerulus", relay)
    network.add_population("granule_cell", granule.cell)
    network.add_edges(
        name="glomerulus_to_granule",
        pre="glomerulus",
        post="granule_cell",
        method=braincell.network.pairs([(0, 0)]),
    )
    network.add_projection(
        name="glomerulus_to_granule_ampa",
        edges="glomerulus_to_granule",
        synapse="glomerulus_ampa",
        method=braincell.network.explicit_contacts(
            source_edge=[0],
            synapse_index=[synapse_index],
        ),
        weight=1.0,
        delay=0.0 * u.ms,
    )
    metadata = PrototypeMetadata(
        connection="glomerulus_to_granule",
        source_contact_index=0,
        source_pre_id=int(source_pair[0]),
        source_post_id=int(source_pair[1]),
        target_compartment=target_compartment,
        target_branch_id=location.branch_id,
        target_branch_name=location.branch_name,
        target_x=location.x,
        synapse_index=synapse_index,
        note="Generic AMPA and deterministic relay; not yet paper-calibrated.",
    )
    return network, metadata


def build_glomerulus_granule_subnetwork(
    spec_path: str | Path,
    source_hdf5: str | Path,
    braincell_repo: str | Path,
    *,
    max_glomeruli: int = 4,
    max_granules: int = 16,
    paper_mechanisms: bool = False,
):
    """Build a small runnable subgraph selected from all published contacts.

    The selected global cell IDs and every retained contact come directly from
    ``glomerulus_to_granule.npz``.  IDs are remapped only inside this pilot so
    BrainCell receives compact population indices.
    """

    import braincell
    import brainunit as u
    from braincell import mech
    from braincell.filter import at

    braincell_repo = Path(braincell_repo)
    if str(braincell_repo) not in sys.path:
        sys.path.insert(0, str(braincell_repo))
    from examples.neuron_compare.cell.grc_ma2020.grc_braincell import GrC
    from examples.neuron_compare.cell.grc_ma2020.parameters import load_grc20_params

    if max_glomeruli <= 0 or max_granules <= 0:
        raise ValueError("max_glomeruli and max_granules must both be positive.")

    spec_path = Path(spec_path)
    with np.load(spec_path / "connectivity/glomerulus_to_granule.npz") as data:
        cell_pairs = np.asarray(data["cell_pairs"], dtype=np.int64)
        target_compartments_all = np.asarray(data["compartment_pairs"][:, 1], dtype=np.int64)

    source_pre_ids = np.unique(cell_pairs[:, 0])[:max_glomeruli]
    pre_mask = np.isin(cell_pairs[:, 0], source_pre_ids)
    source_post_ids = np.unique(cell_pairs[pre_mask, 1])[:max_granules]
    selected = pre_mask & np.isin(cell_pairs[:, 1], source_post_ids)
    selected_pairs = cell_pairs[selected]
    target_compartments = target_compartments_all[selected]
    if not len(selected_pairs):
        raise ValueError("The requested limits selected no glomerulus→Granule contacts.")

    pre_lookup = {int(value): index for index, value in enumerate(source_pre_ids)}
    post_lookup = {int(value): index for index, value in enumerate(source_post_ids)}
    local_contact_pairs = np.asarray(
        [
            (pre_lookup[int(pre)], post_lookup[int(post)])
            for pre, post in selected_pairs
        ],
        dtype=np.int32,
    )
    edge_pairs, contact_to_edge = np.unique(
        local_contact_pairs,
        axis=0,
        return_inverse=True,
    )

    relay = _build_scheduled_spike_source(
        tuple((0.0,) for _ in source_pre_ids),
        name="pilot_glomerulus",
    )
    granule = GrC(
        params=load_grc20_params(),
        temperature_celsius=32.0,
        pop_size=(len(source_post_ids),),
        name="pilot_granule",
    ).build()
    _restrict_spikes_to_soma(granule.cell)
    mapping = build_morphology_mapping(source_hdf5, "GranuleCell", granule.morpho)
    if paper_mechanisms:
        # Import performs BrainCell registry registration for both mechanisms.
        from . import paper_synapses as _paper_synapses  # noqa: F401

        synapses = (
            mech.Synapse(
                "GrCAMPA2020",
                name="glomerulus_ampa",
                g_max=0.0012 * u.uS,
                E_rev=0.0 * u.mV,
            ),
            mech.Synapse(
                "GrCNMDA2020",
                name="glomerulus_nmda",
                g_max=0.0188 * u.uS,
                E_rev=-3.7 * u.mV,
            ),
        )
    else:
        synapses = (
            mech.Synapse(
                "AMPA",
                name="glomerulus_ampa",
                g_max=0.001 * (u.mS / u.cm**2),
            ),
        )
    pools = {
        synapse.instance_name: place_mapped_synapse_pool(
            granule.cell,
            mapping,
            target_compartments,
            synapse,
        )
        for synapse in synapses
    }
    synapse_indices = {
        name: pool.contact_indices(target_compartments)
        for name, pool in pools.items()
    }
    unique_target_compartments = tuple(int(value) for value in np.unique(target_compartments))
    for compartment in unique_target_compartments:
        location = mapping.location(compartment)
        for synapse in synapses:
            granule.cell.place(
                at(location.branch_id, location.x),
                mech.MechanismProbe(
                    name=f"{synapse.instance_name.removeprefix('glomerulus_')}_g",
                    mechanism=synapse.instance_name,
                    field="g",
                ),
            )
    granule.cell.place(at("soma", 0.5), mech.StateProbe(name="v_soma", field="v"))

    network = braincell.Network(name="published-glomerulus-granule-pilot")
    network.add_population("glomerulus", relay)
    network.add_population("granule_cell", granule.cell)
    network.add_edges(
        name="glomerulus_to_granule",
        pre="glomerulus",
        post="granule_cell",
        method=braincell.network.pairs(edge_pairs),
    )
    for synapse in synapses:
        name = synapse.instance_name
        network.add_projection(
            name=f"glomerulus_to_granule_{name.removeprefix('glomerulus_')}",
            edges="glomerulus_to_granule",
            synapse=name,
            method=braincell.network.explicit_contacts(
                source_edge=contact_to_edge,
                synapse_index=synapse_indices[name],
            ),
            weight=1.0,
            delay=0.0 * u.ms,
        )
    metadata = SubnetworkMetadata(
        connection="glomerulus_to_granule",
        source_pre_ids=tuple(map(int, source_pre_ids)),
        source_post_ids=tuple(map(int, source_post_ids)),
        n_edge=int(len(edge_pairs)),
        n_contact=int(len(selected_pairs)),
        target_compartments=unique_target_compartments,
        synapse_mechanisms=tuple(synapse.synapse_type for synapse in synapses),
        note=(
            "Published topology and GrC AMPA/NMDA MOD port; explicit t=0 relay schedules."
            if paper_mechanisms
            else "Published subgraph; generic AMPA and explicit t=0 relay schedules; not calibrated."
        ),
    )
    return network, metadata


def build_mossy_glomerulus_granule_subnetwork(
    spec_path: str | Path,
    source_hdf5: str | Path,
    braincell_repo: str | Path,
    *,
    mossy_spike_times_ms: Mapping[int, Sequence[float]] | None = None,
    max_glomeruli: int = 8,
    max_granules: int = 32,
):
    """Build a runnable paper-topology mossy→glomerulus→Granule subnetwork.

    ``mossy_to_glomerulus`` is a relay branch in the BSB configuration, not a
    chemical synapse.  BrainCell currently has no cell-to-cell relay primitive,
    so this builder preserves the imported edge set and expands each mossy
    source's spike schedule onto its retained child glomeruli.  The resulting
    glomerulus events then enter the published GrC AMPA/NMDA mechanisms through
    the imported ``glomerulus_to_granule`` contacts.

    The default schedule is the paper configuration's sensory-burst pattern,
    shifted from 1100/1103/1106/1131/1151 ms to 1/4/7/32/52 ms so the pilot can
    run quickly.  Its four source IDs are taken from the paper configuration's
    synchronous/Poisson mossy-fiber device targets.
    """

    import braincell
    import brainunit as u
    from braincell import mech
    from braincell.filter import at

    braincell_repo = Path(braincell_repo)
    if str(braincell_repo) not in sys.path:
        sys.path.insert(0, str(braincell_repo))
    from examples.neuron_compare.cell.grc_ma2020.grc_braincell import GrC
    from examples.neuron_compare.cell.grc_ma2020.parameters import load_grc20_params

    if max_glomeruli <= 0 or max_granules <= 0:
        raise ValueError("max_glomeruli and max_granules must both be positive.")
    if mossy_spike_times_ms is None:
        mossy_spike_times_ms = {
            source_id: (1.0, 4.0, 7.0, 32.0, 52.0)
            for source_id in (213, 214, 222, 223)
        }
    source_mossy_ids = tuple(int(value) for value in mossy_spike_times_ms)
    if not source_mossy_ids:
        raise ValueError("mossy_spike_times_ms must contain at least one source ID.")
    if len(set(source_mossy_ids)) != len(source_mossy_ids):
        raise ValueError("mossy source IDs must be unique.")
    normalized_spike_times = tuple(
        _validate_spike_times(mossy_spike_times_ms[source_id], source_id=source_id)
        for source_id in source_mossy_ids
    )

    spec_path = Path(spec_path)
    with np.load(spec_path / "connectivity/mossy_to_glomerulus.npz") as data:
        mossy_glomerulus_pairs = np.asarray(data["cell_pairs"], dtype=np.int64)
    with np.load(spec_path / "connectivity/glomerulus_to_granule.npz") as data:
        glomerulus_granule_pairs = np.asarray(data["cell_pairs"], dtype=np.int64)
        target_compartments_all = np.asarray(
            data["compartment_pairs"][:, 1],
            dtype=np.int64,
        )

    known_mossy_ids = set(map(int, np.unique(mossy_glomerulus_pairs[:, 0])))
    unknown = tuple(value for value in source_mossy_ids if value not in known_mossy_ids)
    if unknown:
        raise ValueError(
            f"Mossy IDs {unknown!r} do not occur in mossy_to_glomerulus.npz. "
            "Expected the original global BSB IDs."
        )

    selected_mg_pairs = _round_robin_mossy_children(
        mossy_glomerulus_pairs,
        source_mossy_ids,
        max_glomeruli,
    )
    if not len(selected_mg_pairs):
        raise ValueError("The selected mossy fibers have no retained glomerulus children.")
    source_glomerulus_ids = tuple(map(int, selected_mg_pairs[:, 1]))
    parent_by_glomerulus = {
        int(glomerulus): int(mossy)
        for mossy, glomerulus in selected_mg_pairs
    }
    glomerulus_parent_mossy_ids = tuple(
        parent_by_glomerulus[glomerulus]
        for glomerulus in source_glomerulus_ids
    )

    gg_pre_mask = np.isin(glomerulus_granule_pairs[:, 0], source_glomerulus_ids)
    source_granule_ids_array = np.unique(
        glomerulus_granule_pairs[gg_pre_mask, 1]
    )[:max_granules]
    gg_selected = gg_pre_mask & np.isin(
        glomerulus_granule_pairs[:, 1],
        source_granule_ids_array,
    )
    selected_gg_pairs = glomerulus_granule_pairs[gg_selected]
    target_compartments = target_compartments_all[gg_selected]
    if not len(selected_gg_pairs):
        raise ValueError("The retained glomeruli have no Granule contacts.")
    source_granule_ids = tuple(map(int, source_granule_ids_array))

    mossy_lookup = {value: index for index, value in enumerate(source_mossy_ids)}
    glomerulus_lookup = {
        value: index for index, value in enumerate(source_glomerulus_ids)
    }
    granule_lookup = {value: index for index, value in enumerate(source_granule_ids)}
    local_mg_pairs = np.asarray(
        [
            (mossy_lookup[int(pre)], glomerulus_lookup[int(post)])
            for pre, post in selected_mg_pairs
        ],
        dtype=np.int32,
    )
    local_gg_contacts = np.asarray(
        [
            (glomerulus_lookup[int(pre)], granule_lookup[int(post)])
            for pre, post in selected_gg_pairs
        ],
        dtype=np.int32,
    )
    gg_edge_pairs, contact_to_edge = np.unique(
        local_gg_contacts,
        axis=0,
        return_inverse=True,
    )

    spike_times_by_mossy = dict(zip(source_mossy_ids, normalized_spike_times))
    glomerulus_spike_times = tuple(
        spike_times_by_mossy[parent]
        for parent in glomerulus_parent_mossy_ids
    )
    mossy = _build_scheduled_spike_source(
        normalized_spike_times,
        name="paper_mossy_fibers",
    )
    # This is an offline expansion of the fixed source schedule, not a native
    # BrainCell runtime event-forwarding relay.
    glomerulus = _build_scheduled_spike_source(
        glomerulus_spike_times,
        name="paper_glomerulus",
    )

    granule = GrC(
        params=load_grc20_params(),
        temperature_celsius=32.0,
        pop_size=(len(source_granule_ids),),
        name="paper_input_granule",
    ).build()
    _restrict_spikes_to_soma(granule.cell)
    mapping = build_morphology_mapping(source_hdf5, "GranuleCell", granule.morpho)
    from . import paper_synapses as _paper_synapses  # noqa: F401

    synapses = (
        mech.Synapse(
            "GrCAMPA2020",
            name="glomerulus_ampa",
            g_max=0.0012 * u.uS,
            E_rev=0.0 * u.mV,
        ),
        mech.Synapse(
            "GrCNMDA2020",
            name="glomerulus_nmda",
            g_max=0.0188 * u.uS,
            E_rev=-3.7 * u.mV,
        ),
    )
    pools = {
        synapse.instance_name: place_mapped_synapse_pool(
            granule.cell,
            mapping,
            target_compartments,
            synapse,
        )
        for synapse in synapses
    }
    synapse_indices = {
        name: pool.contact_indices(target_compartments)
        for name, pool in pools.items()
    }
    for compartment in map(int, np.unique(target_compartments)):
        location = mapping.location(compartment)
        for synapse in synapses:
            granule.cell.place(
                at(location.branch_id, location.x),
                mech.MechanismProbe(
                    name=f"{synapse.instance_name.removeprefix('glomerulus_')}_g",
                    mechanism=synapse.instance_name,
                    field="g",
                ),
            )
    granule.cell.place(at("soma", 0.5), mech.StateProbe(name="v_soma", field="v"))

    network = braincell.Network(name="paper-mossy-glomerulus-granule-pilot")
    network.add_population("mossy_fibers", mossy)
    network.add_population("glomerulus", glomerulus)
    network.add_population("granule_cell", granule.cell)
    network.add_edges(
        name="mossy_to_glomerulus",
        pre="mossy_fibers",
        post="glomerulus",
        method=braincell.network.pairs(local_mg_pairs),
    )
    network.add_edges(
        name="glomerulus_to_granule",
        pre="glomerulus",
        post="granule_cell",
        method=braincell.network.pairs(gg_edge_pairs),
    )
    for synapse in synapses:
        name = synapse.instance_name
        network.add_projection(
            name=f"glomerulus_to_granule_{name.removeprefix('glomerulus_')}",
            edges="glomerulus_to_granule",
            synapse=name,
            method=braincell.network.explicit_contacts(
                source_edge=contact_to_edge,
                synapse_index=synapse_indices[name],
            ),
            weight=1.0,
            delay=0.0 * u.ms,
        )

    metadata = InputDrivenSubnetworkMetadata(
        source_mossy_ids=source_mossy_ids,
        source_glomerulus_ids=source_glomerulus_ids,
        source_granule_ids=source_granule_ids,
        mossy_to_glomerulus_edges=tuple(
            (int(pre), int(post))
            for pre, post in selected_mg_pairs
        ),
        n_glomerulus_to_granule_edge=int(len(gg_edge_pairs)),
        n_glomerulus_to_granule_contact=int(len(selected_gg_pairs)),
        mossy_spike_times_ms=normalized_spike_times,
        glomerulus_parent_mossy_ids=glomerulus_parent_mossy_ids,
        relay_mode="topology-expanded spike schedule",
        synapse_mechanisms=tuple(synapse.synapse_type for synapse in synapses),
        note=(
            "Published global IDs, relay branching, GrC contacts, and AMPA/NMDA mechanisms. "
            "The BSB relay is represented by schedules expanded along mossy_to_glomerulus; "
            "the edge set is retained but has no synthetic chemical projection."
        ),
    )
    return network, metadata


def build_glomerulus_golgi_subnetwork(
    spec_path: str | Path,
    source_hdf5: str | Path,
    braincell_repo: str | Path,
    *,
    glomerulus_spike_times_ms: Mapping[int, Sequence[float]] | None = None,
    max_glomeruli: int = 4,
    max_golgi: int = 4,
):
    """Build a runnable published glomerulus→Golgi subgraph.

    Cell pairs, target compartments, and contact-to-edge ownership come from
    ``glomerulus_to_golgi.npz``.  If an explicit schedule is supplied, its
    keys are interpreted as original BSB glomerulus IDs.  Otherwise the first
    ``max_glomeruli`` connected glomeruli each emit one event at 1 ms.

    The mechanism names are GoC-specific aliases, but their equations are the
    ``AMPA.granule`` and ``NMDA.granule`` mechanisms selected by the paper-era
    ``dbbs_models.GolgiCell`` definition for ``AMPA_MF`` and ``NMDA``.
    """

    import braincell
    import brainunit as u
    from braincell import mech
    from braincell.filter import at

    braincell_repo = Path(braincell_repo)
    if str(braincell_repo) not in sys.path:
        sys.path.insert(0, str(braincell_repo))
    from examples.neuron_compare.cell.goc_ma2020.goc_braincell import GoC
    from examples.neuron_compare.cell.goc_ma2020.parameters import (
        dend_region_name,
        load_goc20_params,
    )

    if max_glomeruli <= 0 or max_golgi <= 0:
        raise ValueError("max_glomeruli and max_golgi must both be positive.")

    spec_path = Path(spec_path)
    with np.load(spec_path / "connectivity/glomerulus_to_golgi.npz") as data:
        cell_pairs = np.asarray(data["cell_pairs"], dtype=np.int64)
        target_compartments_all = np.asarray(
            data["compartment_pairs"][:, 1],
            dtype=np.int64,
        )

    known_glomerulus_ids = tuple(map(int, np.unique(cell_pairs[:, 0])))
    if glomerulus_spike_times_ms is None:
        source_glomerulus_ids = known_glomerulus_ids[:max_glomeruli]
        normalized_spike_times = tuple((1.0,) for _ in source_glomerulus_ids)
    else:
        source_glomerulus_ids = tuple(int(value) for value in glomerulus_spike_times_ms)
        if not source_glomerulus_ids:
            raise ValueError("glomerulus_spike_times_ms must contain at least one source ID.")
        if len(source_glomerulus_ids) > max_glomeruli:
            raise ValueError(
                "The explicit glomerulus schedule contains more sources than max_glomeruli."
            )
        unknown = tuple(
            value for value in source_glomerulus_ids
            if value not in set(known_glomerulus_ids)
        )
        if unknown:
            raise ValueError(
                f"Glomerulus IDs {unknown!r} do not occur in glomerulus_to_golgi.npz."
            )
        normalized_spike_times = tuple(
            _validate_spike_times(
                glomerulus_spike_times_ms[source_id],
                source_id=source_id,
            )
            for source_id in source_glomerulus_ids
        )

    pre_mask = np.isin(cell_pairs[:, 0], source_glomerulus_ids)
    source_golgi_ids_array = np.unique(cell_pairs[pre_mask, 1])[:max_golgi]
    selected = pre_mask & np.isin(cell_pairs[:, 1], source_golgi_ids_array)
    selected_pairs = cell_pairs[selected]
    target_compartments = target_compartments_all[selected]
    if not len(selected_pairs):
        raise ValueError("The selected glomeruli have no Golgi contacts.")

    source_golgi_ids = tuple(map(int, source_golgi_ids_array))
    glomerulus_lookup = {
        value: index for index, value in enumerate(source_glomerulus_ids)
    }
    golgi_lookup = {value: index for index, value in enumerate(source_golgi_ids)}
    local_contacts = np.asarray(
        [
            (glomerulus_lookup[int(pre)], golgi_lookup[int(post)])
            for pre, post in selected_pairs
        ],
        dtype=np.int32,
    )
    edge_pairs, contact_to_edge = np.unique(
        local_contacts,
        axis=0,
        return_inverse=True,
    )

    glomerulus = _build_scheduled_spike_source(
        normalized_spike_times,
        name="paper_golgi_input_glomerulus",
    )
    golgi = GoC(
        params=load_goc20_params(),
        temperature_celsius=32.0,
        pop_size=(len(source_golgi_ids),),
        name="paper_input_golgi",
    ).build()
    _restrict_spikes_to_soma(golgi.cell)
    mapping = build_morphology_mapping(source_hdf5, "GolgiCell", golgi.morph)
    from . import paper_synapses as _paper_synapses  # noqa: F401

    synapses = (
        mech.Synapse(
            "GoCAMPA_MF2020",
            name="glomerulus_ampa_mf",
            g_max=0.0012 * u.uS,
            E_rev=0.0 * u.mV,
        ),
        mech.Synapse(
            "GoCNMDA2020",
            name="glomerulus_nmda",
            g_max=0.0188 * u.uS,
            E_rev=-3.7 * u.mV,
        ),
    )
    pools = {
        synapse.instance_name: place_mapped_synapse_pool(
            golgi.cell,
            mapping,
            target_compartments,
            synapse,
        )
        for synapse in synapses
    }
    synapse_indices = {
        name: pool.contact_indices(target_compartments)
        for name, pool in pools.items()
    }
    unique_target_compartments = tuple(int(value) for value in np.unique(target_compartments))
    for compartment in unique_target_compartments:
        location = mapping.location(compartment)
        golgi.cell.place(
            at(location.branch_id, location.x),
            mech.StateProbe(name="v_contact", field="v"),
        )
        for synapse in synapses:
            golgi.cell.place(
                at(location.branch_id, location.x),
                mech.MechanismProbe(
                    name=(
                        "ampa_mf_g"
                        if synapse.instance_name == "glomerulus_ampa_mf"
                        else "nmda_g"
                    ),
                    mechanism=synapse.instance_name,
                    field="g",
                ),
            )
    golgi.cell.place(at("soma", 0.5), mech.StateProbe(name="v_soma", field="v"))

    network = braincell.Network(name="published-glomerulus-golgi-pilot")
    network.add_population("glomerulus", glomerulus)
    network.add_population("golgi_cell", golgi.cell)
    network.add_edges(
        name="glomerulus_to_golgi",
        pre="glomerulus",
        post="golgi_cell",
        method=braincell.network.pairs(edge_pairs),
    )
    for synapse in synapses:
        network.add_projection(
            name=f"glomerulus_to_golgi_{synapse.instance_name.removeprefix('glomerulus_')}",
            edges="glomerulus_to_golgi",
            synapse=synapse.instance_name,
            method=braincell.network.explicit_contacts(
                source_edge=contact_to_edge,
                synapse_index=synapse_indices[synapse.instance_name],
            ),
            weight=1.0,
            delay=0.0 * u.ms,
        )

    dendrites = [branch for branch in golgi.morph.branches if branch.type == "dendrite"]
    region_by_branch_name = {
        str(branch.name): dend_region_name(index)
        for index, branch in enumerate(dendrites)
    }
    target_region_counts: dict[str, int] = {}
    for compartment in map(int, target_compartments):
        region = region_by_branch_name[mapping.location(compartment).branch_name]
        target_region_counts[region] = target_region_counts.get(region, 0) + 1

    metadata = GolgiSubnetworkMetadata(
        source_glomerulus_ids=source_glomerulus_ids,
        source_golgi_ids=source_golgi_ids,
        n_edge=int(len(edge_pairs)),
        n_contact=int(len(selected_pairs)),
        target_compartments=unique_target_compartments,
        target_region_counts=tuple(sorted(target_region_counts.items())),
        glomerulus_spike_times_ms=normalized_spike_times,
        synapse_mechanisms=tuple(synapse.synapse_type for synapse in synapses),
        note=(
            "Published cell pairs and exact GoC target compartments; paper DBBS "
            "AMPA.granule/NMDA.granule mechanisms exposed through GoC aliases."
        ),
    )
    return network, metadata


def build_mossy_glomerulus_granule_golgi_subnetwork(
    spec_path: str | Path,
    source_hdf5: str | Path,
    braincell_repo: str | Path,
    *,
    mossy_spike_times_ms: Mapping[int, Sequence[float]] | None = None,
    max_glomeruli: int = 8,
    max_granules: int = 32,
    max_golgi: int = 8,
):
    """Build a common-input ``MF→glomerulus→{GrC, GoC}`` subnetwork.

    The Granule pathway is built by
    :func:`build_mossy_glomerulus_granule_subnetwork`.  This function then
    selects every published glomerulus→GoC contact involving the *same*
    retained glomerulus IDs and adds a real multi-compartment GoC population.
    Consequently both postsynaptic populations receive exactly the same
    topology-expanded glomerulus schedules; no second independently selected
    input source is introduced.
    """

    import braincell
    import brainunit as u
    from braincell import mech
    from braincell.filter import at

    braincell_repo = Path(braincell_repo)
    if str(braincell_repo) not in sys.path:
        sys.path.insert(0, str(braincell_repo))
    from examples.neuron_compare.cell.goc_ma2020.goc_braincell import GoC
    from examples.neuron_compare.cell.goc_ma2020.parameters import (
        dend_region_name,
        load_goc20_params,
    )

    if max_golgi <= 0:
        raise ValueError("max_golgi must be positive.")

    network, granule_metadata = build_mossy_glomerulus_granule_subnetwork(
        spec_path,
        source_hdf5,
        braincell_repo,
        mossy_spike_times_ms=mossy_spike_times_ms,
        max_glomeruli=max_glomeruli,
        max_granules=max_granules,
    )

    spec_path = Path(spec_path)
    with np.load(spec_path / "connectivity/glomerulus_to_golgi.npz") as data:
        cell_pairs = np.asarray(data["cell_pairs"], dtype=np.int64)
        target_compartments_all = np.asarray(
            data["compartment_pairs"][:, 1],
            dtype=np.int64,
        )

    source_glomerulus_ids = granule_metadata.source_glomerulus_ids
    pre_mask = np.isin(cell_pairs[:, 0], source_glomerulus_ids)
    source_golgi_ids_array = np.unique(cell_pairs[pre_mask, 1])[:max_golgi]
    selected = pre_mask & np.isin(cell_pairs[:, 1], source_golgi_ids_array)
    selected_pairs = cell_pairs[selected]
    target_compartments = target_compartments_all[selected]
    if not len(selected_pairs):
        raise ValueError(
            "The retained mossy-fiber glomeruli have no published Golgi contacts."
        )

    source_golgi_ids = tuple(map(int, source_golgi_ids_array))
    glomerulus_lookup = {
        value: index for index, value in enumerate(source_glomerulus_ids)
    }
    golgi_lookup = {value: index for index, value in enumerate(source_golgi_ids)}
    local_contacts = np.asarray(
        [
            (glomerulus_lookup[int(pre)], golgi_lookup[int(post)])
            for pre, post in selected_pairs
        ],
        dtype=np.int32,
    )
    edge_pairs, contact_to_edge = np.unique(
        local_contacts,
        axis=0,
        return_inverse=True,
    )

    golgi = GoC(
        params=load_goc20_params(),
        temperature_celsius=32.0,
        pop_size=(len(source_golgi_ids),),
        name="paper_shared_input_golgi",
    ).build()
    _restrict_spikes_to_soma(golgi.cell)
    mapping = build_morphology_mapping(source_hdf5, "GolgiCell", golgi.morph)
    from . import paper_synapses as _paper_synapses  # noqa: F401

    synapses = (
        mech.Synapse(
            "GoCAMPA_MF2020",
            name="glomerulus_ampa_mf",
            g_max=0.0012 * u.uS,
            E_rev=0.0 * u.mV,
        ),
        mech.Synapse(
            "GoCNMDA2020",
            name="glomerulus_nmda",
            g_max=0.0188 * u.uS,
            E_rev=-3.7 * u.mV,
        ),
    )
    pools = {
        synapse.instance_name: place_mapped_synapse_pool(
            golgi.cell,
            mapping,
            target_compartments,
            synapse,
        )
        for synapse in synapses
    }
    synapse_indices = {
        name: pool.contact_indices(target_compartments)
        for name, pool in pools.items()
    }
    unique_target_compartments = tuple(
        int(value) for value in np.unique(target_compartments)
    )
    for compartment in unique_target_compartments:
        location = mapping.location(compartment)
        for synapse in synapses:
            golgi.cell.place(
                at(location.branch_id, location.x),
                mech.MechanismProbe(
                    name=(
                        "ampa_mf_g"
                        if synapse.instance_name == "glomerulus_ampa_mf"
                        else "nmda_g"
                    ),
                    mechanism=synapse.instance_name,
                    field="g",
                ),
            )
    golgi.cell.place(at("soma", 0.5), mech.StateProbe(name="v_soma", field="v"))

    network.add_population("golgi_cell", golgi.cell)
    network.add_edges(
        name="glomerulus_to_golgi",
        pre="glomerulus",
        post="golgi_cell",
        method=braincell.network.pairs(edge_pairs),
    )
    for synapse in synapses:
        network.add_projection(
            name=(
                "glomerulus_to_golgi_"
                f"{synapse.instance_name.removeprefix('glomerulus_')}"
            ),
            edges="glomerulus_to_golgi",
            synapse=synapse.instance_name,
            method=braincell.network.explicit_contacts(
                source_edge=contact_to_edge,
                synapse_index=synapse_indices[synapse.instance_name],
            ),
            weight=1.0,
            delay=0.0 * u.ms,
        )

    dendrites = [branch for branch in golgi.morph.branches if branch.type == "dendrite"]
    region_by_branch_name = {
        str(branch.name): dend_region_name(index)
        for index, branch in enumerate(dendrites)
    }
    target_region_counts: dict[str, int] = {}
    for compartment in map(int, target_compartments):
        region = region_by_branch_name[mapping.location(compartment).branch_name]
        target_region_counts[region] = target_region_counts.get(region, 0) + 1

    metadata = SharedInputSubnetworkMetadata(
        granule_pathway=granule_metadata,
        source_golgi_ids=source_golgi_ids,
        n_glomerulus_to_golgi_edge=int(len(edge_pairs)),
        n_glomerulus_to_golgi_contact=int(len(selected_pairs)),
        golgi_target_compartments=unique_target_compartments,
        golgi_target_region_counts=tuple(sorted(target_region_counts.items())),
        golgi_synapse_mechanisms=tuple(
            synapse.synapse_type for synapse in synapses
        ),
        note=(
            "One topology-expanded MF/glomerulus schedule drives both published "
            "GrC and GoC contacts; MF→glomerulus remains an offline relay expansion."
        ),
    )
    return network, metadata


def build_granular_layer_chemical_subnetwork(
    spec_path: str | Path,
    source_hdf5: str | Path,
    braincell_repo: str | Path,
    *,
    mossy_spike_times_ms: Mapping[int, Sequence[float]] | None = None,
    max_glomeruli: int = 8,
    max_granules: int = 32,
    max_golgi: int = 8,
):
    """Add exact GoC→GrC GABA contacts to the shared-input network.

    This closes the chemical ``glomerulus→GoC→GrC`` feedback path while
    deliberately leaving ``gap_goc`` inactive.  Only contacts whose published
    source GoC and target GrC are already retained by the shared-input
    selection are added; no probabilistic rewiring is performed.
    """

    import braincell
    import brainunit as u
    from braincell import mech
    from braincell.filter import at

    network, shared_metadata = build_mossy_glomerulus_granule_golgi_subnetwork(
        spec_path,
        source_hdf5,
        braincell_repo,
        mossy_spike_times_ms=mossy_spike_times_ms,
        max_glomeruli=max_glomeruli,
        max_granules=max_granules,
        max_golgi=max_golgi,
    )
    source_golgi_ids = shared_metadata.source_golgi_ids
    source_granule_ids = shared_metadata.granule_pathway.source_granule_ids

    spec_path = Path(spec_path)
    with np.load(spec_path / "connectivity/golgi_to_granule.npz") as data:
        cell_pairs = np.asarray(data["cell_pairs"], dtype=np.int64)
        target_compartments_all = np.asarray(
            data["compartment_pairs"][:, 1],
            dtype=np.int64,
        )
    selected = np.isin(cell_pairs[:, 0], source_golgi_ids) & np.isin(
        cell_pairs[:, 1], source_granule_ids
    )
    selected_pairs = cell_pairs[selected]
    target_compartments = target_compartments_all[selected]
    if not len(selected_pairs):
        raise ValueError(
            "The retained GoC and GrC populations share no published GABA contacts."
        )

    golgi_lookup = {value: index for index, value in enumerate(source_golgi_ids)}
    granule_lookup = {
        value: index for index, value in enumerate(source_granule_ids)
    }
    local_contacts = np.asarray(
        [
            (golgi_lookup[int(pre)], granule_lookup[int(post)])
            for pre, post in selected_pairs
        ],
        dtype=np.int32,
    )
    edge_pairs, contact_to_edge = np.unique(
        local_contacts,
        axis=0,
        return_inverse=True,
    )

    granule_cell = network.populations["granule_cell"].cell
    mapping = build_morphology_mapping(source_hdf5, "GranuleCell", granule_cell.morpho)
    from . import paper_synapses as _paper_synapses  # noqa: F401

    gaba = mech.Synapse(
        "GrCGABA2020",
        name="golgi_gaba",
        E_rev=-65.0 * u.mV,
    )
    pool = place_mapped_synapse_pool(
        granule_cell,
        mapping,
        target_compartments,
        gaba,
    )
    synapse_indices = pool.contact_indices(target_compartments)
    unique_target_compartments = tuple(
        int(value) for value in np.unique(target_compartments)
    )
    for compartment in unique_target_compartments:
        location = mapping.location(compartment)
        granule_cell.place(
            at(location.branch_id, location.x),
            mech.MechanismProbe(
                name="gaba_g",
                mechanism="golgi_gaba",
                field="g",
            ),
        )

    network.add_edges(
        name="golgi_to_granule",
        pre="golgi_cell",
        post="granule_cell",
        method=braincell.network.pairs(edge_pairs),
    )
    network.add_projection(
        name="golgi_to_granule_gaba",
        edges="golgi_to_granule",
        synapse="golgi_gaba",
        method=braincell.network.explicit_contacts(
            source_edge=contact_to_edge,
            synapse_index=synapse_indices,
        ),
        weight=1.0,
        delay=0.0 * u.ms,
    )

    metadata = ChemicalLoopSubnetworkMetadata(
        shared_input=shared_metadata,
        n_golgi_to_granule_edge=int(len(edge_pairs)),
        n_golgi_to_granule_contact=int(len(selected_pairs)),
        gaba_target_compartments=unique_target_compartments,
        gaba_synapse_mechanism="GrCGABA2020",
        spike_output_mode="soma-CV upward threshold crossings only",
        note=(
            "Published no-gap granular-layer chemical loop. GoC/GrC network "
            "events are restricted to soma CV crossings so one propagating "
            "action potential is not emitted independently by every CV."
        ),
    )
    return network, metadata


def _validate_spike_times(values: Sequence[float], *, source_id: int) -> tuple[float, ...]:
    times = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or value < 0.0 for value in times):
        raise ValueError(f"Spike times for mossy ID {source_id} must be finite and non-negative.")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError(f"Spike times for mossy ID {source_id} must be strictly increasing.")
    return times


def _restrict_spikes_to_soma(cell) -> None:
    """Expose soma threshold crossings as the cell's network spike output.

    BrainCell stores a crossing value for every CV and its generic network
    reducer treats a crossing in any CV as a cell event.  NEURON/BSB cell
    models use a designated spike detector, so a propagating action potential
    must not be emitted once per axonal/dendritic CV.  This reproduction-local
    adapter retains the full multicompartment dynamics while zeroing non-soma
    entries in the event buffer.
    """

    if getattr(cell, "_paper_soma_spike_adapter", False):
        return
    soma_cv_ids = tuple(int(cv.id) for cv in cell.cvs if cv.branch_type == "soma")
    if not soma_cv_ids:
        raise ValueError("A paper cell requires at least one soma CV for spike output.")
    base_get_spike = cell.get_spike
    mask = np.zeros((cell.n_cv,), dtype=np.float32)
    mask[list(soma_cv_ids)] = 1.0

    def soma_get_spike(self, last_V, next_V):
        _ = self
        return base_get_spike(last_V, next_V) * mask

    cell.get_spike = MethodType(soma_get_spike, cell)
    cell._paper_soma_spike_adapter = True


def _round_robin_mossy_children(
    pairs: np.ndarray,
    source_ids: Sequence[int],
    limit: int,
) -> np.ndarray:
    """Select relay children fairly while preserving each source's file order."""

    children = [pairs[pairs[:, 0] == source_id] for source_id in source_ids]
    selected: list[np.ndarray] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for rows in children:
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64).reshape((-1, 2))


def _build_scheduled_spike_source(
    spike_times_ms: Sequence[Sequence[float]],
    *,
    name: str,
):
    """Return a one-CV spike-source population driven by explicit event times."""

    import brainunit as u
    from braincell import Branch, CVPerBranch, Cell, Morphology

    if not spike_times_ms:
        raise ValueError("A scheduled spike source requires at least one source.")
    max_events = max((len(times) for times in spike_times_ms), default=0)
    padded_times = np.full((len(spike_times_ms), max(max_events, 1)), -1.0, dtype=float)
    valid_events = np.zeros_like(padded_times, dtype=bool)
    for source_index, times in enumerate(spike_times_ms):
        count = len(times)
        if count:
            padded_times[source_index, :count] = times
            valid_events[source_index, :count] = True

    def scheduled_solver(cell) -> None:
        _scheduled_spike_source_solver(
            cell,
            spike_times_ms=padded_times,
            valid_events=valid_events,
        )

    relay_soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    return Cell(
        Morphology.from_root(relay_soma, name="soma"),
        cv_policy=CVPerBranch(),
        pop_size=(len(spike_times_ms),),
        V_init=-10.0 * u.mV,
        V_th=0.0 * u.mV,
        solver=scheduled_solver,
        name=name,
    )


def _scheduled_spike_source_solver(cell, *, spike_times_ms, valid_events) -> None:
    import brainstate
    import brainunit as u
    import jax.numpy as jnp

    time_ms = brainstate.environ.get("t").to_decimal(u.ms)
    dt_ms = brainstate.environ.get("dt").to_decimal(u.ms)
    current_step = jnp.rint(time_ms / dt_ms).astype(jnp.int32)
    event_steps = jnp.rint(jnp.asarray(spike_times_ms) / dt_ms).astype(jnp.int32)
    firing = jnp.any(jnp.asarray(valid_events) & (event_steps == current_step), axis=1)
    voltage = jnp.where(firing[:, None], 10.0, -10.0) * u.mV
    cell.V.value = u.math.broadcast_to(voltage, cell.V.value.shape)
