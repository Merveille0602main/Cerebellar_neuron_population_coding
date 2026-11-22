from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data/reference/balanced.hdf5"
BRAINCELL = ROOT.parent / "braincell"


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="integration data unavailable")
def test_published_glomerulus_granule_subnetwork_runs():
    import brainunit as u

    from cerebellar_reproduction.runnable_prototype import (
        build_glomerulus_granule_subnetwork,
    )

    network, metadata = build_glomerulus_granule_subnetwork(
        ROOT / "data/export/balanced",
        SOURCE,
        BRAINCELL,
        max_glomeruli=2,
        max_granules=4,
    )
    result = network.run(dt=0.025 * u.ms, duration=2.0 * u.ms, event_backend="scatter")
    conductance = np.asarray(result.traces["granule_cell"]["ampa_g"], dtype=float)
    voltage = np.asarray(result.traces["granule_cell"]["v_soma"].to_decimal(u.mV))

    assert metadata.n_contact >= metadata.n_edge > 0
    assert metadata.synapse_mechanisms == ("AMPA",)
    assert len(metadata.source_pre_ids) == 2
    assert set(metadata.target_compartments).issubset({121, 130, 139, 148})
    assert int(np.asarray(result.spikes["glomerulus"]).sum()) == 2
    assert float(conductance.max()) > 0.0
    assert np.isfinite(voltage).all()
    assert float(np.ptp(voltage)) > 0.0


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="integration data unavailable")
def test_published_ampa_nmda_port_runs_with_release_history():
    import brainunit as u

    from cerebellar_reproduction.runnable_prototype import (
        build_glomerulus_granule_subnetwork,
    )

    network, metadata = build_glomerulus_granule_subnetwork(
        ROOT / "data/export/balanced",
        SOURCE,
        BRAINCELL,
        max_glomeruli=1,
        max_granules=2,
        paper_mechanisms=True,
    )
    result = network.run(dt=0.05 * u.ms, duration=3.0 * u.ms, event_backend="scatter")
    ampa = result.traces["granule_cell"]["ampa_g"].to_decimal(u.uS)
    nmda = result.traces["granule_cell"]["nmda_g"].to_decimal(u.uS)

    assert metadata.synapse_mechanisms == ("GrCAMPA2020", "GrCNMDA2020")
    assert float(np.asarray(ampa).max()) > 0.0
    assert float(np.asarray(nmda).max()) > 0.0


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="integration data unavailable")
def test_paper_mossy_relay_drives_published_granule_contacts():
    import brainunit as u

    from cerebellar_reproduction.runnable_prototype import (
        build_mossy_glomerulus_granule_subnetwork,
    )

    network, metadata = build_mossy_glomerulus_granule_subnetwork(
        ROOT / "data/export/balanced",
        SOURCE,
        BRAINCELL,
        mossy_spike_times_ms={213: (1.0, 2.0), 214: (1.5,)},
        max_glomeruli=2,
        max_granules=4,
    )
    result = network.run(dt=0.05 * u.ms, duration=4.0 * u.ms, event_backend="scatter")
    mossy_spikes = np.asarray(result.spikes["mossy_fibers"])
    glomerulus_spikes = np.asarray(result.spikes["glomerulus"])
    ampa = np.asarray(result.traces["granule_cell"]["ampa_g"].to_decimal(u.uS))
    nmda = np.asarray(result.traces["granule_cell"]["nmda_g"].to_decimal(u.uS))

    assert metadata.source_mossy_ids == (213, 214)
    assert metadata.glomerulus_parent_mossy_ids == (213, 214)
    assert metadata.relay_mode == "topology-expanded spike schedule"
    assert metadata.synapse_mechanisms == ("GrCAMPA2020", "GrCNMDA2020")
    assert mossy_spikes.sum(axis=(0, 2)).tolist() == [2.0, 1.0]
    assert glomerulus_spikes.sum(axis=(0, 2)).tolist() == [2.0, 1.0]
    assert float(ampa.max()) > 0.0
    assert float(nmda.max()) > 0.0


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="integration data unavailable")
def test_published_glomerulus_golgi_subnetwork_runs():
    import brainunit as u

    from cerebellar_reproduction import build_glomerulus_golgi_subnetwork

    network, metadata = build_glomerulus_golgi_subnetwork(
        ROOT / "data/export/balanced",
        SOURCE,
        BRAINCELL,
        max_glomeruli=1,
        max_golgi=1,
    )
    result = network.run(
        dt=0.05 * u.ms,
        duration=3.0 * u.ms,
        event_backend="scatter",
        spike_recording="population",
    )
    ampa = np.asarray(result.traces["golgi_cell"]["ampa_mf_g"].to_decimal(u.uS))
    nmda = np.asarray(result.traces["golgi_cell"]["nmda_g"].to_decimal(u.uS))
    voltage = np.asarray(result.traces["golgi_cell"]["v_soma"].to_decimal(u.mV))

    assert metadata.n_edge == metadata.n_contact == 1
    assert metadata.synapse_mechanisms == ("GoCAMPA_MF2020", "GoCNMDA2020")
    assert metadata.target_region_counts == (("dend_basal", 1),)
    assert int(np.asarray(result.spikes["glomerulus"]).sum()) == 1
    assert float(ampa.max()) > 0.0
    assert float(nmda.max()) > 0.0
    assert np.isfinite(voltage).all()


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="integration data unavailable")
def test_one_mossy_schedule_drives_granule_and_golgi_contacts():
    import brainunit as u

    from cerebellar_reproduction import (
        build_mossy_glomerulus_granule_golgi_subnetwork,
    )

    network, metadata = build_mossy_glomerulus_granule_golgi_subnetwork(
        ROOT / "data/export/balanced",
        SOURCE,
        BRAINCELL,
        mossy_spike_times_ms={213: (1.0, 2.0), 214: (1.5,)},
        max_glomeruli=2,
        max_granules=4,
        max_golgi=2,
    )
    result = network.run(
        dt=0.05 * u.ms,
        duration=4.0 * u.ms,
        event_backend="scatter",
        spike_recording="population",
    )
    granule_ampa = np.asarray(
        result.traces["granule_cell"]["ampa_g"].to_decimal(u.uS)
    )
    golgi_ampa = np.asarray(
        result.traces["golgi_cell"]["ampa_mf_g"].to_decimal(u.uS)
    )
    golgi_nmda = np.asarray(
        result.traces["golgi_cell"]["nmda_g"].to_decimal(u.uS)
    )

    assert metadata.granule_pathway.source_glomerulus_ids == (765, 732)
    assert metadata.source_golgi_ids == (24, 52)
    assert metadata.n_glomerulus_to_golgi_contact >= 2
    assert set(network.populations) == {
        "mossy_fibers",
        "glomerulus",
        "granule_cell",
        "golgi_cell",
    }
    assert np.asarray(result.spikes["mossy_fibers"]).sum(axis=0).tolist() == [2.0, 1.0]
    assert np.asarray(result.spikes["glomerulus"]).sum(axis=0).tolist() == [2.0, 1.0]
    assert float(granule_ampa.max()) > 0.0
    assert float(golgi_ampa.max()) > 0.0
    assert float(golgi_nmda.max()) > 0.0


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="integration data unavailable")
def test_no_gap_granular_chemical_loop_delivers_golgi_gaba():
    import brainunit as u

    from cerebellar_reproduction import build_granular_layer_chemical_subnetwork

    network, metadata = build_granular_layer_chemical_subnetwork(
        ROOT / "data/export/balanced",
        SOURCE,
        BRAINCELL,
        mossy_spike_times_ms={214: tuple(np.arange(1.0, 11.0))},
        max_glomeruli=1,
        max_granules=1,
        max_golgi=1,
    )
    result = network.run(
        dt=0.05 * u.ms,
        duration=18.0 * u.ms,
        event_backend="scatter",
        spike_recording="population",
    )
    gaba = np.asarray(
        result.traces["granule_cell"]["gaba_g"].to_decimal(u.uS)
    )

    assert metadata.shared_input.granule_pathway.source_glomerulus_ids == (732,)
    assert metadata.shared_input.source_golgi_ids == (24,)
    assert metadata.shared_input.granule_pathway.source_granule_ids == (4130,)
    assert metadata.n_golgi_to_granule_edge == 1
    assert metadata.n_golgi_to_granule_contact == 2
    assert metadata.gaba_target_compartments == (120, 129)
    assert metadata.gaba_synapse_mechanism == "GrCGABA2020"
    assert int(np.asarray(result.spikes["golgi_cell"]).sum()) == 1
    assert float(gaba.max()) > 0.0
