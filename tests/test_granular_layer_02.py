"""Regression tests for the published granular-layer circuit."""

from pathlib import Path

import pytest

bc = pytest.importorskip("braincell")
if bc.__version__ != "0.2.0" and "+reproduction." not in bc.__version__:
    pytest.skip("requires a supported BrainCell runtime", allow_module_level=True)

import brainunit as u
import numpy as np

from cerebellar_reproduction.granular_layer import GLOM, GOC, GRC, MF, build_granular_layer
from cerebellar_reproduction.paper_synapses import GoCAMPA_AA2020, GoCGABAAlpha1


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data/export/balanced"
REFERENCE = ROOT / "data/reference/balanced.hdf5"
ASSETS = ROOT / "assets/braincell"

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_file() or not SPEC.is_dir() or not ASSETS.is_dir(),
    reason="published network and BrainCell cell assets are required",
)


def test_full_granular_layer_maps_every_published_source_and_target():
    network, report = build_granular_layer(SPEC, REFERENCE, ASSETS, full=True, plan_only=True)
    assert network is None
    assert report["population_counts"] == {MF: 117, GLOM: 2336, GRC: 28615, GOC: 70}
    assert report["control_volumes_per_cell"] == {GRC: 114, GOC: 317}
    assert {name: route["anatomical_contacts"] for name, route in report["routes"].items()} == {
        "mossy_to_glomerulus": 2336,
        "glomerulus_to_granule": 114460,
        "glomerulus_to_golgi": 3939,
        "ascending_axon_to_golgi": 22431,
        "parallel_fiber_to_golgi": 63552,
        "golgi_to_granule": 96887,
        "golgi_to_golgi": 181476,
        "gap_goc": 2082,
    }
    assert report["routes"]["gap_goc"]["runtime_electrical_coupling"] is False


def test_native_circuit_delivers_live_ascending_parallel_and_inhibitory_events():
    network, report = build_granular_layer(SPEC, REFERENCE, ASSETS, exercise_circuit=True)
    assert set(network.populations) == {MF, GLOM, GRC, GOC}
    assert network.connections.n_rows == 509
    assert report["population_counts"][GOC] == 2
    assert "mechanisms" not in report["routes"]["gap_goc"]
    for route in report["routes"].values():
        for receiver in route.get("mechanisms", {}).values():
            assert receiver["event_rows"] == receiver["receivers"] == route["anatomical_contacts"]

    result = network.run(dt=0.05 * u.ms, duration=18 * u.ms)
    assert len(result.events[MF]["event"].source_id) == 40
    assert len(result.events[GLOM]["event"].source_id) == 40
    for population, output in [
        (GRC, "ascending_axon_to_golgi_axon"),
        (GRC, "parallel_fiber_to_golgi_axon"),
        (GOC, "golgi_to_granule_axon"),
        (GOC, "golgi_to_golgi_axon"),
    ]:
        assert len(result.events[population][output].source_id) > 0

    for population in (GRC, GOC):
        for name, block in result.samples[population].items():
            assert np.isfinite(block.values.mantissa).all(), name
            if name != "soma_v":
                assert np.any(block.values.to_decimal(u.uS) > 0), name

    # Separate cross-cell contacts from the self-contacts in the published rows.
    goc_ids = report["selected_global_ids"][GOC]
    with np.load(SPEC / "connectivity/golgi_to_golgi.npz", allow_pickle=False) as data:
        pairs = data["cell_pairs"]
        pairs = pairs[np.isin(pairs, goc_ids).all(axis=1)]
    pairs = pairs[np.argsort(pairs[:, 1], kind="stable")]
    cross_cell = pairs[:, 0] != pairs[:, 1]
    conductance = result.samples[GOC]["golgi_to_golgi_gaba"].values.to_decimal(u.uS)
    assert np.any(conductance[..., cross_cell] > 0)


def test_new_golgi_mechanism_defaults():
    ampa = GoCAMPA_AA2020((1,))
    for value, expected in [(ampa.tau_1, 30.0), (ampa.tau_rec, 35.1), (ampa.tau_facil, 54.0)]:
        np.testing.assert_allclose(value.to_decimal(u.ms), expected)
    np.testing.assert_allclose(ampa.U, 0.4)
    np.testing.assert_allclose(ampa.g_max.to_decimal(u.nS), 1.2)

    gaba = GoCGABAAlpha1((1,))
    gaba.init_state()
    assert not hasattr(gaba, "alpha6_states")
    np.testing.assert_allclose(gaba.g_max_alpha1.to_decimal(u.nS), 3.2)
    gaba.alpha1_states[6].value = gaba.alpha1_states[0].value
    assert np.all(gaba.current(-55.0 * u.mV).to_decimal(u.pA) < 0)
