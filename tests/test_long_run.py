from pathlib import Path

import numpy as np
import pytest

bc = pytest.importorskip("braincell")
pytestmark = pytest.mark.skipif(not hasattr(bc.Network, "add_projection"), reason="requires the legacy projection API")


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data/reference/balanced.hdf5"
BRAINCELL = ROOT / "assets/braincell"


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="integration data unavailable")
def test_output_estimate_and_chunked_run_match_continuous_pathway():
    import brainunit as u

    from cerebellar_reproduction import (
        build_mossy_glomerulus_granule_subnetwork,
        estimate_network_output_memory,
    )

    def build_network():
        return build_mossy_glomerulus_granule_subnetwork(
            ROOT / "data/export/balanced",
            SOURCE,
            BRAINCELL,
            mossy_spike_times_ms={213: (1.0,)},
            max_glomeruli=1,
            max_granules=2,
        )[0]

    reference_network = build_network()
    chunked_network = build_network()
    estimate = estimate_network_output_memory(
        reference_network,
        dt=0.05 * u.ms,
        duration=2.1 * u.ms,
        chunk_duration=1.05 * u.ms,
        spike_recording="population",
    )
    reference = reference_network.run(
        dt=0.05 * u.ms,
        duration=2.1 * u.ms,
        spike_recording="population",
        event_backend="scatter",
    )
    chunks = list(
        chunked_network.run_chunks(
            dt=0.05 * u.ms,
            duration=2.1 * u.ms,
            chunk_duration=1.05 * u.ms,
            spike_recording="population",
            event_backend="scatter",
        )
    )

    assert estimate.n_step == 42
    assert estimate.n_chunk_step == 21
    assert estimate.maximum_chunk_output_bytes * 2 == estimate.total_output_bytes
    assert len(chunks) == 2
    for probe_name in ("ampa_g", "nmda_g", "v_soma"):
        expected = reference.traces["granule_cell"][probe_name]
        unit = expected.unit
        joined = np.concatenate(
            [chunk.traces["granule_cell"][probe_name].to_decimal(unit) for chunk in chunks]
        )
        np.testing.assert_allclose(
            joined,
            expected.to_decimal(unit),
            rtol=2e-6,
            atol=1e-8,
        )
    joined_spikes = np.concatenate([chunk.spikes["glomerulus"] for chunk in chunks])
    np.testing.assert_array_equal(joined_spikes, reference.spikes["glomerulus"])
