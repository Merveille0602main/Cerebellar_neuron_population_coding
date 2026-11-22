from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data/reference/balanced.hdf5"
BRAINCELL = Path("/home/merveille/copneuron/braincell")


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="integration data unavailable")
def test_published_contact_drives_real_granule_cell():
    import brainunit as u

    from cerebellar_reproduction.runnable_prototype import (
        build_glomerulus_granule_prototype,
    )

    network, metadata = build_glomerulus_granule_prototype(
        ROOT / "data/export/balanced",
        SOURCE,
        BRAINCELL,
    )
    result = network.run(
        dt=0.025 * u.ms,
        duration=2.0 * u.ms,
        event_backend="scatter",
    )
    ampa = np.asarray(result.traces["granule_cell"]["ampa_g"], dtype=float)
    voltage = np.asarray(result.traces["granule_cell"]["v_soma"].to_decimal(u.mV))
    assert metadata.target_compartment in (121, 130, 139, 148)
    assert int(np.asarray(result.spikes["glomerulus"]).sum()) == 1
    assert float(ampa.max()) > 0.0
    assert np.isfinite(voltage).all()
    assert float(np.ptp(voltage)) > 0.0
