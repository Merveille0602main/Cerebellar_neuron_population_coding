import numpy as np
import brainunit as u


def _drive_one(synapse):
    synapse.apply_events((synapse.y.value * 0.0 + 1.0) * u.uS)


def test_granule_ampa_first_release_and_cdur_guard():
    import brainstate

    from cerebellar_reproduction.paper_synapses import GrCAMPA2020

    synapse = GrCAMPA2020(size=(1,))
    synapse.init_state()
    with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
        _drive_one(synapse)
        first_y = float(np.asarray(synapse.y.value)[0])
        first_cursor = int(np.asarray(synapse.history_cursor.value)[0])
        _drive_one(synapse)

    assert np.isclose(first_y, 0.43)
    assert first_cursor == 1
    assert np.isclose(float(np.asarray(synapse.y.value)[0]), first_y)
    assert int(np.asarray(synapse.history_cursor.value)[0]) == first_cursor


def test_published_receptor_rates_conserve_probability_and_nmda_block():
    import brainstate

    from cerebellar_reproduction.paper_synapses import GrCAMPA2020, GrCNMDA2020

    ampa = GrCAMPA2020(size=(1,))
    nmda = GrCNMDA2020(size=(1,))
    for synapse in (ampa, nmda):
        synapse.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            _drive_one(synapse)
        with brainstate.environ.context(t=0.05 * u.ms, dt=0.05 * u.ms):
            synapse.pre_integral(-65.0 * u.mV)

    ampa_sum = ampa.C.value + ampa.O.value + ampa.D.value
    nmda_sum = (
        nmda.C0.value
        + nmda.C1.value
        + nmda.C2.value
        + nmda.D.value
        + nmda.O.value
    )
    assert np.allclose(np.asarray(ampa_sum), 1.0, atol=1e-6)
    assert np.allclose(np.asarray(nmda_sum), 1.0, atol=1e-6)
    assert np.isclose(float(np.asarray(nmda.magnesium_block(-20.0 * u.mV)).reshape(())), 0.5)
    assert np.isclose(float(np.asarray(ampa.g_max.to_decimal(u.uS)).reshape(())), 0.0012)
    assert np.isclose(float(np.asarray(nmda.g_max.to_decimal(u.uS)).reshape(())), 0.0188)


def test_golgi_aliases_reuse_paper_dbbs_granule_mechanism_defaults():

    from cerebellar_reproduction.paper_synapses import (
        GoCAMPA_MF2020,
        GoCNMDA2020,
        GrCAMPA2020,
        GrCNMDA2020,
    )

    goc_ampa = GoCAMPA_MF2020(size=(1,))
    goc_nmda = GoCNMDA2020(size=(1,))

    assert isinstance(goc_ampa, GrCAMPA2020)
    assert isinstance(goc_nmda, GrCNMDA2020)
    assert np.isclose(float(np.asarray(goc_ampa.U).reshape(())), 0.42)
    assert np.isclose(float(np.asarray(goc_nmda.U).reshape(())), 0.42)
    assert np.isclose(
        float(np.asarray(goc_ampa.g_max.to_decimal(u.uS)).reshape(())),
        0.0012,
    )
    assert np.isclose(
        float(np.asarray(goc_nmda.g_max.to_decimal(u.uS)).reshape(())),
        0.0188,
    )


def test_granule_adapter_override_is_distinct_from_golgi_mod_default():
    from cerebellar_reproduction.paper_synapses import (
        GoCAMPA_MF2020,
        GoCNMDA2020,
        GrCAMPA2020,
        GrCNMDA2020,
    )

    assert np.isclose(float(np.asarray(GrCAMPA2020(size=(1,)).U).reshape(())), 0.43)
    assert np.isclose(float(np.asarray(GrCNMDA2020(size=(1,)).U).reshape(())), 0.43)
    assert np.isclose(float(np.asarray(GoCAMPA_MF2020(size=(1,)).U).reshape(())), 0.42)
    assert np.isclose(float(np.asarray(GoCNMDA2020(size=(1,)).U).reshape(())), 0.42)


def test_granule_gaba_defaults_release_and_receptor_probability():
    import brainstate

    from cerebellar_reproduction.paper_synapses import GrCGABA2020

    gaba = GrCGABA2020(size=(1,))
    gaba.init_state()
    with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
        _drive_one(gaba)
    for step in range(1, 5):
        with brainstate.environ.context(t=step * 0.05 * u.ms, dt=0.05 * u.ms):
            gaba.pre_integral(-40.0 * u.mV)

    alpha1_sum = sum(state.value for state in gaba.alpha1_states)
    alpha6_sum = sum(state.value for state in gaba.alpha6_states)
    current = gaba.current(-40.0 * u.mV)

    assert np.isclose(float(np.asarray(gaba.U).reshape(())), 0.35)
    assert np.isclose(
        float(np.asarray(gaba.g_max_alpha1.to_decimal(u.uS)).reshape(())),
        0.000918807,
    )
    assert np.isclose(
        float(np.asarray(gaba.g_max_alpha6.to_decimal(u.uS)).reshape(())),
        0.000132842,
    )
    assert np.allclose(np.asarray(alpha1_sum), 1.0, atol=1e-5)
    assert np.allclose(np.asarray(alpha6_sum), 1.0, atol=1e-5)
    assert float(np.asarray(gaba.g.to_decimal(u.uS))[0]) > 0.0
    assert float(np.asarray(current.to_decimal(u.nA))[0]) < 0.0
