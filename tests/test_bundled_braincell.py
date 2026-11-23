"""Regression checks for the calcium fixes ported into the bundled runtime."""

import numpy as np
import pytest

bc = pytest.importorskip("braincell")
if "+reproduction." not in bc.__version__:
    pytest.skip("requires the bundled BrainCell runtime", allow_module_level=True)

import brainstate
import brainunit as u
import jax

from braincell.channel import calcium


def test_dcn_constant_field_drive_has_a_finite_limit_at_zero():
    ci, co = 0.0001, 2.0
    temperature = 307.15

    def drive(millivolts):
        return calcium._su2015_dcn_ghk_drive(
            millivolts * u.mV, ci * u.mM, co * u.mM, temperature * u.kelvin,
        )

    with brainstate.environ.context(precision=64):
        expected_at_zero = (4.47814e6 / 23.20764929) * (ci - co) / 1000.0
        np.testing.assert_allclose(drive(0.0), expected_at_zero, rtol=1e-12)
        assert np.isfinite(jax.grad(drive)(0.0))
        for voltage in (-65.0, 35.0):
            exp_term = np.exp(-23.20764929 * voltage / temperature)
            previous = (4.47814e6 * voltage / temperature) * (ci - co * exp_term) / (1000 * (1 - exp_term))
            np.testing.assert_allclose(drive(voltage), previous, rtol=1e-12)


@pytest.mark.parametrize("flux", [calcium._cav3p1_nmodl_ghk_flux, calcium._cav3p3_nmodl_ghk_flux])
def test_cav3_constant_field_flux_has_a_finite_slope_at_zero(flux):
    def drive(millivolts):
        return u.get_mantissa(flux(millivolts * u.mV, 0.0001 * u.mM, 2.0 * u.mM, 2, 307.15 * u.kelvin))

    with brainstate.environ.context(precision=64):
        assert np.isfinite(drive(0.0))
        differentiated = jax.grad(drive)(0.0)
        assert np.isfinite(differentiated)
        step = 1e-6
        np.testing.assert_allclose(differentiated, (drive(step) - drive(-step)) / (2 * step), rtol=1e-7)
