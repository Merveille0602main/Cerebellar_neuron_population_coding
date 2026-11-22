"""BrainCell ports of the published DBBS glutamatergic synapses.

The equations and mechanism-level defaults in this module follow the NEURON mechanisms
``GRANULE_Ampa_det_vi.mod`` and ``GRANULE_Nmda_det_vi.mod`` shipped with the
BrainCell cerebellar model sources.  The two mechanisms independently retain
the same per-contact Tsodyks--Markram release state, matching the original
pair of point processes attached to every glomerulus→Granule contact.

The final runtime parameters do not come from the MOD files alone.  In the
paper-era ``dbbs_models`` v1.1.2 adapter, ``GranuleCell`` overrides the MOD
default ``U=0.42`` with ``U=0.43`` for both AMPA and NMDA.  Consequently the
``GrC*`` public classes below default to 0.43.  ``GolgiCell`` maps its
``AMPA_MF`` and ``NMDA`` names to the same mechanisms without that override,
so the ``GoC*`` aliases deliberately restore the MOD default 0.42.

The paper-era ``dbbs_models.GolgiCell`` maps its ``AMPA_MF`` and ``NMDA``
names to the same ``dbbs_mod_collection`` ``AMPA.granule`` and
``NMDA.granule`` mechanisms.  The GoC aliases at the bottom therefore reuse
these exact equations with the GoC-effective defaults instead of the different synapse files
bundled with the standalone ModelDB 266806 GoC model.
"""

from __future__ import annotations

from typing import Callable, Union

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

from braincell import HHTypedNeuron, Synapse
from braincell.mech import ScalarEventInput, StateSpec, register_synapse


ArrayLike = Union[brainstate.typing.ArrayLike, Callable]


def _backward_euler_markov(states: tuple, matrix_per_ms):
    """Apply one coupled backward-Euler step to a small Markov system."""

    values = jnp.stack(tuple(state.value for state in states), axis=-1)
    size = len(states)
    identity = jnp.eye(size, dtype=values.dtype)
    dt_ms = brainstate.environ.get_dt().to_decimal(u.ms)
    updated = jnp.linalg.solve(
        identity - dt_ms * matrix_per_ms,
        values[..., None],
    )[..., 0]
    for index, state in enumerate(states):
        state.value = updated[..., index]


def _add_reversible_transition(matrix, source, target, forward, backward):
    """Add ``source <-> target`` rates to a column-source Markov matrix."""

    matrix = matrix.at[..., source, source].add(-forward)
    matrix = matrix.at[..., target, source].add(forward)
    matrix = matrix.at[..., target, target].add(-backward)
    return matrix.at[..., source, target].add(backward)


class _GranuleGlutamateBase(Synapse):
    """Shared release machinery from the two published MOD mechanisms."""

    root_type = HHTypedNeuron
    event_input = ScalarEventInput(u.uS, aggregation="sum")
    states = {name: StateSpec(0.0) for name in ("y", "z", "u_state")}
    history_size = 100
    uses_diffusion = True

    def __init__(
        self,
        size: brainstate.typing.Size,
        *,
        U: ArrayLike = 0.42,
        tau_rec: ArrayLike = 8.0 * u.ms,
        tau_facil: ArrayLike = 5.0 * u.ms,
        tau_1: ArrayLike = 1.0 * u.ms,
        Tmax: ArrayLike = 1.0 * u.mM,
        Cdur: ArrayLike = 0.3 * u.ms,
        M: ArrayLike = 21.515,
        release_distance: ArrayLike = 1.03 * u.um,
        diffusion: ArrayLike = 0.223 * (u.um**2 / u.ms),
        cleft_height: ArrayLike = 0.02 * u.um,
        q10_diffusion: ArrayLike = 1.5,
        q10_channel: ArrayLike = 2.4,
        temperature_celsius: ArrayLike = 32.0,
        name: str | None = None,
    ):
        super().__init__(size=size, name=name)
        self.U = braintools.init.param(U, self.varshape, allow_none=False)
        self.tau_rec = braintools.init.param(tau_rec, self.varshape, allow_none=False)
        self.tau_facil = braintools.init.param(tau_facil, self.varshape, allow_none=False)
        self.tau_1 = braintools.init.param(tau_1, self.varshape, allow_none=False)
        self.Tmax = braintools.init.param(Tmax, self.varshape, allow_none=False)
        self.Cdur = braintools.init.param(Cdur, self.varshape, allow_none=False)
        self.M = braintools.init.param(M, self.varshape, allow_none=False)
        self.release_distance = braintools.init.param(
            release_distance, self.varshape, allow_none=False
        )
        self.diffusion = braintools.init.param(diffusion, self.varshape, allow_none=False)
        self.cleft_height = braintools.init.param(
            cleft_height, self.varshape, allow_none=False
        )
        self.q10_diffusion = braintools.init.param(
            q10_diffusion, self.varshape, allow_none=False
        )
        self.q10_channel = braintools.init.param(
            q10_channel, self.varshape, allow_none=False
        )
        self.temperature_celsius = braintools.init.param(
            temperature_celsius, self.varshape, allow_none=False
        )

    def init_state(self, V_post=None, batch_size=None):
        super().init_state(V_post=V_post, batch_size=batch_size)
        shape = self.varshape if batch_size is None else (batch_size,) + self.varshape
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.last_release = brainstate.ShortTermState(zeros)
        self.last_event_ms = brainstate.ShortTermState(zeros - 1.0e12)
        self.last_tm_update_ms = brainstate.ShortTermState(zeros)
        history_shape = shape + (self.history_size,)
        self.release_history = brainstate.ShortTermState(u.math.zeros(history_shape))
        self.release_time_history_ms = brainstate.ShortTermState(
            u.math.zeros(history_shape) - 1.0e12
        )
        self.history_cursor = brainstate.ShortTermState(
            u.math.asarray(zeros, dtype=int)
        )

    def reset_state(self, V_post=None, batch_size=None):
        super().reset_state(V_post=V_post, batch_size=batch_size)
        shape = self.varshape if batch_size is None else (batch_size,) + self.varshape
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.last_release.value = zeros
        self.last_event_ms.value = zeros - 1.0e12
        self.last_tm_update_ms.value = zeros
        self.release_history.value = u.math.zeros(shape + (self.history_size,))
        self.release_time_history_ms.value = (
            u.math.zeros(shape + (self.history_size,)) - 1.0e12
        )
        self.history_cursor.value = u.math.asarray(zeros, dtype=int)

    def apply_events(self, payload, V_post=None):
        _ = V_post
        incoming = payload.to_decimal(u.uS) > 0
        t_ms = brainstate.environ.get("t", 0.0 * u.ms).to_decimal(u.ms)
        pulse_age = (t_ms - self.last_event_ms.value) * u.ms
        # The MOD point process ignores a second event while its Cdur pulse is
        # still on (``if (!on)`` in NET_RECEIVE).
        boundary_epsilon = 1.0e-6 * u.ms
        event = incoming & (pulse_age + boundary_epsilon >= self.Cdur)
        elapsed = (t_ms - self.last_tm_update_ms.value) * u.ms
        y_decay = self.y.value * u.math.exp(-elapsed / self.tau_1)
        z_decay = self.z.value * u.math.exp(-elapsed / self.tau_rec)
        z_decay = z_decay + self.y.value * (
            u.math.exp(-elapsed / self.tau_1)
            - u.math.exp(-elapsed / self.tau_rec)
        ) / ((self.tau_1 / self.tau_rec) - 1.0)
        u_decay = self.u_state.value * u.math.exp(-elapsed / self.tau_facil)
        facilitated = self._facilitate(u_decay)
        available = 1.0 - y_decay - z_decay
        released = available * facilitated
        y_after = y_decay + released
        self.u_state.value = u.math.where(event, facilitated, self.u_state.value)
        self.y.value = u.math.where(event, y_after, self.y.value)
        self.z.value = u.math.where(event, z_decay, self.z.value)
        self.last_release.value = u.math.where(event, y_after, self.last_release.value)

        self.last_event_ms.value = u.math.where(event, t_ms, self.last_event_ms.value)
        self.last_tm_update_ms.value = u.math.where(
            event, t_ms, self.last_tm_update_ms.value
        )
        if not self.uses_diffusion:
            return
        slots = u.math.arange(self.history_size)
        write_slot = slots == u.math.expand_dims(self.history_cursor.value, axis=-1)
        write_mask = u.math.expand_dims(event, axis=-1) & write_slot
        self.release_history.value = u.math.where(
            write_mask,
            u.math.expand_dims(y_after, axis=-1),
            self.release_history.value,
        )
        self.release_time_history_ms.value = u.math.where(
            write_mask,
            t_ms,
            self.release_time_history_ms.value,
        )
        next_cursor = (self.history_cursor.value + 1) % self.history_size
        self.history_cursor.value = u.math.where(
            event, next_cursor, self.history_cursor.value
        )

    def _facilitate(self, u_decay):
        return u_decay + self.U * (1.0 - u_decay)

    def compute_derivative(self, V_post=None):
        _ = V_post
        # These NET_RECEIVE arguments are analytically decayed only when the
        # next event arrives in the original MOD mechanism. Keep zero-valued
        # derivatives so BrainCell's synapse scheduler still invokes
        # ``pre_integral`` for the coupled receptor-state solve.
        self.y.derivative = self.y.value * 0.0 / u.ms
        self.z.derivative = self.z.value * 0.0 / u.ms
        self.u_state.derivative = self.u_state.value * 0.0 / u.ms

    def _q10_conductance(self):
        return self.q10_diffusion ** ((self.temperature_celsius - 30.0) / 10.0)

    def _q10_rate(self):
        return self.q10_channel ** ((self.temperature_celsius - 30.0) / 10.0)

    def _diffused_transmitter(self):
        """Return the MOD diffusion-history sum as a concentration."""

        t_ms = brainstate.environ.get("t", 0.0 * u.ms).to_decimal(u.ms)
        age_ms = t_ms - self.release_time_history_ms.value
        valid = age_ms > 0.0
        safe_age = u.math.where(valid, age_ms, 1.0)
        rd_um = u.math.expand_dims(
            self.release_distance.to_decimal(u.um), axis=-1
        )
        diff_um2_ms = u.math.expand_dims(
            self.diffusion.to_decimal(u.um**2 / u.ms), axis=-1
        )
        cleft_um = u.math.expand_dims(
            self.cleft_height.to_decimal(u.um), axis=-1
        )
        molecule_factor = u.math.expand_dims(
            1.0e3 * (1.0e3 * 1.0e15 / 6.022e23) * self.M,
            axis=-1,
        )
        exponent = -(rd_um**2) / (4.0 * diff_um2_ms * safe_age)
        denominator = 4.0 * u.math.pi * diff_um2_ms * cleft_um * safe_age
        contribution = self.release_history.value * molecule_factor * u.math.exp(
            exponent
        ) / denominator
        return u.math.sum(u.math.where(valid, contribution, 0.0), axis=-1) * u.mM

    def _local_transmitter_pulse(self):
        t_ms = brainstate.environ.get("t", 0.0 * u.ms).to_decimal(u.ms)
        age = (t_ms - self.last_event_ms.value) * u.ms
        # NMODL's self-event at exactly ``event_time + Cdur`` clears T before
        # the following kinetic solve, hence the upper boundary is open.
        active = (age >= 0.0 * u.ms) & (age + 1.0e-6 * u.ms < self.Cdur)
        return u.math.where(active, self.Tmax * self.last_release.value, 0.0 * u.mM)


@register_synapse("GrCAMPA2020")
class GrCAMPA2020(_GranuleGlutamateBase):
    """Published glomerulus→Granule AMPA point-process mechanism."""

    def __init__(
        self,
        size: brainstate.typing.Size,
        *,
        U: ArrayLike = 0.43,
        g_max: ArrayLike = 0.0012 * u.uS,
        E_rev: ArrayLike = 0.0 * u.mV,
        r1_fixed: ArrayLike = 5.4 / u.ms,
        r2: ArrayLike = 0.82 / u.ms,
        r5: ArrayLike = 0.013 / u.ms,
        r6_fixed: ArrayLike = 1.12 / u.ms,
        kB: ArrayLike = 0.44 * u.mM,
        **kwargs,
    ):
        super().__init__(size=size, U=U, **kwargs)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E_rev = braintools.init.param(E_rev, self.varshape, allow_none=False)
        self.r1_fixed = braintools.init.param(r1_fixed, self.varshape, allow_none=False)
        self.r2 = braintools.init.param(r2, self.varshape, allow_none=False)
        self.r5 = braintools.init.param(r5, self.varshape, allow_none=False)
        self.r6_fixed = braintools.init.param(r6_fixed, self.varshape, allow_none=False)
        self.kB = braintools.init.param(kB, self.varshape, allow_none=False)

    def init_state(self, V_post=None, batch_size=None):
        super().init_state(V_post=V_post, batch_size=batch_size)
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.C = brainstate.ShortTermState(zeros + 1.0)
        self.O = brainstate.ShortTermState(zeros)
        self.D = brainstate.ShortTermState(zeros)

    def reset_state(self, V_post=None, batch_size=None):
        super().reset_state(V_post=V_post, batch_size=batch_size)
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.C.value = zeros + 1.0
        self.O.value = zeros
        self.D.value = zeros

    @property
    def g(self):
        return self.g_max * self._q10_conductance() * self.O.value

    def pre_integral(self, V_post=None):
        _ = V_post
        transmitter = self._local_transmitter_pulse()
        if self.uses_diffusion:
            transmitter = transmitter + self._diffused_transmitter()
        saturation = transmitter**2 / (transmitter + self.kB) ** 2
        q10 = self._q10_rate()
        r1 = (q10 * self.r1_fixed * saturation).to_decimal(u.ms**-1)
        r2 = (q10 * self.r2 + self.C.value * 0.0 / u.ms).to_decimal(u.ms**-1)
        r5 = (q10 * self.r5 + self.C.value * 0.0 / u.ms).to_decimal(u.ms**-1)
        r6 = (q10 * self.r6_fixed * saturation).to_decimal(u.ms**-1)
        matrix = jnp.stack(
            (
                jnp.stack((-r1 - r6, r2, r5), axis=-1),
                jnp.stack((r1, -r2, jnp.zeros_like(r1)), axis=-1),
                jnp.stack((r6, jnp.zeros_like(r1), -r5), axis=-1),
            ),
            axis=-2,
        )
        _backward_euler_markov((self.C, self.O, self.D), matrix)

    def current(self, V_post):
        return self.g * (self.E_rev - V_post)


@register_synapse("GrCNMDA2020")
class GrCNMDA2020(_GranuleGlutamateBase):
    """Published glomerulus→Granule NMDA point-process mechanism."""

    def __init__(
        self,
        size: brainstate.typing.Size,
        *,
        U: ArrayLike = 0.43,
        g_max: ArrayLike = 0.0188 * u.uS,
        E_rev: ArrayLike = -3.7 * u.mV,
        Rb: ArrayLike = 5.0 / (u.ms * u.mM),
        Ru: ArrayLike = 0.1 / u.ms,
        Rd_rate: ArrayLike = 0.012 / u.ms,
        Rr: ArrayLike = 0.009 / u.ms,
        Ro: ArrayLike = 0.03 / u.ms,
        Rc: ArrayLike = 0.966 / u.ms,
        v0_block: ArrayLike = -20.0 * u.mV,
        k_block: ArrayLike = 13.0 * u.mV,
        **kwargs,
    ):
        super().__init__(size=size, U=U, **kwargs)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E_rev = braintools.init.param(E_rev, self.varshape, allow_none=False)
        self.Rb = braintools.init.param(Rb, self.varshape, allow_none=False)
        self.Ru = braintools.init.param(Ru, self.varshape, allow_none=False)
        self.Rd_rate = braintools.init.param(Rd_rate, self.varshape, allow_none=False)
        self.Rr = braintools.init.param(Rr, self.varshape, allow_none=False)
        self.Ro = braintools.init.param(Ro, self.varshape, allow_none=False)
        self.Rc = braintools.init.param(Rc, self.varshape, allow_none=False)
        self.v0_block = braintools.init.param(v0_block, self.varshape, allow_none=False)
        self.k_block = braintools.init.param(k_block, self.varshape, allow_none=False)

    def init_state(self, V_post=None, batch_size=None):
        super().init_state(V_post=V_post, batch_size=batch_size)
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.C0 = brainstate.ShortTermState(zeros + 1.0)
        self.C1 = brainstate.ShortTermState(zeros)
        self.C2 = brainstate.ShortTermState(zeros)
        self.D = brainstate.ShortTermState(zeros)
        self.O = brainstate.ShortTermState(zeros)

    def reset_state(self, V_post=None, batch_size=None):
        super().reset_state(V_post=V_post, batch_size=batch_size)
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.C0.value = zeros + 1.0
        self.C1.value = zeros
        self.C2.value = zeros
        self.D.value = zeros
        self.O.value = zeros

    @property
    def g(self):
        return self.g_max * self._q10_conductance() * self.O.value

    def magnesium_block(self, V_post):
        return 1.0 / (1.0 + u.math.exp(-(V_post - self.v0_block) / self.k_block))

    def pre_integral(self, V_post=None):
        _ = V_post
        transmitter = self._diffused_transmitter()
        q10 = self._q10_rate()
        rb = (q10 * self.Rb * transmitter).to_decimal(u.ms**-1)
        zero = jnp.zeros_like(rb)
        ru = (q10 * self.Ru + self.C0.value * 0.0 / u.ms).to_decimal(u.ms**-1)
        rd = (q10 * self.Rd_rate + self.C0.value * 0.0 / u.ms).to_decimal(u.ms**-1)
        rr = (q10 * self.Rr + self.C0.value * 0.0 / u.ms).to_decimal(u.ms**-1)
        ro = (q10 * self.Ro + self.C0.value * 0.0 / u.ms).to_decimal(u.ms**-1)
        rc = (q10 * self.Rc + self.C0.value * 0.0 / u.ms).to_decimal(u.ms**-1)
        matrix = jnp.stack(
            (
                jnp.stack((-rb, ru, zero, zero, zero), axis=-1),
                jnp.stack((rb, -ru - rb, ru, zero, zero), axis=-1),
                jnp.stack((zero, rb, -ru - rd - ro, rr, rc), axis=-1),
                jnp.stack((zero, zero, rd, -rr, zero), axis=-1),
                jnp.stack((zero, zero, ro, zero, -rc), axis=-1),
            ),
            axis=-2,
        )
        _backward_euler_markov((self.C0, self.C1, self.C2, self.D, self.O), matrix)

    def current(self, V_post):
        return self.g * (self.E_rev - V_post) * self.magnesium_block(V_post)


class _GABAAlpha1(_GranuleGlutamateBase):
    """DBBS release and eight-state alpha1 receptor kinetics."""

    alpha1_rates = (
    17.7,
    2.0,
    21.703,
    0.97325,
    0.0,
    0.0,
    0.4144,
    0.03,
    1.0002,
    10.0,
    7.0e-4,
    0.14208,
    3.3e-4,
    3.4898,
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        *,
        g_max_alpha1: ArrayLike = 0.000918807 * u.uS,
        g_max_factor: ArrayLike = 1.0,
        E_rev: ArrayLike = -65.0 * u.mV,
        U: ArrayLike = 0.35,
        tau_rec: ArrayLike = 36.169 * u.ms,
        tau_facil: ArrayLike = 58.517 * u.ms,
        tau_1: ArrayLike = 0.1 * u.ms,
        M: ArrayLike = 7.506,
        release_distance: ArrayLike = 0.978 * u.um,
        **kwargs,
    ):
        super().__init__(
            size=size, U=U, tau_rec=tau_rec, tau_facil=tau_facil,
            tau_1=tau_1, M=M, release_distance=release_distance, **kwargs,
        )
        self.g_max_alpha1 = braintools.init.param(
            g_max_alpha1, self.varshape, allow_none=False
        )
        self.g_max_factor = braintools.init.param(
            g_max_factor, self.varshape, allow_none=False
        )
        self.E_rev = braintools.init.param(E_rev, self.varshape, allow_none=False)

    def init_state(self, V_post=None, batch_size=None):
        super().init_state(V_post=V_post, batch_size=batch_size)
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.alpha1_states = tuple(
            brainstate.ShortTermState(zeros + (1.0 if index == 0 else 0.0))
            for index in range(8)
        )

    def reset_state(self, V_post=None, batch_size=None):
        super().reset_state(V_post=V_post, batch_size=batch_size)
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        for index, state in enumerate(self.alpha1_states):
            state.value = zeros + (1.0 if index == 0 else 0.0)

    @property
    def open_alpha1(self):
        return self.alpha1_states[6].value + self.alpha1_states[7].value

    @property
    def g_alpha1(self):
        return self.g_max_alpha1 * self._q10_conductance() * self.open_alpha1

    @property
    def g(self):
        return self.g_alpha1 * self.g_max_factor

    def _receptor_matrix(self, transmitter, rates):
        (
            kon,
            koff,
            d3,
            r3,
            d1d2,
            r1r2,
            a1,
            b1,
            a2,
            b2,
            r1,
            r2,
            d1,
            d2,
        ) = rates
        concentration = transmitter.to_decimal(u.mM)
        q10 = self._q10_rate()
        template = concentration + self.alpha1_states[0].value * 0.0
        matrix = jnp.zeros(template.shape + (8, 8), dtype=template.dtype)

        def rate(value):
            return q10 * value + template * 0.0

        matrix = _add_reversible_transition(
            matrix, 0, 1, rate(2.0 * kon) * concentration, rate(koff)
        )
        matrix = _add_reversible_transition(
            matrix, 1, 2, rate(kon) * concentration, rate(2.0 * koff)
        )
        matrix = _add_reversible_transition(matrix, 2, 5, rate(d3), rate(r3))
        matrix = _add_reversible_transition(
            matrix, 3, 4, rate(d1d2) * concentration, rate(r1r2)
        )
        matrix = _add_reversible_transition(matrix, 6, 1, rate(a1), rate(b1))
        matrix = _add_reversible_transition(matrix, 7, 2, rate(a2), rate(b2))
        matrix = _add_reversible_transition(matrix, 3, 1, rate(r1), rate(d1))
        return _add_reversible_transition(matrix, 4, 2, rate(r2), rate(d2))

    def _advance_alpha1(self, diffusion):
        alpha1_matrix = self._receptor_matrix(
            self._local_transmitter_pulse() + diffusion,
            self.alpha1_rates,
        )
        _backward_euler_markov(self.alpha1_states, alpha1_matrix)

    def pre_integral(self, V_post=None):
        self._advance_alpha1(self._diffused_transmitter())

    def current(self, V_post):
        return self.g * (self.E_rev - V_post)


@register_synapse("GrCGABA2020")
class GrCGABA2020(_GABAAlpha1):
    """Paper GoC→GrC GABA with alpha1 and diffusion-driven alpha6 receptors."""

    alpha6_rates = (
        54.8, 0.31061, 15.0, 7.4132, 24.2, 0.091668, 0.06,
        0.03, 0.4, 10.0, 0.040001, 0.4316, 0.86042, 2.7012,
    )

    def __init__(self, size, *, g_max_alpha6: ArrayLike = 0.000132842 * u.uS, **kwargs):
        super().__init__(size, **kwargs)
        self.g_max_alpha6 = braintools.init.param(
            g_max_alpha6, self.varshape, allow_none=False
        )

    def init_state(self, V_post=None, batch_size=None):
        super().init_state(V_post=V_post, batch_size=batch_size)
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.alpha6_states = tuple(
            brainstate.ShortTermState(zeros + (1.0 if index == 0 else 0.0))
            for index in range(8)
        )

    def reset_state(self, V_post=None, batch_size=None):
        super().reset_state(V_post=V_post, batch_size=batch_size)
        zeros = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        for index, state in enumerate(self.alpha6_states):
            state.value = zeros + (1.0 if index == 0 else 0.0)

    @property
    def open_alpha6(self):
        return self.alpha6_states[6].value + self.alpha6_states[7].value

    @property
    def g_alpha6(self):
        return self.g_max_alpha6 * self._q10_conductance() * self.open_alpha6

    @property
    def g(self):
        return (self.g_alpha1 + self.g_alpha6) * self.g_max_factor

    def pre_integral(self, V_post=None):
        diffusion = self._diffused_transmitter()
        self._advance_alpha1(diffusion)
        alpha6_matrix = self._receptor_matrix(diffusion, self.alpha6_rates)
        _backward_euler_markov(self.alpha6_states, alpha6_matrix)


@register_synapse("GoCGABAAlpha1")
class GoCGABAAlpha1(_GABAAlpha1):
    """GoC→GoC proxy with the published 3.2 nS nominal conductance."""

    def __init__(self, size, *, g_max_alpha1: ArrayLike = 0.0032 * u.uS, **kwargs):
        super().__init__(size, g_max_alpha1=g_max_alpha1, **kwargs)


@register_synapse("GoCAMPA_MF2020")
class GoCAMPA_MF2020(GrCAMPA2020):
    """Paper GoC ``AMPA_MF`` alias for DBBS ``AMPA.granule``."""

    def __init__(self, size, *, U: ArrayLike = 0.42, **kwargs):
        super().__init__(size=size, U=U, **kwargs)


@register_synapse("GoCNMDA2020")
class GoCNMDA2020(GrCNMDA2020):
    """Paper GoC ``NMDA`` alias for DBBS ``NMDA.granule``."""

    def __init__(self, size, *, U: ArrayLike = 0.42, **kwargs):
        super().__init__(size=size, U=U, **kwargs)


@register_synapse("GoCAMPAPF2020")
class GoCAMPAPF2020(GrCAMPA2020):
    """DBBS default AMPA with the paper-era GoC AMPA_PF overrides."""

    uses_diffusion = False
    history_size = 1

    def __init__(
        self, size, *, U=0.4, tau_1=30.0 * u.ms, tau_rec=35.1 * u.ms,
        tau_facil=54.0 * u.ms, q10_diffusion=1.1, r2=0.82 / u.ms, **kwargs
    ):
        super().__init__(
            size, U=U, tau_1=tau_1, tau_rec=tau_rec, tau_facil=tau_facil,
            q10_diffusion=q10_diffusion, r2=r2, **kwargs
        )

    def _facilitate(self, u_decay):
        return u_decay + self.U * (1.0 + 30.0 * u_decay) * (
            u.math.exp(-5.0 * u_decay) - u.math.exp(-5.0)
        )


@register_synapse("GoCAMPA_AA2020")
class GoCAMPA_AA2020(GoCAMPAPF2020):
    """Masoli et al. (2020), ModelDB 266806 syn_AA2GoC_det using Golgi_PF_syn."""


__all__ = [
    "GoCAMPAPF2020",
    "GoCAMPA_AA2020",
    "GoCAMPA_MF2020",
    "GoCGABAAlpha1",
    "GoCNMDA2020",
    "GrCAMPA2020",
    "GrCGABA2020",
    "GrCNMDA2020",
]
