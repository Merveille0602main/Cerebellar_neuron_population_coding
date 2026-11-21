# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Stimulus helpers shared by channel_no_conc."""



import math

import brainunit as u

try:
    from .experiment_schema import StimulusSpec
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from experiment_schema import StimulusSpec  # type: ignore


def current_at_ms(stimulus: StimulusSpec, t_ms: float) -> float:
    kind = getattr(stimulus, "kind", None)
    if kind == "dc":
        return stimulus.amp_nA if stimulus.delay_ms <= t_ms < (stimulus.delay_ms + stimulus.dur_ms) else 0.0

    if kind == "sine":
        if not (stimulus.start_ms <= t_ms < (stimulus.start_ms + stimulus.duration_ms)):
            return 0.0
        local_t_sec = (t_ms - stimulus.start_ms) / 1000.0
        angle = (2.0 * math.pi * stimulus.frequency_hz * local_t_sec) + stimulus.phase_rad
        return stimulus.offset_nA + (stimulus.amplitude_nA * math.sin(angle))

    raise TypeError(f"Unsupported stimulus type {type(stimulus).__name__!s}.")


def sine_step_phase_shift(frequency_hz: float, dt_ms: float) -> float:
    """Return the phase that aligns a sampled sine with NEURON's `IClamp`.

    NEURON evaluates a vector-played `IClamp` amplitude at the *start* of the
    step, while BrainCell evaluates a `SineClamp` at the step midpoint. Both
    conventions were measured on a channel-free single compartment by
    differencing the recorded voltage against `I = C dV/dt`:

    ========  =================================  ===============
    side      current applied for the step to t  fit residual
    ========  =================================  ===============
    NEURON    `sin(w (t - dt))`                 `8.8e-15`
    BrainCell `sin(w (t - dt / 2))`             `1.1e-05`
    ========  =================================  ===============

    Advancing the BrainCell phase by `-w dt / 2` therefore makes the two
    applied current sequences identical instead of offset by half a step.
    The shift is a discretization detail, not a change of the stimulus: at
    `dt = 0.01 ms` and `250 Hz` it is `0.0045 rad`, but it moves the
    voltage trace by `dV/dt dt/2`, which is `1e-02 mV` at the `ac`
    template amplitudes.

    Parameters
    ----------
    frequency_hz : float
        Sine frequency in Hz.
    dt_ms : float
        Integration step in ms.

    Returns
    -------
    float
        Phase increment in radians to add to `StimulusSpec.phase_rad`.
    """
    return -math.pi * float(frequency_hz) * float(dt_ms) / 1000.0


def build_braincell_stimulus(stimulus: StimulusSpec, *, dt_ms: float | None = None):
    """Build the BrainCell counterpart of a :class:`StimulusSpec`.

    Parameters
    ----------
    stimulus : StimulusSpec
        Declarative stimulus description shared with the NEURON runner.
    dt_ms : float or None, optional
        Integration step in ms. When given and `stimulus.kind` is
        `'sine'`, the clamp phase is shifted by
        :func:`sine_step_phase_shift` so that the per-step current sequence
        matches the one NEURON applies. `None` keeps the analytic clamp and
        therefore reintroduces the half-step offset.
    """
    import braincell

    kind = getattr(stimulus, "kind", None)
    if kind == "dc":
        return braincell.CurrentClamp(
            delay=stimulus.delay_ms * u.ms,
            durations=stimulus.dur_ms * u.ms,
            amplitudes=stimulus.amp_nA * u.nA,
        )

    if kind == "sine":
        phase = float(stimulus.phase_rad)
        if dt_ms is not None:
            phase += sine_step_phase_shift(stimulus.frequency_hz, dt_ms)
        return braincell.SineClamp(
            amplitude=stimulus.amplitude_nA * u.nA,
            frequency=stimulus.frequency_hz * u.Hz,
            phase=phase,
            offset=stimulus.offset_nA * u.nA,
            delay=stimulus.start_ms * u.ms,
            duration=stimulus.duration_ms * u.ms,
        )

    raise TypeError(f"Unsupported stimulus type {type(stimulus).__name__!s}.")
