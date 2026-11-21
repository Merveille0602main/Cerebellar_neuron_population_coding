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

"""Additional channel ports for the Shai et al. (2015) L5PC model.

The classes in this module are additive imports.  They intentionally keep the
source mechanism names in their registry keys so a model assembled from the
ModelDB HOC files can be audited against the original ``.mod`` files.
"""

from typing import Optional

import braintools
import brainstate
import brainunit as u

from braincell._base_channel import IonInfo
from braincell._typing import ArrayLike, Initializer, Size
from braincell._base_neuron import HHTypedNeuron
from braincell.channel._base import Gate, OhmicHH, q10_factor
from braincell.ion import Calcium, Potassium, Sodium
from braincell.mech import register_channel

__all__ = [
    "NaTs2_t_L5PC",
    "NaTa_t_L5PC",
    "Nap_Et2_L5PC",
    "Ca_LVAst_L5PC",
    "Ca_HVA_L5PC",
    "K_Tst_L5PC",
    "K_Pst_L5PC",
    "SKv3_1_L5PC",
    "SK_E2_L5PC",
    "Im_L5PC",
    "Ih_L5PC",
]

_TEMP_REF_21C = u.celsius2kelvin(21.0)
_TEMP_TARGET_34C = u.celsius2kelvin(34.0)
_QT = 2.3 ** ((34.0 - 21.0) / 10.0)
_TEMP_REF_34C = u.celsius2kelvin(34.0)


class _L5AlphaBeta(OhmicHH):
    """Shared constructor for L5PC alpha/beta mechanisms."""

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 0.01 * (u.mS / u.cm**2),
        temp: ArrayLike = _TEMP_TARGET_34C,
        q10: Initializer = 2.3,
        temp_ref: ArrayLike = _TEMP_REF_21C,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)

    def _phi(self):
        return q10_factor(self.q10, self.temp, self.temp_ref)


class _L5InfTau(_L5AlphaBeta):
    """Shared HH gate methods for mechanisms specified by inf/tau curves."""

    def _rate_phi(self):
        return self._phi()


@register_channel("NaTs2_t_L5PC")
class NaTs2_t_L5PC(_L5AlphaBeta):
    """Transient somatic sodium current from ``NaTs2_t.mod``."""

    root_type = Sodium
    gates = (
        Gate("m", power=3, q10="q10", temp_ref="temp_ref"),
        Gate("h", q10="q10", temp_ref="temp_ref"),
    )

    def f_m_alpha(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return 0.182 * (v + 32.0) / (1.0 - u.math.exp(-(v + 32.0) / 6.0))

    def f_m_beta(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return 0.124 * (-v - 32.0) / (1.0 - u.math.exp(-(-v - 32.0) / 6.0))

    def f_h_alpha(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return -0.015 * (v + 60.0) / (1.0 - u.math.exp((v + 60.0) / 6.0))

    def f_h_beta(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return -0.015 * (-v - 60.0) / (1.0 - u.math.exp((-v - 60.0) / 6.0))


@register_channel("NaTa_t_L5PC")
class NaTa_t_L5PC(NaTs2_t_L5PC):
    """Axonal sodium current from ``NaTa_t.mod``."""

    def f_m_alpha(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return 0.182 * (v + 38.0) / (1.0 - u.math.exp(-(v + 38.0) / 6.0))

    def f_m_beta(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return 0.124 * (-v - 38.0) / (1.0 - u.math.exp(-(-v - 38.0) / 6.0))

    def f_h_alpha(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return -0.015 * (v + 66.0) / (1.0 - u.math.exp((v + 66.0) / 6.0))

    def f_h_beta(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return -0.015 * (-v - 66.0) / (1.0 - u.math.exp((-v - 66.0) / 6.0))


@register_channel("Nap_Et2_L5PC")
class Nap_Et2_L5PC(_L5AlphaBeta):
    """Persistent sodium current from ``Nap_Et2.mod``."""

    root_type = Sodium
    gates = (
        Gate("m", power=3, q10="q10", temp_ref="temp_ref"),
        Gate("h", q10="q10", temp_ref="temp_ref"),
    )

    def f_m_inf(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((v + 52.6) / -4.6))

    def f_m_tau(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        alpha = 0.182 * (v + 38.0) / (1.0 - u.math.exp(-(v + 38.0) / 6.0))
        beta = 0.124 * (-v - 38.0) / (1.0 - u.math.exp(-(-v - 38.0) / 6.0))
        return 6.0 / (alpha + beta)

    def f_h_inf(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((v + 48.8) / 10.0))

    def f_h_tau(self, V, Na: IonInfo):
        v = V.to_decimal(u.mV)
        alpha = -2.88e-6 * (v + 17.0) / (1.0 - u.math.exp((v + 17.0) / 4.63))
        beta = 6.94e-6 * (v + 64.4) / (1.0 - u.math.exp(-(v + 64.4) / 2.63))
        return 1.0 / (alpha + beta)


@register_channel("Ca_LVAst_L5PC")
class Ca_LVAst_L5PC(_L5AlphaBeta):
    """Low-voltage calcium current from ``Ca_LVAst.mod``."""

    root_type = Calcium
    gates = (
        Gate("m", power=2, q10="q10", temp_ref="temp_ref"),
        Gate("h", q10="q10", temp_ref="temp_ref"),
    )

    def f_m_inf(self, V, Ca: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 1.0 / (1.0 + u.math.exp((v + 30.0) / -6.0))

    def f_m_tau(self, V, Ca: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return (5.0 + 20.0 / (1.0 + u.math.exp((v + 25.0) / 5.0)))

    def f_h_inf(self, V, Ca: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 1.0 / (1.0 + u.math.exp((v + 80.0) / 6.4))

    def f_h_tau(self, V, Ca: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 20.0 + 50.0 / (1.0 + u.math.exp((v + 40.0) / 7.0))


@register_channel("Ca_HVA_L5PC")
class Ca_HVA_L5PC(_L5AlphaBeta):
    """High-voltage calcium current from ``Ca_HVA.mod``."""

    root_type = Calcium
    gates = (Gate("m", power=2), Gate("h"))

    def f_m_alpha(self, V, Ca: IonInfo):
        v = V.to_decimal(u.mV)
        return 0.055 * (-27.0 - v) / (u.math.exp((-27.0 - v) / 3.8) - 1.0)

    def f_m_beta(self, V, Ca: IonInfo):
        v = V.to_decimal(u.mV)
        return 0.94 * u.math.exp((-75.0 - v) / 17.0)

    def f_h_alpha(self, V, Ca: IonInfo):
        v = V.to_decimal(u.mV)
        return 0.000457 * u.math.exp((-13.0 - v) / 50.0)

    def f_h_beta(self, V, Ca: IonInfo):
        v = V.to_decimal(u.mV)
        return 0.0065 / (u.math.exp((-v - 15.0) / 28.0) + 1.0)


@register_channel("K_Tst_L5PC")
class K_Tst_L5PC(_L5InfTau):
    """Transient potassium current from ``K_Tst.mod``."""

    root_type = Potassium
    gates = (
        Gate("m", power=4, q10="q10", temp_ref="temp_ref"),
        Gate("h", q10="q10", temp_ref="temp_ref"),
    )

    def f_m_inf(self, V, K: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 1.0 / (1.0 + u.math.exp(-v / 19.0))

    def f_m_tau(self, V, K: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 0.34 + 0.92 * u.math.exp(-((v + 71.0) / 59.0) ** 2)

    def f_h_inf(self, V, K: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 1.0 / (1.0 + u.math.exp(-(v + 66.0) / -10.0))

    def f_h_tau(self, V, K: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 8.0 + 49.0 * u.math.exp(-((v + 73.0) / 23.0) ** 2)


@register_channel("K_Pst_L5PC")
class K_Pst_L5PC(_L5InfTau):
    """Persistent potassium current from ``K_Pst.mod``."""

    root_type = Potassium
    gates = (
        Gate("m", power=2, q10="q10", temp_ref="temp_ref"),
        Gate("h", q10="q10", temp_ref="temp_ref"),
    )

    def f_m_inf(self, V, K: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 1.0 / (1.0 + u.math.exp(-(v + 1.0) / 12.0))

    def f_m_tau(self, V, K: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return u.math.where(v < -50.0, 1.25 + 175.03 * u.math.exp(0.026 * v), 1.25 + 13.0 * u.math.exp(-0.026 * v))

    def f_h_inf(self, V, K: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 1.0 / (1.0 + u.math.exp(-(v + 54.0) / -11.0))

    def f_h_tau(self, V, K: IonInfo):
        v = V.to_decimal(u.mV) + 10.0
        return 360.0 + (1010.0 + 24.0 * (v + 55.0)) * u.math.exp(-((v + 75.0) / 48.0) ** 2)


@register_channel("SKv3_1_L5PC")
class SKv3_1_L5PC(_L5InfTau):
    """Kv3.1 potassium current from ``SKv3_1.mod``."""

    root_type = Potassium
    gates = (Gate("m"),)

    def f_m_inf(self, V, K: IonInfo):
        v = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((v - 18.7) / -9.7))

    def f_m_tau(self, V, K: IonInfo):
        v = V.to_decimal(u.mV)
        return 4.0 / (1.0 + u.math.exp((v + 46.56) / -44.14))


@register_channel("Im_L5PC")
class Im_L5PC(_L5AlphaBeta):
    """M-current from ``Im.mod``."""

    root_type = Potassium
    gates = (Gate("m", q10="q10", temp_ref="temp_ref"),)

    def f_m_alpha(self, V, K: IonInfo):
        v = V.to_decimal(u.mV)
        return 3.3e-3 * u.math.exp(0.1 * (v + 35.0))

    def f_m_beta(self, V, K: IonInfo):
        v = V.to_decimal(u.mV)
        return 3.3e-3 * u.math.exp(-0.1 * (v + 35.0))


@register_channel("Ih_L5PC")
class Ih_L5PC(OhmicHH):
    """Hyperpolarization-activated current from ``Ih.mod``."""

    root_type = HHTypedNeuron
    gates = (Gate("m"),)

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 0.01 * (u.mS / u.cm**2),
        E: Initializer = -45.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)

    def f_m_alpha(self, V):
        v = V.to_decimal(u.mV)
        return 0.001 * 6.43 * (v + 154.9) / (u.math.exp((v + 154.9) / 11.9) - 1.0)

    def f_m_beta(self, V):
        v = V.to_decimal(u.mV)
        return 0.001 * 193.0 * u.math.exp(v / 33.1)


@register_channel("SK_E2_L5PC")
class SK_E2_L5PC(OhmicHH):
    """Calcium-activated SK current from ``SK_E2.mod``."""

    root_type = brainstate.mixin.JointTypes[Potassium, Calcium]
    current_owner_type = Potassium
    gates = (Gate("z"),)

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 0.001 * (u.mS / u.cm**2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

    def f_z_inf(self, V, K: IonInfo, Ca: IonInfo):
        cai = u.math.maximum(Ca.Ci.to_decimal(u.mM), 1e-7)
        return 1.0 / (1.0 + (0.00043 / cai) ** 4.8)

    def f_z_tau(self, V, K: IonInfo, Ca: IonInfo):
        return 1.0
