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

"""Calcium pool used by the Shai et al. (2015) L5PC model."""

from typing import Optional

import braintools
import brainunit as u

from braincell._typing import Initializer, Size
from braincell.ion import Calcium
from braincell.ion._base import DynamicNernstIon
from braincell.mech import register_ion

__all__ = ["CaDynamics_E2_L5PC"]


@register_ion("CaDynamics_E2_L5PC")
class CaDynamics_E2_L5PC(Calcium, DynamicNernstIon):
    r"""Port ``CaDynamics_E2.mod`` for the Shai L5PC model.

    The calcium concentration follows the source mechanism's shell equation,
    while ``E`` remains the fixed NEURON calcium reversal potential.  The
    latter is intentional: ``CaDynamics_E2.mod`` writes ``cai`` but does not
    update ``eca``.
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(34.0),
        gamma: Initializer = 0.05,
        decay: Initializer = 80.0 * u.ms,
        depth: Initializer = 0.1 * u.um,
        minCai: Initializer = 1e-4 * u.mM,
        E: Initializer = 140.0 * u.mV,
        Co: Optional[Initializer] = None,
        Ci_initializer: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        if Ci_initializer is None:
            Ci_initializer = minCai
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )
        self.gamma = braintools.init.param(gamma, self.varshape, allow_none=False)
        self.decay = braintools.init.param(decay, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)
        self.minCai = braintools.init.param(minCai, self.varshape, allow_none=False)
        self._fixed_E = braintools.init.param(E, self.varshape, allow_none=False)

    @property
    def E(self):
        """Return the fixed calcium reversal used by the HOC model."""
        return self._fixed_E

    def derivative(self, Ci, V, total_current=None):
        _ = V
        total_current = self._drive_current(total_current)
        drive = u.math.maximum(
            self.gamma
            * total_current.to_decimal(u.mA / u.cm**2)
            * 1e4
            / (2.0 * 96485.0 * self.depth.to_decimal(u.um))
            * (u.mM / u.ms),
            u.math.zeros_like(total_current.to_decimal(u.mA / u.cm**2)) * (u.mM / u.ms),
        )
        return drive - (Ci - self.minCai) / self.decay
