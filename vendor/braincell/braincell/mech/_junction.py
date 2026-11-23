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

"""Point declarations for bidirectional electrical connections."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

from braincell.mech._params import Params
from braincell.mech._point import Point

__all__ = ["Junction", "JunctionEndpoint"]


@dataclass(frozen=True, eq=False)
class JunctionEndpoint:
    """Select ordered morphology locations on members of a Cell population.

    Parameters
    ----------
    cell : Cell
        Cell registered in a Network.
    locations : LocsetExpr or LocsetMask
        One morphology location per contact row in order.
    population_indices : int or sequence of int, optional
        Population member per row. A scalar applies to every location.
    """

    cell: object
    locations: object
    population_indices: int | tuple[int, ...] = 0

    def __post_init__(self) -> None:
        indices = self.population_indices
        if isinstance(indices, Integral) and not isinstance(indices, bool):
            return
        try:
            indices = tuple(indices)
        except TypeError as exc:
            raise TypeError("Junction population indices must be integers.") from exc
        if any(not isinstance(index, Integral) or isinstance(index, bool) for index in indices):
            raise TypeError("Junction population indices must be integers.")
        object.__setattr__(self, "population_indices", indices)


@dataclass(frozen=True)
class Junction(Point):
    """Declare a symmetric, ohmic gap junction.

    Parameters
    ----------
    params : Params or Mapping
        Parameter g is the non-negative conductance with brainunit units.
        It can be a scalar or supply one value per contact.
    partner : JunctionEndpoint, optional
        Endpoint opposite the endpoint passed to Network.connect_junction.
        A declaration without a partner remains valid as placement metadata.
    """

    params: Params = field(default_factory=Params)
    partner: JunctionEndpoint | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", Params.coerce(self.params))
        if self.partner is not None and not isinstance(self.partner, JunctionEndpoint):
            raise TypeError("Junction partner must be a JunctionEndpoint.")
