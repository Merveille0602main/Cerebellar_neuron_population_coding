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

"""Lower and integrate network-owned electrical contacts."""

from __future__ import annotations

from dataclasses import dataclass

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from braincell.event import _resolve_cell_location_cvs
from braincell.mech import JunctionEndpoint
from braincell.quad._staggered import _prepare_dhs_voltage_system, _solve_dhs_voltage_system, staggered_step


def _endpoint_rows(endpoint: JunctionEndpoint, size: int):
    cv = _resolve_cell_location_cvs(endpoint.cell, endpoint.locations)
    indices = np.asarray(endpoint.population_indices, dtype=np.int64)
    if indices.ndim == 0:
        indices = np.full(cv.size, indices, dtype=np.int64)
    if cv.size == 0 or indices.shape != cv.shape:
        raise ValueError("Each junction endpoint requires one population index per location.")
    if np.any((indices < 0) | (indices >= size)):
        raise ValueError("Junction population index is out of range.")
    return indices, cv


def _conductance_rows(conductance, size):
    if not isinstance(conductance, u.Quantity) or not u.have_same_dim(conductance, u.nS):
        raise TypeError("Junction conductance g must have conductance units.")
    values = np.asarray(conductance.to_decimal(u.nS), dtype=float)
    if values.ndim == 0:
        values = np.full(size, values)
    if values.shape != (size,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Junction g must be a finite non-negative scalar or one value per contact.")
    return values


@dataclass(frozen=True)
class JunctionBatch:
    """One named batch of physical electrical contacts."""

    left_name: str
    right_name: str
    left: JunctionEndpoint
    right: JunctionEndpoint
    conductance_ns: np.ndarray

    @classmethod
    def create(cls, left_population, right_population, left, junction):
        """Validate aligned endpoint lists and a conductance per row."""
        left_member, _ = _endpoint_rows(left, left_population.size)
        right_member, _ = _endpoint_rows(junction.partner, right_population.size)
        if left_member.size != right_member.size:
            raise ValueError("Junction endpoints must have the same number of rows.")
        if set(junction.params) != {"g"}:
            raise ValueError("A connected Junction requires exactly the conductance parameter g.")
        conductance = _conductance_rows(junction.params["g"], left_member.size)
        conductance = conductance.copy()
        conductance.flags.writeable = False
        return cls(left_population.name, right_population.name, left, junction.partner, conductance)


@dataclass(frozen=True)
class JunctionSolve:
    """Compact, bidirectional conductance graph for the network voltage step."""

    rows: tuple
    left: np.ndarray
    right: np.ndarray
    conductance_ns: np.ndarray
    capacitance_pf: np.ndarray

    @classmethod
    def build(cls, batches, populations):
        """Resolve current discretizations and construct the compact graph."""
        nodes = {}
        edges = []
        conductances = []
        for batch in batches:
            left = _endpoint_rows(batch.left, populations[batch.left_name].size)
            right = _endpoint_rows(batch.right, populations[batch.right_name].size)
            for (a, i), (b, j), g in zip(zip(*left), zip(*right), batch.conductance_ns, strict=True):
                first = (batch.left_name, int(a), int(i))
                second = (batch.right_name, int(b), int(j))
                if first == second or g == 0:
                    continue
                edges.append((nodes.setdefault(first, len(nodes)), nodes.setdefault(second, len(nodes))))
                conductances.append(g)
        if not edges:
            return None

        groups = {}
        capacitance = []
        for (name, member, cv_id), row in nodes.items():
            groups.setdefault(name, []).append((member, cv_id, row))
            cv = populations[name].cell.cvs[cv_id]
            capacitance.append(float((cv.cm * cv.area).to_decimal(u.pF)))
        capacitance = np.asarray(capacitance)
        if not np.all(np.isfinite(capacitance)) or np.any(capacitance <= 0):
            raise ValueError("Junction endpoints require positive finite membrane capacitance.")

        left, right = np.asarray(edges, dtype=np.int32).T
        groups = tuple((name, *np.asarray(rows, dtype=np.int32).T) for name, rows in groups.items())
        return cls(groups, left, right, np.asarray(conductances), capacitance)

    def prepare_voltages(self, populations):
        """Promote integer initial voltages before the fixed-dtype scan."""
        for name, *_ in self.rows:
            cell = populations[name].cell
            voltage = cell.V.value
            if not jnp.issubdtype(voltage.dtype, jnp.inexact):
                cell.V.value = voltage.astype(brainstate.environ.dftype())
                cell._event_previous_V.value = cell.V.value

    def validate_solver(self, populations) -> None:
        """Require the staggered integrator for every coupled population."""
        for name, *_ in self.rows:
            if populations[name].cell.solver is not staggered_step:
                raise ValueError(f"Coupled junctions require the staggered solver on population {name!r}.")

    def advance(self, populations, t, dt):
        """Solve the locally linearized cables and all gap terms together."""
        systems = {}
        base = {}
        for name, *_ in self.rows:
            cell = populations[name].cell
            initial = cell.V.value
            cell.cache_ion_total_currents(initial)
            source, numeric = _prepare_dhs_voltage_system(cell, t, dt)
            systems[name] = source, numeric, initial.shape
            base[name] = _solve_dhs_voltage_system(source, numeric, initial.shape)

        dtype = jnp.result_type(*(value.dtype for value in base.values()))
        weight = jnp.sqrt(dt.to_decimal(u.ms) * jnp.asarray(self.conductance_ns, dtype=dtype))
        capacitance = jnp.asarray(self.capacitance_pf, dtype=dtype)

        def edge_voltage(voltages):
            endpoints = jnp.zeros(self.capacitance_pf.size, dtype=dtype)
            for name, member, cv, row in self.rows:
                value = voltages[name].to_decimal(u.mV)
                endpoints = endpoints.at[row].set(value.reshape((-1, value.shape[-1]))[member, cv])
            return weight * (endpoints[self.left] - endpoints[self.right])

        def cable_response(edge_values):
            # Weighted incidence maps edge amplitudes to equal/opposite currents.
            # Scale by capacitance here because the local DHS rows are normalized.
            flow = weight * edge_values
            endpoints = jnp.zeros_like(capacitance).at[self.left].add(flow)
            endpoints = endpoints.at[self.right].add(-flow) / capacitance
            response = {}
            for name, member, cv, row in self.rows:
                source, numeric, shape = systems[name]
                rhs = jnp.zeros(numeric.solves.shape, dtype=numeric.solves.dtype)
                rhs = rhs.at[member, source.dynamic_rows_np[cv]].set(endpoints[row].astype(rhs.dtype))
                response[name] = _solve_dhs_voltage_system(source, numeric, shape, rhs=u.Quantity(rhs, u.mV))
            return response

        # Woodbury reduction of the full cable matrix plus the gap Laplacian.
        rhs = edge_voltage(base)
        if self.left.size == 1:
            response = cable_response(jnp.ones_like(rhs))
            amplitude = (rhs / (1.0 + edge_voltage(response)))[0]
            return {name: voltage - amplitude * response[name] for name, voltage in base.items()}

        correction, _ = cg(lambda amplitude: amplitude + edge_voltage(cable_response(amplitude)), rhs, tol=1e-7)
        response = cable_response(correction)
        return {name: voltage - response[name] for name, voltage in base.items()}
