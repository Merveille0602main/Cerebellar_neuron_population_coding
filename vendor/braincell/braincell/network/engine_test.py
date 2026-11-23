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

"""Integration tests for network-owned junctions."""

import unittest

import brainstate
import brainunit as u
import numpy as np

from braincell import CVPerBranch, Cell
from braincell.filter import AllRegion, LocsetMask, at
from braincell.mech import CableProperty, Junction, JunctionEndpoint
from braincell.network import Network
from braincell.network._testing import make_soma_tree, make_two_point_tree


def _identity_solver(cell):
    pass


def _make_cell(initial_voltage, *, size=1, morphology=None, threshold=0.0, solver=_identity_solver):
    return Cell(
        make_soma_tree() if morphology is None else morphology,
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=initial_voltage,
        V_th=threshold * u.mV,
        solver=solver,
    )


def _soma(cell, member=0):
    return JunctionEndpoint(cell, at("soma", 0.5), member)


class GapJunctionTest(unittest.TestCase):
    def test_parallel_contacts_match_two_cell_solution(self):
        with brainstate.environ.context(precision=64):
            left = _make_cell(-50.0 * u.mV, solver="staggered")
            right = _make_cell(-70.0 * u.mV, threshold=-65, solver="staggered")
            net = Network()
            net.add_population("right", right)
            net.add_population("left", left)
            locations = LocsetMask.from_columns([0, 0], [0.5, 0.5])
            net.connect_junction(
                "gap",
                endpoint=JunctionEndpoint(left, locations),
                junction=Junction(
                    params={"g": np.array([0.0004, 0.0006]) * u.uS},
                    partner=JunctionEndpoint(right, locations),
                ),
            )
            step = 10 * u.ms
            capacitance_pf = (left.cvs[0].cm * left.cvs[0].area).to_decimal(u.pF)
            decay = 1.0 / (1.0 + 2.0 * step.to_decimal(u.ms) / capacitance_pf)
            result = net.run(dt=step, duration=step)
            self.assertEqual(len(result.events["right"]["spike"].source_id), 1)
            for n in (1, 2):
                v_left = left.V.value.to_decimal(u.mV)
                v_right = right.V.value.to_decimal(u.mV)
                np.testing.assert_allclose(v_left + v_right, -120.0, atol=2e-4)
                np.testing.assert_allclose(v_left - v_right, 20.0 * decay ** n, atol=2e-4)
                if n == 1:
                    net.run(dt=step, duration=step)

    def test_same_cell_different_compartments_can_couple(self):
        net = Network()
        cell = _make_cell(-65 * u.mV, morphology=make_two_point_tree(), solver='staggered')
        net.add_population("cell", cell)
        net.connect_junction(
            "self",
            endpoint=_soma(cell),
            junction=Junction(params={"g": 2.0 * u.nS}, partner=JunctionEndpoint(cell, at("dend", 0.5))),
        )
        net.init_state()
        cell.V.value = np.array([[-55.0, -75.0]]) * u.mV
        before = cell.V.value.to_decimal(u.mV)
        net.run(dt=0.05 * u.ms, duration=0.05 * u.ms)
        after = cell.V.value.to_decimal(u.mV)
        self.assertLess(after[0, 0], before[0, 0])
        self.assertGreater(after[0, 1], before[0, 1])

    def test_validation_and_zero_conductance(self):
        net = Network()
        cell = _make_cell(np.array([[-54.0], [-70.0]]) * u.mV, size=2)
        net.add_population("cells", cell)
        left, right = _soma(cell, 0), _soma(cell, 1)
        with self.assertRaisesRegex(ValueError, "partner"):
            net.connect_junction("gap", endpoint=left, junction=Junction())
        with self.assertRaisesRegex(ValueError, "out of range"):
            net.connect_junction(
                "gap", endpoint=left, junction=Junction(params={"g": 1 * u.nS}, partner=_soma(cell, 2))
            )
        with self.assertRaisesRegex(TypeError, "units"):
            net.connect_junction("gap", endpoint=left, junction=Junction(params={"g": 1}, partner=right))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            net.connect_junction("gap", endpoint=left, junction=Junction(params={"g": -1 * u.nS}, partner=right))
        net.connect_junction("gap", endpoint=left, junction=Junction(params={"g": 0 * u.nS}, partner=right))
        net.run(dt=0.1 * u.ms, duration=0.1 * u.ms)
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), [[-54.0], [-70.0]], atol=1e-4)

    def test_coupled_step_matches_monolithic_cable_matrix(self) -> None:
        with brainstate.environ.context(precision=64):
            voltages = np.array([[-55.0, -72.0], [-68.0, -80.0]])
            left, right = (
                _make_cell(v[None, :] * u.mV, morphology=make_two_point_tree(), solver="staggered")
                for v in voltages
            )
            right.paint(
                AllRegion(),
                CableProperty(
                    resting_potential=-65.0 * u.mV,
                    axial_resistivity=100.0 * u.ohm * u.cm,
                    membrane_capacitance=2.0 * u.uF / u.cm**2,
                ),
            )
            net = Network()
            net.add_population("right", right)
            net.add_population("left", left)
            net.connect_junction(
                "gap",
                endpoint=JunctionEndpoint(left, LocsetMask([(1, 0.5), (0, 0.5)])),
                junction=Junction(
                    params={"g": np.array([7.0, 3.0]) * u.nS},
                    partner=JunctionEndpoint(right, LocsetMask([(1, 0.5), (1, 0.5)])),
                ),
            )
            net.init_state()
            dt_ms = 0.25
            mass = np.array([float((cv.area * cv.cm).to_decimal(u.pF)) for cell in (left, right) for cv in cell.cvs])
            matrix = np.diag(mass / dt_ms)
            for start, cell in ((0, left), (2, right)):
                span = slice(start, start + 2)
                axial = np.asarray(cell._get_axial_operator().to_decimal(u.ms**-1))
                matrix[span, span] += mass[span, None] * axial
            for a, b, g in ((1, 3, 7.0), (0, 3, 3.0)):
                matrix[np.ix_([a, b], [a, b])] += g * np.array([[1.0, -1.0], [-1.0, 1.0]])
            expected = np.linalg.solve(matrix, mass / dt_ms * voltages.ravel())
            net.run(dt=dt_ms * u.ms, duration=dt_ms * u.ms)
            measured = np.concatenate([cell.V.value.to_decimal(u.mV)[0] for cell in (left, right)])
            np.testing.assert_allclose(measured, expected, atol=1e-7, rtol=0.0)
            np.testing.assert_allclose(mass @ measured, mass @ voltages.ravel(), atol=1e-6, rtol=0.0)

    def test_junction_requires_staggered_solver_only_for_active_populations(self) -> None:
        for conductance in (0.0, 1.0):
            net = Network()
            cell = _make_cell(np.array([[-55.0], [-70.0]]) * u.mV, size=2)
            net.add_population("cells", cell)
            net.connect_junction(
                "gap", endpoint=_soma(cell, 0),
                junction=Junction(params={"g": conductance * u.nS}, partner=_soma(cell, 1)),
            )
            if conductance:
                with self.assertRaisesRegex(ValueError, "requires? the staggered"):
                    net.run(dt=0.05 * u.ms, duration=0.05 * u.ms)
            else:
                net.run(dt=0.05 * u.ms, duration=0.05 * u.ms)
                np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), [[-55.0], [-70.0]])
