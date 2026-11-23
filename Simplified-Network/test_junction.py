"""Check mirrored contact pairing and current delivery through BrainCell."""

import unittest

import braincell as bc
import brainunit as u
import numpy as np
from braincell.filter import at

from junction import reciprocal_contacts


class JunctionTest(unittest.TestCase):
    def test_reciprocal_rows_preserve_parallel_and_same_cell_contacts(self):
        cells = np.array([[2, 1], [1, 2], [1, 2], [2, 1], [2, 2], [2, 2]])
        compartments = np.array([[8, 7], [7, 8], [7, 8], [8, 7], [8, 9], [9, 8]])
        expected = [[[1, 7], [2, 8]], [[1, 7], [2, 8]], [[2, 8], [2, 9]]]
        np.testing.assert_array_equal(reciprocal_contacts(cells, compartments), expected)
        with self.assertRaisesRegex(ValueError, "reciprocally mirrored"):
            reciprocal_contacts(cells[1:], compartments[1:])

    @unittest.skipUnless(bc.__version__ == "0.2.0", "requires the archived BrainCell 0.2.0 runtime")
    def test_network_delivers_equal_and_opposite_current(self):
        from explicit_junction import ExplicitJunction, RECORDING

        soma = bc.Branch.from_lengths(lengths=[20] * u.um, radii=[8, 8] * u.um, type="soma")
        cell = bc.Cell(bc.Morphology.from_root(soma, name="soma"), pop_size=(2,), V_init=-65.0 * u.mV)
        junction = ExplicitJunction(
            np.array([[[0, 0], [1, 0]]]), np.array([[0, 1]]), np.zeros((1, 2), dtype=int)
        )
        junction._place(cell, [[at("soma", 0.5), at("soma", 0.5)]])
        net = bc.Network("passive_pair")
        net.add_population("golgi", cell)
        net.init_state()
        cell.V.value = np.array([[-70.0], [-50.0]]) * u.mV

        result = net.run(dt=0.05 * u.ms, duration=0.05 * u.ms)
        current = result.samples["golgi"][RECORDING].values.to_decimal(u.pA)
        np.testing.assert_allclose(current, [[20.0, -20.0]], atol=1e-5)
        voltage = np.asarray(cell.V.value.to_decimal(u.mV))[:, 0]
        self.assertGreater(voltage[0], -70.0)
        self.assertLess(voltage[1], -50.0)


if __name__ == "__main__":
    unittest.main()
