from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest

bc = pytest.importorskip("braincell")
pytestmark = pytest.mark.skipif(not hasattr(bc.Network, "add_edges"), reason="requires the legacy edge-set API")

from cerebellar_reproduction.braincell_adapter import (
    add_edge_sets,
    add_populations,
    place_mapped_synapse_pool,
)


@dataclass
class StubCell:
    pop_size: tuple[int, ...]


def test_lowers_exported_edges_to_braincell(tmp_path):
    import braincell

    populations = tmp_path / "populations"
    connectivity = tmp_path / "connectivity"
    populations.mkdir()
    connectivity.mkdir()
    np.savez_compressed(populations / "a.npz", source_ids=np.array([10, 11]))
    np.savez_compressed(populations / "b.npz", source_ids=np.array([20, 21, 22]))
    np.savez_compressed(
        connectivity / "a_to_b.npz",
        cell_pairs=np.array([[10, 20], [11, 22], [11, 22]]),
        local_pairs=np.array([[0, 0], [1, 2], [1, 2]], dtype=np.int32),
        edge_pairs=np.array([[0, 0], [1, 2]], dtype=np.int32),
        contact_to_edge=np.array([0, 1, 1], dtype=np.int32),
    )
    (tmp_path / "manifest.json").write_text(
        """{
          "populations": {
            "a": {"file": "populations/a.npz", "n_cell": 2},
            "b": {"file": "populations/b.npz", "n_cell": 3}
          },
          "connectivity": {
            "a_to_b": {
              "file": "connectivity/a_to_b.npz", "pre_population": "a",
              "post_population": "b", "n_edge": 2, "n_contact": 3
            }
          }
        }"""
    )
    network = braincell.Network(name="adapter-test")
    add_populations(network, tmp_path, {"a": StubCell((2,)), "b": StubCell((3,))})
    edges = add_edge_sets(network, tmp_path)
    assert edges["a_to_b"].n_edge == 2
    assert np.array_equal(edges["a_to_b"].pre_index, [0, 1])
    assert np.array_equal(edges["a_to_b"].post_index, [0, 2])


def test_mapped_granule_compartments_form_a_real_synapse_pool():
    root = Path(__file__).parents[1]
    braincell_repo = root / "assets/braincell"
    if not braincell_repo.exists():
        return
    if str(braincell_repo) not in sys.path:
        sys.path.insert(0, str(braincell_repo))

    from braincell import mech
    from examples.neuron_compare.cell.grc_ma2020.grc_braincell import GrC
    from examples.neuron_compare.cell.grc_ma2020.parameters import load_grc20_params
    from cerebellar_reproduction.morphology_mapping import build_morphology_mapping

    source = root / "data/reference/balanced.hdf5"
    if not source.exists():
        return
    granule = GrC(params=load_grc20_params(), pop_size=(1,)).build()
    mapping = build_morphology_mapping(source, "GranuleCell", granule.morpho)
    compartments = np.asarray([121, 130, 139, 148])
    pool = place_mapped_synapse_pool(
        granule.cell,
        mapping,
        compartments,
        mech.Synapse("AMPA", name="glomerulus_ampa"),
    )
    indices = pool.contact_indices(compartments)
    assert pool.n_active == 4
    assert np.array_equal(indices, [0, 1, 2, 3])
