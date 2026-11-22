from __future__ import annotations

import h5py
import numpy as np

from cerebellar_reproduction.bsb_legacy import read_published_network
from cerebellar_reproduction.export import write_spec
from cerebellar_reproduction.validation import validate_export


def _fixture(path):
    with h5py.File(path, "w") as handle:
        cells = handle.create_group("cells")
        cells.create_dataset(
            "positions",
            data=np.array([[10, 1, 1.0, 2.0, 3.0], [11, 2, 4.0, 5.0, 6.0], [12, 2, 7.0, 8.0, 9.0]]),
        )
        type_maps = cells.create_group("type_maps")
        type_maps.create_dataset("glomerulus_map", data=np.array([10]))
        type_maps.create_dataset("granule_cell_map", data=np.array([11, 12]))
        type_maps.create_dataset("mossy_fibers_map", data=np.array([], dtype=np.int64))
        placement = cells.create_group("placement")
        for name, start, positions in (
            ("glomerulus", 10, [[1.0, 2.0, 3.0]]),
            ("granule_cell", 11, [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ):
            group = placement.create_group(name)
            group.create_dataset("identifiers", data=np.array([start, len(positions)]))
            group.create_dataset("positions", data=np.asarray(positions))
        mossy = placement.create_group("mossy_fibers")
        mossy.create_dataset("identifiers", data=np.array([100, 2]))
        connections = cells.create_group("connections")
        connections.create_dataset("glomerulus_to_granule", data=np.array([[10.0, 11.0], [10.0, 12.0]]))
        connections.create_dataset("mossy_to_glomerulus", data=np.array([[100.0, 10.0], [101.0, 10.0]]))
        compartments = cells.create_group("connection_compartments")
        compartments.create_dataset("glomerulus_to_granule", data=np.array([[0, 4], [0, 5]]))


def test_reads_and_exports_legacy_bsb_network(tmp_path):
    source = tmp_path / "balanced.hdf5"
    _fixture(source)
    spec = read_published_network(source)

    assert spec.populations["glomerulus"].size == 1
    assert np.array_equal(spec.populations["granule_cell"].positions[1], [7.0, 8.0, 9.0])
    assert np.array_equal(spec.populations["mossy_fibers"].source_ids, [100, 101])
    connection = spec.connectivity["glomerulus_to_granule"]
    assert connection.size == 2
    assert np.array_equal(connection.compartment_pairs[:, 1], [4, 5])

    manifest = write_spec(spec, tmp_path / "export")
    assert manifest["connectivity"]["glomerulus_to_granule"]["n_contact"] == 2
    data = np.load(tmp_path / "export/connectivity/glomerulus_to_granule.npz")
    assert np.array_equal(data["cell_pairs"], [[10, 11], [10, 12]])
    assert np.array_equal(data["local_pairs"], [[0, 0], [0, 1]])
    assert np.array_equal(data["edge_pairs"], [[0, 0], [0, 1]])
    assert np.array_equal(data["contact_to_edge"], [0, 1])
    validate_export(tmp_path / "export")
