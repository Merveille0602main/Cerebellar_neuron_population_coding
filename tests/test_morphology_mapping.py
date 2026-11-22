from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from cerebellar_reproduction.morphology_mapping import (
    build_morphology_mapping,
    target_mapping_summary,
)


SOURCE = Path(__file__).parents[1] / "data/reference/balanced.hdf5"
BRAINCELL = Path("/home/merveille/copneuron/braincell")


@pytest.mark.skipif(not SOURCE.exists() or not BRAINCELL.exists(), reason="published integration data unavailable")
def test_all_published_postsynaptic_compartments_are_mappable():
    from braincell import Morphology

    cases = {
        "granule_cell": ("GranuleCell", "GrC/morphology/GrC.asc"),
        "golgi_cell": ("GolgiCell", "GoC/morphology/GoC.asc"),
        "purkinje_cell": ("PurkinjeCell", "PC/morphology/PC.asc"),
        "stellate_cell": ("StellateCell", "SC/morphology/SC.asc"),
        "basket_cell": ("BasketCell", "BC/morphology/BC.asc"),
    }
    mappings = {}
    for population, (name, relative) in cases.items():
        morphology = Morphology.from_asc(BRAINCELL / "examples/neuron_compare/Cerebellum_mod" / relative)
        mappings[population] = build_morphology_mapping(SOURCE, name, morphology)

    population_by_connection = {
        "ascending_axon_to_golgi": "golgi_cell",
        "ascending_axon_to_purkinje": "purkinje_cell",
        "basket_to_basket": "basket_cell",
        "basket_to_purkinje": "purkinje_cell",
        "gap_goc": "golgi_cell",
        "glomerulus_to_golgi": "golgi_cell",
        "glomerulus_to_granule": "granule_cell",
        "golgi_to_golgi": "golgi_cell",
        "golgi_to_granule": "granule_cell",
        "parallel_fiber_to_basket": "basket_cell",
        "parallel_fiber_to_golgi": "golgi_cell",
        "parallel_fiber_to_purkinje": "purkinje_cell",
        "parallel_fiber_to_stellate": "stellate_cell",
        "stellate_to_purkinje": "purkinje_cell",
        "stellate_to_stellate": "stellate_cell",
    }
    with h5py.File(SOURCE, "r") as handle:
        for connection, population in population_by_connection.items():
            targets = handle[f"cells/connection_compartments/{connection}"][:, 1]
            summary = target_mapping_summary(mappings[population], targets)
            assert summary["contact_coverage"] == 1.0, connection
            assert summary["unique_compartment_coverage"] == 1.0, connection


def test_granule_target_mapping_distinguishes_terminal_and_preterminal():
    if not SOURCE.exists() or not BRAINCELL.exists():
        pytest.skip("published integration data unavailable")
    from braincell import Morphology

    morphology = Morphology.from_asc(
        BRAINCELL / "examples/neuron_compare/Cerebellum_mod/GrC/morphology/GrC.asc"
    )
    mapping = build_morphology_mapping(SOURCE, "GranuleCell", morphology)
    terminal = mapping.location(121)
    preterminal = mapping.location(120)
    assert terminal.branch_id == preterminal.branch_id
    assert terminal.x > preterminal.x
    assert terminal.mapping_mode == "semantic_dendrite_ordinal"
