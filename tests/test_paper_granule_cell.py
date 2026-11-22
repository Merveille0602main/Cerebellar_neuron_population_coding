from pathlib import Path
import sys

import numpy as np
import pytest


BRAINCELL = Path(__file__).resolve().parents[2] / "braincell"
sys.path.append(str(BRAINCELL))


@pytest.mark.skipif(not BRAINCELL.exists(), reason="BrainCell checkout unavailable")
def test_paper_granule_morphology_matches_historical_builder():
    import brainunit as u

    from cerebellar_reproduction.paper_granule_cell import (
        PAPER_BRANCH_COUNT,
        PAPER_CV_COUNT,
        build_paper_granule_morphology,
        paper_granule_structure,
    )

    morphology = build_paper_granule_morphology()
    branches = {branch.name: branch for branch in morphology.branches}
    assert morphology.n_branches == PAPER_BRANCH_COUNT == 108
    assert paper_granule_structure()["control_volumes"]["total"] == PAPER_CV_COUNT == 114
    assert sum(name.startswith("dendrite_") for name in branches) == 4
    assert sum(name.startswith("parallel_fiber_") for name in branches) == 100
    assert np.isclose(branches["ascending_axon"].length.to_decimal(u.um), 126.0)
    assert np.isclose(branches["dendrite_0"].length.to_decimal(u.um), 13.5)
    assert np.isclose(branches["parallel_fiber_99"].length.to_decimal(u.um), 20.0)
    assert np.isclose(branches["parallel_fiber_99"].radii_proximal[0].to_decimal(u.um), 0.075)
    assert branches["parallel_fiber_0"].parent.name == "ascending_axon"
    assert branches["parallel_fiber_1"].parent.name == "ascending_axon"
    assert branches["parallel_fiber_2"].parent.name == "parallel_fiber_0"
    assert branches["parallel_fiber_3"].parent.name == "parallel_fiber_1"


@pytest.mark.skipif(not BRAINCELL.exists(), reason="BrainCell checkout unavailable")
def test_paper_granule_cell_builds_with_paper_regions_and_population_axis():
    from cerebellar_reproduction.paper_granule_cell import PaperGranuleCell

    model = PaperGranuleCell(pop_size=(2,), name="paper_granule_cell").build()
    assert model.cell is not None
    assert model.morpho.n_branches == 108
    assert set(model.regions) == {"soma", "dend", "hilock", "ais", "aa", "pf"}
    assert tuple(model.cell.pop_size) == (2,)
