"""Paper-era ``dbbs_models.GranuleCell`` morphology for BrainCell.

This module deliberately lives in the reproduction repository.  It corrects
the morphology construction of BrainCell's experimental ``GrCFull`` example
without changing the jointly maintained BrainCell repository.  The membrane
painting implementation and its audited Ma et al. 2020 parameter table are
reused from that example; only the paper-specific procedural morphology,
discretisation, population support, and region selection are supplied here.

The source of truth is ``dbbs_models`` v1.1.2, commit ``5f920d5``, which is
the version pinned by the De Schepper et al. simulation environment.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import brainunit as u
import numpy as np
from braincell import Axon, Cell, CVPerBranchList, Dendrite, Morphology, Soma
from braincell.filter import branch_in
from examples.neuron_compare.cell.grc_ma2020.grc_full_braincell import GrCFull
from examples.neuron_compare.cell.grc_ma2020.grc_full_parameters import (
    GrCFullParameters,
)


PAPER_DBBS_MODELS_COMMIT = "5f920d5"
PAPER_SOMA_LENGTH_UM = 5.62232
PAPER_SOMA_DIAMETER_UM = 5.8
PAPER_DENDRITE_LENGTH_SETTING_UM = 15.0
PAPER_DENDRITE_POINT_LENGTH_UM = 13.5
PAPER_DENDRITE_DIAMETER_UM = 0.75
PAPER_ASCENDING_AXON_LENGTH_UM = 126.0
# Arborize 1.1.1 reapplies the historical nseg rule after morphology construction.
PAPER_ASCENDING_AXON_NSEG = 7
PAPER_ASCENDING_AXON_DIAMETER_UM = 0.3
PAPER_PARALLEL_FIBER_SECTION_LENGTH_UM = 20.0
PAPER_PARALLEL_FIBER_SECTION_COUNT = 100
PAPER_PARALLEL_FIBER_ARM_SECTION_COUNT = 50
# The final section attributes and the published HDF5 use a 0.15 um diameter.
PAPER_PARALLEL_FIBER_DIAMETER_UM = 0.15
PAPER_BRANCH_COUNT = 108
PAPER_CV_COUNT = 114


class PaperGranuleCell(GrCFull):
    """BrainCell port of the complete procedural paper-era granule cell.

    One BrainCell branch corresponds to one NEURON section.  The ascending
    axon therefore remains one branch with 7 control volumes, while each of
    the 100 parallel-fibre sections is one branch with one control volume.
    This distinction is essential for matching the original model.
    """

    def __init__(
        self,
        params: GrCFullParameters | None = None,
        *,
        temperature_celsius: float = 25.0,
        v_init_mV: float = -65.0,
        pop_size=(),
        name: str | None = None,
    ):
        # Do not call GrCFull.__init__: that class requires an ASC path and
        # builds a different, non-paper morphology.
        self.params = params if params is not None else GrCFullParameters()
        self.temperature_celsius = float(temperature_celsius)
        self.v_init_mV = float(v_init_mV)
        self.pop_size = pop_size
        self.name = name
        self.morpho = None
        self.cell = None
        self.regions: dict[str, Any] = {}

    def build(self) -> "PaperGranuleCell":
        self.morpho = build_paper_granule_morphology()
        cv_counts = tuple(
            PAPER_ASCENDING_AXON_NSEG if branch.name == "ascending_axon" else 1
            for branch in self.morpho.branches
        )
        if sum(cv_counts) != PAPER_CV_COUNT:
            raise RuntimeError(f"Unexpected paper GrC CV count: {sum(cv_counts)}.")
        self.cell = Cell(
            self.morpho,
            pop_size=self.pop_size,
            cv_policy=CVPerBranchList(cv_counts),
            V_init=self.v_init_mV * u.mV,
            solver="staggered",
            cache_ion_total_current=True,
            ion_channel_update_order="family",
            name=self.name,
        )
        self._define_regions()
        self._paint_cable()
        self._paint_ions()
        self._paint_channels()
        return self

    def _define_regions(self) -> None:
        if self.morpho is None:
            raise RuntimeError("Morphology must be built before defining regions.")
        counts = Counter(branch.type for branch in self.morpho.branches)
        expected = {"soma": 1, "dendrite": 4, "axon": 103}
        if any(counts.get(kind, 0) != count for kind, count in expected.items()):
            raise RuntimeError(f"Unexpected paper GrC branch counts: {dict(counts)}.")
        self.regions = {
            "soma": branch_in("name", "soma"),
            "dend": _branch_names_region(f"dendrite_{index}" for index in range(4)),
            "hilock": branch_in("name", "axon_hillock"),
            "ais": branch_in("name", "axon_initial_segment"),
            "aa": branch_in("name", "ascending_axon"),
            "pf": _branch_names_region(
                f"parallel_fiber_{index}" for index in range(PAPER_PARALLEL_FIBER_SECTION_COUNT)
            ),
        }


def build_paper_granule_morphology() -> Morphology:
    """Construct the geometry and section topology in ``GranuleCell.builder``."""
    soma = Soma.from_points(
        points=np.asarray(((0.0, 0.0, 0.0), (0.0, PAPER_SOMA_LENGTH_UM, 0.0))) * u.um,
        radii=np.full(2, PAPER_SOMA_DIAMETER_UM / 2.0) * u.um,
    )
    morphology = Morphology.from_root(soma, name="soma")

    # The historical loop creates ten points at j=0..9.  Consequently the
    # pt3d path is 13.5 um even though Section.L was initially set to 15 um.
    for index in range(4):
        x = (index - 1.5) * 2.0
        points = np.asarray(
            [(x, -PAPER_DENDRITE_LENGTH_SETTING_UM * j / 10.0, 0.0) for j in range(10)]
        )
        dendrite = Dendrite.from_points(
            points=points * u.um,
            radii=np.full(len(points), PAPER_DENDRITE_DIAMETER_UM / 2.0) * u.um,
        )
        morphology.attach(
            parent="soma",
            child_branch=dendrite,
            child_name=f"dendrite_{index}",
            parent_x=0.0,
        )

    hillock = _axon_line(
        (0.0, PAPER_SOMA_LENGTH_UM, 0.0),
        (0.0, PAPER_SOMA_LENGTH_UM + 1.0, 0.0),
        diameter_um=1.5,
    )
    morphology.attach(
        parent="soma", child_branch=hillock, child_name="axon_hillock", parent_x=0.0
    )
    ais_y = PAPER_SOMA_LENGTH_UM + 11.0
    ais = _axon_line(
        (0.0, PAPER_SOMA_LENGTH_UM + 1.0, 0.0),
        (0.0, ais_y, 0.0),
        diameter_um=0.7,
    )
    morphology.attach(
        parent="axon_hillock",
        child_branch=ais,
        child_name="axon_initial_segment",
        parent_x=1.0,
    )

    aa_points = np.asarray(
        [(0.0, ais_y + fraction * PAPER_ASCENDING_AXON_LENGTH_UM, 0.0) for fraction in np.linspace(0, 1, 11)]
    )
    ascending_axon = Axon.from_points(
        points=aa_points * u.um,
        radii=np.full(len(aa_points), PAPER_ASCENDING_AXON_DIAMETER_UM / 2.0) * u.um,
    )
    morphology.attach(
        parent="axon_initial_segment",
        child_branch=ascending_axon,
        child_name="ascending_axon",
        parent_x=1.0,
    )

    pf_y = ais_y + PAPER_ASCENDING_AXON_LENGTH_UM
    for index in range(PAPER_PARALLEL_FIBER_SECTION_COUNT):
        sign = 1.0 if index % 2 == 0 else -1.0
        z = (index // 2) * PAPER_PARALLEL_FIBER_SECTION_LENGTH_UM
        pf = _axon_line(
            (0.0, pf_y, sign * z),
            (0.0, pf_y, sign * (z + PAPER_PARALLEL_FIBER_SECTION_LENGTH_UM)),
            diameter_um=PAPER_PARALLEL_FIBER_DIAMETER_UM,
        )
        parent = "ascending_axon" if index < 2 else f"parallel_fiber_{index - 2}"
        morphology.attach(
            parent=parent,
            child_branch=pf,
            child_name=f"parallel_fiber_{index}",
            parent_x=1.0,
        )

    if morphology.n_branches != PAPER_BRANCH_COUNT:
        raise RuntimeError(f"Unexpected paper GrC branch count: {morphology.n_branches}.")
    return morphology


def paper_granule_structure() -> dict[str, Any]:
    """Return machine-readable structural invariants for reports and tests."""
    return {
        "source_commit": PAPER_DBBS_MODELS_COMMIT,
        "branches": {
            "soma": 1,
            "dendrite": 4,
            "axon_hillock": 1,
            "axon_initial_segment": 1,
            "ascending_axon": 1,
            "parallel_fiber": PAPER_PARALLEL_FIBER_SECTION_COUNT,
            "total": PAPER_BRANCH_COUNT,
        },
        "control_volumes": {
            "ascending_axon": PAPER_ASCENDING_AXON_NSEG,
            "total": PAPER_CV_COUNT,
        },
        "length_um": {
            "soma": PAPER_SOMA_LENGTH_UM,
            "dendrite_pt3d": PAPER_DENDRITE_POINT_LENGTH_UM,
            "ascending_axon": PAPER_ASCENDING_AXON_LENGTH_UM,
            "parallel_fiber_section": PAPER_PARALLEL_FIBER_SECTION_LENGTH_UM,
            "parallel_fiber_each_arm": PAPER_PARALLEL_FIBER_ARM_SECTION_COUNT
            * PAPER_PARALLEL_FIBER_SECTION_LENGTH_UM,
        },
    }


def _axon_line(
    proximal: tuple[float, float, float],
    distal: tuple[float, float, float],
    *,
    diameter_um: float,
) -> Axon:
    return Axon.from_points(
        points=np.asarray((proximal, distal)) * u.um,
        radii=np.full(2, diameter_um / 2.0) * u.um,
    )


def _branch_names_region(names: Any) -> Any:
    names = tuple(names)
    if not names:
        raise ValueError("Paper granule-cell region cannot be empty.")
    return branch_in("name", names)
