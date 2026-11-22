#!/usr/bin/env python3
"""Report BSB target-compartment coverage by BrainCell morphologies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from braincell import Morphology

from cerebellar_reproduction.morphology_mapping import (
    build_morphology_mapping,
    target_mapping_summary,
)
from cerebellar_reproduction.validation import validate_export


POPULATION_MORPHOLOGY = {
    "granule_cell": ("GranuleCell", "GrC/morphology/GrC.asc"),
    "golgi_cell": ("GolgiCell", "GoC/morphology/GoC.asc"),
    "purkinje_cell": ("PurkinjeCell", "PC/morphology/PC.asc"),
    "stellate_cell": ("StellateCell", "SC/morphology/SC.asc"),
    "basket_cell": ("BasketCell", "BC/morphology/BC.asc"),
}


def build_report(spec: Path, source_hdf5: Path, braincell_repo: Path) -> dict:
    manifest = validate_export(spec)
    morph_root = braincell_repo / "examples/neuron_compare/Cerebellum_mod"
    mappings = {}
    morphology_report = {}
    for population, (morphology_name, relative_path) in POPULATION_MORPHOLOGY.items():
        morphology = Morphology.from_asc(morph_root / relative_path)
        mapping = build_morphology_mapping(source_hdf5, morphology_name, morphology)
        mappings[population] = mapping
        morphology_report[population] = {
            "morphology": morphology_name,
            "transform": mapping.transform_name,
            "n_branch_matched": len(mapping.branch_matches),
            "n_compartment_mapped": len(mapping.locations),
            "n_compartment_unmapped": len(mapping.unmapped_compartments),
        }

    connectivity_report = {}
    for name, info in sorted(manifest["connectivity"].items()):
        population = info["post_population"]
        if population not in mappings:
            connectivity_report[name] = {
                "post_population": population,
                "status": "relay_or_entity_target",
            }
            continue
        with np.load(spec / info["file"]) as data:
            if "compartment_pairs" not in data:
                connectivity_report[name] = {
                    "post_population": population,
                    "status": "no_compartment_table",
                }
                continue
            summary = target_mapping_summary(mappings[population], data["compartment_pairs"][:, 1])
        connectivity_report[name] = {
            "post_population": population,
            "status": "mapped" if summary["contact_coverage"] == 1.0 else "partially_mapped",
            **summary,
        }

    return {
        "source_hdf5": str(source_hdf5),
        "spec": str(spec),
        "morphologies": morphology_report,
        "connectivity": connectivity_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("source_hdf5", type=Path)
    parser.add_argument("braincell_repo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.spec, args.source_hdf5, args.braincell_repo)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
