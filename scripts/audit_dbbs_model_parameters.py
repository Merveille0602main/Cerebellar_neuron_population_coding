#!/usr/bin/env python3
"""Fetch paper-era dbbs_models sources and audit the local BrainCell GoC template."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np


ROOT = Path(__file__).parents[1]
COMMIT = "5f920d5"
FILES = {
    "granule_cell_models.py": "d17215a2a9945bee237cf1f62120baa1e3ff449d75256ee5e41f9e5c4c5eaf4c",
    "golgi_cell_models.py": "5c6d3ac21f5ee47f6368098f2212ee5720f541a55da99d742e30c12feb201044",
    "requirements.txt": "6c2493e9e027f11d6c12c7be5b5b471ab41634f4f04a275df4af5feaa315828e",
}
GOC_MORPHOLOGY = "dbbs_models/morphologies/pair-140514-C2-1_split_1.asc"
STELLATE_MORPHOLOGY = "dbbs_models/morphologies/stellate.asc"
STELLATE_MORPHOLOGY_SHA256 = "f6f838f35f0182e6e784ced7c74443916a955f4d60d64fddb1db7268c84beb9f"


def _download(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/dbbs-lab/dbbs-models/{COMMIT}/{path}"
    with urlopen(url) as response:  # noqa: S310 - fixed project/commit URL
        return response.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _class_assignment(source: str, class_name: str, field: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if any(isinstance(target, ast.Name) and target.id == field for target in statement.targets):
                    return ast.literal_eval(statement.value)
    raise ValueError(f"Could not find literal {class_name}.{field}.")


def _attribute(section_types, region: str, mechanism: str, attribute: str) -> float:
    return float(section_types[region]["attributes"][(attribute, mechanism)])


def _paper_goc_channel_values(section_types) -> np.ndarray:
    entries = (
        ("apical_dendrites", "Nav1_6", "gbar"),
        ("apical_dendrites", "Kca1_1", "gbar"),
        ("apical_dendrites", "Kca2_2", "gkbar"),
        ("apical_dendrites", "Cav2_3", "gcabar"),
        ("apical_dendrites", "Cav3_1", "pcabar"),
        ("basal_dendrites", "Nav1_6", "gbar"),
        ("basal_dendrites", "Kca1_1", "gbar"),
        ("basal_dendrites", "Kca2_2", "gkbar"),
        ("basal_dendrites", "Ca", "gcabar"),
        ("soma", "Nav1_6", "gbar"),
        ("soma", "Kv1_1", "gbar"),
        ("soma", "Kv3_4", "gkbar"),
        ("soma", "Kv4_3", "gkbar"),
        ("soma", "Kca1_1", "gbar"),
        ("soma", "Kca3_1", "gkbar"),
        ("soma", "Ca", "gcabar"),
        ("soma", "Cav3_1", "pcabar"),
        ("axon_initial_segment", "HCN1", "gbar"),
        ("axon_initial_segment", "HCN2", "gbar"),
        ("axon_initial_segment", "Nav1_6", "gbar"),
        ("axon_initial_segment", "Km", "gkbar"),
        ("axon_initial_segment", "Kca1_1", "gbar"),
        ("axon_initial_segment", "Ca", "gcabar"),
        ("axon", "Nav1_6", "gbar"),
        ("axon", "Kv3_4", "gkbar"),
    )
    return np.asarray(
        [_attribute(section_types, region, mechanism, attribute) for region, mechanism, attribute in entries],
        dtype=float,
    )


def _paper_grc_channel_values(section_types) -> np.ndarray:
    entries = (
        ("soma", "Leak", "gmax"),
        ("soma", "Kv3_4", "gkbar"),
        ("soma", "Kv4_3", "gkbar"),
        ("soma", "Kir2_3", "gkbar"),
        ("soma", "Ca", "gcabar"),
        ("soma", "Kv1_1", "gbar"),
        ("soma", "Kv1_5", "gKur"),
        ("soma", "Kv2_2", "gKv2_2bar"),
        ("dendrites", "Leak", "gmax"),
        ("dendrites", "Ca", "gcabar"),
        ("dendrites", "Kca1_1", "gbar"),
        ("dendrites", "Kv1_1", "gbar"),
    )
    return np.asarray(
        [_attribute(section_types, region, mechanism, attribute) for region, mechanism, attribute in entries],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--braincell-repo", type=Path, default=ROOT.parent / "braincell")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "data/reference/dbbs_models_v1.1.2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/export/dbbs_model_parameter_audit.json",
    )
    args = parser.parse_args()

    args.reference_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, bytes] = {}
    source_hashes = {}
    for name, expected_sha in FILES.items():
        remote_path = f"dbbs_models/{name}" if name.endswith("_models.py") else name
        content = _download(remote_path)
        actual_sha = _sha256(content)
        if actual_sha != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch for {remote_path}: {actual_sha} != {expected_sha}")
        (args.reference_dir / name).write_bytes(content)
        payloads[name] = content
        source_hashes[remote_path] = actual_sha

    morphology_content = _download(GOC_MORPHOLOGY)
    source_morphology_sha = _sha256(morphology_content)
    local_morphology = (
        args.braincell_repo
        / "examples/neuron_compare/Cerebellum_mod/GoC/morphology/GoC.asc"
    ).read_bytes()
    local_morphology_sha = _sha256(local_morphology)

    stellate_content = _download(STELLATE_MORPHOLOGY)
    paper_stellate_sha = _sha256(stellate_content)
    if paper_stellate_sha != STELLATE_MORPHOLOGY_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {STELLATE_MORPHOLOGY}: "
            f"{paper_stellate_sha} != {STELLATE_MORPHOLOGY_SHA256}"
        )
    paper_stellate_path = args.reference_dir / "stellate.asc"
    paper_stellate_path.write_bytes(stellate_content)
    paper_stellate_utf8_path = args.reference_dir / "stellate_utf8.asc"
    paper_stellate_utf8_path.write_text(stellate_content.decode("latin-1"))
    local_stellate_path = (
        args.braincell_repo
        / "examples/neuron_compare/Cerebellum_mod/SC/morphology/SC.asc"
    )
    local_stellate_sha = _sha256(local_stellate_path.read_bytes())

    granule_source = payloads["granule_cell_models.py"].decode()
    golgi_source = payloads["golgi_cell_models.py"].decode()
    granule_synapses = _class_assignment(granule_source, "GranuleCell", "synapse_types")
    granule_sections = _class_assignment(granule_source, "GranuleCell", "section_types")
    golgi_synapses = _class_assignment(golgi_source, "GolgiCell", "synapse_types")
    golgi_sections = _class_assignment(golgi_source, "GolgiCell", "section_types")

    source_values = _paper_goc_channel_values(golgi_sections)
    local_values = np.genfromtxt(
        args.braincell_repo
        / "examples/neuron_compare/cell/goc_ma2020/Optimization_result.txt"
    ).reshape(-1)
    if local_values.shape != source_values.shape:
        raise RuntimeError(
            f"GoC parameter shape mismatch: {local_values.shape} != {source_values.shape}"
        )

    if str(args.braincell_repo) not in sys.path:
        sys.path.insert(0, str(args.braincell_repo))
    from examples.neuron_compare.cell.grc_ma2020.parameters import load_grc20_params

    local_grc = load_grc20_params()
    paper_grc_values = _paper_grc_channel_values(granule_sections)
    local_grc_values = np.asarray(
        (
            local_grc.soma.leak,
            local_grc.soma.kv3p4,
            local_grc.soma.kv4p3,
            local_grc.soma.kir2p3,
            local_grc.soma.cahva,
            local_grc.soma.kv1p1,
            local_grc.soma.kv1p5,
            local_grc.soma.kv2p2,
            local_grc.dend.leak,
            local_grc.dend.cahva,
            local_grc.dend.kca1p1,
            local_grc.dend.kv1p1,
        ),
        dtype=float,
    )

    requirements = payloads["requirements.txt"].decode().splitlines()
    report = {
        "dbbs_models_commit": COMMIT,
        "dbbs_models_version": "1.1.2",
        "source_sha256": source_hashes,
        "paper_dependencies": requirements,
        "granule_synapse_types": granule_synapses,
        "golgi_synapse_types": golgi_synapses,
        "grc_template_audit": {
            "soma_and_dendrite_channel_parameter_count": int(len(paper_grc_values)),
            "soma_and_dendrite_channel_parameters_exact_match": bool(
                np.array_equal(paper_grc_values, local_grc_values)
            ),
            "maximum_channel_parameter_abs_difference": float(
                np.max(np.abs(paper_grc_values - local_grc_values))
            ),
            "geometry_exact_match": False,
            "current_reproduction_geometry": "standalone ModelDB GrC.asc soma+dendrite morphology",
            "paper_geometry": (
                "procedural dbbs_models GranuleCell: one soma, four synthetic dendrites, "
                "hillock, AIS, ascending axon, and two parallel-fiber chains"
            ),
            "current_runtime_scope": (
                "The runnable granular-layer prototype uses the current BrainCell "
                "grc_ma2020 soma+dendrite template. Its 12 soma/dendrite densities "
                "match, but geometry and omitted axonal load are not yet paper-equivalent."
            ),
        },
        "goc_template_audit": {
            "morphology": {
                "paper_sha256": source_morphology_sha,
                "braincell_sha256": local_morphology_sha,
                "exact_match": source_morphology_sha == local_morphology_sha,
            },
            "channel_parameter_count": int(len(source_values)),
            "channel_parameters_exact_match": bool(np.array_equal(source_values, local_values)),
            "maximum_channel_parameter_abs_difference": float(
                np.max(np.abs(source_values - local_values))
            ),
            "interpretation": (
                "The current BrainCell GoC morphology and 25 regional channel values "
                "match paper-era dbbs_models exactly. Remaining whole-cell trace "
                "differences must be audited at mechanism initialization, current laws, "
                "ion state, update ordering, and numerical integration layers."
            ),
        },
        "stellate_template_audit": {
            "paper_morphology_path": str(paper_stellate_path),
            "paper_morphology_utf8_path": str(paper_stellate_utf8_path),
            "paper_morphology_sha256": paper_stellate_sha,
            "braincell_morphology_path": str(local_stellate_path),
            "braincell_morphology_sha256": local_stellate_sha,
            "exact_match": paper_stellate_sha == local_stellate_sha,
            "interpretation": (
                "The current shared BrainCell SC.asc is not the paper-era "
                "dbbs_models StellateCell morphology. The reproduction must use "
                "the pinned paper asset or a verified mapping; it must not silently "
                "treat current SC compartment IDs as equivalent."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
