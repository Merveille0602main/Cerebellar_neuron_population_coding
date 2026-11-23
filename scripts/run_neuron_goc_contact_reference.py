#!/usr/bin/env python3
"""Generate a NEURON whole-GoC trace for one published glomerular contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from cerebellar_reproduction.morphology_mapping import build_morphology_mapping


ROOT = Path(__file__).parents[1]


def _first_contact(spec: Path) -> tuple[int, int, int]:
    with np.load(spec / "connectivity/glomerulus_to_golgi.npz") as data:
        pairs = np.asarray(data["cell_pairs"], dtype=np.int64)
        targets = np.asarray(data["compartment_pairs"][:, 1], dtype=np.int64)
    glomerulus_id = int(np.unique(pairs[:, 0])[0])
    rows = np.flatnonzero(pairs[:, 0] == glomerulus_id)
    golgi_id = int(np.unique(pairs[rows, 1])[0])
    row = int(rows[np.flatnonzero(pairs[rows, 1] == golgi_id)[0]])
    return glomerulus_id, golgi_id, int(targets[row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "data/export/balanced")
    parser.add_argument(
        "--source-hdf5",
        type=Path,
        default=ROOT / "data/reference/balanced.hdf5",
    )
    parser.add_argument("--braincell-repo", type=Path, default=ROOT / "assets/braincell")
    parser.add_argument(
        "--mechanism-library",
        type=Path,
        default=ROOT / "data/reference/neuron_goc_mechanisms/x86_64/libnrnmech.so",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/reference/neuron_goc_contact.npz",
    )
    parser.add_argument("--spike-time-ms", type=float, default=1.0)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--duration-ms", type=float, default=3.0)
    args = parser.parse_args()

    import sys

    if str(args.braincell_repo) not in sys.path:
        sys.path.insert(0, str(args.braincell_repo))
    from examples.neuron_compare.cell.goc_ma2020.goc_braincell import GoC as BrainCellGoC
    from examples.neuron_compare.cell.goc_ma2020.goc_neuron import GoC as NeuronGoC
    from examples.neuron_compare.cell.goc_ma2020.parameters import load_goc20_params
    from neuron import h

    glomerulus_id, golgi_id, compartment = _first_contact(args.spec)
    mapping_cell = BrainCellGoC(
        params=load_goc20_params(),
        temperature_celsius=32.0,
        pop_size=(1,),
        name="mapping_only_golgi",
    ).build()
    mapping = build_morphology_mapping(
        args.source_hdf5,
        "GolgiCell",
        mapping_cell.morph,
    )
    location = mapping.location(compartment)
    match = re.fullmatch(r"dendrite_(\d+)", location.branch_name)
    if match is None:
        raise SystemExit(f"Expected a GoC dendrite target, got {location.branch_name!r}.")
    dendrite_index = int(match.group(1))

    h.celsius = 32.0
    cell = NeuronGoC(
        params=load_goc20_params(),
        nrnmech_path=args.mechanism_library,
    ).build()
    target = cell.dend_sections[dendrite_index](location.x)
    ampa = h.GRANULE_Ampa_det_vi(target)
    nmda = h.GRANULE_Nmda_det_vi(target)
    ampa_connection = h.NetCon(None, ampa)
    nmda_connection = h.NetCon(None, nmda)
    for connection in (ampa_connection, nmda_connection):
        connection.delay = 0.0
        connection.weight[0] = 1.0

    recordings = {
        "time_ms": h.Vector().record(h._ref_t),
        "v_soma_mV": h.Vector().record(cell.root_soma(0.5)._ref_v),
        "v_contact_mV": h.Vector().record(target._ref_v),
        "ampa_g_pS": h.Vector().record(ampa._ref_g),
        "ampa_current_nA": h.Vector().record(ampa._ref_i),
        "nmda_g_pS": h.Vector().record(nmda._ref_g),
        "nmda_current_nA": h.Vector().record(nmda._ref_i),
    }
    h.load_file("stdrun.hoc")
    h.cvode_active(0)
    h.dt = args.dt_ms
    h.steps_per_ms = 1.0 / args.dt_ms
    h.finitialize(-65.0)
    ampa_connection.event(args.spike_time_ms)
    nmda_connection.event(args.spike_time_ms)
    h.continuerun(args.duration_ms)

    data = {name: np.asarray(vector, dtype=float) for name, vector in recordings.items()}
    data["ampa_g_uS"] = data.pop("ampa_g_pS") * 1.0e-6
    data["nmda_g_uS"] = data.pop("nmda_g_pS") * 1.0e-6
    data.update(
        {
            "source_glomerulus_id": np.asarray(glomerulus_id),
            "source_golgi_id": np.asarray(golgi_id),
            "target_compartment": np.asarray(compartment),
            "target_dendrite_index": np.asarray(dendrite_index),
            "target_x": np.asarray(location.x),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_sample": len(data["time_ms"]),
                "source_glomerulus_id": glomerulus_id,
                "source_golgi_id": golgi_id,
                "target_compartment": compartment,
                "target_dendrite_index": dendrite_index,
                "target_x": location.x,
                "ampa_g_peak_uS": float(data["ampa_g_uS"].max()),
                "nmda_g_peak_uS": float(data["nmda_g_uS"].max()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    cell.cleanup()


if __name__ == "__main__":
    main()
