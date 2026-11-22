#!/usr/bin/env python3
"""Estimate and benchmark the no-gap granular-layer chemical loop."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import brainunit as u
import numpy as np

from cerebellar_reproduction import (
    build_granular_layer_chemical_subnetwork,
    build_paper_mossy_input_schedule,
    estimate_network_output_memory,
    run_network_in_chunks,
)


ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = ROOT.parent / "deschepper-etal-2022-source/configs/balanced.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spec", type=Path, default=ROOT / "data/export/balanced")
    parser.add_argument(
        "--source-hdf5",
        type=Path,
        default=ROOT / "data/reference/balanced.hdf5",
    )
    parser.add_argument("--braincell-repo", type=Path, default=ROOT.parent / "braincell")
    parser.add_argument("--devices", nargs="+", default=["initial_mf", "background"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-glomeruli", type=int, default=2)
    parser.add_argument("--max-granules", type=int, default=8)
    parser.add_argument("--max-golgi", type=int, default=2)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--chunk-duration-ms", type=float, default=20.0)
    parser.add_argument(
        "--spike-recording",
        choices=("full", "population", "none"),
        default="population",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optionally write the estimate/run summary as JSON.",
    )
    args = parser.parse_args()

    with np.load(args.spec / "connectivity/mossy_to_glomerulus.npz") as data:
        source_ids = np.unique(data["cell_pairs"][:, 0]).astype(int).tolist()
    inputs = build_paper_mossy_input_schedule(
        args.config,
        source_ids,
        device_names=args.devices,
        seed=args.seed,
    )
    network, metadata = build_granular_layer_chemical_subnetwork(
        args.spec,
        args.source_hdf5,
        args.braincell_repo,
        mossy_spike_times_ms=inputs.spike_times_ms,
        max_glomeruli=args.max_glomeruli,
        max_granules=args.max_granules,
        max_golgi=args.max_golgi,
    )
    estimate = estimate_network_output_memory(
        network,
        dt=args.dt_ms * u.ms,
        duration=args.duration_ms * u.ms,
        chunk_duration=args.chunk_duration_ms * u.ms,
        spike_recording=args.spike_recording,
    )
    shared = metadata.shared_input
    payload = {
        "network": repr(network),
        "source_mossy_count": len(source_ids),
        "input_event_count": sum(map(len, inputs.spike_times_ms.values())),
        "retained": {
            "glomeruli": len(shared.granule_pathway.source_glomerulus_ids),
            "granules": len(shared.granule_pathway.source_granule_ids),
            "golgi": len(shared.source_golgi_ids),
            "glomerulus_to_granule_contacts": (
                shared.granule_pathway.n_glomerulus_to_granule_contact
            ),
            "glomerulus_to_golgi_contacts": shared.n_glomerulus_to_golgi_contact,
            "golgi_to_granule_contacts": metadata.n_golgi_to_granule_contact,
        },
        "estimate": asdict(estimate),
    }
    if not args.estimate_only:
        payload["run"] = asdict(
            run_network_in_chunks(
                network,
                dt=args.dt_ms * u.ms,
                duration=args.duration_ms * u.ms,
                chunk_duration=args.chunk_duration_ms * u.ms,
                spike_recording=args.spike_recording,
            )
        )
    report = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
