#!/usr/bin/env python3
"""Run the first paper-input mossy→glomerulus→Granule BrainCell pathway."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import brainunit as u
import numpy as np

from cerebellar_reproduction import (
    build_mossy_glomerulus_granule_subnetwork,
    build_paper_mossy_input_schedule,
)


ROOT = Path(__file__).parents[1]


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
        "--mossy-ids",
        type=int,
        nargs="+",
        default=[213, 214, 222, 223],
        help="Original global BSB mossy-fiber IDs.",
    )
    parser.add_argument(
        "--spike-times-ms",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 7.0, 32.0, 52.0],
        help="Spike schedule applied to every selected mossy source.",
    )
    parser.add_argument(
        "--input-config",
        type=Path,
        help="Generate mossy schedules from a BSB config instead of --spike-times-ms.",
    )
    parser.add_argument(
        "--input-devices",
        nargs="+",
        default=["initial_mf", "background"],
        help="BSB mossy input devices used with --input-config.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-mossy-fibers",
        action="store_true",
        help="Use all 117 imported mossy sources (mainly for the background protocol).",
    )
    parser.add_argument("--max-glomeruli", type=int, default=8)
    parser.add_argument("--max-granules", type=int, default=32)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument(
        "--duration-ms",
        type=float,
        help="Defaults to 60 ms for explicit bursts; required for config-driven runs.",
    )
    args = parser.parse_args()

    source_ids = args.mossy_ids
    if args.all_mossy_fibers:
        with np.load(args.spec / "connectivity/mossy_to_glomerulus.npz") as data:
            source_ids = np.unique(data["cell_pairs"][:, 0]).astype(int).tolist()
    input_provenance = None
    if args.input_config is not None:
        generated = build_paper_mossy_input_schedule(
            args.input_config,
            source_ids,
            device_names=args.input_devices,
            seed=args.seed,
        )
        schedule = generated.spike_times_ms
        if args.duration_ms is None:
            parser.error(
                "--duration-ms is required with --input-config; the full 10 s run "
                "has not been resource-qualified yet."
            )
        duration_ms = args.duration_ms
        input_provenance = {
            "config": str(args.input_config),
            "simulation": generated.simulation,
            "devices": generated.device_names,
            "seed": generated.seed,
            "ignored_sd_devices": generated.ignored_sd_devices,
            "note": generated.note,
        }
    else:
        schedule = {
            source_id: tuple(args.spike_times_ms)
            for source_id in source_ids
        }
        duration_ms = 60.0 if args.duration_ms is None else args.duration_ms
    network, metadata = build_mossy_glomerulus_granule_subnetwork(
        args.spec,
        args.source_hdf5,
        args.braincell_repo,
        mossy_spike_times_ms=schedule,
        max_glomeruli=args.max_glomeruli,
        max_granules=args.max_granules,
    )
    result = network.run(
        dt=args.dt_ms * u.ms,
        duration=duration_ms * u.ms,
        event_backend="scatter",
    )
    mossy_spikes = np.asarray(result.spikes["mossy_fibers"])
    glomerulus_spikes = np.asarray(result.spikes["glomerulus"])
    ampa = np.asarray(
        result.traces["granule_cell"]["ampa_g"].to_decimal(u.uS),
        dtype=float,
    )
    nmda = np.asarray(
        result.traces["granule_cell"]["nmda_g"].to_decimal(u.uS),
        dtype=float,
    )
    voltage = np.asarray(
        result.traces["granule_cell"]["v_soma"].to_decimal(u.mV),
        dtype=float,
    )
    summary = {
        "network": repr(network),
        "selection": asdict(metadata),
        "mossy_spikes_per_source": mossy_spikes.sum(axis=(0, 2)).astype(int).tolist(),
        "glomerulus_spikes_per_source": (
            glomerulus_spikes.sum(axis=(0, 2)).astype(int).tolist()
        ),
        "ampa_g_max_uS": float(ampa.max()),
        "nmda_g_max_uS": float(nmda.max()),
        "granule_v_soma_min_mV": float(voltage.min()),
        "granule_v_soma_max_mV": float(voltage.max()),
    }
    if input_provenance is not None:
        summary["input_provenance"] = input_provenance
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
