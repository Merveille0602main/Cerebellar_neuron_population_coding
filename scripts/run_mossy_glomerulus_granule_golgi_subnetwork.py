#!/usr/bin/env python3
"""Run one MF schedule through the published GrC and GoC feedforward paths."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import brainunit as u
import numpy as np

from cerebellar_reproduction import (
    build_mossy_glomerulus_granule_golgi_subnetwork,
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
    parser.add_argument("--mossy-ids", type=int, nargs="+", default=[213, 214])
    parser.add_argument(
        "--spike-times-ms",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 7.0],
    )
    parser.add_argument("--max-glomeruli", type=int, default=2)
    parser.add_argument("--max-granules", type=int, default=4)
    parser.add_argument("--max-golgi", type=int, default=2)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--duration-ms", type=float, default=10.0)
    args = parser.parse_args()

    schedule = {
        source_id: tuple(args.spike_times_ms)
        for source_id in args.mossy_ids
    }
    network, metadata = build_mossy_glomerulus_granule_golgi_subnetwork(
        args.spec,
        args.source_hdf5,
        args.braincell_repo,
        mossy_spike_times_ms=schedule,
        max_glomeruli=args.max_glomeruli,
        max_granules=args.max_granules,
        max_golgi=args.max_golgi,
    )
    result = network.run(
        dt=args.dt_ms * u.ms,
        duration=args.duration_ms * u.ms,
        event_backend="scatter",
        spike_recording="population",
    )

    def trace_max(population: str, field: str, unit) -> float:
        values = result.traces[population][field]
        if unit is not None:
            values = values.to_decimal(unit)
        return float(np.asarray(values, dtype=float).max())

    summary = {
        "network": repr(network),
        "selection": asdict(metadata),
        "mossy_spikes_per_source": np.asarray(
            result.spikes["mossy_fibers"]
        ).sum(axis=0).astype(int).tolist(),
        "glomerulus_spikes_per_source": np.asarray(
            result.spikes["glomerulus"]
        ).sum(axis=0).astype(int).tolist(),
        "granule_ampa_g_max_uS": trace_max("granule_cell", "ampa_g", u.uS),
        "granule_nmda_g_max_uS": trace_max("granule_cell", "nmda_g", u.uS),
        "golgi_ampa_mf_g_max_uS": trace_max("golgi_cell", "ampa_mf_g", u.uS),
        "golgi_nmda_g_max_uS": trace_max("golgi_cell", "nmda_g", u.uS),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
