#!/usr/bin/env python3
"""Run a small published glomerulus→Granule subnetwork in BrainCell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import brainunit as u
import numpy as np

from cerebellar_reproduction.runnable_prototype import (
    build_glomerulus_granule_subnetwork,
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
    parser.add_argument(
        "--braincell-repo",
        type=Path,
        default=ROOT / "assets/braincell",
    )
    parser.add_argument("--max-glomeruli", type=int, default=4)
    parser.add_argument("--max-granules", type=int, default=16)
    parser.add_argument("--duration-ms", type=float, default=2.0)
    parser.add_argument(
        "--generic-synapse",
        action="store_true",
        help="Use the old generic AMPA integration check instead of the paper AMPA+NMDA port.",
    )
    args = parser.parse_args()

    network, metadata = build_glomerulus_granule_subnetwork(
        args.spec,
        args.source_hdf5,
        args.braincell_repo,
        max_glomeruli=args.max_glomeruli,
        max_granules=args.max_granules,
        paper_mechanisms=not args.generic_synapse,
    )
    result = network.run(
        dt=0.025 * u.ms,
        duration=args.duration_ms * u.ms,
        event_backend="scatter",
    )
    ampa_trace = result.traces["granule_cell"]["ampa_g"]
    if hasattr(ampa_trace, "to_decimal"):
        ampa_trace = ampa_trace.to_decimal(u.uS)
    conductance = np.asarray(ampa_trace, dtype=float)
    voltage = np.asarray(
        result.traces["granule_cell"]["v_soma"].to_decimal(u.mV),
        dtype=float,
    )
    summary = {
        "network": repr(network),
        "selection": asdict(metadata),
        "relay_spike_count": int(np.asarray(result.spikes["glomerulus"]).sum()),
        "ampa_g_max": float(conductance.max()),
        "granule_v_soma_min_mV": float(voltage.min()),
        "granule_v_soma_max_mV": float(voltage.max()),
    }
    if "nmda_g" in result.traces["granule_cell"]:
        nmda_trace = result.traces["granule_cell"]["nmda_g"]
        if hasattr(nmda_trace, "to_decimal"):
            nmda_trace = nmda_trace.to_decimal(u.uS)
        summary["nmda_g_max_uS"] = float(np.asarray(nmda_trace, dtype=float).max())
        summary["ampa_g_max_uS"] = summary.pop("ampa_g_max")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
