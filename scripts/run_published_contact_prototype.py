#!/usr/bin/env python3
"""Run one published glomerulus→Granule contact in BrainCell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import brainunit as u
import numpy as np

from cerebellar_reproduction.runnable_prototype import (
    build_glomerulus_granule_prototype,
)


ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spec",
        type=Path,
        nargs="?",
        default=ROOT / "data/export/balanced",
    )
    parser.add_argument(
        "source_hdf5",
        type=Path,
        nargs="?",
        default=ROOT / "data/reference/balanced.hdf5",
    )
    parser.add_argument(
        "braincell_repo",
        type=Path,
        nargs="?",
        default=ROOT.parent / "braincell",
    )
    parser.add_argument("--duration-ms", type=float, default=2.0)
    args = parser.parse_args()

    network, metadata = build_glomerulus_granule_prototype(
        args.spec,
        args.source_hdf5,
        args.braincell_repo,
    )
    result = network.run(
        dt=0.025 * u.ms,
        duration=args.duration_ms * u.ms,
        event_backend="scatter",
    )
    ampa = np.asarray(result.traces["granule_cell"]["ampa_g"], dtype=float)
    voltage = np.asarray(
        result.traces["granule_cell"]["v_soma"].to_decimal(u.mV),
        dtype=float,
    )
    relay_spikes = np.asarray(result.spikes["glomerulus"])
    summary = {
        "network": repr(network),
        "contact": asdict(metadata),
        "relay_spike_count": int(relay_spikes.sum()),
        "ampa_g_max": float(ampa.max()),
        "granule_v_soma_min_mV": float(voltage.min()),
        "granule_v_soma_max_mV": float(voltage.max()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
