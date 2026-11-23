#!/usr/bin/env python3
"""Run a published glomerulus→Golgi BrainCell subnetwork."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import brainunit as u
import numpy as np

from cerebellar_reproduction import build_glomerulus_golgi_subnetwork


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
    parser.add_argument("--max-glomeruli", type=int, default=4)
    parser.add_argument("--max-golgi", type=int, default=4)
    parser.add_argument("--spike-times-ms", type=float, nargs="+", default=[1.0])
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--duration-ms", type=float, default=3.0)
    args = parser.parse_args()

    with np.load(args.spec / "connectivity/glomerulus_to_golgi.npz") as data:
        connected_glomeruli = np.unique(data["cell_pairs"][:, 0]).astype(int)
    selected_glomeruli = connected_glomeruli[: args.max_glomeruli]
    schedule = {
        int(source_id): tuple(args.spike_times_ms)
        for source_id in selected_glomeruli
    }
    network, metadata = build_glomerulus_golgi_subnetwork(
        args.spec,
        args.source_hdf5,
        args.braincell_repo,
        glomerulus_spike_times_ms=schedule,
        max_glomeruli=args.max_glomeruli,
        max_golgi=args.max_golgi,
    )
    result = network.run(
        dt=args.dt_ms * u.ms,
        duration=args.duration_ms * u.ms,
        event_backend="scatter",
        spike_recording="population",
    )
    ampa = np.asarray(
        result.traces["golgi_cell"]["ampa_mf_g"].to_decimal(u.uS),
        dtype=float,
    )
    nmda = np.asarray(
        result.traces["golgi_cell"]["nmda_g"].to_decimal(u.uS),
        dtype=float,
    )
    voltage = np.asarray(
        result.traces["golgi_cell"]["v_soma"].to_decimal(u.mV),
        dtype=float,
    )
    summary = {
        "network": repr(network),
        "selection": asdict(metadata),
        "glomerulus_spike_count": int(np.asarray(result.spikes["glomerulus"]).sum()),
        "golgi_spike_count": int(np.asarray(result.spikes["golgi_cell"]).sum()),
        "ampa_mf_g_max_uS": float(ampa.max()),
        "nmda_g_max_uS": float(nmda.max()),
        "golgi_v_soma_min_mV": float(voltage.min()),
        "golgi_v_soma_max_mV": float(voltage.max()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
