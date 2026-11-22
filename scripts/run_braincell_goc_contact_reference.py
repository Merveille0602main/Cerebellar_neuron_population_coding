#!/usr/bin/env python3
"""Generate the matching BrainCell whole-GoC published-contact trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import brainunit as u
import numpy as np

from cerebellar_reproduction import build_glomerulus_golgi_subnetwork


ROOT = Path(__file__).parents[1]


def _squeeze_trace(value, unit) -> np.ndarray:
    return np.asarray(value.to_decimal(unit), dtype=float).reshape(len(value), -1)[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "data/export/balanced")
    parser.add_argument(
        "--source-hdf5",
        type=Path,
        default=ROOT / "data/reference/balanced.hdf5",
    )
    parser.add_argument("--braincell-repo", type=Path, default=ROOT.parent / "braincell")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/reference/braincell_goc_contact.npz",
    )
    parser.add_argument("--spike-time-ms", type=float, default=1.0)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--duration-ms", type=float, default=3.0)
    args = parser.parse_args()

    with np.load(args.spec / "connectivity/glomerulus_to_golgi.npz") as source:
        glomerulus_id = int(np.unique(source["cell_pairs"][:, 0])[0])
    network, metadata = build_glomerulus_golgi_subnetwork(
        args.spec,
        args.source_hdf5,
        args.braincell_repo,
        glomerulus_spike_times_ms={glomerulus_id: (args.spike_time_ms,)},
        max_glomeruli=1,
        max_golgi=1,
    )
    result = network.run(
        dt=args.dt_ms * u.ms,
        duration=args.duration_ms * u.ms,
        event_backend="scatter",
        spike_recording="population",
    )
    traces = result.traces["golgi_cell"]
    v_soma = _squeeze_trace(traces["v_soma"], u.mV)
    v_contact = _squeeze_trace(traces["v_contact"], u.mV)
    ampa_g = _squeeze_trace(traces["ampa_mf_g"], u.uS)
    nmda_g = _squeeze_trace(traces["nmda_g"], u.uS)
    mg_block = 1.0 / (1.0 + np.exp(-(v_contact + 20.0) / 13.0))
    data = {
        "time_ms": np.asarray(result.time.to_decimal(u.ms), dtype=float),
        "v_soma_mV": v_soma,
        "v_contact_mV": v_contact,
        "ampa_g_uS": ampa_g,
        "ampa_current_nA": ampa_g * v_contact,
        "nmda_g_uS": nmda_g,
        "nmda_current_nA": nmda_g * (v_contact + 3.7) * mg_block,
        "source_glomerulus_id": np.asarray(metadata.source_glomerulus_ids[0]),
        "source_golgi_id": np.asarray(metadata.source_golgi_ids[0]),
        "target_compartment": np.asarray(metadata.target_compartments[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_sample": len(data["time_ms"]),
                "source_glomerulus_id": metadata.source_glomerulus_ids[0],
                "source_golgi_id": metadata.source_golgi_ids[0],
                "target_compartment": metadata.target_compartments[0],
                "ampa_g_peak_uS": float(ampa_g.max()),
                "nmda_g_peak_uS": float(nmda_g.max()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
