#!/usr/bin/env python3
"""Run the extracted network and save somatic traces, spikes, and a summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import braincell as bc
import brainunit as u
import numpy as np

from cerebellar_reproduction.granular_layer import GOC, GRC, MF, ROUTES, build_granular_layer
from cerebellar_reproduction.validation import validate_export


ROOT = Path(__file__).resolve().parent
NEURON_POPULATIONS = (GRC, GOC)


def run_forward(duration_ms: float, dt_ms: float, output: Path) -> dict:
    spec = ROOT / "data/spec"
    validate_export(spec)
    with np.load(spec / f"populations/{MF}.npz", allow_pickle=False) as data:
        mossy_ids = data["source_ids"]
    burst = np.arange(0.25, 8.0, 0.4)
    started = perf_counter()
    network, report = build_granular_layer(
        spec,
        ROOT / "data/morphologies.hdf5",
        ROOT / "assets/braincell",
        full=True,
        mossy_spike_times_ms=dict.fromkeys(mossy_ids.tolist(), burst),
    )
    report.pop("braincell_module", None)
    report["selection"] = "seed_318_simplified_network"
    build_seconds = perf_counter() - started
    print(f"Built {network.connections.n_rows:,} event routes in {build_seconds:.1f} s.", flush=True)

    for population in NEURON_POPULATIONS:
        network.populations[population].cell.soma.record("soma_v", bc.observe.state("v"))
    for route in ROUTES:
        for receptor in report["routes"][route.name].get("mechanisms", {}):
            name = f"{route.name}_{receptor}"
            network.populations[route.post].cell.record(name, bc.observe.synapse(name=name).state("g"))

    started = perf_counter()
    result = network.run(dt=dt_ms * u.ms, duration=duration_ms * u.ms)
    simulation_seconds = perf_counter() - started

    arrays = {}
    soma_ranges = {}
    peak_conductance = {}
    for population in NEURON_POPULATIONS:
        with np.load(spec / f"populations/{population}.npz", allow_pickle=False) as data:
            source_ids = np.sort(data["source_ids"])
        samples = result.samples[population]
        soma_record = samples["soma_v"].values.to_decimal(u.mV)
        soma_v = np.asarray(soma_record).reshape(-1, len(source_ids))
        if soma_v.shape != (len(result.time), len(source_ids)) or not np.isfinite(soma_v).all():
            raise RuntimeError(f"{population} produced invalid somatic voltage traces.")
        spikes = result.events[population]["spike"]
        arrays[f"{population}_source_ids"] = source_ids
        arrays[f"{population}_soma_v_mV"] = soma_v
        arrays[f"{population}_spike_time_ms"] = np.asarray(spikes.time.to_decimal(u.ms))
        arrays[f"{population}_spike_source_ids"] = source_ids[spikes.source_id]
        soma_ranges[population] = [float(soma_v.min()), float(soma_v.max())]
        for name, block in samples.items():
            if name == "soma_v":
                continue
            values = np.asarray(block.values.to_decimal(u.nS))
            if not np.isfinite(values).all():
                raise RuntimeError(f"{name} produced non-finite synaptic conductances.")
            peak_conductance[name] = float(values.max())

    arrays["time_ms"] = np.asarray(result.time.to_decimal(u.ms))
    report["simulation"] = {
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "steps": len(result.time),
        "build_seconds": round(build_seconds, 3),
        "run_seconds_including_initialization_and_compilation": round(simulation_seconds, 3),
        "chemical_event_rows": network.connections.n_rows,
        "stimulus": {"kind": "synchronous_mossy_burst", "times_ms": burst.round(6).tolist()},
        "somatic_voltage_range_mV": soma_ranges,
        "peak_synaptic_conductance_nS": peak_conductance,
        "event_counts": {
            population: {port: len(series.source_id) for port, series in ports.items()}
            for population, ports in result.events.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "forward_traces.npz", **arrays)
    (output / "forward_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-ms", type=float, default=18.0)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if not (np.isfinite(args.dt_ms) and np.isfinite(args.duration_ms) and 0 < args.dt_ms <= args.duration_ms):
        parser.error("Require a finite duration at least as long as one positive integration step.")
    report = run_forward(args.duration_ms, args.dt_ms, args.output)
    print(json.dumps(report["simulation"], indent=2, sort_keys=True))
    print(f"Saved traces and summary to {args.output}.")


if __name__ == "__main__":
    main()
