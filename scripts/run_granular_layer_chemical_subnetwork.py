#!/usr/bin/env python3
"""Build or simulate the granular layer with the official BrainCell 0.2 API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import brainunit as u
import numpy as np

from cerebellar_reproduction.granular_layer import MF, build_granular_layer
from cerebellar_reproduction.paper_inputs import build_paper_mossy_input_schedule


ROOT = Path(__file__).resolve().parents[1]
BUNDLED_ASSETS = ROOT / "assets/braincell"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "data/export/balanced")
    parser.add_argument("--source-hdf5", type=Path, default=ROOT / "data/reference/balanced.hdf5")
    parser.add_argument(
        "--braincell-repo", type=Path,
        default=BUNDLED_ASSETS if BUNDLED_ASSETS.is_dir() else ROOT.parent / "braincell",
        help="override the bundled BrainCell morphology and cell templates",
    )
    parser.add_argument("--full", action="store_true", help="retain every published granular-layer cell and contact")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan-only", action="store_true", help="validate full morphology and contact mapping")
    actions.add_argument("--build-only", action="store_true", help="also materialize all native connections")
    parser.add_argument("--exercise-circuit", action="store_true", help="apply diagnostic somatic clamps in the pilot")
    parser.add_argument("--receiver-mode", choices=("section", "contact"), default="contact")
    parser.add_argument("--input-config", type=Path, help="read BSB initial_mf/background with replacement random seeds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--temperature-c", type=float, default=32.0)
    args = parser.parse_args()
    if args.dt_ms <= 0.0 or (args.duration_ms is not None and args.duration_ms <= 0.0):
        parser.error("dt and duration must be positive.")
    if args.full and not (args.plan_only or args.build_only or args.duration_ms is not None):
        parser.error("Set an explicit --duration-ms for a full simulation.")
    network_options = dict(
        full=args.full,
        temperature_celsius=args.temperature_c,
        receiver_mode=args.receiver_mode,
        exercise_circuit=args.exercise_circuit,
    )
    if args.input_config and not args.plan_only:
        if args.full:
            with np.load(args.spec / "populations" / f"{MF}.npz", allow_pickle=False) as data:
                ids = data["source_ids"].tolist()
        else:
            _, plan = build_granular_layer(
                args.spec, args.source_hdf5, args.braincell_repo, plan_only=True, **network_options
            )
            ids = plan["selected_global_ids"][MF]
        schedule = build_paper_mossy_input_schedule(args.input_config, ids, seed=args.seed)
        network_options["mossy_spike_times_ms"] = dict(schedule.spike_times_ms)
    network, report = build_granular_layer(
        args.spec, args.source_hdf5, args.braincell_repo, plan_only=args.plan_only, **network_options
    )
    if network is not None:
        report["network"] = repr(network)
    if network is not None and not args.build_only:
        duration = args.duration_ms if args.duration_ms is not None else 18.0
        result = network.run(dt=args.dt_ms * u.ms, duration=duration * u.ms)
        report["simulation"] = {
            "dt_ms": args.dt_ms,
            "duration_ms": duration,
            "steps": len(result.time),
            "events": {
                kind: {name: len(series.source_id) for name, series in outputs.items()}
                for kind, outputs in result.events.items()
            },
            "somatic_voltage_mV": {
                kind: [float(np.min(values)), float(np.max(values))]
                for kind, outputs in result.samples.items()
                if "soma_v" in outputs
                for values in [np.asarray(outputs["soma_v"].values.to_decimal(u.mV))]
            },
            "peak_conductance_uS": {
                name: float(np.max(block.values.to_decimal(u.uS)))
                for outputs in result.samples.values()
                for name, block in outputs.items()
                if name != "soma_v"
            },
        }
        if not all(np.isfinite(block.values.to_decimal(u.mV)).all()
                   for outputs in result.samples.values()
                   for name, block in outputs.items() if name == "soma_v"):
            raise RuntimeError("The simulation produced non-finite soma voltages.")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
