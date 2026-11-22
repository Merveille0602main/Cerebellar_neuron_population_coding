#!/usr/bin/env python3
"""Generate fixed-voltage traces from the BrainCell GrC synapse ports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import brainstate
import brainunit as u
import numpy as np

from braincell.quad import ind_exp_euler_step
from cerebellar_reproduction.paper_synapses import (
    GrCAMPA2020,
    GrCGABA2020,
    GrCNMDA2020,
)


ROOT = Path(__file__).parents[1]


def _decimal(value, unit=None) -> float:
    if unit is not None:
        value = value.to_decimal(unit)
    return float(np.asarray(value).reshape(()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mechanism", choices=("ampa", "nmda", "gaba"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spike-times-ms", default="1.0")
    parser.add_argument("--voltage-mv", type=float, default=-65.0)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    args = parser.parse_args()

    cls = {
        "ampa": GrCAMPA2020,
        "nmda": GrCNMDA2020,
        "gaba": GrCGABA2020,
    }[args.mechanism]
    synapse = cls(size=(1,))
    synapse.init_state()
    spike_times = tuple(
        float(value) for value in args.spike_times_ms.split(",") if value.strip()
    )
    spike_steps = {int(round(value / args.dt_ms)) for value in spike_times}
    times = np.arange(
        0.0,
        args.duration_ms + (0.5 * args.dt_ms),
        args.dt_ms,
        dtype=float,
    )
    voltage = args.voltage_mv * u.mV
    traces: dict[str, list[float]] = {
        "g_pS": [],
        "current_nA": [],
        "open": [],
        "transmitter_mM": [],
        "release_y": [],
    }
    if args.mechanism == "ampa":
        traces["desensitized"] = []
        traces["closed"] = []
    elif args.mechanism == "nmda":
        traces["desensitized"] = []
        traces.update({"C0": [], "C1": [], "C2": [], "mg_block": []})
    else:
        traces.update(
            {
                "open_alpha6": [],
                "g_alpha1_pS": [],
                "g_alpha6_pS": [],
                "diffused_mM": [],
            }
        )

    with brainstate.environ.context(dt=args.dt_ms * u.ms):
        for step, time_ms in enumerate(times):
            with brainstate.environ.context(t=time_ms * u.ms):
                drive = np.asarray([1.0 if step in spike_steps else 0.0])
                synapse.apply_events(drive * u.uS, voltage)
                transmitter = synapse._diffused_transmitter()
                if args.mechanism in ("ampa", "gaba"):
                    transmitter = transmitter + synapse._local_transmitter_pulse()
                traces["g_pS"].append(_decimal(synapse.g, u.uS) * 1.0e6)
                traces["current_nA"].append(_decimal(-synapse.current(voltage), u.nA))
                traces["transmitter_mM"].append(_decimal(transmitter, u.mM))
                traces["release_y"].append(_decimal(synapse.last_release.value))
                if args.mechanism == "ampa":
                    traces["open"].append(_decimal(synapse.O.value))
                    traces["desensitized"].append(_decimal(synapse.D.value))
                    traces["closed"].append(_decimal(synapse.C.value))
                elif args.mechanism == "nmda":
                    traces["open"].append(_decimal(synapse.O.value))
                    traces["desensitized"].append(_decimal(synapse.D.value))
                    traces["C0"].append(_decimal(synapse.C0.value))
                    traces["C1"].append(_decimal(synapse.C1.value))
                    traces["C2"].append(_decimal(synapse.C2.value))
                    traces["mg_block"].append(_decimal(synapse.magnesium_block(voltage)))
                else:
                    diffusion = synapse._diffused_transmitter()
                    traces["open"].append(_decimal(synapse.open_alpha1))
                    traces["open_alpha6"].append(_decimal(synapse.open_alpha6))
                    traces["g_alpha1_pS"].append(
                        _decimal(synapse.g_alpha1, u.uS) * 1.0e6
                    )
                    traces["g_alpha6_pS"].append(
                        _decimal(synapse.g_alpha6, u.uS) * 1.0e6
                    )
                    traces["diffused_mM"].append(_decimal(diffusion, u.mM))
                if step < len(times) - 1:
                    ind_exp_euler_step(synapse, voltage)

    data = {
        "time_ms": times,
        "voltage_mV": np.full(times.shape, args.voltage_mv),
        **{name: np.asarray(values, dtype=float) for name, values in traces.items()},
    }
    output = args.output or ROOT / f"data/reference/braincell_grc_{args.mechanism}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(
        json.dumps(
            {
                "mechanism": cls.__name__,
                "output": str(output),
                "n_sample": len(times),
                "spike_times_ms": spike_times,
                "g_peak_pS": float(data["g_pS"].max()),
                "current_abs_peak_nA": float(np.abs(data["current_nA"]).max()),
                "transmitter_peak_mM": float(data["transmitter_mM"].max()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
