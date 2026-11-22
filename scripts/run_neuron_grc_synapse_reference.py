#!/usr/bin/env python3
"""Generate voltage-clamped NEURON traces from the published GrC MOD files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
DEFAULT_MOD_DIR = ROOT / "data/reference/neuron_goc_mechanisms"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mechanism", choices=("ampa", "nmda", "gaba"))
    parser.add_argument("--mod-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spike-times-ms", default="1.0")
    parser.add_argument("--voltage-mv", type=float, default=-65.0)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    args = parser.parse_args()

    from neuron import h, load_mechanisms

    mod_dir = args.mod_dir
    if mod_dir is None:
        mod_dir = DEFAULT_MOD_DIR
    load_mechanisms(str(mod_dir))
    h.load_file("stdrun.hoc")
    section = h.Section(name=f"grc_{args.mechanism}_reference")
    section.L = 20.0
    section.diam = 20.0
    section.nseg = 1
    segment = section(0.5)
    clamp = h.SEClamp(segment)
    clamp.dur1 = args.duration_ms + args.dt_ms
    clamp.amp1 = args.voltage_mv
    clamp.rs = 1.0e-4

    class_name = {
        "ampa": "GRANULE_Ampa_det_vi",
        "nmda": "GRANULE_Nmda_det_vi",
        "gaba": "glia__dbbs_mod_collection__GABA__granule",
    }[args.mechanism]
    synapse = getattr(h, class_name)(segment)
    if args.mechanism in ("ampa", "nmda"):
        # dbbs_models v1.1.2 overrides the MOD default (0.42) for GrC only.
        synapse.U = 0.43
    connection = h.NetCon(None, synapse)
    connection.delay = 0.0
    connection.weight[0] = 1.0
    spike_times = tuple(
        float(value) for value in args.spike_times_ms.split(",") if value.strip()
    )

    fields = (("g_pS", "g"), ("current_nA", "i"))
    if args.mechanism == "gaba":
        fields += (
            ("open", "Open"),
            ("open_alpha6", "Open_a6"),
            ("g_alpha1_pS", "gA1"),
            ("g_alpha6_pS", "gA6"),
            ("transmitter_mM", "Trelease"),
            ("local_T_mM", "T"),
            ("diffused_mM", "diffus"),
            ("release_y", "yout"),
        )
    else:
        fields += (
            ("open", "O"),
            ("desensitized", "D"),
            ("transmitter_mM", "Trelease"),
            ("local_T_mM", "T"),
        )
    if args.mechanism == "ampa":
        fields += (("closed", "C"), ("release_y", "yuno"))
    elif args.mechanism == "nmda":
        fields += (
            ("C0", "C0"),
            ("C1", "C1"),
            ("C2", "C2"),
            ("mg_block", "MgBlock"),
        )
    recordings = {
        output_name: h.Vector().record(getattr(synapse, f"_ref_{field}"))
        for output_name, field in fields
    }
    time = h.Vector().record(h._ref_t)
    voltage = h.Vector().record(segment._ref_v)

    h.celsius = 32.0
    h.cvode_active(0)
    h.dt = args.dt_ms
    h.steps_per_ms = 1.0 / args.dt_ms
    h.finitialize(args.voltage_mv)
    for spike_time in spike_times:
        connection.event(spike_time)
    h.continuerun(args.duration_ms)

    data = {
        "time_ms": np.asarray(time, dtype=float),
        "voltage_mV": np.asarray(voltage, dtype=float),
        **{name: np.asarray(vector, dtype=float) for name, vector in recordings.items()},
    }
    output = args.output or ROOT / f"data/reference/neuron_grc_{args.mechanism}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    summary = {
        "mechanism": class_name,
        "output": str(output),
        "n_sample": len(data["time_ms"]),
        "spike_times_ms": spike_times,
        "g_peak_pS": float(data["g_pS"].max()),
        "current_abs_peak_nA": float(np.abs(data["current_nA"]).max()),
        "transmitter_peak_mM": float(data["transmitter_mM"].max()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    h.delete_section(sec=section)


if __name__ == "__main__":
    main()
