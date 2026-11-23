#!/usr/bin/env python3
"""Run the extracted network and save somatic traces, spikes, and a summary."""

from __future__ import annotations

import argparse
from hashlib import sha256
from inspect import signature
import json
from pathlib import Path
from time import perf_counter

import brainunit as u
import numpy as np

from junction import CONDUCTANCE_NS, CV_VOLTAGE_RECORDING, NativeJunction


ROOT = Path(__file__).resolve().parent
RESULT_FOLDERS = {"native": "with_native_junction", "explicit": "with_junction", "off": "without_junction"}


def _runtime_digest(root: Path) -> str:
    digest = sha256()
    for path in sorted((root / "braincell").rglob("*.py")):
        if path.name.endswith("_test.py") or path.name.startswith("test_"):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _load_braincell(mode: str):
    import braincell as bc

    if bc.__version__ == "0.2.0" and mode != "native":
        return bc, {"source": "pypi_release", "version": bc.__version__}
    if mode == "explicit":
        raise RuntimeError("The historical explicit mode requires the released BrainCell 0.2.0 environment.")
    if "+reproduction." not in bc.__version__ or not hasattr(bc.Network, "connect_junction"):
        raise RuntimeError("Install the BrainCell runtime bundled in vendor/braincell as shown in the README.")
    package_root = Path(bc.__file__).resolve().parent.parent
    return bc, {
        "source": "reproduction_bundled",
        "version": bc.__version__,
        "runtime_python_sha256": _runtime_digest(package_root),
    }


def run_forward(
    duration_ms: float, dt_ms: float, output: Path, *,
    junction_mode: str = "native",
) -> dict:
    bc, runtime = _load_braincell(junction_mode)
    from cerebellar_reproduction.granular_layer import GOC, GRC, MF, ROUTES, build_granular_layer
    from cerebellar_reproduction.validation import validate_export

    spec = ROOT / "data/spec"
    morphologies = ROOT / "data/morphologies.hdf5"
    validate_export(spec)
    with np.load(spec / f"populations/{MF}.npz", allow_pickle=False) as data:
        mossy_ids = data["source_ids"]
    burst = np.arange(0.25, 8.0, 0.4)
    started = perf_counter()
    builder_options = {}
    if "gap_conductance_ns" in signature(build_granular_layer).parameters:
        # Some local parent worktrees can wire gap contacts themselves. This
        # runner owns Junction installation so each contact is registered once.
        builder_options["gap_conductance_ns"] = 0.0
    network, report = build_granular_layer(
        spec,
        morphologies,
        ROOT / "assets/braincell",
        full=True,
        mossy_spike_times_ms=dict.fromkeys(mossy_ids.tolist(), burst),
        **builder_options,
    )
    report.pop("braincell_module", None)
    report["selection"] = "seed_318_simplified_network"
    report["braincell_runtime"] = runtime
    report["junction_mode"] = junction_mode
    junction = None
    recording = None
    if junction_mode == "native":
        junction = NativeJunction.install(network, network.populations[GOC].cell, spec, morphologies)
        recording = CV_VOLTAGE_RECORDING
    elif junction_mode == "explicit":
        from explicit_junction import RECORDING, ExplicitJunction

        junction = ExplicitJunction.install(network.populations[GOC].cell, spec, morphologies)
        recording = RECORDING
    if junction is not None:
        report["routes"]["gap_goc"].update(junction.metadata())
        report["excluded"].pop("gap_goc", None)
    build_seconds = perf_counter() - started
    print(f"Built {network.connections.n_rows:,} event routes in {build_seconds:.1f} s.", flush=True)
    if junction is not None:
        print(f"Registered {len(junction.endpoints)} {junction_mode} Junction contacts.", flush=True)

    for population in (GRC, GOC):
        network.populations[population].cell.soma.record("soma_v", bc.observe.state("v"))
    for route in ROUTES:
        for receptor in report["routes"][route.name].get("mechanisms", {}):
            name = f"{route.name}_{receptor}"
            network.populations[route.post].cell.record(name, bc.observe.synapse(name=name).state("g"))

    started = perf_counter()
    result = network.run(dt=dt_ms * u.ms, duration=duration_ms * u.ms)
    simulation_seconds = perf_counter() - started

    arrays = {}
    junction_current = None
    if junction is not None:
        sample = result.samples[GOC][recording]
        if junction_mode == "native":
            endpoint_v = junction.endpoint_voltage(sample, len(result.time))
            into_first = CONDUCTANCE_NS * (endpoint_v[..., 1] - endpoint_v[..., 0])
            junction_current = np.stack((into_first, -into_first), axis=-1)
            arrays["junction_endpoint_voltage_mV"] = endpoint_v
            arrays["junction_endpoint_cv_ids"] = junction.cv_ids
        else:
            junction_current = np.asarray(sample.values.to_decimal(u.pA)).reshape(len(result.time), -1, 2)
        if not np.isfinite(junction_current).all():
            raise RuntimeError("Junction currents contain non-finite values.")
        np.testing.assert_allclose(junction_current[..., 0], -junction_current[..., 1], rtol=0, atol=1e-5)
        arrays["junction_contact_endpoints"] = junction.endpoints
        arrays["junction_current_pA"] = junction_current

    soma_ranges = {}
    peak_conductance = {}
    for population in (GRC, GOC):
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
            if name in {"soma_v", recording}:
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
    if junction_current is not None:
        report["simulation"]["junction_current"] = {
            "peak_absolute_pA": float(np.abs(junction_current).max()),
            "max_pair_residual_pA": float(np.abs(junction_current.sum(axis=-1)).max()),
            "contacts_with_nonzero_current": int(np.count_nonzero(np.any(junction_current[..., 0], axis=0))),
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--junction-mode", choices=tuple(RESULT_FOLDERS), default="native")
    modes.add_argument("--no-junctions", dest="junction_mode", action="store_const", const="off",
                       help="reproduce the uncoupled baseline")
    parser.add_argument("--output", type=Path, help="default: runs/<selected-mode folder>")
    args = parser.parse_args()
    if not (np.isfinite(args.dt_ms) and np.isfinite(args.duration_ms) and 0 < args.dt_ms <= args.duration_ms):
        parser.error("Require a finite duration at least as long as one positive integration step.")
    output = args.output if args.output is not None else ROOT / "runs" / RESULT_FOLDERS[args.junction_mode]
    report = run_forward(args.duration_ms, args.dt_ms, output, junction_mode=args.junction_mode)
    print(json.dumps(report["simulation"], indent=2, sort_keys=True))
    print(f"Saved traces and summary to {output}.")


if __name__ == "__main__":
    main()
