#!/usr/bin/env python3
"""Compare paper GoC intrinsic baselines mechanism-by-mechanism.

Use the reproduction environment and pass the GoC debug reference assets
and matching compiled NEURON libraries with ``--braincell-repo``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import brainstate
import brainunit as u
import numpy as np
from braincell import mech
from braincell.filter import at
from neuron import h

ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / "data/export/balanced/benchmarks/goc_intrinsic_baseline.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-ms", type=float, default=2.0)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--temperature-c", type=float, default=34.0)
    parser.add_argument("--v-init-mv", type=float, default=-65.0)
    parser.add_argument("--braincell-repo", type=Path, required=True, help="GoC debug reference assets")
    parser.add_argument(
        "--cases",
        help="Comma-separated case names; defaults to every single-mechanism case plus all.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    assets = args.braincell_repo.resolve()
    sys.path.append(str(assets))
    from examples.neuron_compare.cell.goc_ma2020.debug.goc_braincell_debug import GoC as BrainCellGoC
    from examples.neuron_compare.cell.goc_ma2020.debug.goc_neuron_debug import GoC as NeuronGoC
    from examples.neuron_compare.cell.goc_ma2020.debug.goc_parameters import (
        GoCConfig, GoCToggles, load_goc20_params,
    )

    brainstate.environ.set(precision=64)
    _load_neuron_libraries(assets / "examples/neuron_compare/Cerebellum_mod/GoC")
    parameter = load_goc20_params(assets / "examples/neuron_compare/cell/goc_ma2020/Optimization_result.txt")
    selected = None if args.cases is None else frozenset(args.cases.split(","))
    reports = []
    for case_name, enabled in _cases():
        if selected is not None and case_name not in selected:
            continue
        toggles = GoCToggles(**{field.name: field.name in enabled for field in fields(GoCToggles)})
        config = GoCConfig(
            toggles=toggles,
            temperature_celsius=args.temperature_c,
            v_init_mV=args.v_init_mv,
        )
        neuron_trace = _run_neuron(
            NeuronGoC(params=parameter, config=config, nrnmech_path=None).build(),
            config,
            dt_ms=args.dt_ms,
            duration_ms=args.duration_ms,
        )
        braincell_trace = _run_braincell(
            BrainCellGoC(params=parameter, config=config, frozen=True).build(),
            dt_ms=args.dt_ms,
            duration_ms=args.duration_ms,
        )
        report = _compare(case_name, enabled, neuron_trace, braincell_trace)
        reports.append(report)
        print(
            f"{case_name:20s} rmse={report['rmse_mV']:.6g} mV "
            f"max={report['max_abs_error_mV']:.6g} mV "
            f"end_delta={report['end_delta_mV']:.6g} mV"
        )

    payload = {
        "purpose": "GoC no-stimulus intrinsic baseline toggle audit",
        "precision": 64,
        "dt_ms": args.dt_ms,
        "duration_ms": args.duration_ms,
        "temperature_c": args.temperature_c,
        "v_init_mV": args.v_init_mv,
        "alignment": "BrainCell post-step sample i vs NEURON sample i+1",
        "neuron_version": _neuron_version(),
        "cases": reports,
    }
    if not reports:
        raise ValueError(f"No cases selected by --cases={args.cases!r}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


def _cases() -> tuple[tuple[str, frozenset[str]], ...]:
    active = (
        "nav",
        "kv1p1",
        "kv3p4",
        "kv4p3",
        "km",
        "kca1p1",
        "kca2p2",
        "kca3p1",
        "cahva",
        "cav2p3",
        "cav3p1",
        "hcn1",
        "hcn2",
        "cdp",
    )
    single = tuple((f"leak+{name}", frozenset(("leak", name))) for name in active)
    return (
        ("leak", frozenset(("leak",))),
        *single,
        ("all", frozenset(("leak", *active))),
    )


def _load_neuron_libraries(goc_root: Path) -> None:
    for component in ("channel", "ion"):
        library = goc_root / component / "x86_64/libnrnmech.so"
        if not library.exists():
            raise FileNotFoundError(f"Missing GoC NEURON mechanism library: {library}")
        h.nrn_load_dll(str(library))


def _run_neuron(model, config, *, dt_ms: float, duration_ms: float) -> np.ndarray:
    try:
        voltage = h.Vector().record(model.root_soma(0.5)._ref_v)
        h.cvode_active(0)
        h.dt = dt_ms
        h.steps_per_ms = 1.0 / dt_ms
        h.celsius = config.temperature_celsius
        h.v_init = config.v_init_mV
        h.finitialize(h.v_init)
        h.continuerun(duration_ms)
        return np.asarray(voltage, dtype=float).copy()
    finally:
        model.cleanup()


def _run_braincell(model, *, dt_ms: float, duration_ms: float) -> np.ndarray:
    model.cell.place(at("soma", 0.5), mech.StateProbe(name="v_soma"))
    model.cell.init_state()
    model.cell.reset_state()
    result = model.cell.run(dt=dt_ms * u.ms, duration=duration_ms * u.ms)
    return np.asarray(result.traces["v_soma"].to_decimal(u.mV), dtype=float).reshape(-1)


def _compare(
    name: str,
    enabled: frozenset[str],
    neuron_trace: np.ndarray,
    braincell_trace: np.ndarray,
) -> dict[str, Any]:
    # NEURON records the initialized t=0 sample; BrainCell's first StateProbe
    # value is after its first step.  Compare equal physical times.
    reference = neuron_trace[1 : 1 + len(braincell_trace)]
    if len(reference) != len(braincell_trace):
        raise RuntimeError(
            f"Trace length mismatch: NEURON={len(neuron_trace)}, BrainCell={len(braincell_trace)}."
        )
    error = braincell_trace - reference
    return {
        "name": name,
        "enabled": sorted(enabled),
        "n_sample": int(len(braincell_trace)),
        "neuron_first_post_step_mV": float(reference[0]),
        "braincell_first_post_step_mV": float(braincell_trace[0]),
        "first_step_delta_mV": float(error[0]),
        "neuron_end_mV": float(reference[-1]),
        "braincell_end_mV": float(braincell_trace[-1]),
        "end_delta_mV": float(error[-1]),
        "rmse_mV": float(np.sqrt(np.mean(error * error))),
        "max_abs_error_mV": float(np.max(np.abs(error))),
    }


def _neuron_version() -> str:
    import neuron

    return str(neuron.__version__)


if __name__ == "__main__":
    main()
