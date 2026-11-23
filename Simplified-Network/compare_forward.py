#!/usr/bin/env python3
"""Compare the uncoupled, explicit, and native Golgi junction runs."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parent / "results"
POPULATIONS = ("granule_cell", "golgi_cell")
INPUTS = ("mossy_fibers", "glomerulus")
NETWORK_SETTINGS = (
    "selection", "receiver_mode", "population_counts", "control_volumes_per_cell",
    "morphology_branches_per_cell", "diagnostic_current_clamps", "ascending_axon_ampa",
    "golgi_to_golgi_gaba", "mossy_to_glomerulus",
)
SIMULATION_SETTINGS = ("duration_ms", "dt_ms", "steps", "chemical_event_rows", "stimulus")


def _population_stats(before, after, population: str) -> dict:
    ids = before[f"{population}_source_ids"]
    np.testing.assert_array_equal(after[f"{population}_source_ids"], ids)
    before_v = before[f"{population}_soma_v_mV"].astype(np.float64)
    after_v = after[f"{population}_soma_v_mV"].astype(np.float64)
    expected_shape = (len(before["time_ms"]), len(ids))
    if before_v.shape != expected_shape or after_v.shape != expected_shape:
        raise ValueError(f"Unexpected somatic trace shape for {population}.")
    if not np.isfinite(before_v).all() or not np.isfinite(after_v).all():
        raise ValueError(f"Non-finite somatic voltage for {population}.")
    difference = after_v - before_v
    row, column = np.unravel_index(np.abs(difference).argmax(), difference.shape)
    before_spikes = before[f"{population}_spike_source_ids"]
    after_spikes = after[f"{population}_spike_source_ids"]
    before_times = before[f"{population}_spike_time_ms"].astype(np.float64)
    after_times = after[f"{population}_spike_time_ms"].astype(np.float64)
    changed_counts = []
    shifts = []
    for source_id in ids:
        first = np.sort(before_times[before_spikes == source_id])
        second = np.sort(after_times[after_spikes == source_id])
        if len(first) == len(second):
            shifts.extend(second - first)
        else:
            changed_counts.append(int(source_id))

    return {
        "somatic_spikes_before": len(before_spikes),
        "somatic_spikes_after": len(after_spikes),
        "cells_firing_before": len(set(before_spikes.tolist())),
        "cells_firing_after": len(set(after_spikes.tolist())),
        "source_ids_with_changed_spike_count": changed_counts,
        "paired_spikes_in_cells_with_unchanged_count": len(shifts),
        "max_absolute_paired_spike_shift_ms": float(np.abs(shifts).max()) if shifts else None,
        "voltage_rms_difference_mV": float(np.sqrt(np.mean(difference**2))),
        "max_absolute_voltage_difference_mV": float(np.abs(difference[row, column])),
        "max_difference_source_id": int(ids[column]),
        "max_difference_time_ms": float(before["time_ms"][row]),
    }


def _matched_settings(reference: dict, candidate: dict, name: str) -> None:
    for scope, keys in ((None, NETWORK_SETTINGS), ("simulation", SIMULATION_SETTINGS)):
        before = reference if scope is None else reference[scope]
        after = candidate if scope is None else candidate[scope]
        for key in keys:
            if before[key] != after[key]:
                raise ValueError(f"{name} differs from the uncoupled baseline in {key}.")
    before_routes = {route: data for route, data in reference["routes"].items() if route != "gap_goc"}
    after_routes = {route: data for route, data in candidate["routes"].items() if route != "gap_goc"}
    if before_routes != after_routes:
        raise ValueError(f"{name} has different chemical or relay routes.")
    for population in INPUTS:
        if reference["simulation"]["event_counts"][population] != candidate["simulation"]["event_counts"][population]:
            raise ValueError(f"{name} has different input event counts for {population}.")


def _pairwise(before, after, before_sim: dict, after_sim: dict) -> dict:
    return {
        "populations": {name: _population_stats(before, after, name) for name in POPULATIONS},
        "chemical_conductance_peak_change_nS": {
            name: value - before_sim["peak_synaptic_conductance_nS"][name]
            for name, value in after_sim["peak_synaptic_conductance_nS"].items()
        },
    }


def compare(baseline: Path, explicit: Path, native: Path) -> dict:
    folders = {"baseline": baseline, "explicit": explicit, "native": native}
    summaries = {
        name: json.loads((folder / "forward_summary.json").read_text(encoding="utf-8"))
        for name, folder in folders.items()
    }
    reference = summaries["baseline"]
    for name in ("explicit", "native"):
        _matched_settings(reference, summaries[name], name)

    gaps = {name: summary["routes"]["gap_goc"] for name, summary in summaries.items()}
    if gaps["baseline"]["runtime_electrical_coupling"]:
        raise ValueError("The baseline must have electrical coupling disabled.")
    for name, integration in (
        ("explicit", "explicit_start_of_step_point_voltage"),
        ("native", "native_coupled_implicit_cable_voltage"),
    ):
        if not gaps[name]["runtime_electrical_coupling"] or gaps[name]["integration"] != integration:
            raise ValueError(f"Unexpected junction solver for {name}.")
    for key in ("anatomical_contacts", "physical_contacts", "conductance_nS_per_contact"):
        if gaps["native"][key] != gaps["explicit"][key]:
            raise ValueError(f"Native and explicit junctions differ in {key}.")

    runtimes = {
        name: summary.get("braincell_runtime", {"source": "pypi_release", "version": summary["braincell_version"]})
        for name, summary in summaries.items()
    }
    if runtimes["native"]["source"] not in {"local_native_junction_worktree", "reproduction_bundled"}:
        raise ValueError("The native result must use a verified BrainCell Junction runtime.")
    simulations = {name: summary["simulation"] for name, summary in summaries.items()}
    pairs = (("baseline", "explicit"), ("baseline", "native"), ("explicit", "native"))
    with ExitStack() as stack:
        traces = {
            name: stack.enter_context(np.load(folder / "forward_traces.npz", allow_pickle=False))
            for name, folder in folders.items()
        }
        time_ms = traces["baseline"]["time_ms"]
        for name in ("explicit", "native"):
            # The two runtimes can round the same float32 time grid differently.
            np.testing.assert_allclose(traces[name]["time_ms"], time_ms, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(
            traces["explicit"]["junction_contact_endpoints"], traces["native"]["junction_contact_endpoints"]
        )
        comparisons = {
            f"{second}_vs_{first}": _pairwise(traces[first], traces[second], simulations[first], simulations[second])
            for first, second in pairs
        }

    return {
        "matched_run_settings": {key: simulations["baseline"][key] for key in SIMULATION_SETTINGS},
        "input_event_counts": {name: simulations["baseline"]["event_counts"][name] for name in INPUTS},
        "braincell_runtimes": runtimes,
        "junctions": {
            "baseline": gaps["baseline"],
            "explicit": gaps["explicit"] | {"current_measurement": "applied_start_of_step_point_currents"}
            | simulations["explicit"]["junction_current"],
            "native": gaps["native"] | simulations["native"]["junction_current"],
        },
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=RESULTS)
    parser.add_argument("--explicit", "--coupled", type=Path, default=RESULTS / "with_junction")
    parser.add_argument("--native", type=Path, default=RESULTS / "with_native_junction")
    parser.add_argument("--output", type=Path, default=RESULTS / "native_junction_comparison.json")
    args = parser.parse_args()
    report = json.dumps(compare(args.baseline, args.explicit, args.native), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
