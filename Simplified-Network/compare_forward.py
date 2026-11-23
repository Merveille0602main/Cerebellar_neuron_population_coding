#!/usr/bin/env python3
"""Compare the matched forward runs with and without Golgi junctions."""

import argparse
import json
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parent / "results"


def _population_stats(before, after, population: str) -> dict:
    ids = before[f"{population}_source_ids"]
    np.testing.assert_array_equal(after[f"{population}_source_ids"], ids)
    before_v = before[f"{population}_soma_v_mV"].astype(np.float64)
    after_v = after[f"{population}_soma_v_mV"].astype(np.float64)
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


def compare(baseline: Path, coupled: Path) -> dict:
    with (baseline / "forward_summary.json").open(encoding="utf-8") as f:
        before = json.load(f)
    with (coupled / "forward_summary.json").open(encoding="utf-8") as f:
        after = json.load(f)
    if before["routes"]["gap_goc"]["runtime_electrical_coupling"] or not after["routes"]["gap_goc"]["runtime_electrical_coupling"]:
        raise ValueError("Comparison requires an uncoupled baseline and a Junction-enabled run.")
    if any(before[key] != after[key] for key in ("braincell_version", "selection", "receiver_mode", "population_counts")):
        raise ValueError("Comparison requires the same BrainCell version, subset, and chemistry mode.")
    before_sim, after_sim = before["simulation"], after["simulation"]
    matched = ("duration_ms", "dt_ms", "steps", "chemical_event_rows", "stimulus")
    if any(before_sim[key] != after_sim[key] for key in matched):
        raise ValueError("Comparison requires identical stimulation, simulation clocks, and chemical routing.")
    for route, details in before["routes"].items():
        if route != "gap_goc" and details != after["routes"][route]:
            raise ValueError(f"Chemical or relay route changed: {route}.")
    for name in ("mossy_fibers", "glomerulus"):
        if before_sim["event_counts"][name] != after_sim["event_counts"][name]:
            raise ValueError(f"Input event counts differ for {name}.")

    with np.load(baseline / "forward_traces.npz", allow_pickle=False) as first, np.load(
        coupled / "forward_traces.npz", allow_pickle=False
    ) as second:
        np.testing.assert_array_equal(first["time_ms"], second["time_ms"])
        populations = {
            name: _population_stats(first, second, name)
            for name in ("granule_cell", "golgi_cell")
        }
    return {
        "matched_run_settings": {key: before_sim[key] for key in matched},
        "populations": populations,
        "junction": after["routes"]["gap_goc"] | after_sim["junction_current"],
        "chemical_conductance_peak_change_nS": {
            name: value - before_sim["peak_synaptic_conductance_nS"][name]
            for name, value in after_sim["peak_synaptic_conductance_nS"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=RESULTS)
    parser.add_argument("--coupled", type=Path, default=RESULTS / "with_junction")
    parser.add_argument("--output", type=Path, default=RESULTS / "junction_comparison.json")
    args = parser.parse_args()
    report = json.dumps(compare(args.baseline, args.coupled), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
