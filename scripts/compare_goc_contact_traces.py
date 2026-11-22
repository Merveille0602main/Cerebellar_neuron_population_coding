#!/usr/bin/env python3
"""Compare one published GoC contact while separating intrinsic baseline drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def _on_candidate_grid(reference, candidate, field: str) -> tuple[np.ndarray, np.ndarray]:
    reference_time = np.asarray(reference["time_ms"], dtype=float)
    candidate_time = np.asarray(candidate["time_ms"], dtype=float)
    values = np.interp(candidate_time, reference_time, np.asarray(reference[field], dtype=float))
    return values, np.asarray(candidate[field], dtype=float)


def _best_alignment(reference: np.ndarray, candidate: np.ndarray, max_lag: int) -> dict:
    best = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = reference[-lag:], candidate[:lag]
        elif lag > 0:
            left, right = reference[:-lag], candidate[lag:]
        else:
            left, right = reference, candidate
        rmse = float(np.sqrt(np.mean((left - right) ** 2)))
        peak = max(float(np.max(np.abs(reference))), 1.0e-15)
        item = {
            "lag_step": lag,
            "rmse": rmse,
            "normalized_rmse": rmse / peak,
            "max_abs": float(np.max(np.abs(left - right))),
        }
        if best is None or item["rmse"] < best["rmse"]:
            best = item
    assert best is not None
    reference_peak = float(np.max(np.abs(reference)))
    candidate_peak = float(np.max(np.abs(candidate)))
    best.update(
        {
            "neuron_abs_peak": reference_peak,
            "braincell_abs_peak": candidate_peak,
            "peak_relative_error": abs(candidate_peak - reference_peak)
            / max(reference_peak, 1.0e-15),
        }
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neuron",
        type=Path,
        default=ROOT / "data/reference/neuron_goc_contact.npz",
    )
    parser.add_argument(
        "--braincell",
        type=Path,
        default=ROOT / "data/reference/braincell_goc_contact.npz",
    )
    parser.add_argument(
        "--neuron-baseline",
        type=Path,
        default=ROOT / "data/reference/neuron_goc_contact_baseline.npz",
    )
    parser.add_argument(
        "--braincell-baseline",
        type=Path,
        default=ROOT / "data/reference/braincell_goc_contact_baseline.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/export/goc_contact_comparison.json",
    )
    parser.add_argument("--max-lag-step", type=int, default=2)
    args = parser.parse_args()

    with (
        np.load(args.neuron) as neuron,
        np.load(args.braincell) as braincell,
        np.load(args.neuron_baseline) as neuron_baseline,
        np.load(args.braincell_baseline) as braincell_baseline,
    ):
        identifiers = {}
        for field in ("source_glomerulus_id", "source_golgi_id", "target_compartment"):
            left = int(np.asarray(neuron[field]).reshape(()))
            right = int(np.asarray(braincell[field]).reshape(()))
            if left != right:
                raise ValueError(f"Reference ID mismatch for {field}: {left} != {right}")
            identifiers[field] = left

        mechanism_metrics = {}
        mechanism_passed = True
        for field in (
            "ampa_g_uS",
            "nmda_g_uS",
            "ampa_current_nA",
            "nmda_current_nA",
        ):
            reference, candidate = _on_candidate_grid(neuron, braincell, field)
            metric = _best_alignment(reference, candidate, args.max_lag_step)
            peak_limit = 0.05 if field == "nmda_current_nA" else 0.01
            metric["thresholds"] = {
                "peak_relative_error": peak_limit,
                "normalized_rmse": 0.01,
            }
            metric["passed"] = (
                metric["peak_relative_error"] <= peak_limit
                and metric["normalized_rmse"] <= 0.01
            )
            mechanism_passed = mechanism_passed and metric["passed"]
            mechanism_metrics[field] = metric

        evoked_voltage_metrics = {}
        evoked_passed = True
        intrinsic_baseline_metrics = {}
        for field in ("v_soma_mV", "v_contact_mV"):
            neuron_stim, braincell_stim = _on_candidate_grid(neuron, braincell, field)
            neuron_base, braincell_base = _on_candidate_grid(
                neuron_baseline,
                braincell_baseline,
                field,
            )
            neuron_evoked = neuron_stim - neuron_base
            braincell_evoked = braincell_stim - braincell_base
            metric = _best_alignment(neuron_evoked, braincell_evoked, 0)
            metric["thresholds"] = {
                "peak_relative_error": 0.02,
                "normalized_rmse": 0.02,
            }
            metric["passed"] = (
                metric["peak_relative_error"] <= 0.02
                and metric["normalized_rmse"] <= 0.02
            )
            evoked_passed = evoked_passed and metric["passed"]
            evoked_voltage_metrics[field] = metric

            difference = neuron_base - braincell_base
            intrinsic_baseline_metrics[field] = {
                "rmse_mV": float(np.sqrt(np.mean(difference**2))),
                "max_abs_mV": float(np.max(np.abs(difference))),
                "neuron_end_mV": float(neuron_base[-1]),
                "braincell_end_mV": float(braincell_base[-1]),
                "gated": False,
                "note": "Reported separately; intrinsic GoC parity is not yet accepted.",
            }

    response_passed = mechanism_passed and evoked_passed
    report = {
        "identifiers": identifiers,
        "mechanism_metrics": mechanism_metrics,
        "evoked_voltage_metrics": evoked_voltage_metrics,
        "intrinsic_baseline_metrics": intrinsic_baseline_metrics,
        "mechanism_and_evoked_response_passed": response_passed,
        "intrinsic_whole_cell_parity": "unresolved",
        "status": (
            "synaptic_response_passed_intrinsic_baseline_unresolved"
            if response_passed
            else "synaptic_response_failed"
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
    print(encoded)
    if not response_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
