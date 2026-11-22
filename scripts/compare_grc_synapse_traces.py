#!/usr/bin/env python3
"""Compare voltage-clamped BrainCell and NEURON GrC synapse traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _best_alignment(reference: np.ndarray, candidate: np.ndarray, max_lag: int):
    best = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = reference[-lag:], candidate[:lag]
        elif lag > 0:
            left, right = reference[:-lag], candidate[lag:]
        else:
            left, right = reference, candidate
        rmse = float(np.sqrt(np.mean((left - right) ** 2)))
        max_abs = float(np.max(np.abs(left - right)))
        if best is None or rmse < best["rmse"]:
            best = {"lag_step": lag, "rmse": rmse, "max_abs": max_abs}
    assert best is not None
    scale = max(float(np.max(np.abs(reference))), 1.0e-15)
    best["normalized_rmse"] = best["rmse"] / scale
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("neuron", type=Path)
    parser.add_argument("braincell", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-lag-step", type=int, default=2)
    parser.add_argument("--max-peak-relative-error", type=float, default=0.01)
    parser.add_argument("--max-normalized-rmse", type=float, default=0.01)
    args = parser.parse_args()

    with np.load(args.neuron) as source, np.load(args.braincell) as target:
        keys = tuple(
            key
            for key in (
                "transmitter_mM",
                "open",
                "open_alpha6",
                "g_alpha1_pS",
                "g_alpha6_pS",
                "g_pS",
                "current_nA",
            )
            if key in source.files and key in target.files
        )
        metrics = {}
        passed = True
        for key in keys:
            reference = np.asarray(source[key], dtype=float)
            candidate = np.asarray(target[key], dtype=float)
            if reference.shape != candidate.shape:
                raise ValueError(
                    f"Trace {key!r} shape mismatch: {reference.shape} != {candidate.shape}."
                )
            aligned = _best_alignment(reference, candidate, args.max_lag_step)
            reference_peak = float(np.max(np.abs(reference)))
            candidate_peak = float(np.max(np.abs(candidate)))
            peak_relative_error = abs(candidate_peak - reference_peak) / max(
                reference_peak, 1.0e-15
            )
            item_passed = (
                peak_relative_error <= args.max_peak_relative_error
                and aligned["normalized_rmse"] <= args.max_normalized_rmse
            )
            passed = passed and item_passed
            metrics[key] = {
                **aligned,
                "reference_abs_peak": reference_peak,
                "braincell_abs_peak": candidate_peak,
                "peak_relative_error": peak_relative_error,
                "passed": item_passed,
            }

    report = {
        "neuron": str(args.neuron),
        "braincell": str(args.braincell),
        "thresholds": {
            "max_lag_step": args.max_lag_step,
            "max_peak_relative_error": args.max_peak_relative_error,
            "max_normalized_rmse": args.max_normalized_rmse,
        },
        "metrics": metrics,
        "passed": passed,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
